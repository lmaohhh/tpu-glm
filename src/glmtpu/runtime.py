"""pmap runtime for glm5_next TPU serving.

Structure (all static shapes, ~6 compiled executables):
  site_kda(p, streams, valid, rec, conv)         per KDA layer
  site_dsa(p, streams, valid, kv, bitmap, cl)    per DSA layer
  site_dense(p, streams)                          per dense-MLP layer
  site_moe(p, bank, streams [, collect_ids])      per MoE layer

- p (params), rec/conv/kv/bitmap (states), bank: per-chip shards scattered by
  pmap; streams/valid: replicated between sites.
- collect_ids: MoE site returns router ids [B,S,K] for host bank management.
- Prefill: chunks of 256 left-padded; MoE = full sweep (disjoint bank passes
  until every expert has been resident; exact for any routing).
- Decode: one token; hot banks (n_slots >= top_k * d) with exact two-phase
  refresh (snapshot state -> run -> if misses, refresh + re-run).
- Embeddings: host (numpy gather + zero-pad masking).  lm_head: host matmul.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, pmap

from .layers import (attn_site_dsa, attn_site_kda, ffn_site_dense,
                     ffn_site_moe, moe_router, rms_norm)
from .config import GlmConfig


def _dg(a):
    return np.asarray(jax.device_get(a))


class Runner:
    # ================================================================ setup
    def __init__(self, cfg: GlmConfig, params_by_chip, embed_np, lm_head_np,
                 expert_host, log=print):
        self.cfg = cfg
        self.log = log
        self.devs = jax.devices()
        self.d = len(self.devs)
        self.expert_host = expert_host
        self.embed_np = embed_np.astype(np.float32)   # host embed (f32 copy)
        self.lm_head_np = lm_head_np.astype(np.float32)

        # ---- place dense params per chip (pmap-scattered by leading axis) ----
        # params_by_chip: list of dicts (one per chip):
        #   {l: {"attn_hc":…, "ffn_hc":…, "input_ln":…, "post_ln":…,
        #        "kda"/"dsa"/"moe"/"mlp": {...}}}
        stacked = {}   # name -> list of per-chip values (unused; kept for clarity)
        self.P = {}    # pmap-ready param pytree: {l: {site: pytree}}

        def to_jax(v):
            if isinstance(v, np.ndarray):
                return jnp.asarray(v)
            if isinstance(v, dict):
                return {k: to_jax(x) for k, x in v.items()}
            return v

        # build per-layer stacked trees: {l: {site: [chip0_val, ...chip7]}}
        self.PL = {}
        for l in range(cfg.n_layers):
            per_chip = [params_by_chip[c][l] for c in range(self.d)]
            tree = {}
            for key in per_chip[0]:
                vals = [pc[key] for pc in per_chip]
                if isinstance(vals[0], dict):
                    tree[key] = {sub: [v[sub] for v in vals] for sub in vals[0]}
                else:
                    tree[key] = vals
            self.PL[l] = tree
        # to jax arrays with leading device axis: stack leaves across chips
        def stack_tree(chip_dicts):
            return jax.tree.map(lambda *xs: jnp.stack(
                [jnp.asarray(x) for x in xs], axis=0), *chip_dicts)

        self.PJ = {l: stack_tree([params_by_chip[c][l] for c in range(self.d)])
                   for l in range(cfg.n_layers)}

        # replicated final_ln
        self.final_ln = jnp.asarray(params_by_chip[0]["final_ln"])

        # states
        self.sharding_tp = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(self.devs), ("tp",)),
            jax.sharding.PartitionSpec("tp"))
        self.sharding_rep = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(self.devs), ("tp",)),
            jax.sharding.PartitionSpec())
        self.state = self._init_state()
        self.n_slots = cfg.n_slots
        self.banks = {}          # l -> pmap-scattered bank pytree
        self.bank_ids = {}       # l -> [per-chip ids list]
        self._last_routed = {}   # l -> last chunk routed ids (np [B,S,K])
        self._init_banks()

        # compiled sites
        self._compile()

    # ------------------------------------------------------------- states
    def _init_state(self):
        """States as {key: [per-layer] -> per-chip shard list}.  Each entry
        states[key][l] is a list of d arrays (one per device) or a single
        device_put-per-device pytree; pmap scatters the stacked axis 0."""
        cfg = self.cfg
        B = 1
        nh_l = -(-cfg.n_kda_heads // self.d)
        qkvd_l = nh_l * cfg.kda_head_dim
        rec = [self._shard(np.zeros((B, nh_l, cfg.kda_head_dim,
                                     cfg.kda_head_dim), np.float32))
               for _ in range(cfg.n_layers)]
        conv = [self._shard(np.zeros((B, 3 * qkvd_l, cfg.conv_kernel - 1),
                                     np.float32))
                for _ in range(cfg.n_layers)]
        kv = [self._shard(np.zeros((B, cfg.max_ctx, cfg.kv_lora_rank),
                                   np.float32))
              for _ in range(len(cfg.dsa_layers))]
        bm = [self._shard(np.zeros((B, cfg.max_ctx), np.float32))
              for _ in range(len(cfg.dsa_layers))]
        return {"rec": rec, "conv": conv, "kv": kv, "bitmap": bm,
                "cache_len": 0}

    def _shard(self, arr):
        """Per-chip state shards stacked on axis 0, sharded P('tp') (pmap
        semantics: device i reads slice i; replication = identical slices)."""
        return jax.device_put(np.stack([arr] * self.d), self.sharding_tp)

    def _put(self, arr):
        """Replicated across chips (P())."""
        return jax.device_put(np.stack([arr] * self.d), self.sharding_rep)

    # ------------------------------------------------------------- banks
    def _bank_np(self, layer, ids_per_chip):
        I, D = self.cfg.moe_inter, self.cfg.hidden_size
        n = self.n_slots
        outs = []
        for dev_ids in ids_per_chip:
            gu = np.zeros((n, 2 * I, D), np.uint8)
            gus = np.ones((n, -(-2 * I // 128), -(-D // 128)), np.float32)
            dd = np.zeros((n, D, I), np.uint8)
            dds = np.ones((n, -(-D // 128), -(-I // 128)), np.float32)
            ids_arr = np.asarray(dev_ids, np.int32)
            for s in range(n):
                e = int(ids_arr[s]) if s < len(ids_arr) else -1
                if e < 0:
                    continue
                ex = self.expert_host[(layer, e)]
                gu[s] = ex["gu"]          # fused gate|up [2I, D]
                gus[s] = ex["gu_s"]       # fused scale grid
                dd[s] = ex["d"]
                dds[s] = ex["d_s"]
            outs.append({"gu": gu, "gu_s": gus, "d": dd, "d_s": dds,
                         "ids": ids_arr})
        return outs

    def _install_bank(self, layer, ids_per_chip):
        per_chip = self._bank_np(layer, ids_per_chip)
        # single pytree with leading device axis (like PJ params)
        self.banks[layer] = jax.device_put(
            {k: np.stack([pc[k] for pc in per_chip], axis=0)
             for k in per_chip[0]},
            self.sharding_tp)
        self.bank_ids[layer] = [list(map(int, pc["ids"])) for pc in per_chip]

    def _init_banks(self):
        for l in self.cfg.moe_layers:
            self._install_bank(l, [[-1] * self.n_slots for _ in range(self.d)])

    # ------------------------------------------------------------- compile
    def _compile(self):
        cfg = self.cfg
        self.site_kda = pmap(
            lambda p, streams, valid, rec, conv:
                attn_site_kda(p, streams, valid, rec, conv, cfg),
            axis_name="tp")
        self.site_dsa = pmap(
            lambda p, streams, valid, kv, bm, cl:
                attn_site_dsa(p, streams, valid, kv, bm, cl, cfg),
            axis_name="tp")
        self.site_dense = pmap(
            lambda p, streams: ffn_site_dense(p, streams, cfg),
            axis_name="tp")
        self._site_moe_plain = pmap(
            lambda p, bank, streams: ffn_site_moe(p, bank, streams, cfg),
            axis_name="tp")
        self._site_moe_collect = pmap(
            lambda p, bank, streams: ffn_site_moe(p, bank, streams, cfg,
                                                  collect_router=True),
            axis_name="tp")

        def _moe(p, bank, streams, collect_router=False):
            if collect_router:
                return self._site_moe_collect(p, bank, streams)
            return self._site_moe_plain(p, bank, streams)
        self.site_moe = _moe

        # final: mean over streams + final norm (single chip 0; replicated)
        self._final = jax.jit(lambda streams, w: rms_norm(
            jnp.mean(streams, axis=2), w, cfg.rms_norm_eps))

    # ------------------------------------------------------------- helpers
    def _run_attn_site(self, l, streams, valid):
        cfg = self.cfg
        p = self.PJ[l]
        if cfg.is_kda(l):
            rec = self.state["rec"][l]        # [d, B, nh_l, hd, hd]
            conv = self.state["conv"][l]
            streams, rec, conv = self.site_kda(p, streams, valid, rec, conv)
            self.state["rec"][l] = rec
            self.state["conv"][l] = conv
        else:
            di = cfg.dsa_layers.index(l)
            kv = self.state["kv"][di]
            bm = self.state["bitmap"][di]
            cl = jax.device_put(
                np.stack([np.asarray(self.state["cache_len"], np.int32)] * self.d),
                self.sharding_tp)
            streams, kv, bm = self.site_dsa(p, streams, valid, kv, bm, cl)
            self.state["kv"][di] = kv
            self.state["bitmap"][di] = bm
        return streams

    def _run_ffn_site(self, l, streams, collect_ids=False):
        cfg = self.cfg
        p = self.PJ[l]
        if cfg.is_moe(l):
            if collect_ids:
                streams, ids = self.site_moe(p, self.banks[l], streams,
                                             collect_router=True)
                return streams, ids
            return self.site_moe(p, self.banks[l], streams)
        return self.site_dense(p, streams)

    # ------------------------------------------------------------- prefill
    def reset(self):
        """Fresh recurrent/KV state (prefill starts a new sequence)."""
        self.state = self._init_state()
        self._last_routed = {}

    def prefill(self, tokens, collect_last_ids=False):
        cfg = self.cfg
        self.reset()
        S = cfg.prefill_chunk
        n = len(tokens)
        pad_total = (-n) % S if n % S else 0
        pad_total = (-n) % S
        toks = [0] * pad_total + list(tokens)
        assert len(toks) % S == 0

        last_ids = None
        for ci in range(0, len(toks), S):
            chunk = toks[ci:ci + S]
            n_real = min(S, n - (ci - pad_total))
            valid_l = [0.0] * (S - n_real) + [1.0] * n_real
            x = self.embed_np[np.asarray(chunk, np.int32)] * \
                np.asarray(valid_l, np.float32)[:, None]
            # streams: replicated [d, B, S, H, D] via identical slices P('tp')
            streams = np.broadcast_to(
                x[None, :, None, :], (1, S, cfg.hc_mult, cfg.hidden_size))
            streams = jax.device_put(
                np.stack([np.ascontiguousarray(streams, dtype=np.float32)] * self.d),
                self.sharding_tp)

            valid = jax.device_put(
                np.stack([np.asarray(valid_l, np.float32).reshape(1, S)] * self.d),
                self.sharding_tp)

            for l in range(cfg.n_layers):
                streams = self._run_attn_site(l, streams, valid)
                if l in cfg.moe_layers:
                    # exact routed-expert compute: full sweep of ALL experts
                    # in disjoint bank passes (sum over passes = exact MoE)
                    streams, ids = self._prefill_moe(l, streams)
                    self._last_routed[l] = _dg(ids[0])     # [B,S,K] chip 0
                else:
                    streams = self._run_ffn_site(l, streams)

            # final: mean + final_ln -> host
            h_last = self._final(streams, self.final_ln)   # [d, B, S, D]
            self.state["cache_len"] += n_real
            h_last_np = _dg(h_last[0, 0, -1])              # [D] last real token
            self._last_hidden = h_last_np

        self._refresh_all_banks_for_decode()
        return h_last_np

    def _prefill_moe(self, l, streams):
        """Exact MoE during prefill.

        mHC makes the site output affine in the sublayer output:
            site(y) = post * y + comb . streams      (post/comb depend only
                                                      on the INPUT streams)
        and y = shared + moe_pass.  Sweeping disjoint expert banks over
        passes p=1..P:
            sum_p site(shared + moe_p) - (P-1) * site_empty
          = P*post*shared + post*sum_p moe_p + P*(comb.streams)
            - (P-1)*(post*shared + comb.streams)
          = post*(shared + sum_p moe_p) + comb.streams
          = site(shared + ALL routed experts)         <- exact
        where site_empty runs with an empty bank (ids all -1).
        """
        cfg = self.cfg
        E = cfg.n_experts
        step = self.d * self.n_slots
        n_passes = -(-E // step)

        # empty-bank baseline (isolates hc + shared)
        self._install_bank(l, [[-1] * self.n_slots for _ in range(self.d)])
        streams_empty, _ = self._run_ffn_site(l, streams, collect_ids=True)

        total = None
        ids_out = None
        for p_i in range(n_passes):
            start = p_i * step
            ids_per_chip = []
            for dev in range(self.d):
                lo = start + dev * self.n_slots
                hi = min(lo + self.n_slots, E)
                ids_per_chip.append(list(range(lo, hi))
                                    + [-1] * (self.n_slots - max(0, hi - lo)))
            self._install_bank(l, ids_per_chip)
            streams_p, ids = self._run_ffn_site(l, streams, collect_ids=True)
            total = streams_p if total is None else total + streams_p
            ids_out = ids

        result = total - (n_passes - 1) * streams_empty
        return result, ids_out

    def _refresh_all_banks_for_decode(self):
        """Populate decode hot banks after prefill.

        Coverage target: the routed ids of the LAST real token (the next
        decode step routes exactly one token); remaining slots filled with
        the chunk's most frequent ids.  Two-phase decode guarantees exactness
        for any later miss, so no capacity error is ever raised."""
        for l in self.cfg.moe_layers:
            ids_l = self._last_routed.get(l)
            if ids_l is None:
                continue
            arr = np.asarray(ids_l).reshape(-1, self.cfg.top_k)  # [S,K]
            last = [int(i) for i in arr[-1]]
            freq = {}
            for i in arr.flatten():
                freq[int(i)] = freq.get(int(i), 0) + 1
            # priority: last token's ids first, then by frequency
            priority = list(dict.fromkeys(last))
            for i, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
                if i not in priority:
                    priority.append(i)
            cap = self.d * self.n_slots
            chosen = priority[:cap]
            per_chip = [chosen[c::self.d][:self.n_slots] for c in range(self.d)]
            # pad short chips with -1
            per_chip = [pc + [-1] * (self.n_slots - len(pc)) for pc in per_chip]
            self._install_bank(l, per_chip)

    # ------------------------------------------------------------- decode
    def _decode_one(self, token, temperature, top_p):
        cfg = self.cfg
        # embed + streams (single token, always valid)
        x = self.embed_np[token]                     # [D]
        streams = np.zeros((1, 1, cfg.hc_mult, cfg.hidden_size), np.float32)
        streams[0, 0] = x[None, None, :]
        streams = jax.device_put(
            np.stack([streams] * self.d), self.sharding_tp)
        valid = jax.device_put(
            np.stack([np.ones((1, 1), np.float32)] * self.d), self.sharding_tp)

        # exact decode via bank-coverage fixpoint:
        #   run -> collected routed ids must be covered by the banks USED in
        #   that run; if not, refresh banks to cover them, roll back state,
        #   repeat.  Monotone: layer 1's router is always exact (no MoE
        #   before it), and once layers 1..j are exact they stay exact (true
        #   ids stay resident via need+old refresh), so the exact prefix
        #   grows by >= 1 per iteration -> converges in <= n_moe iterations
        #   (1-2 in practice after prefill warm-up).
        max_iters = len(cfg.moe_layers) + 1
        for it in range(max_iters):
            snap = self._snapshot_state()
            hidden, routed = self._run_all_layers(streams, valid, collect=True)
            missing = self._missing_banks(routed)
            if not missing:
                logits = self._lm_head(hidden)
                return logits
            self._refresh_hot_banks(missing)
            self._restore_state(snap)
        raise RuntimeError("decode fixpoint did not converge "
                           f"in {max_iters} iterations")

    def _run_all_layers(self, streams, valid, collect):
        routed = {}
        for l in range(self.cfg.n_layers):
            streams = self._run_attn_site(l, streams, valid)
            if self.cfg.is_moe(l) and collect:
                streams, ids = self._run_ffn_site(l, streams, collect_ids=True)
                routed[l] = _dg(ids[0])
            else:
                streams = self._run_ffn_site(l, streams)
        h = self._final(streams, self._final_ln_arg())
        return h, routed

    def _final_ln_arg(self):
        return self.final_ln

    def _missing_banks(self, routed):
        miss = {}
        for l, ids in routed.items():
            have = set()
            for pc in self.bank_ids[l]:
                have.update(e for e in pc if e >= 0)
            need = set(int(i) for i in np.asarray(ids).flatten())
            if not need <= have:
                miss[l] = need
        return miss

    def _refresh_hot_banks(self, missing):
        for l, need in missing.items():
            have = [e for pc in self.bank_ids[l] for e in pc if e >= 0]
            old = list(dict.fromkeys(have))  # LRU order preserved
            new_ids = list(dict.fromkeys(list(need) + old))[: self.d * self.n_slots]
            per_chip = [new_ids[c::self.d] for c in range(self.d)]
            self._install_bank(l, per_chip)

    def _snapshot_state(self):
        """Shallow copy: per-layer entries are immutable jax arrays, so
        replacing the list entries (not mutating) makes rollback exact."""
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in self.state.items()}

    def _restore_state(self, snap):
        self.state = {k: v for k, v in snap.items()}

    # ------------------------------------------------------------- logits
    def _lm_head(self, hidden):
        """hidden: [d, B, S, D] final-normed (or [d, B, D]); returns [V]."""
        h = _dg(hidden)
        h = h.reshape(-1, h.shape[-1])[-1]        # last row = last token
        return h @ self.lm_head_np.T              # [V]

    # ------------------------------------------------------------- gen
    def generate(self, tokens, max_new_tokens=256, temperature=0.7, top_p=0.95,
                 stop_ids=None, on_token=None):
        logits = self.prefill(tokens)
        out = []
        for i in range(max_new_tokens):
            t = self._sample(logits, temperature, top_p)
            if stop_ids and t in stop_ids:
                break
            out.append(t)
            if on_token:
                on_token(t)
            if i + 1 < max_new_tokens:
                logits = self._decode_one(t, temperature, top_p)
        return out

    def _sample(self, logits, temperature, top_p):
        v = np.asarray(logits).reshape(-1)
        if temperature <= 1e-6:
            return int(np.argmax(v))
        v = v / temperature
        v = v - v.max()
        p = np.exp(v)
        p = p / p.sum()
        if top_p and top_p < 1.0:
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            cut = np.searchsorted(cum, top_p) + 1
            keep = order[:cut]
            p2 = p[3 if False else keep]
            p2 = p[keep] / p[keep].sum()
            return int(np.random.choice(keep, p=p2))
        return int(np.random.choice(len(p), p=p))
