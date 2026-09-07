"""pmap runtime for DeepSeek-V4-Flash TPU serving (de-simplified).

Sites (per layer type; static shapes; prefill-S and decode-S=1 compile
separately):
  site_attn(p, streams, valid, pos0, <layer state>)   sliding/CSA/HCA
  site_ffn(p, bank, streams, input_ids [, collect])   MoE (every layer)
  site_mtp(p, bank, streams_h, embed, id, pos0, ring) MTP draft layer
  _final(streams, ...)                                hc_head collapse

Per-layer KV state (replicated — MQA; every chip needs all KV):
  ring:   u8 [B,W,448] + scales f32 [B,W,7] + rope bf16 [B,W,64]
  comp:   u8 [B,C,448] + scales [B,C,7] + rope [B,C,64]  (C=max_ctx//r)
  idxr:   bf16 [B,C,128] keys-only (CSA layers)
  cstate: compressor carry (kv, score) [B, coff*ratio, coff*Dc] f32
  icstate: indexer compressor carry (CSA layers)
Positions tracked host-side (cache_len); passed as traced pos0.

Prefill: chunks of cfg.prefill_chunk (multiple of every ratio), MoE =
affine-correction full sweep (disjoint banks; exact for any routing).
Decode: 1 token; hot banks + exact two-phase refresh fixpoint
(snapshot -> run -> miss? refresh + rollback + re-run).
MTP-1: draft(streams, next_token) -> logits; verify loop in generate().
"""
from __future__ import annotations

import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, pmap

from . import dsv4_layers as dl
from .dsv4_config import Dsv4Config
from .dsv4_fp4 import dequant_jax, fp4_sim_jax


def _dg(a):
    return np.asarray(jax.device_get(a))


# ===========================================================================
# attention core (one chip, local heads)
# ===========================================================================

def _attn_core(p, streams, valid, pos0, ring, cstate, comp, idxr, icstate,
               cfg, d, mf, cf, ratio):
    """One attention site on one chip (local heads = n_heads // d).
    ring = (u8, s, rope); cstate = (kv_state, sc_state);
    comp = (u8, s, rope); idxr [B,C,ih]; icstate = (kv, sc).
    ratio: STATIC int.  Returns (streams', new_state tuple)."""
    post, comb, collapsed = dl.hc_site(p["attn_hc"], streams, cfg)
    h = dl.rms_norm(collapsed, p["attn_norm"], cfg.rms_norm_eps)
    B, S = h.shape[0], h.shape[1]
    Hl = cfg.n_heads // d
    dh = cfg.head_dim
    W = cfg.window_size
    freqs = cf if ratio else mf
    pos = pos0 + jnp.arange(S)

    # ---- q: low-rank + per-head RMSNorm + rope (trailing rd dims) ----
    qr = dl.rms_norm(h @ p["wq_a"].astype(jnp.bfloat16).T, p["q_norm"],
                     cfg.rms_norm_eps)
    q = (qr @ p["wq_b"].astype(jnp.bfloat16).T).reshape(B, S, Hl, dh)
    q32 = q.astype(jnp.float32)
    q32 = q32 * lax.rsqrt(jnp.mean(q32 * q32, -1, keepdims=True)
                          + cfg.rms_norm_eps)
    q = dl.apply_rope(q32.astype(jnp.bfloat16), pos, freqs)

    # ---- shared-KV MQA ----
    kv = dl.rms_norm(h @ p["wkv"].astype(jnp.bfloat16).T, p["kv_norm"],
                     cfg.rms_norm_eps)
    kv = dl.apply_rope(kv, pos, freqs)

    # ---- ring write (chunk's last min(S, W) tokens) ----
    u8, sc, rp = dl.kv_pack(kv, cfg.rope_head_dim)
    n_write = min(S, W)
    u8w, scw = u8[:, -n_write:], sc[:, -n_write:]
    rpw = rp[:, -n_write:]
    slots = (pos0 + S - n_write + jnp.arange(n_write)) % W
    r_u8 = ring[0].at[:, slots].set(u8w)
    r_s = ring[1].at[:, slots].set(scw)
    r_r = ring[2].at[:, slots].set(rpw.astype(jnp.float32))

    # ---- ring entry positions ----
    last = pos0 - 1
    g = last - ((last - jnp.arange(W)) % W)
    g = jnp.where(g >= 0, g, -1)

    ring_kv = dl.kv_unpack(ring[0], ring[1], ring[2].astype(jnp.bfloat16))
    ctx_kv = jnp.concatenate([ring_kv, kv], axis=1)
    ctx_pos = jnp.concatenate([g, pos])
    W_ = ctx_pos.shape[0]

    c_u8, c_s, c_r = comp[0], comp[1], comp[2]
    idx_new = idxr
    in_state = icstate
    cnk, cns = cstate
    comp_kv_sel = None

    if ratio:
        # ------------- compressor (CSA ratio 4 / HCA 8|128) -------------
        if S == 1:
            ent, should, (nk, ns) = dl.compressor_decode(
                p["comp"], h, cstate, pos0, cfg)
            n_new = 1
        else:
            ent, (nk, ns) = dl.compressor_prefill(
                p["comp"], h, cstate, pos0, cfg)
            n_new = ent.shape[1]
        # compressed-entry positions: e*ratio for the n_new new entries
        cpos = (pos0 // ratio + jnp.arange(n_new)) * ratio
        ent = dl.apply_rope(ent, cpos, cf)
        e0 = pos0 // ratio
        cu8, csc, crp = dl.kv_pack(ent, cfg.rope_head_dim)
        c_u8 = lax.dynamic_update_slice(comp[0], cu8, (0, e0, 0))
        c_s = lax.dynamic_update_slice(comp[1], csc, (0, e0, 0))
        c_r = lax.dynamic_update_slice(
            comp[2], crp.astype(jnp.float32), (0, e0, 0))
        comp_kv = dl.kv_unpack(c_u8, c_s, c_r)          # [B,C,dh]
        comp_idx = jnp.arange(comp_kv.shape[1])
        cnk, cns = nk, ns
        Ctot = comp_kv.shape[1]

        if ratio == 4:
            # ---------------- indexer (CSA) ----------------
            if S == 1:
                ient, ish, (ink, ins) = dl.compressor_decode(
                    p["icomp"], h, icstate, pos0, cfg)
                n_in = 1
            else:
                ient, (ink, ins) = dl.compressor_prefill(
                    p["icomp"], h, icstate, pos0, cfg)
                n_in = ient.shape[1]
            ipos = (pos0 // ratio + jnp.arange(n_in)) * ratio
            ient = dl.apply_rope(ient, ipos, cf)
            ih = dl.hadamard(ient)
            ihq = fp4_sim_jax(ih.astype(jnp.float32), 32)
            i0 = pos0 // ratio
            idx_new = lax.dynamic_update_slice(
                idxr, ihq.astype(jnp.float32), (0, i0, 0))
            iscores = _indexer_scores(
                p["idx"], qr, h, idx_new, pos0, cfg, d, cf)
            thr = (pos[:, None] + 1) // ratio
            valid_e = comp_idx[None, :] < thr
            iscores = jnp.where(valid_e, iscores, -1e30)
            k = min(cfg.index_topk, Ctot)
            tvals, sel = lax.top_k(iscores, k)          # [B,S,k]
            sel = jnp.where(tvals > -1e29, sel, -1)
            safe = jnp.where(sel >= 0, sel, 0)
            # gath[b,s,j,:] = comp_kv[b, safe[b,s,j], :]
            bb = jnp.broadcast_to(jnp.arange(B)[:, None, None],
                                  (B, S, k))
            gath = comp_kv[bb, safe]                     # [B,S,k,dh]
            gath = jnp.where((sel >= 0)[:, :, :, None], gath, 0.0)
            gath_pos = comp_idx[safe]                  # [B,S,k]
            gath_pos = jnp.where(sel >= 0, gath_pos, -1)
            comp_kv_sel = gath.reshape(B, S * k, dh)
            comp_pos_sel = gath_pos.reshape(B, S * k)
            in_state = (ink, ins)
        else:
            comp_kv_sel = comp_kv
            comp_pos_sel = jnp.broadcast_to(comp_idx[None, :],
                                            (B, Ctot))

        kv_sel = jnp.concatenate([ctx_kv, comp_kv_sel], axis=1)
        pos_sel = jnp.concatenate(
            [ctx_pos,
             comp_pos_sel[0] if comp_pos_sel.ndim == 2 else comp_pos_sel],
            axis=0)
        is_comp = jnp.concatenate([jnp.zeros(W_, jnp.bool_),
                                   jnp.ones(kv_sel.shape[1] - W_,
                                            jnp.bool_)])
    else:
        kv_sel, pos_sel = ctx_kv, ctx_pos
        is_comp = jnp.zeros(kv_sel.shape[1], jnp.bool_)

    N = kv_sel.shape[1]
    pq = pos[:, None]                                  # [S,1]
    pk = pos_sel[None, :]                              # [1,N]
    causal = pk <= pq                                  # [S,N]
    in_win = (pk > pq - W) & (pk >= 0)
    vis = jnp.where(is_comp[None, :], causal, causal & in_win)
    valid_ctx = jnp.broadcast_to(vis[None], (B, S, N))

    o = dl.sparse_attn_core(q, kv_sel, valid_ctx.astype(jnp.float32),
                            p["attn_sink"], dh ** -0.5)
    o = dl.apply_rope(o, pos, freqs, inverse=True)
    out = dl.grouped_o_proj(o, p["wo_a"], p["wo_b"])
    out = lax.psum(out, "tp")
    streams2 = dl.hc_apply(post, comb, out.astype(streams.dtype), streams)
    streams2 = streams2 * valid[..., None, None].astype(streams2.dtype)

    new_state = (r_u8, r_s, r_r, c_u8, c_s, c_r, idx_new,
                 in_state[0], in_state[1], cnk, cns)
    return streams2, new_state


def _indexer_scores(p_idx, qr, x, idx_cache, pos0, cfg, d, cf):
    """Head-sharded indexer scores (psum over chips).  qr [B,S,ql];
    idx_cache [B,Ci,ih] bf16 -> scores [B,S,Ci] f32."""
    q = qr @ p_idx["wq_b"].astype(jnp.bfloat16).T
    Hl = p_idx["wq_b"].shape[0] // cfg.index_head_dim
    q = q.reshape(qr.shape[0], qr.shape[1], Hl, cfg.index_head_dim)
    pos = pos0 + jnp.arange(qr.shape[1])
    q = dl.apply_rope(q, pos, cf)
    q = dl.hadamard(q)
    q = fp4_sim_jax(q.astype(jnp.float32), 32).astype(jnp.bfloat16)
    w = (x @ p_idx["weights_proj"].astype(jnp.bfloat16).T) \
        * (cfg.index_head_dim ** -0.5 * (Hl * d) ** -0.5)
    s = jnp.einsum("bshd,btd->bsht", q.astype(jnp.float32),
                   idx_cache.astype(jnp.float32))
    s = jnp.maximum(s, 0.0)
    s = jnp.einsum("bsht,bsh->bst", s, w.astype(jnp.float32))
    return lax.psum(s, "tp")


_INDEX_FREQS = None   # set by runner at compile time (module-level hack
                     # avoided: passed via p dict below instead)



def _hc_head_final(streams, hfn, hbase, hscale, w, cfg):
    """Model head: hc_head collapse (sigmoid pre-weights, no Sinkhorn) +
    final RMSNorm.  streams [..., Hc, D] (any leading axes)."""
    lead = streams.shape[:-2]
    H, D = streams.shape[-2:]
    flat = streams.reshape(*lead, H * D).astype(jnp.float32)
    flat = dl.rms_norm_no_w(flat, cfg.rms_norm_eps)
    mixes = flat @ hfn.astype(jnp.float32).T
    pre = jax.nn.sigmoid(mixes * hscale[0] + hbase) + cfg.hc_eps
    y = jnp.sum(pre[..., None] * streams, axis=2)
    return dl.rms_norm(y.astype(streams.dtype), w, cfg.rms_norm_eps)


class Dsv4Runner:
    # ================================================================ setup
    def __init__(self, cfg: Dsv4Config, params_by_chip, embed_np, lm_head_np,
                 expert_host, log=print):
        global _INDEX_FREQS
        self.cfg = cfg
        self.log = log
        self.devs = jax.devices()
        self.d = len(self.devs)
        self.expert_host = expert_host
        self.embed_np = embed_np.astype(np.float32)
        self.lm_head_np = lm_head_np.astype(np.float32)
        self.PJ = {}
        for l in range(cfg.n_layers):
            self.PJ[l] = self._stack(
                [params_by_chip[c][l] for c in range(self.d)])
        self.PJ["mtp"] = self._stack(
            [params_by_chip[c]["mtp"] for c in range(self.d)])
        self.head_params = params_by_chip[0]["head"]
        self.final_norm = jnp.asarray(params_by_chip[0]["final_ln"])
        self.sharding_tp = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(self.devs), ("tp",)),
            jax.sharding.PartitionSpec("tp"))
        self.sharding_rep = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(self.devs), ("tp",)),
            jax.sharding.PartitionSpec())
        self.n_slots = cfg.n_slots
        self.banks = {}
        self.bank_ids = {}
        self._last_routed = {}
        self._build_freqs()
        _INDEX_FREQS = self.cf
        self.state = self._init_state()
        self._init_banks()
        self._build_sites()

    def _build_freqs(self):
        cfg = self.cfg
        self.mf = jnp.asarray(dl.yarn_freqs(cfg, False))
        self.cf = jnp.asarray(dl.yarn_freqs(cfg, True))

    def _mark(self, chip_lay, l):
        out = dict(chip_lay)
        out["_l"] = l if isinstance(l, int) else -1   # mtp = -1
        return out

    def _stack(self, chip_dicts):
        """Stack leaves across chips on a new leading axis.  None leaves
        (unused compressor/indexer on sliding layers) become dummy [1]
        arrays; the int layer marker is kept un-stacked."""
        def st(*xs):
            if xs[0] is None:
                return jnp.zeros((len(xs), 1), jnp.float32)
            return jnp.stack([jnp.asarray(x) for x in xs], axis=0)
        tree = jax.tree.map(st, *chip_dicts,
                            is_leaf=lambda t: t is None)
        return tree

    # ------------------------------------------------------------- states
    def _init_state(self):
        cfg = self.cfg
        B, W = 1, cfg.window_size
        st = {"cache_len": 0}
        st["ring"] = []
        st["comp"] = []
        st["idxr"] = []
        st["cstate"] = []
        st["icstate"] = []
        Dc, rd = cfg.head_dim, cfg.rope_head_dim
        for l in range(cfg.n_layers):
            r = cfg.ratio(l)
            # every layer (sliding included) has the W-entry ring
            st["ring"].append((
                self._sh(np.zeros((B, W, Dc - rd), np.uint8)),
                self._sh(np.zeros((B, W, (Dc - rd) // 64), np.float32)),
                self._sh(np.zeros((B, W, rd), np.float32))))
            if r == 0:
                st["comp"].append(None)
                st["idxr"].append(None)
                st["cstate"].append(None)
                st["icstate"].append(None)
                continue
            C = cfg.n_comp(r)
            coff = 2 if r == 4 else 1
            st["comp"].append((
                self._sh(np.zeros((B, C, Dc - rd), np.uint8)),
                self._sh(np.zeros((B, C, (Dc - rd) // 64), np.float32)),
                self._sh(np.zeros((B, C, rd), np.float32))))
            st["cstate"].append((
                self._sh(np.zeros((B, coff * r, coff * Dc), np.float32)),
                self._sh(np.zeros((B, coff * r, coff * Dc), np.float32))))
            if r == 4:
                ic = 2 * cfg.index_head_dim
                st["idxr"].append(self._sh(np.zeros(
                    (B, C, cfg.index_head_dim), np.float32)))
                st["icstate"].append((
                    self._sh(np.zeros((B, 2 * r, ic), np.float32)),
                    self._sh(np.zeros((B, 2 * r, ic), np.float32))))
            else:
                st["idxr"].append(None)
                st["icstate"].append(None)
        st["mtp_ring"] = (
            self._sh(np.zeros(
                (B, W, cfg.head_dim - cfg.rope_head_dim), np.uint8)),
            self._sh(np.zeros(
                (B, W, (cfg.head_dim - cfg.rope_head_dim) // 64),
                np.float32)),
            self._sh(np.zeros((B, W, cfg.rope_head_dim), np.float32)))
        return st

    def _sh(self, arr):
        return jax.device_put(np.stack([arr] * self.d), self.sharding_tp)

    def _sc(self, v):
        return jax.device_put(
            np.full((self.d,), int(v), np.int32), self.sharding_tp)

    # ------------------------------------------------------------- banks
    def _bank_np(self, key, ids_per_chip):
        cfg = self.cfg
        I, D = cfg.moe_inter, cfg.hidden_size
        n = self.n_slots
        outs = []
        for dev_ids in ids_per_chip:
            bank = {
                "w1": np.zeros((n, I, D // 2), np.uint8),
                "w1_s": np.zeros((n, I, D // 32), np.uint8),
                "w3": np.zeros((n, I, D // 2), np.uint8),
                "w3_s": np.zeros((n, I, D // 32), np.uint8),
                "w2": np.zeros((n, D, I // 2), np.uint8),
                "w2_s": np.zeros((n, D, I // 32), np.uint8),
                "ids": np.asarray(dev_ids[:n], np.int32),
            }
            for s in range(min(n, len(dev_ids))):
                e = int(bank["ids"][s])
                if e < 0:
                    continue
                ex = self.expert_host[(key, e)]
                for t in ("w1", "w3", "w2"):
                    bank[t][s] = ex[t]
                    bank[t + "_s"][s] = ex[t + "_s"]
            outs.append(bank)
        return outs

    def _install_bank(self, key, ids_per_chip):
        per_chip = self._bank_np(key, ids_per_chip)
        self.banks[key] = jax.device_put(
            {k: np.stack([pc[k] for pc in per_chip], axis=0)
             for k in per_chip[0]}, self.sharding_tp)
        self.bank_ids[key] = [list(map(int, pc["ids"])) for pc in per_chip]

    def _init_banks(self):
        empty = [[-1] * self.n_slots] * self.d
        for l in range(self.cfg.n_layers):
            self._install_bank(l, empty)
        self._install_bank("mtp", empty)

    # ------------------------------------------------------------- compile
    def _build_sites(self):
        cfg, d = self.cfg, self.d
        mf, cf = self.mf, self.cf

        def attn_fn(p, streams, valid, pos0, ring_u8, ring_s, ring_r,
                    c_kv, c_sc, i_kv, i_sc, comp_u8, comp_s, comp_r, idxr,
                    ratio):
            return _attn_core(p, streams, valid, pos0,
                              (ring_u8, ring_s, ring_r), (c_kv, c_sc),
                              (comp_u8, comp_s, comp_r), idxr,
                              (i_kv, i_sc), cfg, d, mf, cf, ratio)

        def _ffn(p, bank, streams, input_ids, collect=False):
            post, comb, collapsed = dl.hc_site(p["ffn_hc"], streams, cfg)
            h = dl.rms_norm(collapsed, p["ffn_norm"], cfg.rms_norm_eps)
            w, ids = dl.moe_router(p["gate"], h, input_ids, cfg)
            sh = dl.dense_mlp_core(p["shared"], h, cfg)
            moe = dl.fp4_bank_core(bank, h, w, ids, cfg)
            y = sh + moe
            streams2 = dl.hc_apply(post, comb, y.astype(streams.dtype),
                                   streams)
            if collect:
                return streams2, ids
            return streams2

        def mtp_fn(p, bank, streams_h, embed_tok, input_id, pos0,
                   r_u8, r_s, r_r):
            e = dl.rms_norm(embed_tok[None], p["enorm"], cfg.rms_norm_eps)
            hprev = dl.rms_norm(streams_h[:, 0], p["hnorm"],
                                cfg.rms_norm_eps)
            x = (e @ p["e_proj"].astype(jnp.bfloat16).T
                 + hprev @ p["h_proj"].astype(jnp.bfloat16).T)
            streams = jnp.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], cfg.hc_mult,
                 cfg.hidden_size)).astype(streams_h.dtype)
            post, comb, collapsed = dl.hc_site(p["attn_hc"], streams, cfg)
            hn = dl.rms_norm(collapsed, p["attn_norm"], cfg.rms_norm_eps)
            B, S = hn.shape[0], hn.shape[1]
            Hl = cfg.n_heads // d
            dh = cfg.head_dim
            W = cfg.window_size
            pos = pos0 + jnp.arange(S)
            qr = dl.rms_norm(hn @ p["wq_a"].astype(jnp.bfloat16).T,
                             p["q_norm"], cfg.rms_norm_eps)
            q = (qr @ p["wq_b"].astype(jnp.bfloat16).T).reshape(B, S, Hl, dh)
            q32 = q.astype(jnp.float32)
            q32 = q32 * lax.rsqrt(jnp.mean(q32 * q32, -1, keepdims=True)
                                  + cfg.rms_norm_eps)
            q = dl.apply_rope(q32.astype(jnp.bfloat16), pos, mf)
            kv = dl.rms_norm(hn @ p["wkv"].astype(jnp.bfloat16).T,
                             p["kv_norm"], cfg.rms_norm_eps)
            kv = dl.apply_rope(kv, pos, mf)
            u8, sc, rp = dl.kv_pack(kv, cfg.rope_head_dim)
            slots = pos % W
            n_u8 = r_u8.at[:, slots[0]].set(u8[:, 0])
            n_s = r_s.at[:, slots[0]].set(sc[:, 0])
            n_r = r_r.at[:, slots[0]].set(rp[:, 0].astype(jnp.float32))
            ring_kv = dl.kv_unpack(n_u8, n_s, n_r.astype(jnp.bfloat16))
            last = pos0 - 1
            g = last - ((last - jnp.arange(W)) % W)
            g = jnp.where(g >= 0, g, -1)
            vis = (g[None, :] <= pos[:, None]) \
                & (g[None, :] > pos[:, None] - W) & (g[None, :] >= 0)
            valid_ctx = jnp.broadcast_to(vis[None], (B, S, W))
            o = dl.sparse_attn_core(q, ring_kv,
                                    valid_ctx.astype(jnp.float32),
                                    p["attn_sink"], dh ** -0.5)
            o = dl.apply_rope(o, pos, mf, inverse=True)
            out = dl.grouped_o_proj(o, p["wo_a"], p["wo_b"])
            out = lax.psum(out, "tp")
            streams2 = dl.hc_apply(post, comb, out.astype(streams.dtype),
                                   streams)
            post2, comb2, collapsed2 = dl.hc_site(p["ffn_hc"], streams2, cfg)
            hn2 = dl.rms_norm(collapsed2, p["ffn_norm"], cfg.rms_norm_eps)
            w, ids = dl.moe_router(p["gate"], hn2, input_id[:, None], cfg)
            sh = dl.dense_mlp_core(p["shared"], hn2, cfg)
            moe = dl.fp4_bank_core(bank, hn2, w, ids, cfg)
            y = sh + moe
            streams3 = dl.hc_apply(post2, comb2, y.astype(streams2.dtype),
                                   streams2)
            hcol = _hc_head_final(streams3, p["hc_head_fn"],
                                  p["hc_head_base"], p["hc_head_scale"],
                                  p["norm"], cfg)
            return hcol, ids, (n_u8, n_s, n_r)

        self.site_attn = pmap(attn_fn, axis_name="tp",
                              static_broadcasted_argnums=(15,))
        self._ffn_plain = pmap(
            lambda p, bank, streams, ids: _ffn(p, bank, streams, ids),
            axis_name="tp")
        self._ffn_collect = pmap(
            lambda p, bank, streams, ids: _ffn(p, bank, streams, ids,
                                               True), axis_name="tp")
        self.site_mtp = pmap(mtp_fn, axis_name="tp")
        self._final = jax.jit(
            lambda streams: _hc_head_final(
                streams, self.head_params["fn"],
                self.head_params["base"], self.head_params["scale"],
                self.final_norm, cfg))

    # ------------------------------------------------------------- helpers
    def _attn_call(self, l, streams, valid, pos0):
        cfg = self.cfg
        p = self.PJ[l]
        st = self.state
        r = cfg.ratio(l)
        zc = self._zero_cstate()
        zi = self._zero_icstate()
        zcomp = self._zero_comp()
        zidx = self._zero_idxr()
        ring = st["ring"][l]
        comp = st["comp"][l] if r else zcomp
        cs = st["cstate"][l] if r else zc
        ics = st["icstate"][l] if r == 4 else zi
        idxr = st["idxr"][l] if r == 4 else zidx
        out = self.site_attn(p, streams, valid, self._sc(pos0),
                             ring[0], ring[1], ring[2],
                             cs[0], cs[1], ics[0], ics[1],
                             comp[0], comp[1], comp[2], idxr, r)
        streams2, ns = out
        st["ring"][l] = (ns[0], ns[1], ns[2])
        if r:
            st["comp"][l] = (ns[3], ns[4], ns[5])
            st["cstate"][l] = (ns[9], ns[10])
            if r == 4:
                st["idxr"][l] = ns[6]
                st["icstate"][l] = (ns[7], ns[8])
        return streams2

    def _zero_cstate(self):
        if not hasattr(self, "_zc"):
            cfg = self.cfg
            self._zc = (self._sh(np.zeros((1, 8, 2 * cfg.head_dim),
                                          np.float32)),) * 2
        return self._zc

    def _zero_icstate(self):
        if not hasattr(self, "_zic"):
            cfg = self.cfg
            self._zic = (self._sh(np.zeros(
                (1, 8, 2 * cfg.index_head_dim), np.float32)),) * 2
        return self._zic

    def _zero_comp(self):
        if not hasattr(self, "_zcomp"):
            cfg = self.cfg
            Dc, rd = cfg.head_dim, cfg.rope_head_dim
            self._zcomp = (
                self._sh(np.zeros((1, 1, Dc - rd), np.uint8)),
                self._sh(np.zeros((1, 1, (Dc - rd) // 64), np.float32)),
                self._sh(np.zeros((1, 1, rd), np.float32)))
        return self._zcomp

    def _zero_idxr(self):
        if not hasattr(self, "_zidx"):
            self._zidx = self._sh(np.zeros(
                (1, 1, self.cfg.index_head_dim), np.float32))
        return self._zidx

    def _ffn_call(self, l, streams, input_ids, collect=False):
        p = self.PJ[l]
        if collect:
            return self._ffn_collect(p, self.banks[l], streams, input_ids)
        return self._ffn_plain(p, self.banks[l], streams, input_ids)

    # ------------------------------------------------------------- prefill
    def reset(self):
        self.state = self._init_state()

    def _embed_streams(self, tokens, valid):
        cfg = self.cfg
        x = self.embed_np[np.asarray(tokens, np.int32)] \
            * np.asarray(valid, np.float32)[:, None]
        B, S = 1, len(tokens)
        streams = np.broadcast_to(
            x[None, :, None, :], (B, S, cfg.hc_mult, cfg.hidden_size))
        streams = np.ascontiguousarray(streams, dtype=np.float32)
        return jax.device_put(np.stack([streams] * self.d),
                              self.sharding_tp)

    def prefill(self, tokens):
        cfg = self.cfg
        self.reset()
        S = cfg.prefill_chunk
        n = len(tokens)
        pad = (-n) % S
        toks = [0] * pad + list(tokens)
        pos0 = 0
        last_hidden = None
        for ci in range(0, len(toks), S):
            chunk = toks[ci:ci + S]
            n_real = min(S, n - (ci - pad))
            valid_l = [0.0] * (S - n_real) + [1.0] * n_real
            streams = self._embed_streams(chunk, valid_l)
            valid = self._sh(np.asarray(valid_l, np.float32).reshape(1, S))
            ids = self._sh(np.asarray(chunk, np.int32).reshape(1, S))
            for l in range(cfg.n_layers):
                streams = self._attn_call(l, streams, valid, pos0)
                streams, routed = self._prefill_moe(l, streams, ids)
                self._last_routed[l] = _dg(routed[0])
            h = self._final(streams)
            last_hidden = _dg(h[0, 0, -1])
            self.state["cache_len"] += n_real
            pos0 += n_real
        self._last_hidden = last_hidden
        self._last_streams = streams
        self._refresh_all_banks_for_decode()
        return last_hidden

    def _prefill_moe(self, l, streams, ids):
        """Exact MoE during prefill via the affine-correction sweep
        (mHC site output is affine in the sublayer output; sweeping
        disjoint banks and subtracting (P-1) empty-bank sites is
        exact)."""
        cfg = self.cfg
        E = cfg.n_experts
        step = self.d * self.n_slots
        n_passes = -(-E // step)
        empty = [[-1] * self.n_slots] * self.d
        self._install_bank(l, empty)
        base, routed = self._ffn_call(l, streams, ids, collect=True)
        total = None
        for p_i in range(n_passes):
            start = p_i * step
            per_chip = []
            for dev in range(self.d):
                lo = start + dev * self.n_slots
                hi = min(lo + self.n_slots, E)
                per_chip.append(list(range(lo, hi))
                                + [-1] * max(0, self.n_slots - (hi - lo)))
            self._install_bank(l, per_chip)
            streams_p, routed = self._ffn_call(l, streams, ids,
                                               collect=True)
            total = streams_p if total is None else total + streams_p
        result = total - (n_passes - 1) * base
        return result, routed

    def _refresh_all_banks_for_decode(self):
        for l in range(self.cfg.n_layers):
            ids_l = self._last_routed.get(l)
            if ids_l is None:
                continue
            arr = np.asarray(ids_l).reshape(-1, self.cfg.top_k)
            last = [int(i) for i in arr[-1]]
            freq = {}
            for i in arr.flatten():
                freq[int(i)] = freq.get(int(i), 0) + 1
            priority = list(dict.fromkeys(last))
            for i, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
                if i not in priority:
                    priority.append(i)
            cap = self.d * self.n_slots
            chosen = priority[:cap]
            per_chip = [chosen[c::self.d][:self.n_slots]
                        for c in range(self.d)]
            per_chip = [pc + [-1] * (self.n_slots - len(pc))
                        for pc in per_chip]
            self._install_bank(l, per_chip)

    # ------------------------------------------------------------- decode
    def _decode_one(self, token, temperature=0.0, top_p=1.0):
        cfg = self.cfg
        pos = self.state["cache_len"]
        streams = self._embed_streams([token], [1.0])
        valid = self._sh(np.ones((1, 1), np.float32))
        ids = self._sh(np.asarray([[token]], np.int32))
        max_iters = cfg.n_layers + 2
        for _ in range(max_iters):
            snap = self._snapshot_state()
            streams_run, routed, hidden = self._run_all(streams, valid, ids,
                                                        pos, collect=True)
            missing = self._missing(routed)
            if not missing:
                logits = self._lm_head(hidden)
                self._last_streams = streams_run
                return logits, streams_run
            self._refresh(missing)
            self._restore_state(snap)
        raise RuntimeError("decode bank fixpoint did not converge")

    def _run_all(self, streams, valid, ids, pos, collect):
        cfg = self.cfg
        routed = {}
        for l in range(cfg.n_layers):
            streams = self._attn_call(l, streams, valid, pos)
            if collect:
                streams, r = self._ffn_call(l, streams, ids, collect=True)
                routed[l] = _dg(r[0])
            else:
                streams = self._ffn_call(l, streams, ids)
        self.state["cache_len"] = pos + 1
        h = self._final(streams)
        return streams, routed, _dg(h[0, 0, 0])

    def _missing(self, routed):
        miss = {}
        for l, ids in routed.items():
            have = set()
            for pc in self.bank_ids[l]:
                have.update(e for e in pc if e >= 0)
            need = set(int(i) for i in np.asarray(ids).flatten())
            if not need <= have:
                miss[l] = need
        return miss

    def _refresh(self, missing):
        for l, need in missing.items():
            have = [e for pc in self.bank_ids[l] for e in pc if e >= 0]
            old = list(dict.fromkeys(have))
            new_ids = list(dict.fromkeys(list(need) + old))[
                :self.d * self.n_slots]
            per_chip = [new_ids[c::self.d] for c in range(self.d)]
            per_chip = [pc + [-1] * (self.n_slots - len(pc))
                        for pc in per_chip]
            self._install_bank(l, per_chip)

    def _snapshot_state(self):
        st = self.state
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in st.items()}

    def _restore_state(self, snap):
        self.state = snap

    # ------------------------------------------------------------- MTP
    def draft(self, streams, next_token):
        """MTP-1 draft: target hc-streams + next token -> logits for the
        position after it.  Draft attention position = cache_len."""
        cfg = self.cfg
        p = self.PJ["mtp"]
        pos = self.state["cache_len"]
        embed = jax.device_put(
            np.stack([self.embed_np[next_token]] * self.d),
            self.sharding_tp)
        ids = self._sh(np.asarray([[next_token]], np.int32))
        pos0 = self._sc(pos)
        max_iters = 3
        for _ in range(max_iters):
            ring = self.state["mtp_ring"]
            out = self.site_mtp(p, self.banks["mtp"], streams, embed,
                                ids, pos0, ring[0], ring[1], ring[2])
            hcol, routed, new_ring = out
            need = set(int(i) for i in np.asarray(_dg(routed[0])).flatten())
            have = set()
            for pc in self.bank_ids["mtp"]:
                have.update(e for e in pc if e >= 0)
            if need <= have:
                self.state["mtp_ring"] = new_ring
                return self._lm_head(_dg(hcol[0, 0, 0]))
            old_ids = list(dict.fromkeys(
                [e for pc in self.bank_ids["mtp"] for e in pc if e >= 0]))
            new_ids = list(dict.fromkeys(list(need) + old_ids))[
                :self.d * self.n_slots]
            per_chip = [new_ids[c::self.d] for c in range(self.d)]
            per_chip = [pc + [-1] * (self.n_slots - len(pc))
                        for pc in per_chip]
            self._install_bank("mtp", per_chip)
        raise RuntimeError("mtp draft fixpoint did not converge")

    # ------------------------------------------------------------- logits
    def _lm_head(self, hidden):
        h = np.asarray(hidden).reshape(-1)
        return h @ self.lm_head_np.T

    # ------------------------------------------------------------- gen
    def generate(self, tokens, max_new_tokens=64, temperature=0.0,
                 top_p=1.0, stop_ids=None, on_token=None):
        h = self.prefill(tokens)
        logits = self._lm_head(h)
        out = []
        for i in range(max_new_tokens):
            t = self._sample(logits, temperature, top_p)
            if stop_ids and t in stop_ids:
                break
            out.append(t)
            if on_token:
                on_token(t)
            if i + 1 < max_new_tokens:
                logits, _ = self._decode_one(t, temperature, top_p)
        return out

    def generate_mtp(self, tokens, max_new_tokens=64, temperature=0.0,
                     top_p=1.0, stop_ids=None, on_token=None, stats=None):
        """MTP-1 speculative decode.  Greedy path is lossless: emits
        a=argmax(target L) always, plus d when the draft's d equals the
        target's next argmax (verified by the target step on a)."""
        h = self.prefill(tokens)
        logits = self._lm_head(h)
        streams = self._last_streams
        out = []
        n_acc = n_rej = 0
        while len(out) < max_new_tokens:
            a = self._sample(logits, temperature, top_p)
            if stop_ids and a in stop_ids:
                break
            out.append(a)
            if on_token:
                on_token(a)
            draft_logits = self.draft(streams, a)
            d = int(np.argmax(draft_logits))
            logits, streams = self._decode_one(a, temperature, top_p)
            emitted = a
            if int(np.argmax(logits)) == d:
                # verified: d is exactly the target's next token
                if not (stop_ids and d in stop_ids) \
                        and len(out) < max_new_tokens:
                    out.append(d)
                    emitted = d
                    if on_token:
                        on_token(d)
                n_acc += 1
                # next cycle consumes d (target runs on d next round);
                # but we need logits for the position after d:
                logits, streams = self._decode_one(emitted, temperature,
                                                   top_p)
            else:
                n_rej += 1
        if stats is not None:
            stats.update({"accepts": n_acc, "rejects": n_rej})
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
            p2 = p[keep] / p[keep].sum()
            return int(np.random.default_rng().choice(keep, p=p2))
        return int(np.random.default_rng().choice(len(p), p=p))
