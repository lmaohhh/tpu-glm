"""Tiny-config fake weights for the DSV4 engine (structural tests).

Produces params_by_chip matching dsv4_loader's real-weight layout:

params_by_chip[c] = {
  l: {
    "attn_hc": {"fn","base","scale"},          # replicated f32
    "ffn_hc": {...},
    "attn_norm": [D], "ffn_norm": [D],
    "wq_a": [ql,D] bf16, "q_norm": [ql],
    "wq_b": [H/d*dh, ql] bf16 (head-sharded),
    "wkv": [dh, D] bf16, "kv_norm": [dh],
    "wo_a": [olr, H/d*dh] bf16, "wo_b": [D, olr] bf16 (group-sharded),
    "attn_sink": [H/d] f32,
    "comp": {...} | None,                       # compressor (replicated)
    "idx": {"wq_b","weights_proj"} | None,      # indexer (head-sharded)
    "icomp": {...} | None,                      # indexer compressor
    "gate": {"w","bias"|"tid2eid"},             # router (replicated)
    "shared": {"w1","w3","w2"},                 # shared expert (sharded)
  },
  "mtp": {attn (sliding) + ffn + e/h_proj + norms + hc_head_*},
  "head": {"fn","base","scale"},                # global hc_head
  "final_ln": [D],
}

Also returns (embed [V,D] f32, lm_head [V,D] f32,
expert_host[(key, e)] = {"w1","w1_s","w3","w3_s","w2","w2_s"} with
key = layer int or "mtp"; FP4 packed u8 + e8m0 u8 scales).
"""
from __future__ import annotations

import numpy as np

from .dsv4_config import Dsv4Config
from .dsv4_fp4 import quantize_np


def _bf(rng, shape, scale=0.02):
    a = (rng.standard_normal(shape) * scale).astype(np.float32)
    import jax.numpy as jnp
    return np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(np.float32))


def _f32(rng, shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def make_fake(cfg: Dsv4Config, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    D = cfg.hidden_size
    Hc = cfg.hc_mult
    mix = (2 + Hc) * Hc
    dh, rd = cfg.head_dim, cfg.rope_head_dim
    Hl = cfg.n_heads // d                      # local heads (1 group/chip)
    ql, ol = cfg.q_lora_rank, cfg.o_lora_rank
    E, I = cfg.n_experts, cfg.moe_inter

    def hc():
        return {"fn": _f32(rng, (mix, Hc * D)),
                "base": _f32(rng, (mix,), 0.0),
                "scale": np.array([1.0, 1.0, 1.0], np.float32)}

    def shared_full():
        return {"w1": _bf(rng, (I, D)),
                "w3": _bf(rng, (I, D)),
                "w2": _bf(rng, (D, I))}

    full = {}
    for l in range(cfg.n_layers):
        r = cfg.ratio(l)
        lay = {
            "attn_hc": hc(), "ffn_hc": hc(),
            "attn_norm": np.ones(D, np.float32),
            "ffn_norm": np.ones(D, np.float32),
            "wq_a": _bf(rng, (ql, D)), "q_norm": np.ones(ql, np.float32),
            "wq_b": _bf(rng, (Hl * dh, ql)),
            "wkv": _bf(rng, (dh, D)), "kv_norm": np.ones(dh, np.float32),
            "wo_a": _bf(rng, (ol, Hl * dh)),
            "wo_b": _bf(rng, (D, ol)),
            "attn_sink": _f32(rng, (Hl,), 0.1),
            "comp": None, "idx": None, "icomp": None,
            "gate": None, "shared": None,
        }
        if r:
            Dc = dh
            coff = 2 if r == 4 else 1
            lay["comp"] = {
                "wkv": _bf(rng, (coff * Dc, D)),
                "wgate": _bf(rng, (coff * Dc, D)),
                "ape": _f32(rng, (r, coff * Dc), 0.1),
                "norm": np.ones(Dc, np.float32),
            }
        if r == 4:
            ih = cfg.index_head_dim
            iHl = cfg.index_n_heads // d
            lay["idx"] = {
                "wq_b": _bf(rng, (iHl * ih, ql)),
                "weights_proj": _bf(rng, (iHl, D)),
            }
            lay["icomp"] = {
                "wkv": _bf(rng, (2 * ih, D)),
                "wgate": _bf(rng, (2 * ih, D)),
                "ape": _f32(rng, (r, 2 * ih), 0.1),
                "norm": np.ones(ih, np.float32),
            }
        gate = {"w": _bf(rng, (E, D))}
        if cfg.is_hash(l):
            # hash routing: tid2eid [V, K] int32 in [0, E)
            gate["tid2eid"] = rng.integers(0, E, (cfg.vocab_size,
                                                  cfg.top_k)).astype(np.int32)
        else:
            gate["bias"] = _f32(rng, (E,), 0.0)
        lay["gate"] = gate
        lay["shared"] = shared_full()
        full[l] = lay

    # ---- MTP layer ----
    mtp = {
        "attn_hc": hc(), "ffn_hc": hc(),
        "attn_norm": np.ones(D, np.float32),
        "ffn_norm": np.ones(D, np.float32),
        "wq_a": _bf(rng, (ql, D)), "q_norm": np.ones(ql, np.float32),
        "wq_b": _bf(rng, (Hl * dh, ql)),
        "wkv": _bf(rng, (dh, D)), "kv_norm": np.ones(dh, np.float32),
        "wo_a": _bf(rng, (ol, Hl * dh)),
        "wo_b": _bf(rng, (D, ol)),
        "attn_sink": _f32(rng, (Hl,), 0.1),
        "e_proj": _bf(rng, (D, D)), "h_proj": _bf(rng, (D, D)),
        "enorm": np.ones(D, np.float32), "hnorm": np.ones(D, np.float32),
        "norm": np.ones(D, np.float32),
        "gate": {"w": _bf(rng, (E, D)), "bias": _f32(rng, (E,), 0.0)},
        "shared": shared_full(),
        "hc_head_fn": _f32(rng, (Hc, Hc * D)),
        "hc_head_base": _f32(rng, (Hc,), 0.0),
        "hc_head_scale": np.array([1.0], np.float32),
    }

    head = {"fn": _f32(rng, (Hc, Hc * D)),
            "base": _f32(rng, (Hc,), 0.0),
            "scale": np.array([1.0], np.float32)}
    final_ln = np.ones(D, np.float32)

    # ---- shard across chips ----
    params_by_chip = []
    for c in range(d):
        chip = {}
        for l in range(cfg.n_layers):
            f = full[l]
            # shared expert: w1/w3 row-shard, w2 col-shard
            sh = {}
            for t in ("w1", "w3"):
                chunk = -(-f["shared"][t].shape[0] // d)
                sh[t] = f["shared"][t][c * chunk:(c + 1) * chunk]
            chunkc = -(-f["shared"]["w2"].shape[1] // d)
            sh["w2"] = f["shared"]["w2"][:, c * chunkc:(c + 1) * chunkc]
            lay = dict(f)  # shallow: replicated tensors shared OK
            lay["shared"] = sh
            chip[l] = lay
        msh = {}
        for t in ("w1", "w3"):
            chunk = -(-mtp["shared"][t].shape[0] // d)
            msh[t] = mtp["shared"][t][c * chunk:(c + 1) * chunk]
        chunkc = -(-mtp["shared"]["w2"].shape[1] // d)
        msh["w2"] = mtp["shared"]["w2"][:, c * chunkc:(c + 1) * chunkc]
        m2 = dict(mtp)
        m2["shared"] = msh
        chip["mtp"] = m2
        chip["head"] = head
        chip["final_ln"] = final_ln
        params_by_chip.append(chip)

    # ---- host experts (fp4) ----
    embed = _bf(rng, (cfg.vocab_size, D))
    lm_head = _bf(rng, (cfg.vocab_size, D))
    expert_host = {}
    keys = list(range(cfg.n_layers)) + ["mtp"]
    for key in keys:
        for e in range(E):
            ex = {}
            for t, out, in_ in (("w1", I, D), ("w3", I, D), ("w2", D, I)):
                w = (rng.standard_normal((out, in_)) * 0.1).astype(np.float32)
                p, s = quantize_np(w)
                ex[t] = p
                ex[t + "_s"] = s
            expert_host[(key, e)] = ex
    return params_by_chip, embed, lm_head, expert_host
