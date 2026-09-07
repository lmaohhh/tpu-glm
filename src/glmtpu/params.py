"""Build params_by_chip for the Runner: FAKE weights (local structural test)
or REAL safetensors shards (loader_real.py mirrors this layout exactly).

params_by_chip[c] = {
    l: {
      "attn_hc": {"fn","base","scale"},      # replicated
      "ffn_hc": {...},
      "input_ln": [D], "post_ln": [D],
      "kda": {...} | None,                    # per-chip shards
      "dsa": {...} | None,
      "moe": {"gate_w","e_bias","sh":{...}} | None,   # sh = shared experts, BF16
      "mlp": {...} | None,                    # dense MLP, BF16
    },
    "final_ln": [D],
}

All BF16 tensors are stored as float32 numpy (bf16 values, f32 container) so
numpy can slice them; JAX casts to bf16 at device_put.

Also returns (embed_f32 [V,D], lm_head_f32 [V,D], expert_host).
expert_host[(layer, e)] = {"gu": u8 [2I,D], "gu_s": f32 grid, "d": u8 [D,I],
                           "d_s": f32 grid}  (gate|up fused on axis 0).
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from .config import GlmConfig

BLOCK = 128


def _ceil(n, d):
    return -(-n // d)


def _slice_rows(w, d, c):
    chunk = _ceil(w.shape[0], d)
    return w[c * chunk:(c + 1) * chunk]


def _slice_cols(w, d, c):
    chunk = _ceil(w.shape[1], d)
    return w[:, c * chunk:(c + 1) * chunk]


def make_fake(cfg: GlmConfig, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    D = cfg.hidden_size
    H = cfg.hc_mult
    mix = (2 + H) * H

    def bf(shape, scale=0.02):
        a = (rng.standard_normal(shape) * scale).astype(np.float32)
        return np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))

    def f32(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def u8_pair(rows, cols):
        """FP8 weight [rows, cols] + scale_inv grid (for expert banks)."""
        from .fp8 import F8_LUT
        w = (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)
        s = (np.abs(rng.standard_normal((_ceil(rows, BLOCK), _ceil(cols, BLOCK))))
             * 0.01 + 0.005).astype(np.float32)
        scaled = np.empty_like(w)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                scaled[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK] = \
                    w[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK] / s[i, j]
        scaled = np.clip(scaled, -448.0, 448.0)
        sign = np.sign(scaled)
        a = np.abs(scaled)
        grid = np.sort(np.unique(np.abs(F8_LUT)))
        idx = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
        left, right = grid[idx - 1], grid[idx]
        chosen = np.where(a - left <= right - a, left, right)
        q = np.zeros(scaled.shape, np.uint8)
        for g in np.unique(chosen):
            u_val = int(np.where(np.abs(F8_LUT) == g)[0][0])
            u = np.where(sign < 0, u_val | 0x80, u_val)
            q[chosen == g] = u[chosen == g].astype(np.uint8)
        return q, s

    nh = cfg.n_kda_heads
    hd = cfg.kda_head_dim
    qkvd = nh * hd
    K = cfg.conv_kernel
    E = cfg.n_experts
    I = cfg.moe_inter
    Hh = cfg.dsa_heads
    dk, dv, dkv = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    full = {}
    for l in range(cfg.n_layers):
        lay = {
            "attn_hc": {"fn": bf((mix, H * D)), "base": f32((mix,), 0.0),
                        "scale": np.array([1.0, 1.0, 1.0], np.float32)},
            "ffn_hc": {"fn": bf((mix, H * D)), "base": f32((mix,), 0.0),
                       "scale": np.array([1.0, 1.0, 1.0], np.float32)},
            "input_ln": np.ones(D, np.float32),
            "post_ln": np.ones(D, np.float32),
            "kda": None, "dsa": None, "moe": None, "mlp": None,
        }
        if cfg.is_kda(l):
            lay["kda"] = {
                "q_proj": bf((qkvd, D)), "k_proj": bf((qkvd, D)), "v_proj": bf((qkvd, D)),
                "q_conv": bf((qkvd, K)), "k_conv": bf((qkvd, K)), "v_conv": bf((qkvd, K)),
                "f_a": bf((hd, D)), "f_b": bf((qkvd, hd)),
                "dt_bias": f32((qkvd,), 0.001), "A_log": f32((nh,), 0.0),
                "b_proj": bf((nh, D)),
                "g_a": bf((hd, D)), "g_b": bf((qkvd, hd)),
                "o_norm": np.ones(hd, np.float32),
                "o_proj": bf((D, qkvd)),
            }
        else:
            lay["dsa"] = {
                "q_a": bf((cfg.q_lora_rank, D)),
                "q_a_ln": np.ones(cfg.q_lora_rank, np.float32),
                "q_b": bf((Hh * dk, cfg.q_lora_rank)),
                "kv_a": bf((dkv, D)), "kv_a_ln": np.ones(dkv, np.float32),
                "kv_b": bf((Hh * (dk + dv), dkv)),
                "o_proj": bf((D, Hh * dv)),
            }
        if cfg.is_moe(l):
            sh_g, _ = u8_pair(I, D)
            sh_u, _ = u8_pair(I, D)
            sh_d, _ = u8_pair(D, I)
            lay["moe"] = {
                "gate_w": bf((E, D)), "e_bias": f32((E,), 0.0),
                "sh": {"g": bf((I, D)), "u": bf((I, D)), "d": bf((D, I))},
            }
        else:
            lay["mlp"] = {"g": bf((cfg.dense_inter, D)),
                          "u": bf((cfg.dense_inter, D)),
                          "d": bf((D, cfg.dense_inter))}
        full[l] = lay

    params_by_chip = []
    for c in range(d):
        chip = {}
        for l in range(cfg.n_layers):
            lay = {
                "attn_hc": {k: v.copy() for k, v in full[l]["attn_hc"].items()},
                "ffn_hc": {k: v.copy() for k, v in full[l]["ffn_hc"].items()},
                "input_ln": full[l]["input_ln"].copy(),
                "post_ln": full[l]["post_ln"].copy(),
                "kda": None, "dsa": None, "moe": None, "mlp": None,
            }
            if cfg.is_kda(l):
                k = full[l]["kda"]
                lay["kda"] = {
                    "q_proj": _slice_rows(k["q_proj"], d, c),
                    "k_proj": _slice_rows(k["k_proj"], d, c),
                    "v_proj": _slice_rows(k["v_proj"], d, c),
                    "q_conv": _slice_rows(k["q_conv"], d, c),
                    "k_conv": _slice_rows(k["k_conv"], d, c),
                    "v_conv": _slice_rows(k["v_conv"], d, c),
                    "f_a": k["f_a"],                       # replicated
                    "f_b": _slice_rows(k["f_b"], d, c),
                    "dt_bias": _slice_rows(k["dt_bias"].reshape(-1, 1), d, c).reshape(-1),
                    "A_log": _slice_rows(k["A_log"].reshape(-1, 1), d, c).reshape(-1),
                    "b_proj": _slice_rows(k["b_proj"], d, c),
                    "g_a": k["g_a"],
                    "g_b": _slice_rows(k["g_b"], d, c),
                    "o_norm": k["o_norm"],
                    "o_proj": _slice_cols(k["o_proj"], d, c),
                }
            else:
                s = full[l]["dsa"]
                lay["dsa"] = {
                    "q_a": s["q_a"], "q_a_ln": s["q_a_ln"],
                    "q_b": _slice_rows(s["q_b"], d, c),
                    "kv_a": s["kv_a"], "kv_a_ln": s["kv_a_ln"],
                    "kv_b": _slice_rows(s["kv_b"], d, c),
                    "o_proj": _slice_cols(s["o_proj"], d, c),
                }
            if cfg.is_moe(l):
                m = full[l]["moe"]
                sh = m["sh"]
                lay["moe"] = {
                    "gate_w": m["gate_w"], "e_bias": m["e_bias"],
                    "sh": {
                        "g": _slice_rows(sh["g"], d, c),
                        "u": _slice_rows(sh["u"], d, c),
                        "d": _slice_cols(sh["d"], d, c),
                    },
                }
            else:
                m = full[l]["mlp"]
                lay["mlp"] = {
                    "g": _slice_rows(m["g"], d, c),
                    "u": _slice_rows(m["u"], d, c),
                    "d": _slice_cols(m["d"], d, c),
                }
            chip[l] = lay
        chip["final_ln"] = np.ones(D, np.float32)
        params_by_chip.append(chip)

    embed = bf((cfg.vocab_size, D))
    lm_head = bf((cfg.vocab_size, D))
    expert_host = {}
    for l in cfg.moe_layers:
        for e in range(E):
            # fused gate|up [2I, D] with ONE fused scale grid
            gu, gu_s = u8_pair(2 * I, D)
            dd, dd_s = u8_pair(D, I)
            expert_host[(l, e)] = {
                "gu": gu, "gu_s": gu_s, "d": dd, "d_s": dd_s,
            }
    return params_by_chip, embed, lm_head, expert_host
