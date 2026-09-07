"""glm5_next per-layer math in pure JAX — FINAL.

Mirrored 1:1 from transformers v5.16.0 modeling_glm5_next.py (line refs).
Every function runs on ONE chip with LOCAL shards, inside jax.pmap(axis_name
='tp').  Collectives: lax.psum(..., 'tp') for column-parallel outputs.

Sharding:
  KDA (34 layers)   heads 64 -> 8 per chip: q/k/v_proj, f_b, g_b, dt_bias
                    row-sharded; o_proj col-sharded + psum; A_log/b_proj/
                    conv channel-sharded; f_a/g_a/o_norm replicated.
  DSA (11 layers)   q_b/kv_b row-sharded by head; o_proj col-sharded + psum;
                    q_a/kv_a/norms replicated; latent cache replicated.
  MLP dense/shared  FP8 Megatron pairing: gate/up row-sharded, down
                    col-sharded + psum; in-graph dequant.
  MoE routed        per-chip expert banks (FP8, static n_slots); every chip
                    applies its resident experts to ALL tokens with router
                    coeff masks; psum.  Exact for any bank contents (misses
                    contribute 0; host guarantees coverage before the call).
  Router / hc / norms / embed(host) / final_ln: replicated.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .fp8 import fp8_matmul

# ===========================================================================
# basic ops
# ===========================================================================


def rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    x32 = x32 * lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (w.astype(jnp.float32) * x32).astype(x.dtype)


def rms_norm_no_w(x, eps):
    x32 = x.astype(jnp.float32)
    return (x32 * lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def l2norm(x, eps=1e-6):
    # modeling lines 418-426 (FLA style: divide by sqrt(sum^2 + eps))
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def swiglu_clamped(gate, up, limit):
    # modeling lines 99-105 / 138-143
    gate = jnp.clip(gate, None, limit)
    up = jnp.clip(up, -limit, limit)
    return jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up


# ===========================================================================
# hyper-connections (modeling lines 220-303, 1318-1329)
# ===========================================================================

def hc_site(hc, streams, cfg):
    """hc = {"fn": [mix, H*D] bf16, "base": [mix] f32, "scale": [3] f32}
    (replicated).  streams [B,S,H,D] -> (post [B,S,H] f32, comb [B,S,H,H] f32
    indexed [b,s,j,h], collapsed [B,S,D] bf16)."""
    B, S, H, D = streams.shape
    flat = streams.reshape(B, S, H * D).astype(jnp.float32)
    flat = rms_norm_no_w(flat, cfg.rms_norm_eps)          # input_norm
    logits = flat @ hc["fn"].astype(jnp.float32).T        # [B,S,(2+H)*H]
    pre_w, post_w, comb_w = jnp.split(logits, [H, 2 * H], axis=-1)
    pre_b, post_b, comb_b = jnp.split(hc["base"].astype(jnp.float32), [H, 2 * H])
    pre_s, post_s, comb_s = hc["scale"][0], hc["scale"][1], hc["scale"][2]

    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + cfg.hc_eps   # line 284
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)      # line 285
    cl = comb_w.reshape(B, S, H, H) * comb_s + comb_b.reshape(H, H)
    comb = jax.nn.softmax(cl, axis=-1) + cfg.hc_eps            # line 287
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + cfg.hc_eps)  # 288
    for _ in range(cfg.hc_sinkhorn_iters - 1):                 # 289-291
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + cfg.hc_eps)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + cfg.hc_eps)
    collapsed = jnp.sum(pre[..., None] * streams, axis=2)      # line 295
    return post, comb, collapsed.astype(streams.dtype)


def hc_apply(post, comb, sub_out, streams):
    """out[h] = post[h]*sub_out + sum_j comb[j,h]*stream[j]  (lines 1318-1320)."""
    term_a = post[..., None].astype(sub_out.dtype) * sub_out[..., None, :]
    term_b = jnp.einsum("bsjh,bsjd->bshd", comb.astype(streams.dtype), streams)
    return term_a + term_b


# ===========================================================================
# KDA (modeling lines 306-735)
# ===========================================================================

def _dwconv_silu(u, w, cs, K):
    """Depthwise causal conv + silu.  u [B,S,C] bf16, w [C,K] bf16,
    cs [B,C,K-1] bf16 (channels-FIRST).  up = concat(cs, u^T) on time:
    out[t] = sum_i w[:,i] * up[:, t+i]  (lines 376-415).
    Zero left-pad == zero conv state, so pad rows never affect real rows."""
    S = u.shape[1]
    up = jnp.concatenate([cs, u.transpose(0, 2, 1)], axis=2)   # [B,C,S+K-1]
    acc = jnp.zeros((u.shape[0], u.shape[2], S), dtype=jnp.float32)
    for i in range(K):
        acc = acc + up[:, :, i:i + S].astype(jnp.float32) * w[:, i].astype(jnp.float32)[None, :, None]
    out = jax.nn.silu(acc).transpose(0, 2, 1)                  # [B,S,C]
    new_cs = up[:, :, S:]                                       # [B,C,K-1]
    return out.astype(u.dtype), new_cs


def kda_core(p, x, rec, conv, valid, cfg):
    """x [B,S,D] bf16 (zeroed at pads), valid [B,S] f32 (1 real / 0 pad).
    rec [B,nh_l,hd,hd] f32 LOCAL; conv [B,3*qkvd_l,K-1] bf16 LOCAL.
    Returns (out [B,S,D] bf16 psum'ed, rec', conv')."""
    B, S, D = x.shape
    nh, hd = rec.shape[1], rec.shape[2]                   # LOCAL heads
    qkvd = nh * hd
    K = cfg.conv_kernel

    q = x @ p["q_proj"].T
    k = x @ p["k_proj"].T
    v = x @ p["v_proj"].T
    q, csq = _dwconv_silu(q, p["q_conv"], conv[:, :qkvd], K)
    k, csk = _dwconv_silu(k, p["k_conv"], conv[:, qkvd:2 * qkvd], K)
    v, csv = _dwconv_silu(v, p["v_conv"], conv[:, 2 * qkvd:], K)

    q = l2norm(q.reshape(B, S, nh, hd).astype(jnp.float32))
    k = l2norm(k.reshape(B, S, nh, hd).astype(jnp.float32))
    v = v.reshape(B, S, nh, hd).astype(jnp.float32)
    q = q * (hd ** -0.5)                                  # after l2norm (453-454)

    # forget gate (lines 306-336): g = lb * sigmoid(exp(A_log)*(f_b f_a x + dt_bias))
    h1 = x @ p["f_a"].T                                   # [B,S,hd] replicated
    fg = (h1 @ p["f_b"].T).astype(jnp.float32) + p["dt_bias"].astype(jnp.float32)
    fg = fg.reshape(B, S, nh, hd)
    decay = jnp.exp(p["A_log"].astype(jnp.float32)).reshape(1, 1, nh, 1)
    g = cfg.gate_lower_bound * jax.nn.sigmoid(decay * fg)  # in (lb, 0)

    beta = jax.nn.sigmoid((x @ p["b_proj"].T).astype(jnp.float32))  # [B,S,nh]

    def step(state, inp):
        qi, ki, vi, gi, bi, val = inp
        s = state * jnp.exp(gi)[..., None]                # line 473
        kv_mem = jnp.sum(s * ki[..., None, :], axis=-2)   # line 474
        delta = (vi - kv_mem) * bi[..., None]             # line 475
        s = s + ki[..., :, None] * delta[..., None, :]    # line 477
        out = jnp.sum(s * qi[..., None, :], axis=-2)      # line 478
        s = jnp.where(val > 0.5, s, state)                # pads leave state
        out = out * val                                    # pads output 0
        return s, out

    # scan over the sequence axis: transpose to [S, B, ...]
    def t(a):
        return a.transpose(1, 0, *range(2, a.ndim))
    rec, core_t = lax.scan(step, rec, (t(q), t(k), t(v), t(g),
                                       t(beta) if beta.ndim > 2 else beta.transpose(1, 0),
                                       t(valid)))
    core = core_t.transpose(1, 0, 2, 3)                   # [B,S,nh,hd]

    # gated output norm (lines 341-360) + col-parallel o_proj
    og = ((x @ p["g_a"].T) @ p["g_b"].T).reshape(B, S, nh, hd)
    n32 = core.astype(jnp.float32)                       # [B,S,nh,hd]
    n32 = n32 * lax.rsqrt(jnp.mean(n32 * n32, axis=-1, keepdims=True) + cfg.rms_norm_eps)
    n32 = p["o_norm"].astype(jnp.float32) * n32
    n32 = n32 * jax.nn.sigmoid(og.astype(jnp.float32))
    coreb = n32.astype(x.dtype).reshape(B, S, qkvd)
    out = coreb @ p["o_proj"].T                           # col-sharded partial
    out = lax.psum(out, "tp")
    new_conv = jnp.concatenate([csq, csk, csv], axis=1)
    return out, rec, new_conv


# ===========================================================================
# DSA / MLA layer: full causal attention over the replicated latent cache with
# absorbed W_UK / W_UV (modeling lines 1066-1258).  v1: indexer skipped
# (documented quality caveat; exactly correct causal masking retained).
# ===========================================================================

def dsa_core(p, x, kv, bitmap, cache_len, chunk_valid, cfg):
    """x [B,S,D] bf16; kv [B,max_ctx,dkv] bf16 replicated (pre-chunk state);
    bitmap [B,max_ctx] f32 (1 = valid key), updated with chunk_valid at
    [cache_len : cache_len+S] inside; cache_len traced i32; chunk_valid
    [B,S] f32.  Returns (out, kv', bitmap')."""
    B, S, D = x.shape
    H = p["q_b"].shape[0] // cfg.qk_nope_head_dim          # LOCAL heads
    dk, dv, dkv = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    T = kv.shape[1]

    q_res = rms_norm(x @ p["q_a"].T, p["q_a_ln"], cfg.rms_norm_eps)   # [B,S,qlora]
    q = (q_res @ p["q_b"].T).reshape(B, S, H, dk)
    c = rms_norm(x @ p["kv_a"].T, p["kv_a_ln"], cfg.rms_norm_eps)     # [B,S,dkv]

    w = p["kv_b"].reshape(H, dk + dv, dkv)
    w_uk = w[:, :dk]
    w_uv = w[:, dk:]

    # write chunk latents into the cache at [cache_len : cache_len+S]
    c = jnp.where(chunk_valid[..., None] > 0.5, c, 0.0).astype(kv.dtype)
    kv = lax.dynamic_update_slice(kv, c, (0, cache_len, 0))
    bitmap = lax.dynamic_update_slice(
        bitmap, chunk_valid.astype(bitmap.dtype), (0, cache_len))

    # absorbed query: q' = W_UK^T q -> [B,S,H,dkv]
    qp = jnp.einsum("bshd,hdk->bshk",
                    q.astype(jnp.float32), w_uk.astype(jnp.float32))
    # full-cache attention with causal + bitmap masks
    qpos = cache_len + jnp.arange(S)
    kpos = jnp.arange(T)
    causal = kpos[None, :] <= qpos[:, None]               # [S,T]
    mask = causal[None, :, :] & (bitmap[:, :T] > 0.5)     # [B,S,T]
    scores = jnp.einsum("bshk,btk->bhst",
                        qp.astype(kv.dtype), kv) * (dk ** -0.5)
    scores = scores.astype(jnp.float32)
    scores = jnp.where(mask[:, None, :, :], scores, -1e30)
    probs = jax.nn.softmax(scores, axis=-1)
    U = jnp.einsum("bhst,btk->bshk", probs.astype(kv.dtype), kv)
    vo = jnp.einsum("bshk,hdk->bshd",
                    U.astype(jnp.float32), w_uv.astype(jnp.float32))
    out = vo.reshape(B, S, H * dv).astype(x.dtype) @ p["o_proj"].T
    out = lax.psum(out, "tp")
    return out, kv, bitmap


# ===========================================================================
# MLPs
# ===========================================================================

def dense_mlp_core(p, h, cfg):
    """BF16 Megatron pairing (weights host-dequantized at load):
    gate/up row-sharded, down col-sharded.  Returns [B,S,D] bf16 (psum'ed)."""
    g = h @ p["g"].T
    u = h @ p["u"].T
    hid = swiglu_clamped(g, u, cfg.swiglu_limit)
    out = hid @ p["d"].T
    return lax.psum(out, "tp")


def moe_router(gate_w, e_bias, h, cfg):
    """Replicated router (lines 146-184; n_group=topk_group=1 => identity).
    h [B,S,D] -> (weights [B,S,K] f32, ids [B,S,K] i32)."""
    logits = h.astype(jnp.float32) @ gate_w.astype(jnp.float32).T   # [B,S,E]
    scores = jax.nn.sigmoid(logits)                        # line 162
    choice = scores + e_bias.astype(jnp.float32)           # line 163 (noaux_tc)
    ids = lax.top_k(choice, cfg.top_k)[1]                  # line 178
    w = jnp.take_along_axis(scores, ids, axis=-1)          # line 179
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-20)   # line 181-182
    w = w * cfg.routed_scaling_factor                      # line 183
    return w, ids


def moe_bank_core(bank, h, w, ids, cfg):
    """Per-chip expert bank application.  bank: {"gu": [n,2I,D] u8,
    "gu_s": [n, ceil(2I/128), ceil(D/128)] f32, "d": [n,D,I] u8, "d_s": ...,
    "ids": [n] i32}.  Slot ids of -1 are padding (never match).  Every chip
    applies its resident experts to ALL tokens; psum makes the sum exact
    given host-side coverage of the union of routed ids."""
    B, S, D = h.shape
    n = bank["ids"].shape[0]
    out = jnp.zeros((B, S, D), dtype=jnp.float32)
    for s in range(n):
        e = bank["ids"][s]
        match = (ids == e).astype(jnp.float32)             # [B,S,K]
        coeff = jnp.sum(match * w, axis=-1)                # [B,S]
        gu = fp8_matmul(h, bank["gu"][s], bank["gu_s"][s])  # [B,S,2I]
        gate, up = jnp.split(gu, 2, axis=-1)
        hid = swiglu_clamped(gate, up, cfg.swiglu_limit)   # [B,S,I]
        y = fp8_matmul(hid, bank["d"][s], bank["d_s"][s])  # [B,S,D]
        out = out + y.astype(jnp.float32) * coeff[..., None]
    return lax.psum(out.astype(h.dtype), "tp")


# ===========================================================================
# site wrappers (one pmap'ed call per attention/FFN site)
# ===========================================================================

def attn_site_kda(p, streams, valid, rec, conv, cfg):
    """p['attn_hc'], p['kda'], p['input_ln'].  streams [B,S,H,D] bf16;
    valid [B,S] f32; rec/conv LOCAL per chip.  Returns (streams', rec', conv').

    Pad-row invariant: pad rows of streams stay exactly zero."""
    post, comb, collapsed = hc_site(p["attn_hc"], streams, cfg)
    h = rms_norm(collapsed, p["input_ln"], cfg.rms_norm_eps)
    out, rec2, conv2 = kda_core(p["kda"], h, rec, conv, valid, cfg)
    streams2 = hc_apply(post, comb, out, streams)
    streams2 = streams2 * valid[..., None, None].astype(streams2.dtype)
    return streams2, rec2, conv2


def attn_site_dsa(p, streams, valid, kv, bitmap, cache_len, cfg):
    """p['attn_hc'], p['dsa'], p['input_ln'].  kv/bitmap LOCAL replicas
    (identical on all chips); cache_len traced i32 scalar."""
    post, comb, collapsed = hc_site(p["attn_hc"], streams, cfg)
    h = rms_norm(collapsed, p["input_ln"], cfg.rms_norm_eps)
    out, kv2, bm2 = dsa_core(p["dsa"], h, kv, bitmap, cache_len, valid, cfg)
    streams2 = hc_apply(post, comb, out, streams)
    streams2 = streams2 * valid[..., None, None].astype(streams2.dtype)
    return streams2, kv2, bm2


def ffn_site_dense(p, streams, cfg):
    """p['ffn_hc'], p['mlp'], p['post_ln']."""
    post, comb, collapsed = hc_site(p["ffn_hc"], streams, cfg)
    h = rms_norm(collapsed, p["post_ln"], cfg.rms_norm_eps)
    y = dense_mlp_core(p["mlp"], h, cfg)
    streams2 = hc_apply(post, comb, y, streams)
    return streams2


def ffn_site_moe(p, bank, streams, cfg, collect_router=False):
    """p['ffn_hc'], p['moe'] (router+shared), bank per chip."""
    post, comb, collapsed = hc_site(p["ffn_hc"], streams, cfg)
    h = rms_norm(collapsed, p["post_ln"], cfg.rms_norm_eps)
    w, ids = moe_router(p["moe"]["gate_w"], p["moe"]["e_bias"], h, cfg)
    sh = dense_mlp_core(p["moe"]["sh"], h, cfg)
    moe = moe_bank_core(bank, h, w, ids, cfg)
    y = sh + moe
    streams2 = hc_apply(post, comb, y, streams)
    if collect_router:
        return streams2, ids
    return streams2
