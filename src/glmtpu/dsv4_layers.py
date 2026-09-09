"""DeepSeek-V4-Flash per-layer math in pure JAX (de-simplified port).

Mirrors research/dsv4-inference-model.py + kernel.py (line-level where
noted) and research/dsv4-port-spec.md §1/§4.  Every function runs on ONE
chip with LOCAL shards inside jax.pmap(axis_name='tp').

Per-chip sharding (d = 8 chips):
  attention   64 heads -> 8 local heads/chip == 1 wo_a group (o_groups 8):
              wq_b/wo_a/wo_b/attn_sink head/group-sliced; wq_a/wkv/norms
              replicated; compressed caches replicated (MQA: every chip
              needs all KV); final wo_b partial -> psum.
  indexer     heads 64 -> 8/chip (scores psum over heads); weights_proj
              head-sliced; its compressor replicated.
  MoE         shared expert w1/w3 row-sharded, w2 col-sharded + psum;
              routed experts in per-chip FP4 banks (fixpoint refresh);
              router + tid2eid replicated.

KV cache formats (per layer):
  ring        [B, W, 512]: nope 448 dims fp8 u8 + per-64 e8m0-ish scales
              stored separately, rope 64 dims bf16.  Kept as one u8 array
              [B, W, 448] + scales [B, W, 7] + rope [B, W, 64].
  compressed  same split, [B, max_ctx//ratio, ...].
  indexer     keys-only bf16 [B, max_ctx//ratio, 128].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .dsv4_fp4 import dequant_jax
from .fp8 import F8_LUT

F8 = jnp.asarray(F8_LUT, jnp.float32)      # [256] e4m3 LUT


# ===========================================================================
# basic ops
# ===========================================================================

def rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    x32 = x32 * lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (w.astype(jnp.float32) * x32).astype(x.dtype)


def rms_norm_no_w(x, eps):
    x32 = x.astype(jnp.float32)
    return (x32 * lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True)
                            + eps)).astype(x.dtype)


def swiglu_clamped(gate, up, limit):
    gate = jnp.clip(gate, None, limit)
    up = jnp.clip(up, -limit, limit)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype)) * up


# ===========================================================================
# rope (dual tables, in-graph from static freq vectors)
# ===========================================================================

def yarn_freqs(cfg, compress: bool) -> jnp.ndarray:
    """Static [rd/2] frequency vector (YaRN interpolation for compress)."""
    rd = cfg.rope_head_dim
    base = cfg.compress_rope_theta if compress else cfg.rope_theta
    freqs = 1.0 / (base ** (jnp.arange(0, rd, 2, dtype=jnp.float32) / rd))
    if compress and cfg.yarn_orig_ctx > 0:
        # find_correction_range (model.py precompute_freqs_cis)
        def corr_dim(rot):
            return (rd * jnp.log(cfg.yarn_orig_ctx /
                                 (rot * 2 * jnp.pi)) / (2 * jnp.log(base)))
        low = jnp.floor(corr_dim(cfg.yarn_beta_fast)).astype(jnp.int32)
        high = jnp.ceil(corr_dim(cfg.yarn_beta_slow)).astype(jnp.int32)
        low, high = jnp.maximum(low, 0), jnp.minimum(high, rd - 1)
        ramp = jnp.clip((jnp.arange(rd // 2, dtype=jnp.float32) - low)
                        / jnp.maximum(high - low, 1), 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / cfg.yarn_factor * (1.0 - smooth) + freqs * smooth
    return freqs                                  # [rd/2] f32 static


def apply_rope(x, pos, freqs, inverse=False):
    """x [B, S, ..., head] (positions on axis 1); pos [S] i32; freqs
    [rd/2] static f32.  Interleaved-pair rope on the TRAILING rd dims
    of the last axis (consecutive channel pairs, model.py
    view_as_complex).  Returns rotated copy."""
    rd2 = freqs.shape[0]
    rd = rd2 * 2
    ang = pos[:, None].astype(jnp.float32) * freqs[None, :]   # [S, rd/2]
    cos = jnp.cos(ang)
    sin = jnp.sin(ang)
    # broadcast over axis 1 (S): [1, S, 1, ..., rd/2] matching x.ndim
    bshape = [1, x.shape[1]] + [1] * (x.ndim - 3) + [rd2]
    cosb = cos.reshape(bshape)
    sinb = sin.reshape(bshape)
    even = x[..., -rd::2].astype(jnp.float32)
    odd = x[..., -rd + 1::2].astype(jnp.float32)
    if inverse:                                   # conj (de-rotation)
        re, im = even * cosb + odd * sinb, odd * cosb - even * sinb
    else:
        re, im = even * cosb - odd * sinb, even * sinb + odd * cosb
    # interleave (re, im) pairs back onto the trailing rd channels:
    # even/odds are [..., rd/2]; build [..., rd] via stack + reshape
    rot = jnp.stack([re, im], axis=-1)            # [..., rd/2, 2]
    rot = rot.reshape(tuple(re.shape[:-1]) + (rd,))
    if rd == x.shape[-1]:
        return rot.astype(x.dtype)
    out = jnp.concatenate([x[..., :-rd].astype(jnp.float32), rot], axis=-1)
    return out.astype(x.dtype)


def hadamard(x):
    """Randomized-Hadamard-style rotation used pre-FP4 in the indexer.
    Deterministic n×n Hadamard (power-of-2 dims), scale n^-0.5."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, "hadamard needs pow2 last dim"
    # build H_n by Kronecker doubling (static, tiny for 128/32)
    h = jnp.array([[1.0, 1.0], [1.0, -1.0]], jnp.float32)
    k = 1
    while (1 << k) < n:
        h = jnp.concatenate([jnp.concatenate([h, h], axis=1),
                             jnp.concatenate([h, -h], axis=1)], axis=0)
        k += 1
    return (x.astype(jnp.float32) @ (h * (n ** -0.5))).astype(x.dtype)


# ===========================================================================
# fp8-sim KV quantization (block 64, ue8m0 power-of-2 scales, inplace)
# ===========================================================================

def kv_fp8_sim(x64):
    """x64 [..., N] (N%64==0) -> quant-dequant f32 (matches kernel
    act_quant inplace, scale_fmt=ue8m0: s = 2^ceil(log2(amax/448)))."""
    shp = x64.shape
    nb = shp[-1] // 64
    xb = x64.astype(jnp.float32).reshape(*shp[:-1], nb, 64)
    amax = jnp.maximum(jnp.abs(xb).max(axis=-1, keepdims=True), 1e-4)
    e = jnp.ceil(jnp.log2(amax / 448.0))
    s = jnp.exp2(e)
    q = jnp.clip(xb / s, -448.0, 448.0)
    # round to e4m3 grid via LUT on nearest-byte: use bitcast round-trip
    q8 = q.astype(jnp.float8_e4m3fn).astype(jnp.float32)
    return (q8 * s).reshape(shp)


def kv_pack(x, rd):
    """x [..., dh] f32 -> (nope u8 [..., dh-rd] fp8 codes (64-blocks),
    scales f32 [..., (dh-rd)//64], rope bf16 [..., rd])."""
    shp = x.shape
    nope = x[..., :-rd]
    rope = x[..., -rd:].astype(jnp.bfloat16)
    nb = (shp[-1] - rd) // 64
    xb = nope.reshape(*shp[:-1], nb, 64)
    amax = jnp.maximum(jnp.abs(xb).max(axis=-1, keepdims=True), 1e-4)
    e = jnp.ceil(jnp.log2(amax / 448.0))
    s = jnp.exp2(e)
    q = jnp.clip(xb / s, -448.0, 448.0)
    b = q.astype(jnp.float8_e4m3fn)
    u8 = lax.bitcast_convert_type(b, jnp.uint8).reshape(*shp[:-1],
                                                        shp[-1] - rd)
    scales = s.reshape(*shp[:-1], nb)
    return u8, scales, rope


def kv_unpack(u8, scales, rope):
    """inverse of kv_pack -> [..., dh] bf16."""
    codes = lax.bitcast_convert_type(
        u8, jnp.float8_e4m3fn).astype(jnp.float32)
    shp = codes.shape
    nope = (codes.reshape(*shp[:-1], -1, 64)
            * scales[..., None]).reshape(shp)
    return jnp.concatenate(
        [nope.astype(jnp.bfloat16), rope], axis=-1)


# ===========================================================================
# mHC hyper-connections (model.py hc_pre/hc_post, kernel hc_split_sinkhorn)
# ===========================================================================

def hc_site(hc, streams, cfg):
    """hc = {"fn": [mix, H*D] f32, "base": [mix] f32, "scale": [3] f32}.
    streams [B,S,H,D] -> (post [B,S,H], comb [B,S,H,H] indexed [b,s,j,k],
    collapsed [B,S,D]).  comb consumed TRANSPOSED downstream."""
    B, S, H, D = streams.shape
    flat = streams.reshape(B, S, H * D).astype(jnp.float32)
    flat = rms_norm_no_w(flat, cfg.rms_norm_eps)
    mixes = flat @ hc["fn"].astype(jnp.float32).T       # [B,S,(2+H)H]
    pre_w, post_w, comb_w = jnp.split(mixes, [H, 2 * H], axis=-1)
    pre_b, post_b, comb_b = jnp.split(hc["base"].astype(jnp.float32),
                                      [H, 2 * H])
    pre_s, post_s, comb_s = hc["scale"][0], hc["scale"][1], hc["scale"][2]
    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + cfg.hc_eps
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)
    cl = comb_w.reshape(B, S, H, H) * comb_s + comb_b.reshape(H, H)
    comb = jax.nn.softmax(cl, axis=-1) + cfg.hc_eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + cfg.hc_eps)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + cfg.hc_eps)
    collapsed = jnp.sum(pre[..., None] * streams, axis=2)
    return post, comb, collapsed.astype(streams.dtype)


def hc_apply(post, comb, sub_out, streams):
    """streams'[k] = post[k]*sub_out + Σ_j comb[j,k]*stream[j]."""
    term_a = post[..., None].astype(sub_out.dtype) * sub_out[..., None, :]
    term_b = jnp.einsum("bsjk,bsjd->bskd", comb.astype(streams.dtype),
                        streams)
    return term_a + term_b


# ===========================================================================
# attention core (shared by sliding / CSA / HCA)
# ===========================================================================

def sparse_attn_core(q, kv_ctx, kv_valid, sink, scale):
    """q [B,S,Hl,dh]; kv_ctx [B,N,dh] bf16 (dequantized); kv_valid
    [B,S,N] f32; sink [Hl] f32.  Softmax with the per-head sink added
    to the denominator (kernel sparse_attn).  Returns o [B,S,Hl,dh]."""
    scores = jnp.einsum("bshd,bnd->bhsn",
                        q.astype(jnp.float32),
                        kv_ctx.astype(jnp.float32)) * scale   # [B,Hl,S,N]
    scores = scores.astype(jnp.float32)
    neg = -1e30
    mask = kv_valid[:, None, :, :] > 0.5                # [B,1,S,N]
    scores = jnp.where(mask, scores, neg)
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    p = jnp.where(mask, p, 0.0)
    sum_p = jnp.sum(p, axis=-1, keepdims=True)
    sum_p = sum_p + jnp.exp(sink[None, :, None, None] - m)   # sink
    o = jnp.einsum("bhsn,bnd->bshd", p, kv_ctx.astype(jnp.float32))
    o = o / jnp.transpose(sum_p, (0, 2, 1, 3))       # [B,S,Hl,1]
    return o.astype(jnp.bfloat16)


def grouped_o_proj(o, wo_a_blk, wo_b_blk):
    """o [B,S,Hl,dh] -> group flatten [B,S,Hl*dh] (Hl*dh == H*dh/G_local
    only when G_local groups per chip; here 1 group/chip) ->
    wo_a_blk [olr, Hl*dh] -> [B,S,olr] -> wo_b_blk [D, olr] partial."""
    B, S, Hl, dh = o.shape
    flat = o.reshape(B, S, Hl * dh)
    mid = flat @ wo_a_blk.astype(jnp.bfloat16).T          # [B,S,olr]
    return mid @ wo_b_blk.astype(jnp.bfloat16).T          # partial [B,S,D]


# ===========================================================================
# compressor (CSA ratio-4 overlap + HCA ratio-128 non-overlap)
# ===========================================================================

def compressor_proj(p, x):
    """fp32 projections (checkpoint stores bf16; reference computes fp32).
    x [B,S,D] -> (kv [B,S,coff*Dc] f32, score [B,S,coff*Dc] f32)."""
    x32 = x.astype(jnp.float32)
    kv = x32 @ p["wkv"].astype(jnp.float32).T
    sc = x32 @ p["wgate"].astype(jnp.float32).T
    return kv, sc


def compressor_ape8(p):
    """ape [ratio, coff*Dc] -> transformed-layout bias table
    [coff*ratio, Dc]: slot k<ratio -> ape[k, :Dc] (Ca cols),
    slot k>=ratio -> ape[k-ratio, Dc:] (Cb cols).  Non-overlap
    (coff=1): ape [ratio, Dc] as-is."""
    ape = p["ape"].astype(jnp.float32)         # [ratio, coff*Dc]
    ratio = ape.shape[0]
    coff = ape.shape[1] // p["norm"].shape[0]
    d = p["norm"].shape[0]
    if coff == 1:
        return ape
    return jnp.concatenate([ape[:, :d], ape[:, d:]], axis=0)  # [2r, d]


def compressor_prefill(p, x, state, pos0, cfg):
    """Vectorized over the chunk's windows.  Requires pos0 % ratio == 0 and
    S % ratio == 0 (runtime guarantees; pos0 traced or static).
    Returns (entries [B, S/ratio, Dc], new_state)."""
    ratio = p["ape"].shape[0]
    coff = p["ape"].shape[1] // p["norm"].shape[0]
    d = p["norm"].shape[0]
    kv, sc = compressor_proj(p, x)                   # [B,S,coff*d]
    B, S = kv.shape[0], kv.shape[1]
    nw = S // ratio
    kvw = kv.reshape(B, nw, ratio, coff * d)         # window-major
    scw = sc.reshape(B, nw, ratio, coff * d)
    ape = p["ape"].astype(jnp.float32)               # [ratio, coff*d]
    scw = scw + ape[None, None]                      # in-window ape bias
    if coff == 2:
        # overlap: entry i pools [Ca(win i-1) cols :d | Cb(win i) cols d:]
        ca_prev = state[0][:, None, :ratio, :d]      # [B,1,r,d]
        sc_prev = state[1][:, None, :ratio, :d]
        ca_kv = jnp.concatenate(
            [ca_prev, kvw[:, :-1, :, :d]], axis=1)   # [B,nw,r,d]
        ca_sc = jnp.concatenate(
            [sc_prev, scw[:, :-1, :, :d]], axis=1)
        cb_kv = kvw[:, :, :, d:]                     # [B,nw,ratio,d]
        cb_sc = scw[:, :, :, d:]
        slots_kv = jnp.concatenate([ca_kv, cb_kv], axis=2)   # [B,nw,2r,d]
        slots_sc = jnp.concatenate([ca_sc, cb_sc], axis=2)
    else:
        slots_kv = kvw                               # [B,nw,ratio,d]
        slots_sc = scw
    wts = jax.nn.softmax(slots_sc, axis=2)           # over slots
    entry = jnp.sum(slots_kv * wts, axis=2)          # [B,nw,d]
    entry = rms_norm(entry, p["norm"].astype(jnp.float32), cfg.rms_norm_eps)
    # new state: last window full projection (kv/score incl. ape).
    # Layout must match compressor_decode's expectation:
    #   coff==2 (CSA): rows [0,ratio) = Ca of the last window,
    #                  rows [ratio,2*ratio) = Cb of the window being
    #                  built (empty after a boundary-aligned chunk).
    #   coff==1 (HCA): exactly `ratio` rows (the last window).
    if coff == 2:
        new_kv_state = jnp.concatenate(
            [kvw[:, -1], jnp.zeros((B, ratio, coff * d), jnp.float32)], axis=1)
        new_sc_state = jnp.concatenate(
            [scw[:, -1], jnp.zeros((B, ratio, coff * d), jnp.float32)], axis=1)
    else:
        new_kv_state = kvw[:, -1]
        new_sc_state = scw[:, -1]
    return entry, (new_kv_state, new_sc_state)


def compressor_decode(p, x, state, pos, cfg):
    """Single token at global position pos.  state as above.  Returns
    (entry [B,1,Dc] or None, new_state).  Mirrors model.py Compressor
    decode branch."""
    ratio = p["ape"].shape[0]
    coff = p["ape"].shape[1] // p["norm"].shape[0]
    d = p["norm"].shape[0]
    kv, sc = compressor_proj(p, x)                   # [B,1,coff*d]
    ape = p["ape"].astype(jnp.float32)
    slot_pos = pos % ratio                           # traced i32
    sc = sc + ape[slot_pos][None, None]              # [B,1,coff*d]
    kv_state, sc_state = state
    B = kv.shape[0]
    slot = (ratio + pos % ratio) if coff == 2 else (pos % ratio)
    onehot = (jnp.arange(kv_state.shape[1]) == slot)          # [rows]
    kv_state = jnp.where(onehot[None, :, None],
                         jnp.broadcast_to(kv, (B, 1, coff * d)), kv_state)
    sc_state = jnp.where(onehot[None, :, None],
                         jnp.broadcast_to(sc, (B, 1, coff * d)), sc_state)
    should = ((pos + 1) % ratio == 0).astype(jnp.float32)
    entry = jnp.zeros((B, 1, d), jnp.float32)
    if coff == 2:
        pool_kv = jnp.concatenate(
            [kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], axis=1)
        pool_sc = jnp.concatenate(
            [sc_state[:, :ratio, :d], sc_state[:, ratio:, d:]], axis=1)
    else:
        pool_kv, pool_sc = kv_state, sc_state
    wts = jax.nn.softmax(pool_sc, axis=1)
    e = jnp.sum(pool_kv * wts, axis=1, keepdims=True)     # [B,1,d]
    e = rms_norm(e, p["norm"].astype(jnp.float32), cfg.rms_norm_eps)
    entry = should * e
    # rotate state when a window completed
    if coff == 2:
        new_kv = jnp.where(should > 0.5,
                           jnp.concatenate([kv_state[:, ratio:],
                                            jnp.zeros_like(kv_state[:, :ratio])],
                                           axis=1), kv_state)
        new_sc = jnp.where(should > 0.5,
                           jnp.concatenate([sc_state[:, ratio:],
                                            jnp.zeros_like(sc_state[:, :ratio])],
                                           axis=1), sc_state)
    else:
        new_kv, new_sc = kv_state, sc_state
    return entry, should, (new_kv, new_sc)


# ===========================================================================
# indexer (CSA layers)
# ===========================================================================

def indexer_scores(p_idx, qr, x, idx_keys, pos0, cfg):
    """qr [B,S,qlora] (shared q-lora latent, post q_norm); x [B,S,D];
    idx_keys [B, Ci, 128] bf16 (compressed indexer cache).  Returns
    scores [B,S,Ci] f32 with -inf at invalid (causal / unwritten)."""
    rd = cfg.rope_head_dim
    freqs = p_idx["_cfreqs"]                          # static [rd/2]
    q = qr @ p_idx["wq_b"].astype(jnp.bfloat16).T     # [B,S,Hl*128]
    Hl = p_idx["wq_b"].shape[0] // cfg.index_head_dim
    q = q.reshape(qr.shape[0], qr.shape[1], Hl, cfg.index_head_dim)
    pos = pos0 + jnp.arange(qr.shape[1])
    q = apply_rope(q, pos, freqs)
    q = hadamard(q.reshape(*q.shape[:-1], -1)).reshape(q.shape)
    from .dsv4_fp4 import fp4_sim_jax
    q = fp4_sim_jax(q.astype(jnp.float32), 32).astype(jnp.bfloat16)
    w = (x @ p_idx["weights_proj"].astype(jnp.bfloat16).T) \
        * (cfg.index_head_dim ** -0.5 * Hl ** -0.5)  # hmm: n_heads total
    # NOTE: softmax_scale uses TOTAL heads for the 64^-0.5 factor; with
    # head sharding we use local count and psum below.
    # score_e = Σ_h w_h ReLU(q_h · k_e)
    s = jnp.einsum("bshd,btd->bsht",
                   q.astype(jnp.float32), idx_keys.astype(jnp.float32))
    s = jnp.maximum(s, 0.0)                           # ReLU
    s = jnp.einsum("bsht,bsh->bst", s, w.astype(jnp.float32))
    return s, pos


# ===========================================================================
# MoE (hash routing + noaux_tc sqrtsoftplus + FP4 banks + shared expert)
# ===========================================================================

def moe_router(p_gate, h, input_ids, cfg):
    """p_gate = {"w": [E,D], "bias": [E] or None, "tid2eid": [V,K] or None}.
    h [B,S,D]; input_ids [B,S] i32.  Returns (weights [B,S,K] f32,
    ids [B,S,K] i32) — model.py Gate."""
    logits = h.astype(jnp.float32) @ p_gate["w"].astype(jnp.float32).T
    scores = jnp.sqrt(jax.nn.softplus(logits))        # sqrtsoftplus
    if p_gate.get("tid2eid") is not None:
        ids = p_gate["tid2eid"][input_ids]            # [B,S,K] hash lookup
    else:
        choice = scores + p_gate["bias"].astype(jnp.float32)
        ids = lax.top_k(choice, cfg.top_k)[1]
    weights = jnp.take_along_axis(scores, ids, axis=-1)
    weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return weights, ids


def fp4_bank_core(bank, h, w, ids, cfg):
    """Per-chip FP4 expert bank.  bank: {"w1": [n,I,D/2] u8,
    "w1_s": [n,I,D/32] u8, "w2"/"w2_s", "w3"/"w3_s", "ids": [n] i32}.
    Every chip applies resident experts to ALL tokens; psum makes it
    exact given host coverage."""
    B, S, D = h.shape
    n = bank["ids"].shape[0]
    out = jnp.zeros((B, S, D), jnp.float32)
    for s_i in range(n):
        e = bank["ids"][s_i]
        match = (ids == e).astype(jnp.float32)         # [B,S,K]
        coeff = jnp.sum(match * w, axis=-1)            # [B,S]
        hb = h.astype(jnp.bfloat16)
        w1 = dequant_jax(bank["w1"][s_i], bank["w1_s"][s_i])   # [I,D]
        w3 = dequant_jax(bank["w3"][s_i], bank["w3_s"][s_i])
        w2 = dequant_jax(bank["w2"][s_i], bank["w2_s"][s_i])   # [D,I]
        gate = hb @ w1.T                                 # [B,S,I]
        up = hb @ w3.T
        hid = swiglu_clamped(gate, up, cfg.swiglu_limit)
        y = hid @ w2.T                                   # [B,S,D]
        out = out + y.astype(jnp.float32) * coeff[..., None]
    return lax.psum(out.astype(h.dtype), "tp")


def dense_mlp_core(p, h, cfg):
    """Shared expert (BF16 after host dequant): w1/w3 row-sharded, w2
    col-sharded + psum.  p = {"w1": [I/d, D], "w3": ..., "w2": [D, I/d]}."""
    gate = h @ p["w1"].T
    up = h @ p["w3"].T
    hid = swiglu_clamped(gate, up, cfg.swiglu_limit)
    return lax.psum(hid @ p["w2"].T, "tp")
