"""FP4 (e2m1) expert format: packing, dequant, simulation, reference decoder.

Layout (research/dsv4-port-spec.md §3, from DeepSeek inference/convert.py +
kernel.py):
  - e2m1: 1 sign + 2 exp + 1 mantissa; 16 values
    [0, .5, 1, 1.5, 2, 3, 4, 6] and their negatives.
  - packed 2 per byte along K, LOW nibble = earlier K element.
  - scale: e8m0 (power-of-2 exponent byte) per 32 fp4 elements along K,
    per output row: scale tensor [out, in//32] uint8.
Dequant in f32 is EXACT (e2m1 mantissa 2 bits, e8m0 power-of-2); bf16 cast
lossless too (values ±{0,.5,1,1.5,2,3,4,6}·2^k fit in 8-bit mantissa).
"""
from __future__ import annotations

import numpy as np

FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)

FP4_BLOCK = 32


# ------------------------------------------------------------------ numpy
def dequant_np(packed_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Reference decoder.  packed [out, in//2] uint8, scale [out, in//32]
    uint8 (e8m0) -> f32 [out, in].  Low nibble first along K."""
    out_dim, half_in = packed_u8.shape
    in_dim = half_in * 2
    low = (packed_u8 & 0x0F).astype(np.int64)
    high = (packed_u8 >> 4).astype(np.int64)
    vals = np.empty((out_dim, in_dim), np.float32)
    vals[:, 0::2] = FP4_TABLE[low]
    vals[:, 1::2] = FP4_TABLE[high]
    # e8m0 scale: value = 2^(byte - 127)
    scales = np.exp2(scale_u8.astype(np.float32) - 127.0)  # [out, in//32]
    nblocks = in_dim // FP4_BLOCK
    for j in range(nblocks):
        vals[:, j * FP4_BLOCK:(j + 1) * FP4_BLOCK] *= scales[:, j:j + 1]
    return vals


def quantize_np(w: np.ndarray) -> tuple:
    """Quantize f32 [out, in] -> (packed u8 [out, in//2], scale u8
    [out, in//32]) with per-32 rowwise e8m0 scales (round-to-nearest e2m1).
    Used by dsv4_params to fabricate fp4 test weights."""
    out_dim, in_dim = w.shape
    assert in_dim % FP4_BLOCK == 0 and in_dim % 2 == 0
    blocks = w.reshape(out_dim, in_dim // FP4_BLOCK, FP4_BLOCK)
    amax = np.maximum(np.abs(blocks).max(axis=-1), 1e-30)  # [out, nb]
    # e8m0 scale = 2^ceil(log2(amax / 6)) (power-of-2, like the kernel)
    e = np.ceil(np.log2(amax / 6.0))
    e = np.clip(e, -127, 127)
    scale_u8 = (e + 127.0).astype(np.uint8)
    s = np.exp2(e)
    t = blocks / s[..., None]                     # target range [-6, 6]
    # round to nearest e2m1 magnitude: pick nibble index 0..7
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    a = np.abs(t)
    idx = np.clip(np.searchsorted(grid, a), 1, 7)      # candidate upper
    left, right = grid[idx - 1], grid[idx]
    up = (a - left) > (right - a)                       # round up?
    nib7 = np.where(up, idx, idx - 1)                   # magnitude idx 0..7
    nib7 = np.where(a == 0.0, 0, nib7)
    # e2m1 nibble: magnitude idx | 0x8 if negative
    nib = (nib7 + np.where(t < 0, 8, 0)).astype(np.int64)
    nib = nib.reshape(out_dim, in_dim)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_u8


# ------------------------------------------------------------------ jax
def dequant_jax(packed, scale):
    """In-graph dequant.  packed [out, in//2] uint8, scale [out, in//32]
    uint8 -> bf16 [out, in].  Nibble ops + 16-entry LUT + exp2(scale-127)."""
    import jax.numpy as jnp

    out_dim, half_in = packed.shape
    in_dim = half_in * 2
    lut = jnp.asarray(FP4_TABLE, jnp.float32)          # [16]
    low = (packed & 0x0F).astype(jnp.int32)            # [out, in//2]
    high = (packed >> 4).astype(jnp.int32)
    v0 = lut[low]                                       # even K elems
    v1 = lut[high]                                      # odd K elems
    vals = jnp.empty((out_dim, in_dim), jnp.float32)
    vals = vals.at[:, 0::2].set(v0).at[:, 1::2].set(v1)
    s = jnp.exp2(scale.astype(jnp.float32) - 127.0)     # [out, in//32]
    nb = in_dim // FP4_BLOCK
    cols = jnp.arange(in_dim) // FP4_BLOCK
    vals = vals * s[:, cols]
    return vals.astype(jnp.bfloat16)


def fp4_sim_jax(x, block: int = FP4_BLOCK):
    """FP4 activation simulation (indexer q/k): quant-dequant with per-32
    e8m0 scales, round-to-nearest e2m1 grid.  x [..., N] f32 -> f32."""
    import jax
    import jax.numpy as jnp

    shape = x.shape
    n = shape[-1]
    assert n % block == 0
    xb = x.reshape(-1, n // block, block)
    amax = jnp.maximum(jnp.abs(xb).max(axis=-1, keepdims=True), 1e-30)
    e = jnp.ceil(jnp.log2(amax / 6.0))
    s = jnp.exp2(e)
    t = xb / s
    grid = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], jnp.float32)
    a = jnp.abs(t)
    idx = jnp.clip(jnp.searchsorted(grid, a), 1, 7)
    left, right = grid[idx - 1], grid[idx]
    chosen = jnp.where(a - left <= right - a, left, right)
    mag = jnp.where(a == 0.0, 0.0, chosen)
    y = jnp.sign(t) * mag * s
    return y.reshape(shape)
