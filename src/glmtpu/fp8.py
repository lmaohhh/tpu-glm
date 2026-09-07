"""FP8 (e4m3fn) 128x128-blockwise dequantization.

numpy LUT for host-side dequant (dense/small weights -> BF16 at load time);
JAX in-graph version only for routed-expert banks (whole experts, row0=col0=0).
"""
from __future__ import annotations

import numpy as np

BLOCK = 128


def _build_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for u in range(256):
        sign = -1.0 if (u & 0x80) else 1.0
        exp = (u >> 3) & 0xF
        man = u & 0x7
        if exp == 0:
            val = (2.0 ** -6) * (man / 8.0)          # subnormal
        elif (u & 0x7F) == 0x7F:
            val = 0.0                                 # NaN -> 0 (not in weights)
        else:
            val = (2.0 ** (exp - 7)) * (1.0 + man / 8.0)
        lut[u] = sign * val
    return lut


F8_LUT = _build_lut()


def dequant_np(fp8_u8: np.ndarray, scale_inv: np.ndarray) -> np.ndarray:
    """uint8 [R,C] + f32 scale_inv [ceil(R/128), ceil(C/128)] -> f32 [R,C].
    W = fp8 * scale_inv (scale_inv holds the INVERSE scale)."""
    R, C = fp8_u8.shape
    vals = F8_LUT[fp8_u8]                               # [R,C] f32 LUT gather
    out = np.empty((R, C), dtype=np.float32)
    for i in range(0, R, BLOCK):
        for j in range(0, C, BLOCK):
            out[i:i + BLOCK, j:j + BLOCK] = (
                vals[i:i + BLOCK, j:j + BLOCK] * scale_inv[i // BLOCK, j // BLOCK])
    return out


def fp8_matmul(x, fp8_u8, scale_inv, block=BLOCK):
    """JAX in-graph: x [.., in] bf16 @ dequant(fp8 [out, in])^T -> [.., out].
    Only used for whole-expert banks (global row/col both start at 0)."""
    import jax.numpy as jnp
    from jax import lax
    out_dim, in_dim = fp8_u8.shape
    fp8 = lax.bitcast_convert_type(fp8_u8, jnp.float8_e4m3fn).astype(jnp.float32)
    row = jnp.arange(out_dim) // block
    col = jnp.arange(in_dim) // block
    s = scale_inv[row[:, None], col[None, :]]           # [out, in] f32
    w = (fp8 * s).astype(jnp.bfloat16)
    return x @ w.T
