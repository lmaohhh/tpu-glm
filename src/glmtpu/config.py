"""GLM-5.3-Flash (glm5_next) serving config.

Derived from zai-org/GLM-5.3-Flash config.json (transformers 5.16.0) and
verified against real safetensors shard headers.  The orcarouter ab-literated
FP8 variant shares identical geometry (only weights differ).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Optional


@dataclasses.dataclass
class GlmConfig:
    # ---- layer geometry (real values from config.json) ----
    hidden_size: int = 4096
    vocab_size: int = 154880
    n_layers: int = 45              # main stack 0..44; layer 45 (MTP) never loaded
    n_kda_heads: int = 64
    kda_head_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0
    dsa_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    qk_rope_head_dim: int = 0      # NoPE: no rope anywhere in the text stack
    dense_inter: int = 12288       # first 3 layers
    moe_inter: int = 2048
    n_experts: int = 288
    top_k: int = 8
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-5
    eos_ids: tuple = (154820, 154827, 154829)
    # serving
    max_ctx: int = 32768
    prefill_chunk: int = 256
    n_slots: int = 64           # decode bank slots per chip (>= top_k)
    n_passes: int = 1           # reserved

    kda_layers: tuple = ()
    dsa_layers: tuple = ()
    moe_layers: tuple = ()
    dense_mlp_layers: tuple = ()

    @staticmethod
    def real() -> "GlmConfig":
        c = GlmConfig()
        c.kda_layers = tuple(l for l in range(45) if l % 4 != 3)
        c.dsa_layers = tuple(range(3, 45, 4))
        c.dense_mlp_layers = (0, 1, 2)
        c.moe_layers = tuple(range(3, 45))
        return c

    @staticmethod
    def tiny() -> "GlmConfig":
        c = GlmConfig(
            hidden_size=128, vocab_size=512, n_layers=4,
            n_kda_heads=8, kda_head_dim=16, conv_kernel=4,
            dsa_heads=8, q_lora_rank=32, kv_lora_rank=16,
            qk_nope_head_dim=8, v_head_dim=8,
            dense_inter=128, moe_inter=64, n_experts=16, top_k=2,
            hc_mult=2, hc_sinkhorn_iters=3,
            max_ctx=256, prefill_chunk=64, n_slots=2, n_passes=1,
        )
        c.kda_layers = (0, 1, 2)
        c.dsa_layers = (3,)
        c.dense_mlp_layers = (0, 1)
        c.moe_layers = (2, 3)
        return c

    def is_kda(self, l: int) -> bool:
        return l in self.kda_layers

    def is_moe(self, l: int) -> bool:
        return l in self.moe_layers

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["eos_ids"] = list(self.eos_ids)
        return json.dumps(d, indent=1)

    @staticmethod
    def from_json(s: str) -> "GlmConfig":
        d = json.loads(s)
        d["eos_ids"] = tuple(d["eos_ids"])
        return GlmConfig(**d)
