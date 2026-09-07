"""DeepSeek-V4-Flash serving config (de-simplified port).

Values from research/dsv4-0731-config.json + research/dsv4-port-spec.md §0
(all verified against the HF repo).  compress_ratios has 44 entries:
43 target layers + the MTP layer's attention type at index 43.
"""
from __future__ import annotations

import dataclasses
import json


@dataclasses.dataclass
class Dsv4Config:
    # ---- geometry (real values) ----
    hidden_size: int = 4096
    vocab_size: int = 129280
    n_layers: int = 43                 # target layers 0..42
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64            # trailing dims of q/k/o
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    # indexer (CSA layers)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # moe
    n_experts: int = 256
    top_k: int = 6
    moe_inter: int = 2048
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    n_hash_layers: int = 3             # layers 0-2: tid2eid hash routing
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    # rope
    rope_theta: float = 10000.0        # layers 0/1 + MTP (plain)
    compress_rope_theta: float = 160000.0  # CSA/HCA q/k + compressors
    yarn_factor: float = 16.0
    yarn_orig_ctx: int = 65536
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    eos_ids: tuple = (1,)              # <｜end▁of▁sentence｜>
    # serving
    max_ctx: int = 262144              # 256k default (1M via config)
    prefill_chunk: int = 256           # multiple of every ratio (4,128)
    n_slots: int = 8                   # hot experts per chip per layer

    compress_ratios: tuple = ()        # 44 entries; set by real()/tiny()

    @staticmethod
    def real(max_ctx: int = 262144) -> "Dsv4Config":
        c = Dsv4Config(max_ctx=max_ctx)
        ratios = [0, 0]
        for _ in range(21):
            ratios += [4, 128]
        ratios += [4, 0]               # layer 42 CSA; index 43 = MTP sliding
        c.compress_ratios = tuple(ratios)
        assert len(c.compress_ratios) == 44
        assert c.prefill_chunk % 128 == 0
        return c

    @staticmethod
    def tiny() -> "Dsv4Config":
        c = Dsv4Config(
            hidden_size=128, vocab_size=512, n_layers=6,
            n_heads=8, head_dim=80, rope_head_dim=16,
            q_lora_rank=32, o_lora_rank=32, o_groups=8,
            window_size=8,
            index_n_heads=8, index_head_dim=32, index_topk=4,
            n_experts=16, top_k=2, moe_inter=32,
            hc_mult=2, hc_sinkhorn_iters=3,
            max_ctx=256, prefill_chunk=16, n_slots=2,
        )
        c.compress_ratios = (0, 0, 4, 8, 4, 8, 0)
        return c

    # ------------------------------------------------------------ helpers
    def ratio(self, l: int) -> int:
        return self.compress_ratios[l]

    def layer_type(self, l: int) -> str:
        r = self.ratio(l)
        return {0: "sliding", 4: "csa", 128: "hca"}[r if r in (0, 4, 128)
                                                   else _tiny_ratio_map(r)]

    def is_hash(self, l: int) -> bool:
        return l < self.n_hash_layers

    def n_comp(self, ratio: int) -> int:
        """Compressed-cache entries per compressed layer (static size)."""
        return self.max_ctx // ratio

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["eos_ids"] = list(self.eos_ids)
        d["compress_ratios"] = list(self.compress_ratios)
        return json.dumps(d, indent=1)

    @staticmethod
    def from_json(s: str) -> "Dsv4Config":
        d = json.loads(s)
        d["eos_ids"] = tuple(d["eos_ids"])
        d["compress_ratios"] = tuple(d["compress_ratios"])
        return Dsv4Config(**d)


def _tiny_ratio_map(r: int) -> int:
    # tiny config uses ratio 8 as the HCA stand-in
    return 128 if r == 8 else r
