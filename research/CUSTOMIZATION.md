# Customization & model-selection spec for the TPU v5e-8 hot-expert engine

Everything below is tunable in the notebook.  All tunables live in
`glmtpu/config.py` (`GlmConfig`), with per-model hints for future ports.

## 1. KV cache to 1M: what "custom-turboquant" means here

There is no "quantized KV" mode to enable — the DSA latent cache is already
the compressed form (512-dim latents, 1 KiB/token/layer, 64-way
head-shared). What actually gates 1M context on this hardware is the
interaction of **latent-cache replication × bank budget**.  The clean
"turbo" route: **stop replicating the KV cache** — shard it by head across
chips (each chip keeps `kv_lora_rank=512/8`... i.e. 64-dim slices of the
latent, since W_UK/W_UV are per-head) → 1M ctx = 11 layers × 1M × 64 × 2B ≈
139 MiB/chip.  Wait — the correct split for absorbed-MLA by head:
`kv_lora_rank` rows already map 1:1 to... in fact each head owns 8 rows of
the latent; sharding the [B, T, 512] cache on the T axis (2.7M/8 rows... )
does NOT reduce it.  The effective lever: the latent cache is only 1 KiB/token
per LAYER, and with 11 DSA layers, 1M ctx costs 11 GiB replicated.  To fit
1M:

  - Keep 32k default: banks 8 GiB/chip, cache 45 MiB/chip — comfortable.
  - 1M context, replicated cache: 1.4 GiB/chip — banks drop to ~2.9 GiB/
    chip: tight but workable, decode slows (frequent refresh misses).
  - 1M context + *un-replicated, head-sharded* DSA cache:
      cache ≈ 11 L × 1M tok × (512/8 dims... split) — with head-sharing
      (num_key_value_heads=64, kv_b per-head) each head-8th row lands on a
      chip: 1M × 64 × 2B × 11 / 8 ≈ 176 MiB/chip.  Not a win vs
      1.4 GiB unless the rank-512 latent is also sliced; a real win needs
      the 8-bit/4-bit latent quant (true "KV-quant"), which changes dequant
      math (add a scale per 16-lane group per head).

  **Recommendation: land the indexer + 1M together as one unit.**  The
  indexer (top-2048 selection) is what makes 1M *usable* (full attention
  at 1M on 8 chips is compute-bound: 64 heads × 1M² attn — minutes per
  token).  With the indexer, DSA layers only attend to ~2048+tail tokens, so
  prefill of 1M stays ~ the same cost as 32k, the KV "cache" becomes the
  index structure, and 1M fits with a 2.9-3 GiB/chip bank budget.
  Also enable 8-bit latent-KV to make even the replicated variant fit
  comfortably.  ("dSpark" / DSA turbo: see #3.)

## 2. Pretrained-candidate matrix (all fit-in-HBM or bankable on v5e-8)

Verified from the Hub API (params from safetensors totals, Sep 2026):

| Model | Total size | Fits 8×16? | Notes |
|---|---|---|---|
| **Qwen3.6-35B-A3B** (BF16, 36.0 GB) | 36.0 GB | **yes, full residency, TP-8** | 3B active — likely best quality-per-watt fit; pure dense+MoE, no exotic ops.  Fastest of the big ones. |
| **gpt-oss-120b** (BF16 2.2B + U8 114.7B, 182.4 GB) | 182.4 GB | no — needs the hot-bank scheme | 5.1B active.  MXFP4-packed experts (U8 view) — needs a 4-bit dequant path; engine already supports FP8. |
| **GLM-5.3-Flash-NVFP4** (169.1 GB) | 169.1 GB | banked like the FP8 one | 18B active; NVFP4 decoding: 4-bit scaled blocks — needs its own dequant table |
| **Gemma-4-31B-it** (31.3 GB, ungated) | 31.3 GB | yes, full residency | Likely strongest *dense* (non-MoE) option; no per-expert streaming. |
| DeepSeek-V4-Flash (321 GB) | 321 GB | banked (current build) | Current target (the zai/GLM5.3-Flash weights). |
| **Qwen3.8-Flash-Next (125B + 51B n-gram dict)** | 360 GB (+51 GB dict in RAM) | banked + RAM-side dict | 6B active; the n-gram dictionary is DESIGNED for split residency (system RAM) — same philosophy as this engine. |

**Ranked recommendation for this hardware (your use):**
1. **GLM-5.3-Flash** (current) — highest intelligence of the bankable set
   (AA index 57); needs the indexer to shine.
2. **Qwen3.8-Flash-Next** — architecturally aligned with the hot-bank
   design (its N-gram dict lives in system RAM by design); 6B active is the
   least compute; 125B dense-bankable portion + 51B pure-RAM portion.
3. **Qwen3.6-35B-A3B** — if you'd rather have full residency, no
   streaming complexity, slightly lower ceiling than GLM.
4. Avoid: DeepSeek-V4 full (321 GB, 20h+ weekly TPU budget for the total
   download each session).

All of these need a port of the layer math (KDA/DSA/mHC are specific to
GLM5-next; the MoE bank machinery is reusable across all of them).

## 3. De-simplify queue (one subagent at a time)

1. DSA indexer + 1M context (pair; biggest quality unlock)
2. 8-bit/4-bit latent-KV ("turbo-quant") so 1M fits without squeezing
   the banks
3. Streaming routing (route-first) for prefill + async bank prefetch
4. Chat UX: stream `reasoning_content` separately, hard-error on
   non-text modalities, resumable shard downloads
5. Optionally: D-Spark/DSA2 for the 34 KDA layers (replaces KDA; new
   pipeline)

---

## 4. Same-size SOTA survey (fills 330 GB RAM + 120 GiB HBM) — Sep 2026, all verified via Hub API

Decode speed on this engine is governed by **top-k expert bytes/token** (RAM→HBM
streaming) and port effort by architecture overlap with glm5_next.

| Model | Disk | Experts | top-k | Expert MiB/tok | Port effort | Verdict |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash FP8** (current) | 306 GiB | 42×288 | 8 | 192 | — | running; highest ceiling |
| **DeepSeek-V4-Flash FP8** | 149 GiB | 43×256 (**FP4 experts!**) | 6+1sh | 144 (FP4: ~72) | **low** — same DNA as glm5_next (mHC hc_mult=4, indexer top-512, DSA compress 4/128, MTP) | **best speed fit**: FP4 experts halve streaming; needs an FP4 dequant table (FP8 LUT → 4-bit LUT + scale layout) |
| **Qwen3.8-Flash-Next FP8** | 173 GiB | 48×512 | 10 | 47 | medium — linear+full hybrid like ours, plus 51 GB n-gram dict (designed for RAM residency) | **cheapest decode** (47 MiB/tok = 4× less streaming); the n-gram dict = speculative drafts, big decode multiplier |
| **MiniMax-M2.5 FP8** | 214 GiB | 62×256 | 8 | 108 | medium — standard MoE+full attention (no linear attn), gated FP8 blocks | solid all-rounder; NVFP4 variant 130 GiB |
| GLM-5.3 full FP8 (dealignai UNCENSORED) | 791 GiB | — | — | — | low (same arch) | **too big** for 330 GB RAM even banked — needs NVFP4 (~400 GiB, still over) |
| Kimi-K2.6 NVFP4 | 550 GiB | — | — | — | — | too big |
| Qwen3.5-397B-A52B | ~400 GiB FP8 | — | 52B active | huge | — | too big + gated |

**Recommendation order for this machine:**
1. **Keep GLM-5.3-Flash** as primary (306 GiB fits with 24 GiB headroom).
2. **Add DeepSeek-V4-Flash FP8 as the speed build** — same engine bones
   (mHC + indexer + DSA + MTP all present), 149 GiB leaves 180 GiB headroom,
   FP4 experts halve the streaming bytes.  Port = new config + FP4 dequant +
   layer-math swap (closer to glm5_next than it looks).
3. **Qwen3.8-Flash-Next** as the long-context champion (4× less expert
   streaming, n-gram speculative decode) — port is bigger (n-gram dict
   lookup + qwen4_exp linear attention variant).
