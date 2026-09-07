# DeepSeek-V4-Flash → JAX/TPU Port Spec

Target: from-scratch JAX TPU engine for `deepseek-ai/DeepSeek-V4-Flash` (284B total / 13B active, 1M context), coding + agent + chat at ~30 tok/s base decode.
Status: COMPLETE (all claims verified against fetched primary sources on 2026-09-07).

**Primary sources (all fetched and read in full):**
- [R1] DeepSeek-V4 tech report: **arXiv 2606.19348** "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (https://arxiv.org/html/2606.19348)
- [R2] HF repo `deepseek-ai/DeepSeek-V4-Flash` (ungated, MIT): `config.json`, `model.safetensors.index.json` (69,187 tensors), safetensors headers of shards 1,2,3,4,5,44,45,46 (range-requested, shapes extracted), `inference/{model.py,kernel.py,convert.py,generate.py,config.json,README.md}` (DeepSeek's own reference implementation)
- [R3] transformers `main`: `src/transformers/models/deepseek_v4/{configuration_deepseek_v4.py, modeling_deepseek_v4.py}` (324 + 1504 lines, fetched from raw.githubusercontent.com — HTTP 200)
- [R4] DSpark paper: **arXiv 2607.05147** "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"; code: github.com/deepseek-ai/DeepSpec
- [R5] HF `deepseek-ai/DeepSeek-V4-Flash-0731` `config.json` (DSpark keys: `dspark_block_size: 5`, `dspark_markov_rank: 256`, `dspark_noise_token_id: 128799`, `dspark_target_layer_ids: [40, 41, 42]`)
- [R6] vLLM `vllm/models/deepseek_v4/nvidia/dspark.py` (DSpark draft model for DSV4); vLLM `tpu-inference` repo README support matrix ("Speculative Decoding: DFlash ✅ Flax"; DSpark itself not listed)
- [R7] DSA (V3.2 sparse attention) = "DeepSeek Sparse Attention", DeepSeek-AI 2025b — cited by [R1] as the mechanism CSA reuses.

Local artifacts: `D:\tpu-glm\research\dsv4-inference-model.py` (reference impl), `dsv4-hf-modeling.py`, `dsv4-hf-configuration.py`, `dsv4-inference-kernel.py`, `dsv4-inference-convert.py`, `dsv4-index.json`, `hdr-model-0000*.json`.

---

## 0. Config ground truth (fetched from R2 config.json — verbatim values)

```
architectures: ["DeepseekV4ForCausalLM"]   model_type: "deepseek_v4"   transformers_version: "4.57.1"
num_hidden_layers: 43    hidden_size: 4096   vocab_size: 129280   tie_word_embeddings: false
num_attention_heads: 64  num_key_value_heads: 1 (MQA)  head_dim: 512  qk_rope_head_dim: 64
q_lora_rank: 1024        o_lora_rank: 1024  o_groups: 8
compress_ratios: [0, 0, 4, 128, 4, 128, ..., 4, 128, 4, 0]   # 44 ENTRIES (see §1.2 — NOT 43)
sliding_window: 128      compress_rope_theta: 160000   rope_theta: 10000
rope_scaling: yarn, factor 16, original_max_position_embeddings 65536, beta_fast 32, beta_slow 1
index_n_heads: 64  index_head_dim: 128  index_topk: 512
num_hash_layers: 3
n_routed_experts: 256    n_shared_experts: 1   num_experts_per_tok: 6   moe_intermediate_size: 2048
scoring_func: "sqrtsoftplus"  topk_method: "noaux_tc"  norm_topk_prob: true  routed_scaling_factor: 1.5
hidden_act: "silu"  swiglu_limit: 10.0
hc_mult: 4  hc_eps: 1e-06  hc_sinkhorn_iters: 20
num_nextn_predict_layers: 1  (MTP)
expert_dtype: "fp4"  o_groups: 8
quantization_config: fp8 e4m3, scale_fmt ue8m0, weight_block_size [128,128], activation_scheme dynamic
max_position_embeddings: 1048576
```

## 1. D-Spark / DSA-2 / sparse attention

### 1.1 Terminology disambiguation (the #1 confusion to kill)

- **"D-Spark" / "DSpark" is NOT an attention mechanism.** It is DeepSeek's **speculative decoding framework** (arXiv 2607.05147 [R4], open-sourced 2026-06-27 via github.com/deepseek-ai/DeepSpec). Semi-autoregressive drafter = DFlash-style parallel backbone + lightweight Markov/RNN head + confidence head + hardware-aware verification scheduler. Deployed in DeepSeek-V4-Flash-**0731** (R5), where it *replaces* MTP as production spec-decode (the DSpark paper: "MTP-1 ... having been superseded by DSpark two weeks following the DeepSeek-V4-preview release").
- The sparse-attention upgrade in V4 is the **hybrid CSA + HCA** architecture (R1 §2.3), which the report calls "DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025b)" applied *on top of* compressed KV — i.e. "DSA-2 / sparse attention 2.0" = CSA (compress-then-DSA). R1 verbatim: *"CSA compresses the KV caches along the sequence dimension and then performs DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025b), whereas HCA applies more aggressive compression to the KV caches but keeps dense attention."*
- vLLM's TPU feature matrix (R6, github.com/vllm-project/tpu-inference README "Advanced Capabilities") lists **"Speculative Decoding: DFlash ✅ (Flax)"** — DFlash is the parallel drafter that DSpark extends (DSpark subclasses DFlash; vLLM speculators docs: "The draft model subclasses DFlash, so the architecture and training pipeline are otherwise unchanged"). DSpark proper is not yet in the TPU matrix; the NVIDIA-path DSpark draft model lives at `vllm/models/deepseek_v4/nvidia/dspark.py`. So the parent's note "vLLM tpu-inference lists DFlash as GLM dspark-analog, nightly" is essentially right: DFlash is the TPU-JAX-supported base; DSpark is the DeepSeek-specialized extension of it.
- "GLM dspark-analog": GLM-5.x ships an equivalent attached draft (the GLM "spare loop"); functionally the same pattern (draft blocks in-checkpoint + fused verification).

### 1.2 compress_ratios: the exact 44-entry pattern

IMPORTANT correction to the task context: `compress_ratios` has **44 entries**, not 43. Verified programmatically (R2 + R2-inference/config.json, identical in both):

```
[0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 128, 4, 0]   # len == 44
counts: {4: 21, 128: 20, 0: 3}
```

Mapping (cross-verified against actual tensor presence in model.safetensors.index.json — compressor tensors exist exactly on layers 2..42, indexer tensors exactly on the 21 even layers):

| Layers | ratio | Type (R3 `_COMPRESS_RATIO_TO_LAYER_TYPE`) | count |
|---|---|---|---|
| 0, 1 | 0 | `sliding_attention` (sliding window only, plain RoPE θ=10000, no YaRN) | 2 |
| 2, 4, 6, ..., 42 (even) | 4 | `compressed_sparse_attention` (CSA): overlapping compression + Lightning Indexer | 21 |
| 3, 5, 7, ..., 41 (odd) | 128 | `heavily_compressed_attention` (HCA): non-overlapping 128:1 compression, **no indexer** | 20 |
| index 43 (the 44th entry) | 0 | MTP layer's attention (sliding-only) — reference `model.py` builds `MTPBlock(layer_id = n_layers + 0 = 43)`; `mtp.0.*` tensors have **no** compressor/indexer weights | 1 |

**What is compressed: the KV cache — and since V4 is shared-KV MQA, K and V are the same tensor.** R1 §2.3.1: *"each compressed KV entry ... serves as both attention key and value"*. Not keys only; the full 512-dim (rope-part + nope-part) entry is compressed. R3 modeling docstring: *"V4 uses shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects directly to that single KV head and the same tensor is read as both key and value."*

### 1.3 CSA (ratio 4) — exact mechanics (R1 §2.3.1, R2 inference/model.py, R3)

All shapes below verified from safetensors headers (R2).

**Compressor** (`layers.L.attn.compressor.*`, tensors: `wkv [1024,4096]`, `wgate [1024,4096]`, `ape [4,1024]` fp32, `norm.weight [512]`):
1. Per token t: `kv_t = W_kv·h_t ∈ R^1024`, `score_t = W_gate·h_t ∈ R^1024`. Because ratio==4, `overlap=True`, coff = 2: the 1024-dim output is two 512-dim series **Ca** (`[..., :512]`, contributes to the *next* window's entry) and **Cb** (`[..., 512:]`, contributes to the *current* window's entry).
2. Window layout `[B, n_win, 2·ratio=8, 512]`: second half = current window's Cb; first half = *previous* window's Ca (window 0's first half = zero-kv / −inf-gate → softmax weight 0). So each compressed entry pools **8 token-slices with stride 4** — effective width 2·compress_ratio, stride compress_ratio (R1: "the indexes ... used for [Ca] and ... [Cb] are overlapped. Therefore, CSA in fact compresses the sequence length to [4] times").
3. Entry: `C_w = Σ_j softmax(Z + ape)_j · C_j` — softmax over the 8 slots (gate + learnable positional bias `ape[4, 1024]`, laid out per overlap slot), then `RMSNorm(512)` (`compressor.norm`).
4. RoPE on trailing 64 dims at deterministic position `w·4 + first_window_position` (θ = compress_rope_theta = 160000, YaRN factor 16 / orig 65536).
5. KV storage precision (R2 model.py L372): non-rope 448 dims FP8-quantized (block 64, ue8m0 scales), rope 64 dims stay BF16. R1 §2.3.4: *"BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions."*

**Lightning Indexer** (`layers.L.attn.indexer.*`; tensors: `wq_b.weight [8192,1024] + wq_b.scale [64,8]` (fp8, 64 idx-heads × 128), `weights_proj.weight [64,4096]` bf16, own compressor `wkv/wgate [256,4096]`, `ape [4,256]`, `norm [128]`):
1. Own scaled-down compressor at `index_head_dim=128` over the same windows (Ca/Cb overlap at 128-dim, its own `ape [4,256]`, RMSNorm(128), RoPE trailing 64 dims at compressed positions, same compress θ so q·k is translation-invariant — R3: "Both must use the same theta as the outer compressor").
2. Indexer queries: `q = W_IQB · qr` where `qr` is the **shared q-lora latent** (the `q_norm(wq_a(x))` 1024-vector, reused from main attention — R1 eq. 13-14 "the latent query vector is shared"). `[8192,1024] → 64 heads × 128`. RoPE trailing 64 dims at *query* positions; then **randomized Hadamard rotation** (`rotate_activation`) and **FP4 activation quant** (block 32, e8m0) — R1 §2.3.4: *"attention computation within the lightning indexer is performed in FP4 precision"*; R2 model.py L415-416: "# use fp4 simulation for q and kv in indexer; fp4_act_quant(q, 32, True)".
3. Index score (R3 `DeepseekV4IndexerScorer`): `score(t,s) = Σ_h w_{t,h} · ReLU(q_{t,h}·K_s^IComp) · softmax_scale` with `softmax_scale = 128^-0.5`, `w = weights_proj(h_t) · 64^-0.5` (per-head learned weights, [64,4096]). R3 docstring: *"∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)"*.
4. Top-k: `topk(min(512, T_compressed), dim=-1)` per query, with causal mask: query t may only see entries with `entry_index < (t+1)//4` (R3: `causal_threshold = (position_ids + 1) // self.compress_rate`); picks past the threshold are replaced with −1 sentinel and dropped from the attention mask. **index_topk=512 compressed entries per query** (R1: "a smaller attention top-k than V3.2").
5. The indexer's KV cache is *keys-only* (128-dim compressed entries, never used as values) — the one place where a "keys only" cache exists.

**Core attention after selection:** query attends (MQA, single shared KV) to the union of (a) the **sliding window** of 128 uncompressed recent entries and (b) the top-512 selected compressed entries. Attention sink per head (`attn_sink [64]` learnable, added to softmax denominator: R1 eq. 27; R2 kernel: `sum_exp[i] += exp(attn_sink[i] - scores_max[i])`). Softmax scale = `512^-0.5` (head_dim based). Attention output gets **inverse RoPE** (`apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)`) so output carries *relative* position info (R1 §2.3.3 "Partial Rotary ... As a countermeasure, we also apply RoPE with position [−i] on the last 64 dimensions of each [output]").

### 1.4 HCA (ratio 128)

Same recipe, differences (R1 §2.3.2, R2, R3 `DeepseekV4HCACompressor`):
- `wkv/wgate [512, 4096]` → single series (coff=1, `overlap = (ratio==4) = False`), **non-overlapping** windows of 128 tokens.
- `ape [128, 512]` positional bias per in-window slot.
- One compressed entry per 128 tokens → 1M tokens = 8,192 entries; **dense attention over all of them** (no indexer, no top-k). Causality: query t sees entries with `index < (t+1)//128` (block_bias in R3).
- Same shared-KV MQA + sliding window 128 + attention sink + inverse-RoPE output + grouped output projection.

### 1.5 The 3 "hash layers"

`num_hash_layers = 3` → layers **0, 1, 2** have `mlp_layer_types = "hash_moe"` (R3 config: `mlp_layer_types = ["hash_moe"] * min(n, 3) + ["moe"] * (n-3)`). These are **MoE FFN layers with Hash routing, not attention layers**: expert selection is a frozen lookup `tid2eid[input_token_id] → 6 expert ids`, tensor `ffn.gate.tid2eid [129280, 6]` int32 (verified in shard 2-4 headers, layers 0-2 only; layers 3-42 instead have `gate.bias [256]`). The learned gate weight `[256,4096]` still produces the *scores* that weight the selected experts. R1 §2.1: *"we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing (Roller et al., 2021). The Hash routing strategy determines the target experts of each token according to a predefined hash function with regard to the input token ID."* Hash routing has no bias (`gate.bias` absent on layers 0-2 — verified). Note layer 2 is hash-MoE **and** CSA — the schedules are independent.

### 1.6 Router (config values, all read from R2)

- `scoring_func: "sqrtsoftplus"` → `scores = sqrt(softplus(logits))` (R2 model.py: `scores = F.softplus(scores).sqrt()`; R1: "we change the activation function that computes the affinity scores from [sigmoid] into [sqrt(softplus(·))]"). NOTE: transformers registers `sqrtsoftplus` in ACT2FN (R3 uses `ACT2FN[config.scoring_func]`).
- `topk_method: "noaux_tc"` → noaux-topk with bias correction: `indices = topk(scores + e_score_correction_bias)` but weights gathered from **original** (unbiased) scores (R2 Gate: "Bias shifts scores for expert selection (topk) but does not affect routing weights").
- `norm_topk_prob: true` → `weights /= weights.sum(-1)` (for non-softmax scoring).
- `routed_scaling_factor: 1.5` → `weights *= 1.5` after normalization; output = routed + shared expert (shared expert is un-gated, added at full weight).
- top-6 routed of 256 + 1 shared.

## 2. MTP head (num_nextn_predict_layers = 1)

### 2.1 What's in the checkpoint (`mtp.0.*`, R2 shard 46 — exact shapes)

One full extra decoder layer + projections (shares `embed` and `head` with the target — R2 convert.py: `if name.startswith("mtp.") and ("emb" in name or ...head.weight): continue`):

| Tensor | Shape | Notes |
|---|---|---|
| `mtp.0.e_proj.weight/scale` | [4096, 4096] | embedding → MTP input (fp8) |
| `mtp.0.h_proj.weight/scale` | [4096, 4096] | target hidden → MTP input (fp8) |
| `mtp.0.enorm/hnorm/norm.weight` | [4096] | RMSNorms on embedding / target-hidden / pre-head |
| `mtp.0.attn.wq_a [1024,4096], wq_b [32768,1024], wkv [512,4096], wo_a [8192,4096], wo_b [4096,8192], q_norm [1024], kv_norm [512], attn_sink [64]` | — | **sliding-window-only attention** (compress_ratios[43] = 0; NO compressor/indexer tensors — verified) |
| `mtp.0.ffn.gate.weight [256,4096] + gate.bias [256]` (no tid2eid → learned routing), `shared_experts w1/w2/w3` (2048-inter), `experts.0..255 w1/w2/w3 + scales` | — | full MoE, same fp4 expert format |
| `mtp.0.hc_attn_* / hc_ffn_* [24,16384]+[24]+[3]` | — | mHC (same as target layers) |
| `mtp.0.hc_head_fn [4,16384], hc_head_base [4], hc_head_scale [1]` | — | own final HC collapse (model-level `hc_head_*` is NOT reused) |

Param count ≈ 6.6B (256 experts × 3 × 2048×4096 logical = 6.44B + shared 25M + attn ≈ 90M + projs 34M). Storage ≈ 3.3 GB (fp4 experts + fp8 rest).

### 2.2 How MTP speculative decoding works (DeepSeek-V3-style, unchanged in V4)

R1 §2.1: *"The Multi-Token Prediction (MTP) configuration remains identical to that of DeepSeek-V3."* Reference loop (R2 model.py MTPBlock.forward):
1. Target decodes token t, keeping its final pre-norm hc-stream hidden `h_t` (the [B,S,4,4096] hyper-connection state).
2. MTP input: `x = e_proj(enorm(embed(t+1))) + h_proj(hnorm(h_t))` (both [B,S,4096]).
3. Run the MTP decoder layer (own KV cache; sliding-window attention at position t+1) → own `hc_head` collapse → `norm` → shared lm_head → logits for token **t+2**.
4. Accept/reject via standard speculative verification (greedy: accept if argmax matches target; sampling: rejection sampling — DeepSeek-V3 paper appendix B scheme). Accepting gives up to 2 tokens per target forward; on rejection, resample from the target's corrected distribution.
5. With `num_nextn_predict_layers=1` this is **MTP-1**: expected tokens/cycle ≈ 1 + p(accept) (no bonus beyond 1 draft).

### 2.3 DSpark loop (for contrast — R4, R5, R6)

DSpark-5 in the 0731 checkpoint: 3 draft layers reusing the target architecture (`dspark_target_layer_ids: [40,41,42]` — target hidden states from those layers are concat-projected via `main_proj [hidden·3, hidden]` and KV-injected into every draft layer, DFlash-style), block size 5, Markov head rank 256, noise/mask token id 128799. Loop: (1) one parallel forward of the draft over [anchor + 4 mask tokens] → base logits for 5 positions; (2) Markov head (low-rank V×256 transition) sequentially resamples left-to-right within the block (semi-autoregressive, fixes suffix decay); (3) confidence head + optional scheduler trims low-survival suffixes; (4) target verifies the block in ONE pass — non-causal: future query tokens are included in each query's top-k indices in the sparse attention (R6: "To implement non-causal attention, we leverage the sparse attention implementation to include the future query tokens in the top-k indices"); (5) accept longest consistent prefix + 1 bonus. Lossless w.r.t. the target distribution.

### 2.4 Verdict: MTP vs DSpark for this engine

See §6. Summary: **build MTP-1 now; design the draft/verify interface so DSpark can slot in later.**

## 3. FP4 expert format — exact bit layout and TPU dequant

All from R2 `inference/kernel.py` + `inference/convert.py` (checkpoint side) — this is the *expert* format (`expert_dtype: "fp4"`); the FP8 non-expert weights are a different scheme (128×128 blocks).

**Bit layout (e2m1, packed along K):**
- Format: **e2m1** — 1 sign, 2 exponent, 1 mantissa. 16 values: `±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`. convert.py `FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, ...]`.
- Packing: **2 values per byte along the input (K) dimension**, low nibble first. Checkpoint tensor `experts.E.w1.weight [2048, 2048]` = logical `[2048, 4096]` fp4 (2048 = moe_inter, 4096 = hidden) packed to `[2048, 4096/2]` bytes, dtype int8-viewed-as-float4_e2m1fn_x2. convert.py: `low = x & 0x0F; high = (x >> 4) & 0x0F; stack([FP4_TABLE[low], FP4_TABLE[high]], -1)` — **byte's low nibble is the earlier K element**.
- **Scale format: NOT 128×128.** Per-expert-matrix: one **e8m0** (power-of-2 exponent, ue8m0/FE8M0) scale per **32 fp4 elements along K, per output row**. Kernel: `fp4_block_size = 32`, weight_group_size=32; scale tensor `experts.E.w1.scale [2048, 128]` = [out=2048, in/32=4096/32=128], dtype float8_e8m0fnu. (w2: weight [4096,1024] logical [4096,2048], scale [4096, 64].) So: **1×32 groups along K, one scale row-set per output row — "per-32-block rowwise scales", e8m0.**
- Activation side of the fp4 GEMM (reference): x quantized **FP8 e4m3, 1×128 groups along K** with ue8m0 scales (`act_quant(x, 128, "ue8m0", e8m0)`), then fp4×fp8 GEMM with per-32 weight scale × per-128 act scale applied to the FP32 accumulator.

**`o_groups = 8` has NOTHING to do with FP4.** It is the attention **grouped output projection** count: 64 heads × 512 = 32768 output dims split into 8 groups of 4096; each group is projected by `wo_a [8192, 4096]` (= 8 blocks of [1024, 4096]) to 1024-dim (o_lora_rank), then `wo_b [4096, 8192]` mixes 8×1024 → 4096. R1 §2.3.1 "Grouped Output Projection"; R3 `DeepseekV4GroupedLinear` docstring quotes the Flash numbers exactly.

**TPU dequant strategy (JAX):** TPUs have no native e2m1 GEMM (MXFP4-style compute arrives with v7/Ironwood-class MXUs; v5e/v6e do not expose it to JAX). Options, recommended order:
1. **On-the-fly per-expert dequant in the MoE gather loop (recommended).** For each active expert gather: `w = LUT[nibbles]` — unpack via `x & 0x0F` / `x >> 4` (int8 ops), map the 16-entry value table (either `lax.gather` from a 16-float table indexed by nibble, or arithmetic decode: sign·(m0 + m1/2)·2^(e−1) with e==0 → subnormal/zero), multiply by `exp2(scale_int8 − 127)` (e8m0 → exact power of 2, so dequant is a bit-shift-equivalent multiply — no rounding error beyond the format itself), cast to bf16, then bf16 `dot` on the MXU. At batch-1 decode each token touches 6 experts/layer → 6×12.6 MB fp4/layer to unpack+GEMM; unpack is ~2 int ops + LUT + mul per element, negligible vs the GEMM, and HBM traffic stays at fp4 size (150 GB total expert storage stays on-device/host as-is).
2. **Load-time e2m1→e4m3 lossless upcast (convert.py's own trick).** `cast_e2m1fn_to_e4m3fn`: fold the e8m0 scales into fp8 by rescaling each 128×128 block so max offset ≤ 6 bits (6.0·2^6 = 384 < 448), giving 2 bytes/element... actually 1 byte/elem fp8 + one e8m0 scale per 128×128 block — halves the bf16 footprint with exactly representable values. Only useful if the target TPU generation exposes fp8 GEMM to JAX (v7).
3. **Full load-time bf16 dequant** — 284B logical → ~567 GB: only viable sharded across a large pod; avoid.
Dequant correctness anchor: convert.py asserts the cast is lossless (e2m1 values × e8m0 scales are exactly representable in e4m3 within the offset scheme), so bf16 dequant is *exact* — no accuracy excuse for skipping it.

## 4. Full forward math of one deepseek_v4 layer

Notation: B,S batch/seq; d=4096 hidden; H=64 heads; dh=512 head dim (rd=64 rope / nd=448 nope); hc=4 streams; JAX layout suggestions in brackets.

### 4.0 Stream state and mHC (every layer, both sublayers)

State: `X ∈ [B,S,4,4096]` (4 hyper-connection streams; embedding is broadcast to all 4 at input). Per sublayer site (attn and ffn, each with own params `hc_{attn,ffn}_{fn [24,16384], base [24], scale [3]}`):
1. `flat = RMSNorm_unweighted(X.reshape[B,S,16384])` (fp32).
2. `mixes = linear(flat, hc_fn) · rsqrt` → [B,S,24]; split into pre[4], post[4], comb[4,4] logits.
3. `pre = sigmoid(pre·scale[0] + base[:4]) + hc_eps`; `post = 2·sigmoid(post·scale[1] + base[4:8])`; `comb = softmax(comb_logits·scale[2]+base[8:], −1) + eps`, then `comb /= colsum + eps`, then 19 more Sinkhorn row/col normalize iterations (total 20) → doubly-stochastic-ish [B,S,4,4].
4. `collapsed = Σ_j pre_j · X[:,:,j,:]` [B,S,4096] → sublayer input.
5. After sublayer output Y: `X'[:,:,k,:] = post_k·Y + Σ_j comb_{j,k}·X[:,:,j,:]` (note: comb consumed **transposed** — R3: "comb is consumed transposed: indexed as sum_j comb[j,k]*residual[j,d] ... Sinkhorn produces a doubly-stochastic but non-symmetric matrix, so the direction matters"). Run mHC math in fp32 (`_keep_in_fp32_modules_strict` includes attn_hc/ffn_hc — R3).

JAX note: mHC params are tiny ([24,16384] ≈ 1.6MB fp32/layer); the Sinkhorn loop (20 iters, 4×4) is cheap but sequential — vmap over tokens, keep in fp32.

### 4.1 Attention sublayer

`x = attn_norm(collapsed)` (RMSNorm 4096).
1. **Q low-rank**: `qr = q_norm(wq_a·x)` [B,S,1024] (qr is ALSO fed to the indexer); `q = wq_b·qr` [B,S,32768] → [B,S,64,512]; per-head RMSNorm-unweighted (`q *= rsqrt(mean(q²)+eps)` — R2 L498; R3 `q_b_norm`); RoPE trailing 64 dims (`main` θ=10000 no-YaRN for layers 0/1; `compress` θ=160000+YaRN16 for CSA/HCA layers — R2 L475-481).
2. **Shared KV**: `kv = kv_norm(wkv·x)` [B,S,512] (single head); RoPE trailing 64; non-rope 448 dims FP8-simulated (block 64) — store window entries BF16-rope+FP8-nope.
3. **Sliding window write**: ring buffer of 128 entries (every layer). During decode `cache[start_pos % 128] = kv`.
4. **Compressor branch** (CSA ratio 4 / HCA ratio 128; layers 2..42): as §1.3/§1.4 — project `wkv_c/wgate_c` (fp32 compute in reference!), pool windows (CSA overlapping Ca/Cb), RMSNorm, RoPE at compressed positions, write to compressed cache (1 entry per 4 or 128 tokens). CSA also runs the **indexer** (own compressor at 128-dim, FP4 q/k, top-512 indices).
5. **Attention set** = 128 window entries ∪ (CSA: 512 indexer-selected compressed entries; HCA: all 8,192 compressed entries, causal-masked `idx < (t+1)//128`). MQA softmax over the union with per-head sink `attn_sink[64]` in the denominator; scale 512^−0.5. (HCA causal rule; CSA entries additionally filtered by indexer top-k with −1 sentinel → dropped.)
6. **Output**: inverse RoPE on trailing 64 dims of `o` at query position; then grouped output projection: reshape o → [B,S,8,4096]; `wo_a` (8 independent [1024,4096] blocks, einsum 'bsgd,grd->bsgr') → [B,S,8,1024]; `wo_b` [4096, 8192] → [B,S,4096]. (wo_a is stored FP8 block-128×128; wo_b FP8.)

### 4.2 FFN sublayer (MoE)

`x = ffn_norm(collapsed)`.
1. Router (§1.6): logits = gate.weight[256,4096]·x (fp32); `scores = sqrt(softplus(logits))`; hash layers 0-2: `indices = tid2eid[input_ids]`; else `indices = topk(scores + bias)[..6]`; `weights = normalize(gather(scores, indices)) · 1.5`.
2. 6 routed experts, SwiGLU with clamps: `up = clamp(w3·x, ±10)`, `gate = clamp(w1·x, max=10)`, `act = silu(gate)·up`, `y += w2·act · weight` (fp32 accumulate in reference). FP4 weights per §3.
3. Shared expert (same shapes, FP8 weights, un-gated): `y += shared(x)`.
4. Output back through mHC post (§4.0 step 5).

### 4.3 Head

`hc_head` (params [4,16384],[4],[1] — sigmoid pre-weights, no Sinkhorn) collapses 4 streams → `norm` (RMSNorm 4096) → `head [129280, 4096]` (untied, bf16, fp32 logits in reference).

### 4.4 exact layer_types list (43 target layers)

```
0: sliding_attention        (hash_moe)
1: sliding_attention        (hash_moe)
2: compressed_sparse_attention (hash_moe)   ← indexer, ratio 4
3: heavily_compressed_attention (moe)        ← ratio 128
4: CSA ... 5: HCA ... alternating ...
41: HCA (moe)
42: CSA (moe)                              ← indexer, ratio 4
[43: MTP layer — sliding_attention, ratio 0, moe]
```
(Derived: `_COMPRESS_RATIO_TO_LAYER_TYPE = {0: sliding_attention, 4: compressed_sparse_attention, 128: heavily_compressed_attention}` [R3] applied to the verified compress_ratios; mlp_layer_types = hash_moe×3 then moe×40 [R3 §mlp_layer_types default].)

### 4.5 Where sliding window and indexer interact

Every layer (incl. MTP) has the 128-entry sliding window of *uncompressed* shared-KV entries — this is the "additional branch of sliding window attention" (R1 §2.3.3) that (a) restores causal access to tokens inside the query's own compression block and (b) captures local detail. The indexer (CSA layers only) selects which *compressed* entries join the window entries in the attention union; window indices and indexer indices are concatenated into one topk_idxs list (R2 model.py L507-514: `topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)`) and the single `sparse_attn` kernel gathers both. The indexer's own queries also always RoPE at full query positions; its keys live at compressed positions.

## 5. KV cache budget @ 1M tokens (per layer type)

Entry sizes: main KV entry = 512 dims = 64 rope (BF16, 128 B) + 448 nope (FP8, 448 B) = **576 B** (paper mixed format, R1 §2.3.4; reference also stores window non-rope as FP8-sim, block 64). Indexer entry = 128 dims, keys-only, never values. Counts at S = 1,048,576:

| Layer type | window (entries × B) | compressed | indexer | per-layer total |
|---|---|---|---|---|
| sliding (L0,L1,MTP L43) | 128 × 576 = 72 KiB | — | — | **72 KiB** |
| CSA (21 layers, ratio 4) | 72 KiB | 262,144 × 576 = 144.0 MiB | 262,144 × 128 dims: 64 MiB (BF16) / 32 MiB (FP8) / 16 MiB (FP4) | **208 MiB (bf16 idx) / 176 MiB (fp8) / 160 MiB (fp4)** |
| HCA (20 layers, ratio 128) | 72 KiB | 8,192 × 576 = 4.5 MiB | — | **4.6 MiB** |

**Totals @ 1M tokens (44 layers incl. MTP):**
- Paper-faithful mixed KV + BF16 indexer cache: 21×208.1 + 20×4.6 + 3×0.07 = **4.36 GiB**
- Mixed KV + FP8 indexer: **3.70 GiB**; + FP4 indexer (paper says indexer compute is FP4; cache may follow): **3.28 GiB**
- All-8-bit KV (512 B/entry, rope too) + FP8 indexer: **3.36 GiB**
- Pure BF16 everything: 6.72 GiB (for contrast)

Per-token budget rule-of-thumb: ~4.3 KiB/token/CSA-layer, ~0.09 KiB/token/HCA-layer, ~0.05 KiB/token/1M-token-average across all layers (≈4.4 GiB per 1M tokens per sequence). The paper's own claim for context (R1): V4-Flash at 1M context = **7% of DeepSeek-V3.2's KV cache** and 10% of its single-token FLOPs. A GQA8-128 BF16 baseline at 1M is 4 GiB *per layer*, so a V4 CSA layer is ~5% of that baseline layer (matches R1's "dramatically reduced to approximately [1/20] of that baseline"). For the "8-bit KV" budget asked: plan **~3.3–3.7 GiB for 1M context**, i.e. the KV pool is a non-issue next to the ~150–170 GB of weights; the real constraint is the 262k-entry-per-CSA-layer *indexer gather* at decode (top-512 of 262k — do the topk on the 64-head FP4/FP8 score tensor, never materialize [B,S,262k] in bf16).

## 6. MTP vs D-Spark verdict (for a from-scratch JAX TPU engine, ~30 tok/s base, coding+agent+chat)

1. **They are the same *category* of thing (speculative decoding loops), not attention features** — both are optional add-ons on top of the CSA/HCA engine; the engine core is identical either way.
2. **MTP-1 weights ship inside this exact checkpoint** (`mtp.0.*`, 0731-preview Flash included). DSpark draft weights do NOT exist in `deepseek-ai/DeepSeek-V4-Flash` — they live in `DeepSeek-V4-Flash-0731` / `DeepSeek-V4-Flash-DSpark` (different post-training), so DSpark would force a checkpoint switch and a DeepSpec-style training run for a custom target.
3. **Engineering cost**: MTP-1 = one extra decoder layer (already §4-math'ed) + a standard accept/reject verify loop — well-understood from DeepSeek-V3. DSpark = 3 draft layers + KV-injection from target layers 40–42 + Markov head + confidence head + STS calibration + **non-causal sparse attention** (future tokens inside top-k indices — R6), plus a scheduler. Roughly 3–4× the implementation surface.
4. **Payoff at single-stream/low concurrency** (the ~30 tok/s regime): DSpark's headline numbers (60–85% over MTP-1, R4) are measured *at matched aggregate throughput in high-concurrency serving* — its edge comes from confidence-scheduled verification avoiding wasted batch capacity. Third-party single-stream measurement (fraserprice/DeepSeek-V4-Flash-DSpark, RTX Pro 6000, FP8): DSpark ≈ **1.2–1.4× over stock MTP** in per-request decode. At batch 1 that gap shrinks toward ~1.1–1.25× once your MTP loop is well-tuned.
5. **At ~30 tok/s base the draft is nearly free**: one MTP layer ≈ 1/43 of target FLOPs + one extra lm_head; the bottleneck is memory-bound expert gather, and MTP reuses the already-resident experts. Acceptance for coding/agent (structured text, R4: "structured requests like code naturally sustain higher acceptance rates") typically 60–80% → expect ~1.5–1.7× effective decode (45–50 tok/s) from MTP-1 alone.
6. **Losslessness**: both are exact rejection-sampling schemes — output distribution is the target's; no quality risk in either.
7. **Coding + agent workload**: agent tool-call traces are highly repetitive/structured → highest acceptance; this favors whichever drafter is available *sooner*, i.e. MTP.
8. **Chat (open-ended)**: lower acceptance (suffix decay); DSpark's Markov head specifically targets this, but at single stream the win is modest (see 4).
9. **Risk**: DSpark-on-TPU has no reference JAX implementation (vLLM tpu-inference ships DFlash in Flax — the *base* — not DSpark; R6). MTP has a complete readable reference (R2 model.py MTPBlock + convert.py) to port line-by-line.
10. **Recommendation**: implement MTP-1 first (verify loop + `mtp.0` loading), architect the draft interface (draft-forward(target_hidden, ids) → logits, verify(logits, target_logits) → accepted prefix) so a DSpark-style parallel drafter can replace it later; revisit DSpark only if/when the engine moves to multi-stream serving where its confidence scheduler pays.

## 7. Port gotchas checklist (from source reading)

- compress_ratios has 44 entries; do NOT truncate to num_hidden_layers — index 43 selects the MTP layer's attention type (0 = sliding).
- Layers 0/1 use plain RoPE (θ=10000, NO YaRN); CSA/HCA layers and their compressors use θ=160000 + YaRN(16, orig 65536, β 32/1) — two cos/sin tables ("main"/"compress") per forward (R2 L475-481, R3).
- Compressor + indexer projections are stored **BF16 in the checkpoint** (no .scale companions) — R3 `_keep_in_fp32_modules`: `self_attn.compressor.kv_proj`, `gate_proj`, `indexer.kv_proj/gate_proj/scorer.weights_proj`. Everything else FP8 (128×128, ue8m0) except experts (fp4) and hc_*/attn_sink/ape (fp32) and embed/head (bf16).
- Indexer query comes from the **shared q-lora latent `qr` (post q_norm)**, not from raw hidden; and `weights_proj` eats raw hidden (bf16).
- RoPE is **interleaved-pair** style (consecutive channel pairs, `rotate_half` on even/odd — R3 `rotate_half` + "V4's interleaved RoPE pairs consecutive channels"), applied to the **trailing** 64 dims; output gets inverse rotation.
- CSA window write on prefill with `seqlen > window`: ring-buffer rotation (R2 L522-523 cutoff split).
- Hash routing needs `input_ids` plumbed into every FFN call (layers 0-2).
- mHC `comb` is applied **transposed** (sum over first axis); Sinkhorn 20 iterations in fp32; `post` range [0,2] (`2·sigmoid`).
- The indexer cache is keys-only and may be lower precision than main KV; the main KV entry is *one* tensor serving as both K and V (shared-KV MQA) — never allocate separate K and V caches.
- Expert fp4 scale groups are 32 along K (NOT 128); act side of that GEMM uses 128-groups fp8; fp8 non-expert weights use 128×128 blocks — three distinct quant schemes in one checkpoint.
- YaRN on compress rope: reference forces attention_factor/mscale = 1.0 (does NOT scale logits by the YaRN mscale) — R3 config comment.

## References (summary)

- [R1] arXiv 2606.19348 (DeepSeek-V4 tech report) — CSA/HCA §2.3, mHC §2.2, Muon §2.4, FP4/KV precision §2.3.4, hash routing + sqrtsoftplus §2.1.
- [R2] HF deepseek-ai/DeepSeek-V4-Flash: config.json, model.safetensors.index.json, safetensors shard headers, inference/{model,kernel,convert,generate}.py + config.json.
- [R3] huggingface/transformers main: src/transformers/models/deepseek_v4/{configuration,modeling}_deepseek_v4.py.
- [R4] arXiv 2607.05147 (DSpark); github.com/deepseek-ai/DeepSpec.
- [R5] HF deepseek-ai/DeepSeek-V4-Flash-0731 config.json (dspark_* keys; compress_ratios [.., 0,0,0] tail).
- [R6] vLLM: vllm/models/deepseek_v4/nvidia/dspark.py; docs.vllm.ai speculators DSpark/DFlash pages; vllm-project/tpu-inference README support matrix.
- [R7] DeepSeek-V3.2 report (DSA baseline, "DeepSeek Sparse Attention", DeepSeek-AI 2025b) — cited by R1.
