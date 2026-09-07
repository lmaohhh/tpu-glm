# DSV4-Flash-0731 Unc-ensored/Abliterated Source Search — for the JAX safetensors engine

**Task:** Find an 'orcarouter-style' (unc-ensored/abliterated) safetensors source of DeepSeek-V4-Flash-0731
(the agent-tuned DSpark revision), or the best transplant path. Engine constraint: pure-JAX TPU engine that
loads safetensors (FP8 dense + FP4 experts) and **cannot load GGUF**.

**Verdict up front:** A ready-made, ungated, native-precision safetensors abliterated 0731 **already exists**:
**`cebeuq/DeepSeek-V4-Flash-0731-abliterated`** — official 0731 shards (byte-identical, sha256-verified) +
a 1.54 GB abliteration overlay that also covers the DSpark draft heads. No transplant needed. Details and
ranking in Section 4.

All facts below were verified 2026-09-07 against the HF API (`/api/models`, `/api/models/<id>/tree/main`,
`/resolve/main/...` with HTTP Range requests for safetensors headers/tensor bytes, and LFS sha256 from the
tree API). Local raw data: `C:\Users\souffle\dsv4_research\` (hf_details.json, search_expanded.json,
candidates_detail.json, shard_hashes.json, indexes.json, tensor_compare2.json, overlay_and_readme.json).

---

## 1. Inventory of 0731 variants (HF searches + author listings)

Searches run: `DeepSeek-V4-Flash-0731` (100 hits), `DSV4-Flash-0731` (3), `DeepSeek 0731` (105 unique),
`DeepSeek-V4-Flash DSpark` / `DeepSeek-V4-Flash-DSpark` (149 unique), `DeepSeek-V4-Flash abliterated` (196
unique), `DeepSeek-V4-Flash unc-ensored` (6). Author listings pulled for: huihui-ai, orcarouter, dealignai,
unsloth, Guillermofrante (0 models), thedrummer (0 models), cebeuq, prem-research, apetersson, drowzeys,
fraserprice, lovesenko, msuiche, cbert33, squanchyzx, AtlasCloud. **199 unique repos** collected total.

Key facts per repo (gated = HF `gated` field; storage = `usedStorage`; params/dtypes from the API's
safetensors parameter totals; DSpark = `dspark_*` keys present in config.json via /resolve; GGUF-only noted):

### 1a. Official checkpoints

| repo | gated | storage | dtype breakdown (params) | DSpark cfg | vision | notes |
|---|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | **no** | 155.4 GiB | BF16 1.48B, F32 37.7M, F8_E4M3 6.30B, I8 296.35B (FP4 experts packed), I64 2.33M; total 304.18B | yes | no | 48 shards, 4.47M downloads |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` (June) | no | 155.4 GiB | same dtypes but F8_E8M0 9.26B + I8 148.18B; total 165.27B | yes | no | 48 shards; draft-only portion is ~18.5 GiB; 651K dl |
| `deepseek-ai/DeepSeek-V4-Flash` (0622 preview) | no | 148.7 GiB | BF16 1.42B, F32 36.2M, F8_E4M3 6.02B, I8 283.47B; total 290.94B | **no** (MTP-1 only) | no | 46 shards; mtp.0 only |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | no | 156.3 GiB | BF16 1.95B, F32 37.8M, F8_E4M3 6.30B, I8 296.35B; total 304.65B | yes | **yes** (259 vision tensors + aligner, kimi_k25-style tower) | 72,633 tensors; separate post-train branch, NOT 0731 |
| `unsloth/DeepSeek-V4-Flash-0731` | no | 155.4 GiB | identical to official | yes | no | **48/48 shards sha256-identical to official 0731** (pure mirror) |

### 1b. Safetensors abliterated/unc-ensored 0731-family (the target category)

| repo | gated | storage | dtype | DSpark drafts in weights? | relation to official 0731 shards |
|---|---|---|---|---|---|
| **`cebeuq/DeepSeek-V4-Flash-0731-abliterated`** | **no** | 156.9 GiB | native FP8+FP4 (same as official) | **yes** (mtp.0/1/2 present; overlay edits them too) | **48/48 base shards byte-identical (sha256) + `model-overlay-00001-of-00001.safetensors` (92 tensors)** |
| `squanchyzx/DeepSeek-V4-Flash-0731-HERETIC-Abliterated-FP8` | no | 157.6 GiB | native FP8+FP4 | yes (mtp.0/1/2 present, **left stock/unedited**) | 48/48 base shards identical + overlay (66 tensors, layers 10–42 wo_b only) |
| `prem-research/DeepSeek-V4-Flash-0731-abliterated` | no | 155.4 GiB | native | yes | 28/48 shards identical, 20 shards rewritten (baked in place) |
| `apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8` | no | 155.4 GiB | native | yes | 12/48 identical, 36 rewritten |
| `lovesenko/DeepSeek-V4-Flash-0731-Abliterated` | no | 155.4 GiB | native | yes | 2/48 identical, 46 rewritten |
| `amesianx/DeepSeek-V4-Flash-DSpark-Abliterated` | no | 155.4 GiB | native (304B total) | yes | 0/48 identical (full re-bake) |
| `moeshawky/DeepSeek-V4-Flash-Q4-mxfp4-0731-abliterated` | no | 155.4 GiB | native | yes | 0/48 identical |
| `gorbatjovy/DeepSeek-V4-Flash-0731-BahamutRU-t265-abliterated` | no | 156.6 GiB | native | yes | full re-bake (49 shards) |
| `RupertBern/DeepSeek-V4-Flash-0731-HERETIC-Abliterated-FP8` | no | 78.8 GiB | — | cfg yes | 25 shards only (partial/re-packed; low confidence) |
| `Jon-Nielsen/DeepSeek-V4-Flash-0731-Abliterated-MXFP4-INT8` | no | 162.8 GiB | **MXFP4/INT8 re-quant** | yes | incompatible dtype mix |
| `sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4` | no | 163.5 GiB | **NVFP4 re-quant** | yes | incompatible dtype mix |
| `mumitrol/DeepSeek-V4-Flash-0731-Abliterated-NVFP4-vision` | no | 164.4 GiB | **NVFP4 re-quant** + vision | yes | incompatible dtype mix |
| `cbert33/DeepSeek-V4-Flash-0731-abliterated-vision-v2` | no | 156.3 GiB | native + vision tower | yes | 49 shards; `DeepseekV4VisionForCausalLM` |
| `apetersson/DeepSeek-V4-Flash-Vision-Exp-Abliterated` | no | 653.9 GiB | mixed (multiple quants + GGUF) | n/a | Vision-Exp based, not 0731 |

### 1c. Gated (auto/manual) notable repos

- `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors` — gated:auto, 18,229 dl, config dspark keys present, DeepseekV4ForCausalLM. "Anchored Tensors" format (drowzeys' keys layout, for their vLLM/DGX-Spark setup) — not plain HF safetensors layout.
- `drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Unc-ensored` (+ v1.1-alpha) — gated:auto, 9,959 dl; June-DSpark based (created 2026-07-10, before 0731).
- `orcarouter/DeepSeek-V4-Flash-Vision-Unc-ensored` (safetensors, ~156.3 GiB), `-GGUF`, `-MLX` — all gated:auto. **Base model (API-verified via expand=baseModels): `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` — NOT 0731.**
- `msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29` (and GLP-42, Vision-Exp variant) — gated:auto, small (37 dl).
- `windowsxp811203/DeepSeek-V4-Flash-0731-Abliterated` (+GGUF) — gated:auto.
- `DaydreamBlend/DeepSeek-V4-Flash-Vision-Exp-Abliterated-L20-Lambda1.25` — gated:auto (Vision-Exp based).

### 1d. GGUF-only (useless for the JAX engine; listed for completeness)

`huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF` (569K dl, ungated — but GGUF only; **no
companion safetensors repo from huihui-ai exists**; huihui's only other V4F repo is the 0622 ds4 GGUF),
`unsloth/DeepSeek-V4-Flash-0731-GGUF` (286K dl), `bartowski`, `lmstudio-community`, `ggml-org`,
`bullerwins`, `mradermacher/...-Abliterated-FP8-GGUF` (itself derived from apetersson's safetensors),
plus dozens of small quant repos. `Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF`,
`singulared/...-DSpark-GGUF`, `alessandrobologna/...-DSpark-Drafter-GGUF` etc. carry DSpark draft weights
in GGUF form only.

### 1e. DSpark draft-only / FP8 re-quant safetensors (transplant fodder, not unc-ensored)

- `AtlasCloud/DeepSeek-V4-Flash-DSpark-FP8-dspark_only` — ungated, 18.6 GiB, 3 shards, exactly the 4,705 `mtp.0/1/2` draft tensors from the **June** DSpark release (F8_E4M3 19.77B params + BF16/F32 small).
- `AtlasCloud/DeepSeek-V4-Flash-0731-FP8-DSpark` — ungated, 286.3 GiB, everything re-quantized to pure FP8 (F8_E4M3 301B params) — **not** the native FP4-expert layout.
- `nvidia/DeepSeek-V4-Flash-0731-NVFP4`, `dealignai/DeepSeek-V4-Flash-0731-CRACK-NVFP4`, `utarn/...-NVFP4`, `MJPansa/...-NVFP4` (343K dl) — NVFP4 re-quants, wrong dtype layout for this engine.

---

## 2. Official `deepseek-ai/DeepSeek-V4-Flash-0731` — what exactly it is

- **Ungated** (`gated: false`), MIT license, created 2026-07-31, lastModified 2026-08-01, 4,469,466 downloads.
- **usedStorage 166,888,735,421 B = 155.4 GiB**, 48 safetensors shards + repointed index (5.6 MB).
- **Params/dtypes (API safetensors totals):** 304,180,418,494 total = BF16 1,483,567,488 (embed/head/norms/gates/router) + F32 37,741,630 (attn_sink, hc_* scalars) + F8_E4M3 6,304,038,912 (dense attn/ffn weights, ue8m0 128×128 block scales) + I8 296,352,743,424 (FP4-E2M1 routed experts, packed) + I64 2,327,040. Same quantization scheme as 0622 (`quant_method: fp8`, `scale_fmt: ue8m0`, `weight_block_size: [128,128]`).
- **DSpark config keys:** `dspark_block_size=5`, `dspark_markov_rank=256`, `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[40,41,42]`; `num_nextn_predict_layers=1`.
- **Weights layout (index weight_map, 72,317 tensors):** 67,612 target tensors (`embed.weight`, `layers.0–42.*`, `norm.weight`, `head.weight`, `hc_head_*`) + **`mtp.0` (1,568), `mtp.1` (1,565), `mtp.2` (1,572)** tensors. The three DSpark draft stages are stored under the `mtp.0/1/2` prefix (each stage reuses the target arch with KV-injection, incl. `markov_head.markov_w1/w2`, and `main_norm`/`main_proj` injection projections). They live in shards `model-00046/47/48-of-00048.safetensors` (~18.5 GiB).
- **It does NOT ship the 0622 MTP-1 head.** The 0622 `mtp.0` MTP-1 tensors (`e_proj`, `h_proj`, `enorm`, `hnorm`, + MTP-style `hc_head_*`) are gone: 10 tensors exist only in 0622's mtp.0 and 3 only in 0731's mtp.0 (`main_norm.weight`, `main_proj.weight/scale`). 0731 = DSpark drafts **only**.
- **README (fetched, ungated):** "official release of DeepSeek-V4-Flash, superseding the preview version, with substantially enhanced agentic capabilities. Same model structure as DeepSeek-V4-Flash-DSpark" (i.e. speculative module attached). Agentic benchmark gains vs 0622 preview: Terminal-Bench 2.1 82.7 vs 61.8, DeepSWE 54.4 vs 7.3, Cybergym 76.7 vs 38.7, Toolathlon-Verified 70.3 vs 49.7, etc. **No Jinja chat template** — an `encoding/` folder with Python `encode_messages`/`parse_message_from_completion_text` scripts replaces it; `reasoning_effort` now supports `low/high/max`. Recommended sampling: temperature 1.0, top_p 0.95 agentic / 1.0 otherwise; up to 384K output tokens for high/max. vLLM: `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`; SGLang: `--speculative-algorithm DSPARK` (no separate draft path).
- **`tokenizer_config.json` and `generation_config.json` are byte-identical to 0622** (sha-compared). Tokenizer unchanged (vocab 129280; note token 128799 = dspark noise token lies beyond vocab, used as a draft-only sentinel).

### 0622 vs 0731 target-weight comparison (the load-bearing fact)

- **Tensor name sets for target weights are IDENTICAL** (67,612 tensors, same names, same dtypes/shapes — verified on 16 representative tensors incl. `embed.weight`, `head.weight`, `norm.weight`, `layers.*.attn_norm`, `wq_a`, `wo_b`, expert `w1/w2/w3`, `gate`, `shared_experts`, `attn_sink`, `hc_head_base`).
- **But every spot-checked target tensor is byte-DIFFERENT** (full-byte compares on small tensors like `norm.weight` [4096] BF16 and `layers.0.attn.wq_a.scale`; 1–2 MB head compares on large ones; comparison pipeline validated with controls: same-repo-twice = IDENTICAL, unsloth-vs-official = IDENTICAL, cebeuq-base-vs-official = IDENTICAL).
- Also **all 48 shards of 0731 differ (sha256) from June's `DeepSeek-V4-Flash-DSpark`**, and spot tensors differ — 0731 is a further post-trained checkpoint, not a re-release of June.
- **Conclusion: 0731 is a full continued post-train (agent-tuned) of the whole network, NOT "0622 + dspark drafts added".**

---

## 3. Transplant path analysis (dspark drafts onto an abliterated 0622-family base)

Question: graft `mtp.0/1/2` DSpark draft weights from official 0731 onto
`orcarouter/DeepSeek-V4-Flash-Vision-Unc-ensored` (abliterated base)?

Findings:

1. **Mechanically possible but the bases are three different models.** Verified lineage:
   - orcarouter's base is **`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`** (API `baseModels`), which is itself a separate post-train branch from 0731 — spot-compared `norm.weight`, `embed.weight`, `layers.20.ffn.gate.weight`, `mtp.0.attn.wo_b.weight`, `layers.42.ffn.experts.100.w2.weight` all **DIFFER** between 0731 and Vision-Exp. (The parent's "orcarouter = 0622 abliterated" assumption is wrong; it's the Vision-Exp branch, with a vision tower.)
   - 0731 target weights differ from 0622 everywhere (Section 2).
   - So "abliterated 0622/Vision-Exp + 0731 drafts" pairs a **drafter trained against the 0731 target** with a **different target network**.
2. **Consequences:** DSpark drafts KV-inject from target layers 40–42 and predict the 0731 target's distribution. On a different (0622/Vision-Exp-derived, abliterated) target, draft acceptance would drop materially (the cebeuq README quantifies how sensitive acceptance is: ~48% baseline; a drafter mismatched to its training target degrades it further, and an *unedited* drafter on an *edited* target specifically proposes refusal tokens the target no longer produces, hurting acceptance exactly on the unc-ensored use-cases). Output remains **lossless** (the target always verifies), so it is a speed regression, not a correctness one.
3. **Config work needed either way:** 0622 config has no `dspark_*` keys; you must add `dspark_block_size=5`, `dspark_markov_rank=256`, `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[40,41,42]`, and the engine must implement the DSpark 3-stage draft loop (the port currently implements MTP-1 from 0622, whose `mtp.0` tensor names — `e_proj/h_proj/enorm/hnorm` — **do not exist** in 0731; 0731's `mtp.0/1/2` use `main_norm/main_proj` + full attn/ffn blocks + `markov_head`). An MTP-1 loader cannot load DSpark weights; DSpark support is new engine code regardless of which weights you pick.
4. **Draft-tensor sources if a transplant were still wanted:** official 0731 shards 46–48 (~18.5 GiB), or `AtlasCloud/DeepSeek-V4-Flash-DSpark-FP8-dspark_only` (3 shards, 18.6 GiB) — but the AtlasCloud pack is from the **June** DSpark release, whose target also differs from 0731, adding a second mismatch. Use official 0731 shards if ever needed.
5. **Verdict on transplant: technically viable, strategically obsolete.** Since 0731 target ≠ 0622 target ≠ Vision-Exp target, the transplant can never give a matched drafter; and a matched-drafter abliterated 0731 already exists (cebeuq), making the whole exercise unnecessary. Also note orcarouter is gated:auto (needs HF_TOKEN + accepting conditions on Kaggle), while cebeuq is ungated.

---

## 4. Ranked verdict

**The answer to "does an orcarouter-style unc-ensored 0731 in safetensors exist?" is YES — use `cebeuq/DeepSeek-V4-Flash-0731-abliterated`.**

Ranked options for the JAX engine:

1. **`cebeuq/DeepSeek-V4-Flash-0731-abliterated` (BEST).** Ungated, MIT, 156.9 GiB, text-only (no vision tower — exactly matches the port's 0622 structure), native FP8-dense + FP4-experts precision, **48/48 base shards sha256-identical to official 0731** plus a 1.54 GB `model-overlay-00001-of-00001.safetensors` with 92 edited tensors: `layers.0–42.attn.wo_b.{weight,scale}` and `mtp.0/1/2.attn.wo_b.{weight,scale}` (rank-1 refusal projection, λ=2.5, re-quantized in place holding ue8m0 block exponents fixed). The repo's `model.safetensors.index.json` is repointed so the overlay tensors load from the overlay file — **an index-following safetensors loader needs zero special-casing**. Crucially the DSpark draft heads were abliterated with the same direction ("drafter at parity": acceptance 48.7% vs ~48% base; refusal 0% on their AdvBench suite; tool-call compliance 1.000). Loader changes needed vs the current 0622 port: (a) implement DSpark 3-stage drafting (`mtp.0/1/2` with `main_norm/main_proj` KV-injection, `markov_head`, `dspark_*` config keys) — needed for ANY 0731 weights, not specific to this repo; (b) nothing else — dtype layout is identical to 0622 (BF16/F32/F8_E4M3+ue8m0 scales/packed-I8 FP4 experts).
2. **`squanchyzx/DeepSeek-V4-Flash-0731-HERETIC-Abliterated-FP8` (v2)** — same overlay architecture (66 tensors, backbone layers 10–42 `wo_b` only, λ=1.35, base shards identical, DSpark drafts stock/unedited). More conservative edit: their own smoke tests show 37.5% residual refusal vs cebeuq's 0%, but lower long-context degeneration risk (their v1 at λ=1.5 had documented long-agent-history degeneration; v2 fixed it and ran 12/12 clean long-context replays). Tradeoff: stock drafter + edited target = mild acceptance loss on refusal-adjacent content (still lossless). Choose this if maximizing steering/quality retention matters more than zero-refusal.
3. **Other full re-bakes** (ungated, same native dtype layout, DSpark included): `prem-research/DeepSeek-V4-Flash-0731-abliterated` (28/48 shards modified), `apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8` (36 modified, 15K dl), `lovesenko/DeepSeek-V4-Flash-0731-Abliterated` (46 modified). All usable, but none publishes per-tensor receipts as strong as the two overlay repos; cebeuq/squanchyzx's "base shards byte-identical + tiny overlay" is the most auditable and least likely to carry collateral damage.
4. **If vision is ever wanted:** `cbert33/DeepSeek-V4-Flash-0731-abliterated-vision-v2` (ungated, 156.3 GiB, 0731 text + `DeepseekV4VisionForCausalLM` vision tower) — but the engine has no vision path today and the orcarouter-style vision repo itself is gated + based on Vision-Exp, not 0731.
5. **Official 0731 (or the byte-identical `unsloth/DeepSeek-V4-Flash-0731` mirror)** — if unc-ensored turns out to hurt agent/coding quality, the clean fallback; everything else (DSpark layout) identical.
6. **Transplant (orcarouter abliterated base + official 0731 drafts)** — **not recommended**: three-way weight mismatch (orcarouter is Vision-Exp-based, not 0622; 0731's target differs from both), guaranteed drafter/target mismatch → lower draft acceptance, plus orcarouter is gated:auto requiring an HF_TOKEN on Kaggle from an account that accepted the gate. Only worth considering if one specifically wants the vision tower *and* accepts running DSpark with reduced acceptance (or with speculative decoding disabled — the drafts then simply go unused).

**Kaggle notes:**
- cebeuq, squanchyzx, official 0731, and unsloth 0731 are all **ungated** — no HF_TOKEN needed; plain `huggingface_hub`/direct HTTPS downloads work on Kaggle.
- Gated:auto repos (orcarouter*, drowzeys keys-*, msuiche, windowsxp811203, DaydreamBlend) require putting a token from an account that has clicked through the gate into the Kaggle secret `HF_TOKEN` (`import os; os.environ["HF_TOKEN"]` / `HF_HOME` setup) — an extra failure mode for zero benefit here.
- Download budget: cebeuq = 155.4 GiB official shards + 1.54 GiB overlay; if the official 0731 shards are already cached on the Kaggle volume, only the overlay differs.

**Bottom line for the port:** point the JAX engine at `cebeuq/DeepSeek-V4-Flash-0731-abliterated`, keep the FP8/FP4 loading path unchanged, replace MTP-1 with the DSpark 3-stage drafter (`mtp.0/1/2`, `main_norm/main_proj`, `markov_head`, `dspark_*` config), and you get an unc-ensored agent-tuned 0731 with a parity-acceptance drafter and no token-gate hassle.
