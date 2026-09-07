# Kaggle TPU v5e-8 Logistics + GLM-5.3-Flash FP8 Weight Download (RESEARCH — in progress)

Goal: serve GLM-5.3-Flash-FP8 (306 GiB) on Kaggle TPU v5e-8 by streaming weights into host RAM (330 GB class) instead of disk.

## Status
- [x] Mirror verification (orcarouter, zai-org, ModelScope)
- [x] Kaggle session/quota facts
- [x] Download tooling + throughput
- [x] RAM budget math
- [x] UserSecrets for HF token

## 1. Mirrors & gating (VERIFIED 2026-09-07, no token)

| Repo | Gated? | Verdict |
|---|---|---|
| `zai-org/GLM-5.3-Flash` | **NO** (`"gated": false` in API; file list + resolve URLs return 200) | ✅ **BEST ungated source** |
| `orcarouter/GLM-5.3-Flash-FP8` | — | ❌ **Does not exist** (API 401; absent from `?author=orcarouter` listing of 23 repos) |
| `orcarouter/GLM-5.3-Flash-Uncensored-FP8` | **YES — `gated: "auto"`** | ❌ Anonymous resolve → 401 `X-Error-Code: GatedRepo` |

- zai-org/GLM-5.3-Flash: `usedStorage: 656,694,018,754 B = 611.5 GB` (includes non-shard dupes? no — 73 files, 62 shards); safetensors total **328.37 GB = 305.81 GiB**; shards are 62 × ~5.36 GB (`model-00001-of-00062` … `model-00062-of-00062`), last shard 1.26 GB.
- orcarouter uncensored FP8 has identical 62-shard layout, `usedStorage: 328,357,673,346 B ≈ 305.8 GiB`, base_model: zai-org/GLM-5.3-Flash (per API metadata).
- "auto" gate behavior: gate is **auto-approved on request**, but requester must be **logged in** and click through the gate → any valid HF token (even a fresh free account's) can download after accepting. Anonymous gets 401 GatedRepo.

(ModelScope section pending)

## 2. Kaggle TPU facts (pending)

## 3. Download strategy (pending)

## 4. RAM budget math (pending)

## 5. Kaggle UserSecrets / HF token (pending)
