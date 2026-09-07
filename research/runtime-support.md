# Runtime Support Research: glm5_next (GLM-5.3-Flash, FP8, ~306 GiB) on Kaggle TPU v5e-8

**Date:** 2026-09-07 · **Question:** Can any EXISTING runtime serve this model on a Kaggle TPU v5e-8 (8×16 GiB HBM = 112 GiB usable, 330 GB host RAM), or is a custom JAX implementation (HBM hot-expert cache + host-RAM cold experts) required?

**Bottom line (detailed evidence below):** No existing runtime can do this. vLLM's TPU backend has no validated GLM-5.x support (GLM-5 is "Untested" in its matrix), no expert-offload/RAM-streaming feature, and its FP8 quantization support is listed only for v7 hardware (v5e is not listed). llama.cpp has no TPU backend. A custom JAX implementation is required.

---

## 1. vLLM tpu-inference (vllm-project/tpu-inference)

### 1.1 Support matrix: glm5_next / GLM-5.x?

**No.** The official "Recommended Models and Features" page lists `zai-org/GLM-5` with every test marked ❓ Untested, and no GLM-5.3-Flash / glm5_next entry exists at all.

Source: https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features

Quoted rows (nightly matrix, same in stable matrix):

> | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | Text | ❓ | ❓ | ❓ |

and:

> | zai-org/GLM-5 | Text | ❓ Untested | ❓ Untested | ❓ Untested |

The page also explicitly scopes what is and isn't implemented:

> "Although vLLM TPU's new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components. For this reason, until we land more capabilities, we recommend starting from this list of stress tested models and features below."
>
> "We are still landing components in tpu-inference that will improve performance for larger scale, higher complexity models (XL MoE, +vision encoders, MLA, etc.)."

So GLM-5 (and by extension glm5_next/GLM-5.3-Flash, a *larger* 321B MoE) is not in the validated set, and "XL MoE" performance work is explicitly still in progress.

### 1.2 MoE kernel support status

The MoE kernels themselves are untested in the support matrix (same URL):

> | **Moe** | Fused MoE | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
> | gmm | ❓ | ❓ | ❓ | ❓ | ❓ | |
>
> and
>
> | MoE | ❓ | ❓ |

### 1.3 Any TPU expert-offload / expert-RAM-streaming mode?

**No.** The complete feature table on that page (quoted above in the feature section) contains: async scheduler, Chunked Prefill, DCN-based P/D disaggregation, KV Cache Offload, LoRA_Torch, Out-of-tree model support, Prefix Caching, Single Program Multi Data, Speculative Decoding (Eagle3, DFlash, Ngram), Multimodal Inputs, hybrid kv cache, multi-host, runai_model_streamer_loader, sampling_params, Step Pooling (Embedding), structured_decoding.

There is NO expert-offload, expert-streaming, expert-to-CPU-RAM, or weights-to-host feature. The only "offload" feature is **KV Cache Offload** — which offloads the KV cache, not MoE expert weights:

> | KV Cache Offload | ✅ | ✅ | ✅ |

(Support-matrix CSV files live at https://github.com/vllm-project/tpu-inference/tree/main/support_matrices and confirmed the same content.)

### 1.4 Quantization support: FP8 is NOT listed for v5e

From the same page (Quantization Support table):

> | Checkpoint dtype | Method | Supported Hardware Acceleration | Flax | Torchax |
> |---|---|---|---|---|
> | FP4 W4A16 | mxfp4 | v7 | ❓ | ❓ |
> | FP8 W8A16 | compressed-tensor | v7 | ❓ | ❓ |
> | FP8 W8A8 | compressed-tensor | v7 | ❓ | ❓ |
> | INT4 W4A16 | awq | v5, v6 | ❓ | ❓ |
> | INT8 W8A8 | compressed-tensor | v5, v6 | ❓ | ❓ |
> | NVFP4 W4A16 | modelopt_fp4 | v7 | ❓ |

FP8 checkpoint loading is listed ONLY for **v7** (Ironwood) hardware — "Supported Hardware Acceleration: v7". v5e is only listed for INT4/INT8. This means an FP8 GLM-5.3-Flash checkpoint has no validated FP8 loading path on v5e in tpu-inference.

### 1.5 Install path on Kaggle

The upstream quickstart targets Google Cloud TPU VMs with a prebuilt PyTorch/JAX TPU stack (docs: https://docs.vllm.ai/projects/tpu/en/latest — "Compatible TPU Generations: Recommended: v7x, v5e, v6e; Experimental: v3, v4, v5p"). On Kaggle, the practical route is `pip install vllm` with the tpu-inference plugin — but this combination is UNVALIDATED for glm5_next (see §1.1) and there is no known-good pinned recipe. No Kaggle-specific recipe exists in the repo (checked repo root file listing via GitHub API: no `examples/` entry for Kaggle; recipes cover v7x/v6e only per README).

**Verdict: vLLM tpu-inference cannot serve glm5_next on v5e-8 today.** Not in support matrix (untested at best), FP8 checkpoints unsupported on v5e per quantization table, no expert-offload feature, and the model cannot fit in HBM (see §2.4 — models that don't fit get "not enough HBM" in the matrix; Qwen3-Coder-480B-A35B and Kimi-K2.6 already hit that wall on smaller TPU slices).
