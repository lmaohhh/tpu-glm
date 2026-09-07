# GLM-5.3-Flash on Kaggle TPU v5e-8 — hot-expert JAX serving engine

A custom pure-JAX inference engine for the 321B-param FP8 MoE
(`glm5_next` architecture) that **does not fit in TPU HBM** (needs ~306 GiB
vs 8×16 GiB usable). Instead of giving up, it splits the model across the
two memory tiers the Kaggle TPU v5e-8 machine offers:

- **HBM (112 GiB usable)**: TP-sharded dense weights, hot MoE expert banks
  (8 experts/chip/layer, FP8), KV/recurrent state
- **Host RAM (330 GB)**: all 42×288 routed experts as FP8 bytes, streamed
  into HBM banks on demand (exact — a bank-miss triggers refresh + re-run)

The whole stack — 62 shards (~306 GiB) — downloads from HuggingFace into
RAM once per session (parallel curl range stripes, ~20-25 min), then serves
an OpenAI-compatible API through a public cloudflared tunnel, same UX as
the original GPU notebook.

## Architecture notes (glm5_next, transformers v5.16.0 ground truth)

- 45 layers: 34 KDA (Kimi Delta Attention) linear-attention + 11 DSA/MLA
  (every 4th from layer 3), mHC hyper-connections (4 residual streams,
  Sinkhorn-projected), sigmoid router with `noaux_tc` bias, top-8 of 288
  experts + 1 shared expert, **NoPE — zero rotary embeddings anywhere**.
- v1 simplifications (each deliberate, flagged in the notebook):
  - DSA lightning-indexer skipped → full causal attention over the 512-d
    MLA latent cache (exact masking, modest quality cost at long context)
  - MTP head (layer 45) + vision tower never downloaded (~11.6 GiB saved)

## Validation

Every module ran on 8 simulated CPU devices (`--xla_force_host_platform_
device_count=8`) before any Kaggle push:

- prefill (chunked, left-padded) bit-deterministic across runs
- greedy decode reproducible
- hot-bank refresh **exact**: starved-bank (1 slot < top_k) output
  identical to full-bank output via the bank-coverage fixpoint

## Files

- `src/glmtpu/` — engine modules (fp8, config, layers, params, runtime,
  loader_real, openai_api, runner_glue)
- `build_notebook.py` — assembles the self-contained Kaggle notebook
  (embeds all modules; no dataset attachment needed)
- `notebook/glm53-flash-tpu.ipynb` — the notebook (also pulls this repo if
  network allows, falling back to the embedded copy)
- `research/` — ground-truth dumps (modeling source, config, index,
  tokenizer) and feasibility notes

## Usage (Kaggle)

1. Create a TPU v5e-8 notebook, upload `glm53-flash-tpu.ipynb` (or import
   from this repo), enable internet.
2. Run all — weights stream in (~25 min), server comes up, tunnel URL
   printed at the end.
3. Point any OpenAI client at `<tunnel-url>/v1` with key
   `kaggle-sfw-token-9999`.
