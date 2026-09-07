# DeepSeek-V4-Flash → JAX/TPU port — progress log

Status: COMPLETE (subagent run, 2026-09-07/08).  All 8 test groups green
on 8 simulated CPU devices; committed and pushed to main.

## Final architecture (as built)

- `src/glmtpu/dsv4_config.py` — real (43+MTP layers, compress_ratios 44
  entries, 256k default max_ctx, 1M via config) + tiny test config
  (6+MTP layers, ratios [0,0,4,8,4,8,0]).
- `src/glmtpu/dsv4_fp4.py` — e2m1 packing (low nibble first along K),
  e8m0 per-32 rowwise scales, exact in-graph dequant (nibble→16-entry
  LUT→×exp2(scale−127)→bf16), fp4 activation sim, numpy reference
  decoder.
- `src/glmtpu/dsv4_layers.py` — dual rope tables (main θ=10000 plain /
  compress θ=160000 YaRN16 orig 65536 β32/1, in-graph cos/sin, trailing
  interleaved pairs, inverse rope), fp8-sim KV pack/unpack (block 64,
  ue8m0 power-of-2 scales, nope u8 + rope f32 storage), mHC
  hyper-connections (transposed comb, 20 Sinkhorn iters fp32), sink
  softmax, grouped o-proj, CSA compressor (overlap Ca/Cb, ape8 layout),
  HCA compressor, hash/noaux_tc sqrtsoftplus router, FP4 bank core.
- `src/glmtpu/dsv4_params.py` — tiny fake weights incl. fp4-packed
  experts + tid2eid hash table + per-chip shards (8 local heads = 1
  wo_a group per chip).
- `src/glmtpu/dsv4_runtime.py` — pmap sites (attn w/ static ratio,
  ffn, mtp draft), ring/comp/indexer KV state (replicated, MQA), window
  mask by recorded slot positions, causal e < (p+1)//ratio for
  compressed, indexer top-k with -1 sentinel + fp4-sim q/k +
  Hadamard, prefill affine-correction MoE sweep, decode hot-bank
  fixpoint (snapshot→run→miss? refresh+rollback), MTP-1 lossless greedy
  loop + rejection-sampling path.
- `src/glmtpu/dsv4_chat.py` — port of DeepSeek's encoding_dsv4.py
  (the repo ships NO chat_template.jinja; the encoding module IS the
  chat template) + reasoning split for reasoning_content.
- `src/glmtpu/dsv4_glue.py` — ModelRunner + server-side auto-compaction
  (85% threshold, watermark keep-4, model-summarized oldest turns,
  iterative shrink until fit, deterministic greedy).
- `src/glmtpu/dsv4_openai.py` — OpenAI server: reasoning_content
  split in streaming (delta.reasoning_content inside <think>), clean
  400 on image/video parts, model id deepseek-v4-flash.
- `src/glmtpu/dsv4_loader.py` — env-driven DSV4_REPO loader
  (default deepseek-ai/DeepSeek-V4-Flash ungated; orcarouter variant
  via HF_TOKEN with 267 vision tensors stripped: vision.*, aligner.*,
  image_*), fp8 128×128 dequant for dense, experts stay fp4 u8+e8m0.
- `src/glmtpu/test_dsv4.py` — 8 test groups (all passing).
- `build_notebook.py` — extended with build_dsv4() →
  notebook/dsv4-flash-tpu.ipynb (same cell flow as GLM: TPU check,
  git-pull-first + embedded MODS fallback, self-test, tokenizer, load,
  sanity gen, server+tunnel with compaction, smoke test w/
  reasoning split + image-400 check, keep-alive).

## Validation (XLA_FLAGS=--xla_force_host_platform_device_count=8
python -m glmtpu.test_dsv4)

- [x] 1 prefill+decode finite (40-token prefill, sampled decode)
- [x] 2 greedy determinism (twice identical)
- [x] 3 starved-bank (n_slots=1) == full-bank tokens — fixpoint exact
- [x] 4 MTP-1: greedy-draft == plain-greedy prefix (lossless);
      verify predicate accept+reject forced; natural run 0 accepts /
      16 rejects on random weights (accepts expected 60-80% on real
      coding/agent text per spec §6.5)
- [x] 5 hash routing ids == tid2eid lookup; learned ids in range
- [x] 6 FP4 dequant: jax == numpy reference, round-trip idempotent
- [x] 7 chat encoding renders (BOS, <｜Assistant｜>, ends <think>) +
      reasoning/content split
- [x] 8 auto-compaction: 512-token max_ctx conversation compacted
      (424→421 with keep-watermark; smaller summaries in real
      tokenizers), serving continues, summary message inserted
- [x] GLM engine regression: glmtpu.test still ALL PASS

## Notes / deviations

- Chunked prefill invariants: pos0 % ratio == 0 and prefill_chunk %
  ratio == 0 for every ratio (256 % 128 == 0 real, 16 % 8 == 0 tiny).
- Reference has no chunked prefill (single-pass start=0); our chunked
  compressor verified equivalent to per-token decode path to ~1e-6
  (fp32 noise) and chunk-continuity verified (16+16 == 32 single-shot).
- Draft KV on 2-token accept skips one draft position — lossless
  regardless (draft is only a proposal distribution; window mask by
  recorded position drops stale slots).
- Tiny config uses ratio 8 as the HCA stand-in (non-overlap path).
- Test 8 uses max_ctx=512 (120 was below the chat-encoding's fixed
  special-token overhead for the stub tokenizer; real runs use 256k).
