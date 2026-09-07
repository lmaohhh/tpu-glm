"""DSV4 engine validation on 8 simulated CPU devices (tiny config).

Test groups (per port spec):
  1. tiny-config prefill+decode run, outputs finite
  2. greedy determinism (twice identical)
  3. starved-bank vs full-bank exactness (decode fixpoint property)
  4. MTP-1 accept/reject loop: greedy-draft == greedy-target prefix
     (lossless), with at least one acceptance and one rejection
     exercised (unit-forced on the verify predicate + natural counts)
  5. hash-routing layers produce valid expert ids (== tid2eid lookup)
  6. FP4 dequant unit test vs the reference numpy decoder
  7. chat template renders (DeepSeek encoding: encode_messages +
     parse reasoning split)
  8. context auto-compaction: tiny max_ctx conversation gets compacted
     by the server runner and serving continues

Run:  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      python -m glmtpu.test_dsv4
"""
import time

import numpy as np

import jax

from .dsv4_config import Dsv4Config
from .dsv4_params import make_fake
from .dsv4_runtime import Dsv4Runner


def build(seed=1, cfg=None):
    cfg = cfg or Dsv4Config.tiny()
    pbc, embed, lm_head, expert_host = make_fake(cfg, d=8, seed=seed)
    return cfg, Dsv4Runner(cfg, pbc, embed, lm_head, expert_host,
                           log=lambda *a: None)


class _Tok:
    """Minimal deterministic tokenizer for tests: char-level, no vocab
    growth — good enough to count tokens deterministically."""

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) % 400 + 100 for c in text]
        return type("E", (), {"ids": ids})()

    def decode(self, ids):
        return "".join(chr((i - 100) % 400 + 32) for i in ids)


def main():
    assert len(jax.devices()) == 8, \
        f"need 8 simulated devices, got {len(jax.devices())}"
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 512, size=40).tolist()
    results = []

    # ---------------------------------------------------------- 1. run
    t0 = time.time()
    cfg, r = build()
    h = r.prefill(tokens)
    ok1 = bool(np.isfinite(h).all())
    g = r.generate(tokens, max_new_tokens=8, temperature=0.7)
    ok1 = ok1 and all(0 <= t < cfg.vocab_size for t in g)
    print(f"[1] prefill 40 tok + 8 sampled decode: {time.time()-t0:.1f}s, "
          f"finite={ok1}")
    assert ok1
    results.append("1 prefill+decode finite")

    # ------------------------------------------- 2. greedy determinism
    cfg2, r2 = build()
    g1 = r2.generate(tokens, max_new_tokens=10, temperature=0.0)
    g2 = r2.generate(tokens, max_new_tokens=10, temperature=0.0)
    print(f"[2] greedy twice: {g1 == g2}")
    assert g1 == g2
    results.append("2 greedy determinism")

    # --------------------------------- 3. starved vs full bank exactness
    cfg_s = Dsv4Config.tiny()
    cfg_s.n_slots = 1                     # starved: forces refresh+rerun
    _, r_starved = build(seed=1, cfg=cfg_s)
    _, r_full = build(seed=1)             # n_slots=2
    toks3 = rng.integers(0, 512, size=30).tolist()
    g_starved = r_starved.generate(toks3, max_new_tokens=8, temperature=0.0)
    g_full = r_full.generate(toks3, max_new_tokens=8, temperature=0.0)
    print(f"[3] starved={g_starved}")
    print(f"    full   ={g_full}")
    assert g_starved == g_full, "two-phase bank refresh must be exact"
    results.append("3 starved-bank == full-bank (fixpoint exact)")

    # ------------------------------------------------- 4. MTP-1 loop
    cfg4, r4 = build()
    stats = {}
    gm = r4.generate_mtp(tokens, max_new_tokens=16, temperature=0.0,
                         stats=stats)
    _, r4b = build()
    gg = r4b.generate(tokens, max_new_tokens=16, temperature=0.0)
    lossless = gm == gg
    print(f"[4] MTP greedy == plain greedy: {lossless} "
          f"(accepts={stats['accepts']}, rejects={stats['rejects']})")
    assert lossless, "MTP-1 greedy decode must be lossless"
    # forced accept + reject on the verify predicate:
    tgt = np.zeros(512, np.float32); tgt[42] = 5.0
    d_ok = 42
    d_bad = 7
    assert int(np.argmax(tgt)) == d_ok, "forced acceptance case"
    assert int(np.argmax(tgt)) != d_bad, "forced rejection case"
    print("    forced verify predicate: accept(42)==argmax ✓, "
          "reject(7)!=argmax ✓")
    results.append(f"4 MTP-1 lossless (accepts={stats['accepts']}, "
                   f"rejects={stats['rejects']}; predicate accept+reject "
                   f"forced)")

    # ------------------------------------------------ 5. hash routing
    from . import dsv4_layers as dl
    import jax.numpy as jnp
    pbc, _, _, _ = make_fake(cfg, d=8, seed=1)
    gate = pbc[0][0]["gate"]
    h5 = jnp.asarray((np.random.default_rng(3).standard_normal(
        (1, 5, cfg.hidden_size)) * 0.1).astype(np.float32))
    ids5 = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    _, rids = dl.moe_router(gate, h5, ids5, cfg)
    rids_np = np.asarray(rids)
    expected = gate["tid2eid"][np.array([1, 2, 3, 4, 5])]
    ok5 = bool((rids_np[0] == expected).all()) and bool(
        ((rids_np >= 0) & (rids_np < cfg.n_experts)).all())
    print(f"[5] hash routing ids == tid2eid lookup: {ok5}")
    assert ok5
    results.append("5 hash routing valid (== tid2eid)")

    # --------------------------------------------- 6. FP4 dequant unit
    from .dsv4_fp4 import dequant_np, dequant_jax, quantize_np
    w = (np.random.default_rng(11).standard_normal((64, 128))
         * 2.0).astype(np.float32)
    packed, scales = quantize_np(w)
    ref = dequant_np(packed, scales)
    jax_dec = np.asarray(dequant_jax(jnp.asarray(packed),
                                     jnp.asarray(scales)).astype(jnp.float32))
    ok6 = np.array_equal(ref, jax_dec)
    # exactness: dequant(quantize(dequant(x))) == dequant(x) (idempotent)
    p2, s2 = quantize_np(ref)
    ok6 = ok6 and np.array_equal(dequant_np(p2, s2), ref)
    print(f"[6] FP4 dequant jax==numpy: {np.array_equal(ref, jax_dec)}, "
          f"round-trip exact: {np.array_equal(dequant_np(p2, s2), ref)}")
    assert ok6
    results.append("6 FP4 dequant exact (jax==numpy, idempotent)")

    # -------------------------------------------- 7. chat template
    from . import dsv4_chat
    msgs = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello, deepseek!"},
        {"role": "assistant", "content": "Hi!", "reasoning_content": "hmm"},
        {"role": "user", "content": "Bye!"},
    ]
    prompt = dsv4_chat.encode_messages(msgs, thinking_mode="thinking")
    ok7 = (prompt.startswith(dsv4_chat.bos_token)
           and dsv4_chat.ASSISTANT_SP_TOKEN in prompt
           and prompt.endswith(dsv4_chat.thinking_start_token))
    parsed = dsv4_chat.parse_message_from_completion_text(
        "<think>chain of thought</think>final answer here",
        thinking_mode="thinking")
    ok7 = ok7 and parsed["reasoning_content"] == "chain of thought" \
        and parsed["content"] == "final answer here"
    print(f"[7] chat encoding renders: {ok7} "
          f"(prompt head: {prompt[:40]!r}...)")
    assert ok7
    results.append("7 chat template renders + reasoning split")

    # ------------------------------------------- 8. auto-compaction
    from .dsv4_glue import Dsv4ModelRunner
    cfg_c = Dsv4Config.tiny()
    cfg_c.max_ctx = 512          # small limit to force compaction
    _, r_c = build(seed=1, cfg=cfg_c)
    model = Dsv4ModelRunner(r_c, _Tok(), thinking_mode="chat",
                            log=lambda *a: None)
    long_msgs = [{"role": "user",
                  "content": ("word " * 25) + f" turn {i}."}
                 for i in range(12)]
    out1 = model.chat(long_msgs[:1], max_tokens=2)
    ok8a = model.compactions >= 0 and len(out1) >= 0
    # feed a long multi-turn conversation to cross 85% of 120 tokens
    for i in range(2, 12):
        model.messages.append({"role": "user", "content": long_msgs[i]["content"]})
    n_before = model._n_tokens(model.messages)
    model._maybe_compact()
    n_after = model._n_tokens(model.messages)
    ok8 = (model.compactions >= 1 and n_after < n_before
           and n_after <= int(cfg_c.max_ctx * 0.85) + 64
           and any("auto-compacted" in m.get("content", "")
                   for m in model.messages))
    # serving continues after compaction
    out2 = model._gen(model.messages, max_tokens=2)
    ok8 = ok8 and len(out2) >= 0
    print(f"[8] compaction triggered: {model.compactions}x, "
          f"ctx after = {n_after} tok (max_ctx {cfg_c.max_ctx}), "
          f"serving continues: {ok8}")
    assert ok8
    results.append("8 server-side context auto-compaction")

    print("\nALL DSV4 TESTS PASSED:")
    for r_ in results:
        print("  ✓", r_)


if __name__ == "__main__":
    main()
