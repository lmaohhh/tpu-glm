"""ModelRunner glue for the DSV4 engine: tokenizer + chat encoding +
prefix caching.  Stateless: the client owns the conversation (and any
compaction policy) — every request re-renders the full history, and the
prefix cache makes that cheap (only NEW tokens are prefilled when the
conversation extends the previous one verbatim).

Context overflow: raises ValueError with exact counts -> HTTP 400.  The
server never truncates, shifts, or compacts silently; the client app
decides when to compact (mainstream-app behavior).
"""
from __future__ import annotations

import time

import numpy as np

from . import dsv4_chat
from .dsv4_config import Dsv4Config
from .dsv4_runtime import Dsv4Runner

LCP_FLOOR = 64      # shorter common prefixes aren't worth trusting


class Dsv4ModelRunner:
    """OpenAI-compatible ModelRunner over Dsv4Runner with prefix caching."""

    def __init__(self, runner: Dsv4Runner, tokenizer,
                 thinking_mode: str = "chat", log=print):
        self.r = runner
        self.tok = tokenizer
        self.thinking_mode = thinking_mode
        self.log = log
        self.eos = set(runner.cfg.eos_ids)
        self.cache_events = []      # last few cache log lines (debug)

    # ------------------------------------------------------------------
    def _encode(self, messages):
        text = dsv4_chat.encode_messages(
            messages, thinking_mode=self.thinking_mode,
            drop_thinking=True)
        enc = self.tok.encode(text, add_special_tokens=False)
        return list(enc.ids) if hasattr(enc, "ids") else list(enc)

    # ------------------------------------------------------------------
    def chat(self, messages, max_tokens=512, temperature=0.0, top_p=1.0,
             stream_cb=None, reasoning_effort=None):
        if dsv4_chat._has_image_or_video(messages):
            raise ValueError(
                "image/video inputs are not supported by this engine "
                "(vision tower weights are stripped at load; text-only "
                "serving)")
        ids = self._encode(messages)
        max_ctx = self.r.cfg.max_ctx
        if len(ids) + max_tokens > max_ctx:
            raise ValueError(
                f"context length exceeded ({len(ids)} prompt tokens + "
                f"{max_tokens} max_tokens > max_ctx {max_ctx}); compact "
                "the conversation and retry")
        t0 = time.time()

        # ---- prefix-cache-aware generation ----
        retained = getattr(self.r, "_retained", None)
        cache_kind, reused, n_new = "miss (cold)", 0, len(ids)
        if retained is not None:
            old = retained["ids"]
            lcp = 0
            for a, b in zip(old, ids):
                if a != b:
                    break
                lcp += 1
            if lcp == len(old) and len(ids) > len(old) and lcp >= LCP_FLOOR:
                cache_kind, reused, n_new = "hit", len(old), len(ids) - len(old)
                self.r.cache_restore(retained)
                h = self.r.prefill(ids[len(old):], _continue=True)
                self.r._cache_ids = list(ids)
            elif lcp == len(old) and len(ids) == len(old):
                cache_kind, reused, n_new = "hit (identical)", len(old), 0
                self.r.cache_restore(retained)
                h = self.r._last_hidden
            elif lcp >= LCP_FLOOR:
                cache_kind = f"miss (diverged at {lcp}/{len(old)})"
                h = self.r.prefill(ids)
                self.r._cache_ids = list(ids)
            else:
                cache_kind = (f"miss (diverged at {lcp}/{len(old)})" if old
                              else "miss (cold)")
                h = self.r.prefill(ids)
                self.r._cache_ids = list(ids)
        else:
            h = self.r.prefill(ids)
            self.r._cache_ids = list(ids)

        line = f"[cache] {cache_kind}: {reused} reused, {n_new} new"
        self.cache_events.append(line)
        self.cache_events = self.cache_events[-8:]
        self.log(line)

        logits = self.r._lm_head(h)
        out_ids = []
        det_text = ""
        for i in range(max_tokens):
            t = self.r._sample(logits, temperature, top_p)
            if t in self.eos:
                break
            out_ids.append(t)
            if stream_cb and (i + 1) % 4 == 0:
                piece = self.tok.decode(out_ids)
                if piece != det_text:
                    stream_cb(piece[len(det_text):], i)
                    det_text = piece
            if i + 1 < max_tokens:
                logits, _ = self.r._decode_one(t, temperature, top_p)

        # retain prompt + generated: state covers all of them
        self.r._cache_ids = list(ids) + list(out_ids)
        self.r._retained = self.r.cache_retain()

        dt = time.time() - t0
        self.log(f"[gen] {len(out_ids)} tok in {dt:.1f}s")
        return self.tok.decode(out_ids)

    def reset(self):
        self.r.reset()
        self.r._cache_ids = []
        self.r._retained = None
