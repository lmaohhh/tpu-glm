"""ModelRunner glue for the DSV4 engine: tokenizer + chat encoding +
server-side context auto-compaction.

Auto-compaction (spec: prevents platform OOM at long conversations):
when the rendered conversation crosses 85% of max_ctx, the OLDEST turns
(keep a fixed watermark of the most recent messages) are summarized by
the model itself (greedy = deterministic, chunked at 2000 tokens) and
replaced with a single system summary message.  The client never sees
it: the runner stores compacted history internally and serves from it.
"""
from __future__ import annotations

import time

import numpy as np

from . import dsv4_chat
from .dsv4_config import Dsv4Config
from .dsv4_runtime import Dsv4Runner

COMPACT_FRAC = 0.85          # trigger threshold
COMPACT_KEEP = 4             # watermark: messages kept verbatim
COMPACT_CHUNK = 2000         # tokens per summarization chunk
COMPACT_MAX_OUT = 200        # summary length cap (tokens)


class Dsv4ModelRunner:
    """openai_api.ModelRunner implementation over Dsv4Runner."""

    def __init__(self, runner: Dsv4Runner, tokenizer,
                 thinking_mode: str = "chat", log=print):
        self.r = runner
        self.tok = tokenizer
        self.thinking_mode = thinking_mode
        self.log = log
        self.eos = set(runner.cfg.eos_ids)
        self.messages = []         # compacted conversation state
        self.compactions = 0

    # ------------------------------------------------------------------
    def _encode(self, messages):
        text = dsv4_chat.encode_messages(
            messages, thinking_mode=self.thinking_mode,
            drop_thinking=True)
        enc = self.tok.encode(text, add_special_tokens=False)
        ids = list(enc.ids) if hasattr(enc, "ids") else list(enc)
        return ids

    def _n_tokens(self, messages):
        return len(self._encode(messages))

    # ------------------------------------------------------------------
    def _summarize(self, text: str, max_out=None) -> str:
        """Deterministic greedy summarization via the model itself.
        Chunked: long inputs are summarized piecewise, then combined."""
        max_out = max_out or COMPACT_MAX_OUT
        chunks = []
        toks = self.tok.encode(text, add_special_tokens=False)
        toks = list(toks.ids) if hasattr(toks, "ids") else list(toks)
        pieces = [toks[i:i + COMPACT_CHUNK]
                  for i in range(0, len(toks), COMPACT_CHUNK)]
        for piece in pieces:
            part = self.tok.decode(piece)
            msgs = [{"role": "user",
                     "content": "Summarize the following conversation "
                     "turns in under "
                     f"{max_out} tokens, preserving key facts, decisions "
                     "and open tasks:\n\n" + part}]
            out = self._gen(msgs, max_tokens=max_out, temperature=0.0)
            chunks.append(out.strip())
        return "\n".join(chunks)[-max_out * 4:]     # hard char cap

    def _gen(self, messages, max_tokens=512, temperature=0.0, top_p=1.0,
             stream_cb=None):
        ids = self._encode(messages)
        ids = ids[-(self.r.cfg.max_ctx - max_tokens - 8):]
        logits = self.r.prefill(ids)
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
        return self.tok.decode(out_ids)

    # ------------------------------------------------------------------
    def _maybe_compact(self):
        """Server-side auto-compaction: replace oldest turns with a model
        summary when the conversation crosses 85% of max_ctx.  Iterative:
        each pass shrinks the kept history and the summary budget until
        the rendered conversation fits."""
        for _ in range(8):
            n = self._n_tokens(self.messages)
            limit = int(self.r.cfg.max_ctx * COMPACT_FRAC)
            if n <= limit or len(self.messages) <= 1:
                return
            keep_n = min(COMPACT_KEEP, len(self.messages) - 1)
            keep = self.messages[-keep_n:] if keep_n else []
            # shrink the watermark until the kept tail alone fits
            while keep_n > 1 and self._n_tokens(keep) > limit - 64:
                keep_n -= 1
                keep = self.messages[-keep_n:]
            old = self.messages[:-keep_n] if keep_n else self.messages
            lines = []
            for m in old:
                c = m.get("content", "")
                lines.append(f"{m.get('role', 'user')}: {c}")
            text = "\n".join(lines)
            budget = max(16, limit - self._n_tokens(keep) - 64)
            summary = self._summarize(text, max_out=budget)
            head = ("[Conversation summary (auto-compacted by the server "
                    "to stay within the context window)]\n")
            new_msgs = [{"role": "system", "content": head + summary}] + keep
            # hard fit check: shrink the summary until it fits
            while self._n_tokens(new_msgs) > limit and len(summary) > 16:
                summary = summary[:max(16, len(summary) // 2)]
                new_msgs = [{"role": "system",
                             "content": head + summary}] + keep
            self.messages = new_msgs
            self.compactions += 1
            self.log(f"[compact] {n} tok > {limit} ({COMPACT_FRAC:.0%} "
                     f"of max_ctx) -> summarized {len(old)} oldest "
                     f"messages ({self._n_tokens(self.messages)} tok "
                     f"now, compaction #{self.compactions})")

    # ------------------------------------------------------------------
    def chat(self, messages, max_tokens=512, temperature=0.0, top_p=1.0,
             stream_cb=None, reasoning_effort=None):
        if dsv4_chat._has_image_or_video(messages):
            raise ValueError(
                "image/video inputs are not supported by this engine "
                "(vision tower weights are stripped at load; text-only "
                "serving)")
        # append user turn to the compacted history
        self.messages = self.messages + [dict(m) for m in messages
                                         if m.get("role") == "user"] \
            if not self._is_full_conversation(messages) else \
            [dict(m) for m in messages]
        self._maybe_compact()
        t0 = time.time()
        text = self._gen(self.messages, max_tokens=max_tokens,
                         temperature=temperature, top_p=top_p,
                         stream_cb=stream_cb)
        # record the assistant reply in history
        parsed = dsv4_chat.parse_message_from_completion_text(
            text, self.thinking_mode)
        self.messages.append(parsed)
        dt = time.time() - t0
        self.log(f"[gen] {len(text)} chars in {dt:.1f}s")
        return text

    @staticmethod
    def _is_full_conversation(messages):
        # clients that send the whole history each call (stateless use)
        return len(messages) > 1

    def reset(self):
        self.messages = []
        self.compactions = 0
        self.r.reset()
