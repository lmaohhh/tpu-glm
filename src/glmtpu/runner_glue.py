"""ModelRunner: binds the JAX engine + tokenizer + chat template to the
OpenAI API server.  This is the last glue module; the notebook embeds it."""
from __future__ import annotations

import threading
import time

import numpy as np

from .config import GlmConfig
from .runtime import Runner


class GlmModelRunner:
    """openai_api.ModelRunner implementation over the pmap Runner."""

    def __init__(self, runner: Runner, tokenizer, chat_template: str,
                 log=print):
        self.r = runner
        self.tok = tokenizer
        self.chat_template = chat_template
        self.log = log
        self.eos = set(runner.cfg.eos_ids)

    # ------------------------------------------------------------------
    def _render(self, messages, reasoning_effort=None):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"])   # {% break %} support
        tmpl = env.from_string(self.chat_template)
        return tmpl.render(messages=messages, add_generation_prompt=True,
                           reasoning_effort=reasoning_effort, tokenize=False)

    def chat(self, messages, max_tokens=512, temperature=0.7, top_p=0.95,
             stream_cb=None, reasoning_effort=None):
        prompt = self._render(messages, reasoning_effort)
        enc = self.tok.encode(prompt, add_special_tokens=False)
        ids = list(enc.ids) if hasattr(enc, "ids") else list(enc)
        if len(ids) > self.r.cfg.max_ctx - max_tokens - 8:
            ids = ids[-(self.r.cfg.max_ctx - max_tokens - 8):]

        t0 = time.time()
        logits = self.r.prefill(ids)
        n_pref = len(ids)
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
                logits = self.r._decode_one(t, temperature, top_p)
        text = self.tok.decode(out_ids)
        dt = time.time() - t0
        self.log(f"[gen] prefill {n_pref} tok + {len(out_ids)} gen tok "
                 f"in {dt:.1f}s ({len(out_ids)/max(dt-0.01,0.01):.1f} tok/s)")
        return text

    def reset(self):
        self.r.reset()
