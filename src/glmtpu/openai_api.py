"""OpenAI-compatible API server (stdlib only) + ModelRunner interface.

Mirrors the UX of the user's GPU notebook: server on 0.0.0.0:8080, Bearer
auth, /v1/models, /v1/chat/completions (stream + non-stream), /v1/completions,
plus a /healthz probe.  Uses the tokenizers + jinja2 chat template for
formatting (chat_template.jinja from zai-org/GLM-5.3-Flash, embedded in the
notebook).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = "kaggle-sfw-token-9999"
HOST = "0.0.0.0"
PORT = 8080

# The chat template (GLM 5.3) — set by the notebook from the HF repo file.
CHAT_TEMPLATE = ""
TOKENIZER = None          # tokenizers.Tokenizer instance (set by notebook)
EOS_IDS = (154820, 154827, 154829)

_model = None              # ModelRunner instance (set by serve())
_lock = threading.Lock()


class ModelRunner:
    """Interface the JAX engine plugs into."""

    def load(self) -> None: ...
    def chat(self, messages, max_tokens=512, temperature=0.7, top_p=0.95,
             stream_cb=None, reasoning_effort=None) -> str: ...
    def reset(self) -> None: ...


def _render_chat(messages, reasoning_effort=None):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    tmpl = env.from_string(CHAT_TEMPLATE)
    return tmpl.render(messages=messages,
                      add_generation_prompt=True,
                      reasoning_effort=reasoning_effort,
                      tokenize=False)


def _extract_reply(text):
    """Strip reasoning blocks and take the final answer segment."""
    # GLM: <think>...</think> then the reply; split on the LAST </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep the kernel log clean

    def _auth(self):
        h = self.headers.get("Authorization", "")
        return h == f"Bearer {API_KEY}"

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": "glm-5.3-flash-uncensored-fp8", "object": "model",
                 "owned_by": "local"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            self._send_json(401, {"error": "bad api key"})
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/chat/completions":
            self._chat(body)
        elif self.path == "/v1/completions":
            self._completion(body)
        else:
            self._send_json(404, {"error": "not found"})

    def _chat(self, body):
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.95))
        stream = bool(body.get("stream", False))
        effort = body.get("reasoning_effort")

        prompt = _render_chat(messages, effort)
        t0 = time.time()
        with _lock:
            if stream:
                self._stream_chat(prompt, max_tokens, temperature, top_p, t0)
                return
            text = _model.chat(messages, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)
        dt = time.time() - t0
        reply = _extract_reply(text)
        self._send_json(200, {
            "id": "chatcmpl-local", "object": "chat.completion",
            "created": int(time.time()), "model": "glm-5.3-flash-uncensored-fp8",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()),
                      "total_tokens": len(reply.split())},
            "_timing_s": round(dt, 2),
        })

    def _stream_chat(self, prompt, max_tokens, temperature, top_p, t0):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def cb(tok_text, idx):
            chunk = {"id": "chatcmpl-local", "object": "chat.completion.chunk",
                     "created": int(time.time()),
                     "model": "glm-5.3-flash-uncensored-fp8",
                     "choices": [{"index": 0,
                                  "delta": {"content": tok_text}}]}
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

        text = _model.chat(prompt if isinstance(prompt, list) else
                           [{"role": "user", "content": prompt}],
                           max_tokens=max_tokens,
                           temperature=temperature, top_p=top_p,
                           stream_cb=cb)
        # final chunk + done
        done = {"id": "chatcmpl-local", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        try:
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass

    def _completion(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(p if isinstance(p, str) else p.get("text", "")
                               for p in prompt)
        max_tokens = int(body.get("max_tokens", 256))
        with _model_lock():
            text = _model.chat([{"role": "user", "content": prompt}],
                               max_tokens=max_tokens)
        self._send_json(200, {
            "id": "cmpl-local", "object": "text_completion",
            "created": int(time.time()), "model": "glm-5.3-flash-uncensored-fp8",
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        })


def _model_lock():
    return _lock


def serve(model, port=None, host=None):
    """Start the blocking HTTP server with the given ModelRunner."""
    global _model, PORT, HOST, CHAT_TEMPLATE, TOKENIZER, EOS_IDS
    _model = model
    if port:
        PORT = port
    if host:
        HOST = host
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] listening on http://{HOST}:{PORT}/v1  (key: {API_KEY[:8]}...)", flush=True)
    httpd.serve_forever()
