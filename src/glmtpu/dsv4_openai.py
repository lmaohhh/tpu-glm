"""OpenAI-compatible API server for the DSV4 engine (stdlib only).

Differences from the GLM server (openai_api.py):
  - reasoning_content split: DSV4 thinks — streaming emits
    delta.reasoning_content while inside <think>…</think>, then
    delta.content afterwards (DeepSeek style).
  - clean 400 error on image/video content parts.
  - model id deepseek-v4-flash (or -vision-uncensored variant).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import dsv4_chat

API_KEY = "kaggle-sfw-token-9999"
HOST = "0.0.0.0"
PORT = 8080
MODEL_ID = "deepseek-v4-flash"

_model = None
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

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
                {"id": MODEL_ID, "object": "model",
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
        else:
            self._send_json(404, {"error": "not found"})

    def _chat(self, body):
        messages = body.get("messages", [])
        if dsv4_chat._has_image_or_video(messages):
            self._send_json(400, {"error": {
                "message": "Image and video inputs are not supported by "
                           "this engine (vision tower weights are "
                           "stripped; text-only serving).",
                "type": "invalid_request_error", "code": "modalitiy_not_supported"}})
            return
        max_tokens = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stream = bool(body.get("stream", False))
        t0 = time.time()
        with _lock:
            if stream:
                self._stream_chat(messages, max_tokens, temperature, top_p)
                return
            try:
                text = _model.chat(messages, max_tokens=max_tokens,
                                   temperature=temperature, top_p=top_p)
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e),
                                                "type": "invalid_request_error"}})
                return
        parsed = dsv4_chat.parse_message_from_completion_text(
            text, _model.thinking_mode)
        self._send_json(200, {
            "id": "chatcmpl-local", "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {
                             "role": "assistant",
                             "content": parsed["content"],
                             "reasoning_content":
                                 parsed["reasoning_content"] or None}}],
            "usage": {"prompt_tokens": 0,
                      "completion_tokens": len(text.split()),
                      "total_tokens": len(text.split())},
            "_timing_s": round(time.time() - t0, 2),
        })

    def _stream_chat(self, messages, max_tokens, temperature, top_p):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        state = {"in_think": True, "started_think": False}

        def cb(tok_text, idx):
            # split reasoning vs content by </think>
            nonlocal state
            pieces = []
            buf = tok_text
            while buf:
                if state["in_think"]:
                    if not state["started_think"]:
                        state["started_think"] = True
                    if "</think>" in buf:
                        pre, buf = buf.split("</think>", 1)
                        if pre:
                            pieces.append(("reasoning", pre))
                        state["in_think"] = False
                    else:
                        pieces.append(("reasoning", buf))
                        buf = ""
                else:
                    pieces.append(("content", buf))
                    buf = ""
            for kind, txt in pieces:
                if not txt:
                    continue
                delta = {"reasoning_content": txt} if kind == "reasoning" \
                    else {"content": txt}
                chunk = {"id": "chatcmpl-local",
                         "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": MODEL_ID,
                         "choices": [{"index": 0, "delta": delta}]}
                try:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass

        try:
            _model.chat(messages, max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p,
                        stream_cb=cb)
        except Exception:
            pass
        done = {"id": "chatcmpl-local", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop"}]}
        try:
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass


def serve(model, port=None, host=None, model_id=None):
    global _model, PORT, HOST, MODEL_ID
    _model = model
    if port:
        PORT = port
    if host:
        HOST = host
    if model_id:
        MODEL_ID = model_id
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] listening on http://{HOST}:{PORT}/v1  "
          f"(key: {API_KEY[:8]}...)", flush=True)
    httpd.serve_forever()
