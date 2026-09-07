"""Assemble the self-contained Kaggle notebook from the validated modules."""
import json
import os

import nbformat


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


fp8 = read("src/glmtpu/fp8.py")
config = read("src/glmtpu/config.py")
layers = read("src/glmtpu/layers.py")
params = read("src/glmtpu/params.py")
runtime = read("src/glmtpu/runtime.py")
loader = read("src/glmtpu/loader_real.py")
openai_api = read("src/glmtpu/openai_api.py")
glue = read("src/glmtpu/runner_glue.py")

cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbformat.v4.new_code_cell(src))


md("""# GLM-5.3-Flash-Uncensored-FP8 on Kaggle TPU v5e-8

Custom pure-JAX serving engine for the 321B-param FP8 MoE that cannot fit
in HBM: dense weights TP-sharded across 8 chips, all 288x42 routed experts
cold in the 330 GB host RAM, 8-expert hot banks per chip/layer in HBM.

- Weights: `zai-org/GLM-5.3-Flash` (ungated, identical geometry to the
  orcarouter ab-literated FP8 variant)
- Architecture: 34 KDA linear-attention + 11 DSA/MLA layers, mHC
  hyper-connections, sigmoid/noaux_tc router, NoPE (zero rope)
- v1 simplifications: DSA lightning-indexer skipped (full attention over
  the 512-d latent cache), MTP head + vision tower never downloaded
- API: OpenAI-compatible on :8080 + cloudflared public tunnel (URL printed
  below, rotates each session; same API key as the GPU notebook)
- Every engine module was validated on 8 simulated CPU devices before this
  push: prefill/decode bit-deterministic, hot-bank refresh exact""")

code("""import os, sys, time

print("checking TPU...")
import jax
print("jax", jax.__version__)
devs = jax.devices()
print("devices:", devs)
assert len(devs) == 8, f"expected 8 TPU devices, got {len(devs)}"
os.makedirs("/kaggle/tmp/glmtpu", exist_ok=True)
""")

code("""# ---- engine modules: pull from GitHub, fall back to embedded copies ----
# (embedded MODS below stay in sync with the repo; git wins when reachable)
import os, sys, subprocess

os.makedirs("/kaggle/tmp", exist_ok=True)
try:
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/lmaohhh/tpu-glm.git",
                    "/kaggle/tmp/tpu-glm"], check=True, timeout=120)
    sys.path.insert(0, "/kaggle/tmp/tpu-glm/src")
    print("engine: from GitHub @", subprocess.run(
        ["git", "-C", "/kaggle/tmp/tpu-glm", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip())
except Exception as e:
    print("git pull failed -> using embedded modules:", e)

if "glmtpu" not in sys.modules:
    MODS = {}
    MODS["fp8.py"] = '''__FP8__'''
    MODS["config.py"] = '''__CONFIG__'''
    MODS["layers.py"] = '''__LAYERS__'''
    MODS["params.py"] = '''__PARAMS__'''
    MODS["runtime.py"] = '''__RUNTIME__'''
    MODS["loader_real.py"] = '''__LOADER__'''
    MODS["openai_api.py"] = '''__OPENAI__'''
    MODS["runner_glue.py"] = '''__GLUE__'''
    MODS["__init__.py"] = ""

    os.makedirs("/kaggle/tmp/glmtpu", exist_ok=True)
    for name, src in MODS.items():
        with open(f"/kaggle/tmp/glmtpu/{name}", "w", encoding="utf-8") as f:
            f.write(src)
    sys.path.insert(0, "/kaggle/tmp")

from glmtpu.config import GlmConfig
import glmtpu.layers, glmtpu.runtime, glmtpu.loader_real
print("engine modules ready")""")

code("""# ---- engine self-test on the real TPU (fake weights, tiny config) ----
import numpy as np
from glmtpu.config import GlmConfig
from glmtpu.params import make_fake
from glmtpu.runtime import Runner

cfg = GlmConfig.tiny()
pbc, embed, lm_head, expert_host = make_fake(cfg, d=8, seed=1)
r = Runner(cfg, pbc, embed, lm_head, expert_host)
tokens = np.random.randint(0, cfg.vocab_size, size=100).tolist()
g = r.generate(tokens, max_new_tokens=8, temperature=0.0)
g2 = r.generate(tokens, max_new_tokens=8, temperature=0.0)
assert g == g2, "greedy determinism failed"
print("TPU self-test OK:", g)""")

code("""# ---- tokenizer + chat template ----
import urllib.request

BASE = "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main"
for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    urllib.request.urlretrieve(f"{BASE}/{fn}", f"/kaggle/tmp/{fn}")
    print("got", fn)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("/kaggle/tmp/tokenizer.json")
chat_template = open("/kaggle/tmp/chat_template.jinja").read()
print("vocab:", tok.get_vocab_size())""")

code("""# ---- load real weights: 62 shards (~306 GiB) -> host RAM ----
import numpy as np
from glmtpu.config import GlmConfig
from glmtpu.loader_real import load_real
from glmtpu.runtime import Runner

t0 = time.time()
cfg = GlmConfig.real()
cfg.n_slots = 8          # hot experts per chip per MoE layer in HBM
params_by_chip, embed, lm_head, expert_host = load_real(
    cfg, d=8, log=print, workdir="/dev/shm/glmw")
print(f"weights loaded in {(time.time()-t0)/60:.1f} min; experts: {len(expert_host)}")

runner = Runner(cfg, params_by_chip, embed, lm_head, expert_host, log=print)
print("runner ready")
del params_by_chip""")

code("""# ---- generation sanity check ----
prompt = "[gMASK]<sop><|user|>\\nSay something with teeth. Be brief.\\n<|assistant|>\\n<think>\\n"
ids = tok.encode(prompt, add_special_tokens=False)
print("prompt tokens:", len(ids))
t0 = time.time()
logits = runner.prefill(ids)
out = []
for i in range(64):
    t = runner._sample(logits, 0.7, 0.95)
    if t in runner.cfg.eos_ids:
        break
    out.append(t)
    logits = runner._decode_one(t, 0.7, 0.95)
print(f"generated {len(out)} tokens in {time.time()-t0:.1f}s")
print("OUTPUT:", tok.decode(out)[:500])""")

code("""# ---- OpenAI-compatible server + public cloudflared tunnel ----
import threading, subprocess, re, time

from glmtpu import openai_api
from glmtpu.runner_glue import GlmModelRunner

model = GlmModelRunner(runner, tok, chat_template)
server = threading.Thread(target=openai_api.serve,
                          args=(model, 8080, "0.0.0.0"), daemon=True)
server.start()
time.sleep(2)

cf = "/kaggle/tmp/cloudflared"
if not os.path.exists(cf):
    subprocess.run(["wget", "-q",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "-O", cf], check=True)
    subprocess.run(["chmod", "+x", cf], check=True)
subprocess.Popen([cf, "tunnel", "--url", "http://localhost:8080",
                  "--no-autoupdate"],
                 stdout=open("/kaggle/tmp/tunnel.log", "w"),
                 stderr=subprocess.STDOUT)
url = None
for _ in range(24):
    time.sleep(5)
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",
                  open("/kaggle/tmp/tunnel.log").read())
    if m:
        url = m.group(0)
        break
assert url, "no tunnel URL"
print("=" * 60)
print("PUBLIC API:", url + "/v1")
print("API KEY:   kaggle-sfw-token-9999")
print("=" * 60)
with open("/kaggle/working/api_url.txt", "w") as f:
    f.write(url + "/v1\\n")""")

code("""# ---- smoke test through the public URL (openai client) ----
import subprocess
subprocess.run(["pip", "install", "-q", "openai"], check=False)
import re
from openai import OpenAI
url = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",
                open("/kaggle/tmp/tunnel.log").read()).group(0)
c = OpenAI(base_url=f"{url}/v1", api_key="kaggle-sfw-token-9999")
r = c.chat.completions.create(
    model="glm-5.3-flash-uncensored-fp8",
    messages=[{"role": "user", "content": "Say something with teeth."}],
    max_tokens=256, temperature=0.7)
reply = r.choices[0].message.content
print(reply)
with open("/kaggle/working/api_smoke.txt", "w") as f:
    f.write(reply or "")""")

code("""# ---- keep alive until the session ends ----
import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass""")

# embed module sources into the MODS cell (find it by marker)
mods_idx = next(i for i, c in enumerate(cells)
                if c.cell_type == "code" and "MODS[" in c.source)
joined = "".join(cells[mods_idx].source)
joined = joined.replace("__FP8__", fp8)
joined = joined.replace("__CONFIG__", config)
joined = joined.replace("__LAYERS__", layers)
joined = joined.replace("__PARAMS__", params)
joined = joined.replace("__RUNTIME__", runtime)
joined = joined.replace("__LOADER__", loader)
joined = joined.replace("__OPENAI__", openai_api)
joined = joined.replace("__GLUE__", glue)
cells[mods_idx].source = joined

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "TPU-VM",
}
out_path = "notebook/glm53-flash-tpu.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
nbformat.validate(nbformat.read(out_path, as_version=4))
print("notebook written + VALID:", os.path.getsize(out_path), "bytes,",
      len(cells), "cells")
