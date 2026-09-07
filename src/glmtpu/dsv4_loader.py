"""Real-weight loader for DeepSeek-V4-Flash (env-driven repo switch).

Streams safetensors shards from HuggingFace into host RAM and carves
them into the engine's parameter structure (mirrors dsv4_params layout).

Repo selection (like loader_real.py):
  DSV4_REPO env (default deepseek-ai/DeepSeek-V4-Flash, ungated MIT).
  The notebook sets DSV4_REPO=orcarouter/DeepSeek-V4-Flash-Vision-
  Uncensored when an HF_TOKEN is present; that repo = same text tensors
  + a 267-tensor vision tower (vision.*, aligner.*, image_*) which we
  STRIP by name — those shards' vision tensors are simply skipped, so
  the vision weights are never even materialized.

Expert weights stay FP4 on host: expert_host[(layer_or_"mtp", e)] =
{"w1": u8 [I, D/2], "w1_s": u8 [I, D/32] e8m0, "w3", "w3_s",
 "w2": u8 [D, I/2], "w2_s"} — packed along K, low nibble first
(research/dsv4-port-spec.md §3).

Tensor name map (top-level, verified vs research/dsv4-index.json):
  embed.weight, head.weight, norm.weight, hc_head_{fn,base,scale}
  layers.L.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale} (fp8 128x128)
  layers.L.attn.{q_norm,kv_norm}.weight, attn_sink (f32)
  layers.L.attn.compressor.{wkv,wgate}.weight (bf16), ape (f32),
      norm.weight
  layers.L.attn.indexer.{wq_b.weight,wq_b.scale}, weights_proj.weight
  layers.L.attn.indexer.compressor.{wkv,wgate,ape,norm}
  layers.L.{attn_norm,ffn_norm}.weight
  layers.L.ffn.gate.weight, gate.bias (f32) | gate.tid2eid (i32)
  layers.L.ffn.shared_experts.{w1,w2,w3}.{weight,scale} (fp8)
  layers.L.ffn.experts.E.{w1,w2,w3}.weight (fp4 u8) + .scale (e8m0 u8)
  mtp.0.* (same + e_proj/h_proj/enorm/hnorm/norm + own hc_head_*)
"""
from __future__ import annotations

import json
import os
import struct
import time
import urllib.request

import numpy as np

from .dsv4_config import Dsv4Config
from .fp8 import dequant_np

REPO = os.environ.get("DSV4_REPO", "deepseek-ai/DeepSeek-V4-Flash")

# the vision tower (267 tensors) — stripped, never downloaded materialized
STRIP_PREFIXES = ("vision.", "aligner.", "image_")


def _hdrs(token=None):
    h = {"User-Agent": "dsv4-tpu-kernel/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def shard_header(url, token=None):
    req = urllib.request.Request(url, headers={**_hdrs(token),
                                               "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hlen = struct.unpack("<Q", r.read())[0]
    req = urllib.request.Request(url, headers={**_hdrs(token),
                                               "Range": f"bytes=8-{8+hlen-1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), 8 + hlen


def _download_shard(url, out_path, token=None, n_conn=24):
    import concurrent.futures as cf
    import subprocess
    req = urllib.request.Request(url, headers={**_hdrs(token),
                                               "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers["Content-Range"].split("/")[-1])
    if not (os.path.exists(out_path) and os.path.getsize(out_path) == total):
        with open(out_path, "wb") as f:
            f.truncate(total)
    buf = np.memmap(out_path, dtype=np.uint8, mode="r+")
    stripe = max(16 << 20, total // n_conn)
    t0 = time.time()

    def fetch(rng):
        s, e = rng
        part = f"{out_path}.part{s}"
        cmd = ["curl", "-sL", "--fail", "--retry", "4", "-r", f"{s}-{e}"]
        for k, v in _hdrs(token).items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += ["-o", part, url]
        subprocess.run(cmd, check=True)
        with open(part, "rb") as f:
            buf[s:e + 1] = np.frombuffer(f.read(), dtype=np.uint8)
        os.remove(part)

    with cf.ThreadPoolExecutor(max_workers=min(n_conn, 32)) as ex:
        list(ex.map(fetch, [(s, min(s + stripe - 1, total - 1))
                            for s in range(0, total, stripe)]))
    return buf, total, time.time() - t0


class _Views:
    def __init__(self, buf, ds):
        self.buf, self.ds = buf, ds

    def raw(self, meta):
        s, e = meta["data_offsets"]
        return self.buf[self.ds + s:self.ds + e]

    def u8(self, meta):
        return self.raw(meta).reshape(meta["shape"])

    def f32(self, meta):
        return np.frombuffer(self.raw(meta).tobytes(),
                             np.float32).reshape(meta["shape"])

    def bf16(self, meta):
        raw = self.raw(meta)
        arr = (raw.view(np.uint16).reshape(meta["shape"])
               if raw.flags["C_CONTIGUOUS"]
               else np.frombuffer(raw, np.uint16).reshape(meta["shape"]))
        return (arr.astype(np.uint32) << 16).view(np.float32)


def _ceil(n, d):
    return -(-n // d)


def _slice_rows(w, d, c):
    chunk = _ceil(w.shape[0], d)
    return w[c * chunk:(c + 1) * chunk]


def _slice_cols(w, d, c):
    chunk = _ceil(w.shape[1], d)
    return w[:, c * chunk:(c + 1) * chunk]


def carve_shard(header, views, cfg: Dsv4Config, d, chips, expert_host,
                out):
    """Carve all tensors of one shard into the engine layout."""

    def put(l, key, w, col=False, site="attn"):
        for c in range(d):
            tgt = chips[c][l].setdefault(site, {})
            tgt[key] = _slice_cols(w, d, c) if col else _slice_rows(w, d, c)

    # pre-pair fp8 weights with scales
    pairs = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.endswith(".scale"):
            continue
        wname = name[:-len(".scale")] + ".weight"
        if wname in header:
            pairs[wname] = meta

    for name, meta in header.items():
        if name == "__metadata__" or name.endswith(".scale"):
            continue
        if name.startswith(STRIP_PREFIXES):
            continue                                   # vision: stripped

        # ---- global ----
        if name == "embed.weight":
            out["embed"] = views.bf16(meta)
            continue
        if name == "head.weight":
            out["lm_head"] = views.bf16(meta)
            continue
        if name == "norm.weight":
            out["final_ln"] = views.bf16(meta)
            continue
        if name.startswith("hc_head_"):
            out.setdefault("head", {})[
                name[len("hc_head_"):]] = views.f32(meta)
            continue
        if not name.startswith(("layers.", "mtp.")):
            continue

        # ---- layer or mtp ----
        is_mtp = name.startswith("mtp.")
        if is_mtp:
            l, tail = "mtp", name[len("mtp.0."):]
            site_lay = chips[0]["mtp"]
        else:
            parts = name.split(".")
            l = int(parts[1])
            if l >= cfg.n_layers:
                continue
            tail = ".".join(parts[2:])
        lay = chips[0][l] if not is_mtp else site_lay

        def to_all(key, val, site):
            for c in range(d):
                chips[c]["mtp" if is_mtp else l].setdefault(
                    site, {})[key] = val

        # ---- hc + norms ----
        if tail.startswith("hc_attn_") or tail.startswith("hc_ffn_"):
            site = "attn_hc" if tail.startswith("hc_attn") else "ffn_hc"
            key = tail.split("_", 2)[2]
            to_all(key, views.f32(meta) if key != "fn"
                   else views.bf16(meta).astype(np.float32), site)
        elif tail == "attn_norm.weight":
            to_all("attn_norm", views.bf16(meta), "attn")
        elif tail == "ffn_norm.weight":
            to_all("ffn_norm", views.bf16(meta), "ffn")
        elif tail in ("enorm.weight", "hnorm.weight", "norm.weight") \
                and is_mtp:
            to_all({"enorm.weight": "enorm", "hnorm.weight": "hnorm",
                    "norm.weight": "norm"}[tail],
                   views.bf16(meta), "attn")

        # ---- attention weights ----
        elif tail.startswith("attn."):
            short = tail[len("attn."):]
            if short == "attn_sink":
                # f32 [n_heads] -> head-shard
                w = views.f32(meta)
                for c in range(d):
                    chips[c]["mtp" if is_mtp else l].setdefault(
                        "attn", {})["attn_sink"] = _slice_rows(w, d, c)
            elif short == "q_norm.weight":
                to_all("q_norm", views.bf16(meta), "attn")
            elif short == "kv_norm.weight":
                to_all("kv_norm", views.bf16(meta), "attn")
            elif short in ("wq_a.weight", "wq_b.weight", "wkv.weight",
                           "wo_a.weight", "wo_b.weight"):
                key = short[:-len(".weight")]
                smeta = pairs.get(name)
                if smeta is not None:
                    w = dequant_np(views.u8(meta),
                                   views.f32(smeta)).astype(np.float32)
                else:
                    w = views.bf16(meta)
                col = key == "wo_b"      # wo_b: col-shard over olr
                if is_mtp:
                    for c in range(d):
                        chips[c]["mtp"].setdefault("attn", {})[key] = \
                            _slice_cols(w, d, c) if col else \
                            _slice_rows(w, d, c)
                else:
                    put(l, key, w, col=col)

        # ---- compressor + indexer ----
        elif tail.startswith("attn.compressor.") or \
                tail.startswith("attn.indexer.compressor."):
            is_idx = "indexer" in tail
            site = "icomp" if is_idx else "comp"
            short = tail.split("compressor.")[-1]
            if short in ("wkv.weight", "wgate.weight"):
                to_all(short[:-len(".weight")], views.bf16(meta), site)
            elif short == "ape":
                to_all("ape", views.f32(meta), site)
            elif short == "norm.weight":
                to_all("norm", views.bf16(meta), site)
        elif tail.startswith("attn.indexer."):
            short = tail[len("attn.indexer."):]
            if short == "wq_b.weight":
                smeta = pairs.get(name)
                w = dequant_np(views.u8(meta),
                               views.f32(smeta)).astype(np.float32)
                to_all("wq_b", w, "idx")
            elif short == "weights_proj.weight":
                to_all("weights_proj", views.bf16(meta), "idx")

        # ---- FFN ----
        elif tail.startswith("ffn."):
            if tail == "ffn.gate.weight":
                to_all("w", views.bf16(meta), "gate")
            elif tail == "ffn.gate.bias":
                to_all("bias", views.f32(meta), "gate")
            elif tail == "ffn.gate.tid2eid":
                raw = views.raw(meta)
                arr = (raw.view(np.int32).reshape(meta["shape"])
                       if isinstance(raw, np.ndarray)
                       else np.frombuffer(raw, np.int32).reshape(
                           meta["shape"]))
                to_all("tid2eid", arr, "gate")
            elif tail.startswith("ffn.shared_experts."):
                short = tail[len("ffn.shared_experts."):]
                if short.endswith(".weight"):
                    key = short[:-len(".weight")]
                    w = dequant_np(views.u8(meta),
                                   views.f32(pairs[name])).astype(np.float32)
                    for c in range(d):
                        tgt = chips[c]["mtp" if is_mtp else
                                      l].setdefault("shared", {})
                        tgt[key] = _slice_cols(w, d, c) if key == "w2" \
                            else _slice_rows(w, d, c)
            elif tail.startswith("ffn.experts."):
                # experts.E.{w1,w2,w3}.weight (fp4 packed) + .scale (e8m0)
                parts = tail[len("ffn.experts."):].split(".")
                e, base = int(parts[0]), parts[1]
                if base not in ("w1", "w2", "w3"):
                    continue
                key = ("mtp", e) if is_mtp else (l, e)
                ex = expert_host.setdefault(key, {})
                ex[base] = views.u8(meta)             # packed fp4 bytes
                smeta = pairs.get(name)
                if smeta is not None:
                    ex[base + "_s"] = views.u8(smeta)  # e8m0 u8 scales

        # ---- MTP projections ----
        elif is_mtp and tail in ("e_proj.weight", "h_proj.weight"):
            key = tail[:-len(".weight")]
            smeta = pairs.get(name)
            if smeta is not None:
                w = dequant_np(views.u8(meta),
                               views.f32(smeta)).astype(np.float32)
            else:
                w = views.bf16(meta)
            to_all(key, w, "attn")


def finalize_shards(chips, cfg, d, log=print):
    """Convert bf16-valued params to f32 containers and fill structural
    defaults for layers/tails not present in every repo variant."""
    for c in range(d):
        for l in range(cfg.n_layers):
            lay = chips[c][l]
            # gate bias default (hash layers have none)
            g = lay.setdefault("gate", {})
            g.setdefault("bias", None)
            g.setdefault("tid2eid", None)
            # comp/idx/icomp defaults
            lay.setdefault("comp", None)
            lay.setdefault("idx", None)
            lay.setdefault("icomp", None)
        m = chips[c]["mtp"]
        m.setdefault("gate", {}).setdefault("bias", None)
        m["gate"].setdefault("tid2eid", None)


def load_real(cfg: Dsv4Config, d: int, token=None, log=print,
              workdir="/dev/shm/dsv4w", max_shards=None):
    """Full load: returns (params_by_chip, embed, lm_head, expert_host)."""
    if token is None:
        token = os.environ.get("HF_TOKEN") or None
    os.makedirs(workdir, exist_ok=True)
    idx_url = f"https://huggingface.co/{REPO}/resolve/main/" \
              "model.safetensors.index.json"
    with urllib.request.urlopen(
            urllib.request.Request(idx_url, headers=_hdrs(token)),
            timeout=120) as r:
        index = json.loads(r.read())
    files = sorted(set(index["weight_map"].values()))
    if max_shards:
        files = files[:max_shards]
    log(f"[load] repo {REPO}: {len(files)} shards to fetch")

    D = cfg.hidden_size
    chips = [{l: {"attn": {}, "ffn_hc": {}, "attn_hc": {}, "ffn": {},
                  "gate": {}, "shared": {}, "comp": {}, "idx": {},
                  "icomp": {}}
              for l in range(cfg.n_layers)} for _ in range(d)]
    for c in range(d):
        chips[c]["mtp"] = {"attn": {}, "attn_hc": {}, "ffn_hc": {},
                           "ffn": {}, "gate": {}, "shared": {}}
    expert_host = {}
    out = {}

    t0 = time.time()
    n_seen = 0
    for fi, fname in enumerate(files):
        url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
        out_path = os.path.join(workdir, fname)
        buf, total, dt = _download_shard(url, out_path, token)
        header, ds = shard_header(url, token)
        carve_shard(header, _Views(buf, ds), cfg, d, chips, expert_host,
                    out)
        n_seen += 1
        log(f"[load] {fi+1}/{len(files)} {fname} "
            f"({total/1e9:.1f} GB, {dt:.0f}s)")
    finalize_shards(chips, cfg, d, log=log)
    log(f"[load] done in {(time.time()-t0)/60:.1f} min; "
        f"experts: {len(expert_host)}")

    embed = out["embed"]
    lm_head = out["lm_head"]
    for c in range(d):
        chips[c]["head"] = out["head"]
        chips[c]["final_ln"] = out["final_ln"]
    return chips, embed, lm_head, expert_host
