"""Real-weight loader: stream zai-org/GLM-5.3-Flash FP8 shards into host RAM
and carve them into the engine's parameter structure (mirrors make_fake()).

62 shards (~306 GiB) download in parallel (curl range stripes).  Each shard
buffer is parsed via its safetensors JSON header and carved in ONE pass with
a pre-paired (weight, scale) map: the header dict gives every tensor's
offsets up front, so pairing is trivial and order-independent.

Outputs (identical structure to params.make_fake):
  params_by_chip[c][l] = {attn_hc, ffn_hc, input_ln, post_ln,
                          kda|dsa, moe|mlp}
  embed [V,D] f32, lm_head [V,D] f32, final_ln [D] f32
  expert_host[(l, e)] = {"gu": u8[2I,D], "gu_s": f32[blocks],   # gate|up fused
                         "d": u8[D,I], "d_s": f32[blocks]}      # down

Memory: expert FP8 cold storage ~295 GiB lives in mmap'd tmpfs files (page
cache = RAM, no extra copy).  Dense params ~2.7 GiB dequantized to f32.
"""
from __future__ import annotations

import json
import os
import struct
import time
import urllib.request

import numpy as np

from .config import GlmConfig
from .fp8 import dequant_np

REPO = "zai-org/GLM-5.3-Flash"


def _hdrs(token=None):
    h = {"User-Agent": "glm-tpu-kernel/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def shard_header(url, token=None):
    """Returns (header dict, data_start)."""
    req = urllib.request.Request(url, headers={**_hdrs(token), "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hlen = struct.unpack("<Q", r.read())[0]
    req = urllib.request.Request(url, headers={**_hdrs(token),
                                               "Range": f"bytes=8-{8 + hlen - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), 8 + hlen


def _download_shard(url, out_path, token=None, n_conn=24):
    """Parallel curl range stripes into a preallocated sparse tmpfs file.
    Returns mmap'd uint8 view."""
    import concurrent.futures as cf
    import subprocess

    req = urllib.request.Request(url, headers={**_hdrs(token), "Range": "bytes=0-0"})
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


# ---------------------------------------------------------------------------
# carving helpers
# ---------------------------------------------------------------------------

def _ceil(n, d):
    return -(-n // d)


def _slice_rows(w, d, c):
    chunk = _ceil(w.shape[0], d)
    return w[c * chunk:(c + 1) * chunk]


def _slice_cols(w, d, c):
    chunk = _ceil(w.shape[1], d)
    return w[:, c * chunk:(c + 1) * chunk]


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class _Views:
    """Zero-copy tensor views into a shard buffer."""

    def __init__(self, buf, ds):
        self.buf, self.ds = buf, ds

    def u8(self, meta):
        s, e = meta["data_offsets"]
        return self.buf[self.ds + s:self.ds + e].reshape(meta["shape"])

    def f32(self, meta):
        s, e = meta["data_offsets"]
        raw = self.buf[self.ds + s:self.ds + e]
        return np.frombuffer(raw.tobytes(), np.float32).reshape(meta["shape"])

    def bf16(self, meta):
        s, e = meta["data_offsets"]
        raw = self.buf[self.ds + s:self.ds + e]
        arr = (raw.view(np.uint16).reshape(meta["shape"])
               if raw.flags["C_CONTIGUOUS"]
               else np.frombuffer(raw, np.uint16).reshape(meta["shape"]))
        return _bf16_to_f32(arr)


def carve_shard(header, views: _Views, cfg: GlmConfig, d: int, chips, expert_host,
                out):
    """Carve all tensors of one shard.  `out` dict collects embed/lm_head/
    final_ln."""
    LM = "model.language_model."

    def kda_put(l, key, w, col=False):
        for c in range(d):
            tgt = chips[c][l].setdefault("kda", {})
            tgt[key] = _slice_cols(w, d, c) if col else _slice_rows(w, d, c)

    def dsa_put(l, key, w, col=False):
        for c in range(d):
            tgt = chips[c][l].setdefault("dsa", {})
            tgt[key] = _slice_cols(w, d, c) if col else _slice_rows(w, d, c)

    # pre-pair fp8 weights with their scales
    pairs = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.endswith("weight_scale_inv"):
            continue
        wname = name[: -len("weight_scale_inv")] + "weight"
        if wname in header:
            pairs[wname] = meta

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if name.startswith("model.visual.") or ".layers.45." in name:
            continue
        if name.endswith("weight_scale_inv"):
            continue                                  # consumed via pairs

        # ---- global ----
        if name == LM + "embed_tokens.weight":
            out["embed"] = views.bf16(meta)
            continue
        if name == "lm_head.weight":
            out["lm_head"] = views.bf16(meta)
            continue
        if name == LM + "norm.weight":
            out["final_ln"] = views.bf16(meta)
            continue
        if not name.startswith(LM + "layers."):
            continue
        l = int(name[len(LM + "layers."):].split(".")[0])
        if l >= cfg.n_layers:
            continue
        tail = name[len(f"{LM}layers.{l}."):]

        # ---- replicated hc + norms ----
        if tail == "hc_attn_fn":
            for c in range(d):
                chips[c][l]["attn_hc"] = {"fn": views.bf16(meta)}
        elif tail == "hc_attn_base":
            for c in range(d):
                chips[c][l]["attn_hc"]["base"] = views.f32(meta)
        elif tail == "hc_attn_scale":
            for c in range(d):
                chips[c][l]["attn_hc"]["scale"] = views.f32(meta)
        elif tail == "hc_ffn_fn":
            for c in range(d):
                chips[c][l]["ffn_hc"] = {"fn": views.bf16(meta)}
        elif tail == "hc_ffn_base":
            for c in range(d):
                chips[c][l]["ffn_hc"]["base"] = views.f32(meta)
        elif tail == "hc_ffn_scale":
            for c in range(d):
                chips[c][l]["ffn_hc"]["scale"] = views.f32(meta)
        elif tail == "input_layernorm.weight":
            for c in range(d):
                chips[c][l]["input_ln"] = views.bf16(meta)
        elif tail == "post_attention_layernorm.weight":
            for c in range(d):
                chips[c][l]["post_ln"] = views.bf16(meta)

        # ---- KDA (all BF16) ----
        elif tail.startswith("self_attn.") and cfg.is_kda(l):
            short = tail[len("self_attn."):]
            if short.endswith(".weight"):
                short = short[:-len(".weight")]
            key = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj",
                   "f_a_proj": "f_a", "f_b_proj": "f_b", "b_proj": "b_proj",
                   "g_a_proj": "g_a", "g_b_proj": "g_b",
                   "o_proj": "o_proj"}.get(short.split(".")[0])
            if key is None:
                if short == "o_norm":
                    for c in range(d):
                        chips[c][l].setdefault("kda", {})["o_norm"] = views.bf16(meta)
                continue
            w = views.bf16(meta)
            if short.endswith("_conv1d"):
                w = w[:, 0, :]
            if short in ("dt_bias",):
                w = views.f32(meta)
                kda_put(l, "dt_bias", w)
            elif short == "A_log":
                w = views.f32(meta)
                kda_put(l, "A_log", w)
            elif key == "o_proj":
                kda_put(l, key, w, col=True)
            else:
                kda_put(l, key, w)

        # ---- DSA (FP8 with scales, except kv_b BF16) ----
        elif tail.startswith("self_attn.") and not cfg.is_kda(l):
            short = tail[len("self_attn."):]
            if short.endswith(".weight"):
                short = short[:-len(".weight")]
            if short.startswith("indexer"):
                continue
            if short == "q_a_layernorm":
                for c in range(d):
                    chips[c][l].setdefault("dsa", {})["q_a_ln"] = views.bf16(meta)
                continue
            if short == "kv_a_layernorm":
                for c in range(d):
                    chips[c][l].setdefault("dsa", {})["kv_a_ln"] = views.bf16(meta)
                continue
            key = {"q_a_proj": "q_a", "q_b_proj": "q_b",
                   "kv_a_proj_with_mqa": "kv_a", "kv_b_proj": "kv_b",
                   "o_proj": "o_proj"}.get(short)
            if key is None:
                continue
            smeta = pairs.get(name)
            if short == "kv_b_proj":
                w = views.bf16(meta)
            elif smeta is not None:
                w = dequant_np(views.u8(meta), views.f32(smeta)).astype(np.float32)
            else:
                w = views.bf16(meta)
            if key == "o_proj":
                dsa_put(l, key, w, col=True)
            else:
                dsa_put(l, key, w)

        # ---- MLP / MoE ----
        elif tail.startswith("mlp."):
            if tail == "mlp.gate.weight":
                w = views.bf16(meta)
                for c in range(d):
                    chips[c][l].setdefault("moe", {})["gate_w"] = w
            elif tail == "mlp.gate.e_score_correction_bias":
                w = views.f32(meta)
                for c in range(d):
                    chips[c][l].setdefault("moe", {})["e_bias"] = w
            elif tail.startswith("mlp.shared_experts."):
                short = tail[len("mlp.shared_experts."):]
                base = short[:-len(".weight")] if short.endswith(".weight") else short
                key = {"gate_proj": "g", "up_proj": "u", "down_proj": "d"}[base]
                w = dequant_np(views.u8(meta), views.f32(pairs[name])).astype(np.float32)
                sh = None
                for c in range(d):
                    tgt = chips[c][l].setdefault("moe", {}).setdefault("sh", {})
                    tgt[key] = _slice_cols(w, d, c) if key == "d" else _slice_rows(w, d, c)
            elif tail.startswith("mlp.experts."):
                # experts.{e}.{base}.weight  (scale consumed via pairs)
                parts = tail[len("mlp.experts."):].split(".")
                e, base = int(parts[0]), parts[1]
                if base not in ("gate_proj", "up_proj", "down_proj"):
                    continue
                ex = expert_host.setdefault((l, e), {})
                ex[base] = (views.u8(meta), views.f32(pairs.get(name)))
            elif l in cfg.dense_mlp_layers and tail.endswith(".weight"):
                base = tail[len("mlp."):-len(".weight")]
                if base not in ("gate_proj", "up_proj", "down_proj"):
                    continue
                key = {"gate_proj": "g", "up_proj": "u", "down_proj": "d"}[base]
                smeta = pairs.get(name)
                w = dequant_np(views.u8(meta), views.f32(smeta)).astype(np.float32) if smeta is not None else views.bf16(meta)
                for c in range(d):
                    tgt = chips[c][l].setdefault("mlp", {})
                    tgt[key] = _slice_cols(w, d, c) if key == "d" else _slice_rows(w, d, c)


def finalize_experts(expert_host, cfg, log=print):
    """Fuse gate|up into gu + gu_s (concat rows / concat scale grids)."""
    I, D = cfg.moe_inter, cfg.hidden_size
    n = 0
    for (l, e), ex in expert_host.items():
        if "gu" in ex:
            continue
        g, g_s = ex.pop("gate_proj")
        u, u_s = ex.pop("up_proj")
        ex["gu"] = np.concatenate([g, u], axis=0)
        ex["gu_s"] = np.concatenate([g_s, u_s], axis=0)
        ex["d"], ex["d_s"] = ex.pop("down_proj")
        n += 1
    log(f"[load] fused {n} experts (gu + gu_s)")


def load_real(cfg: GlmConfig, d: int, token=None, log=print,
              workdir="/dev/shm/glmw", max_shards=None):
    """Full load: returns (params_by_chip, embed, lm_head, expert_host)."""
    os.makedirs(workdir, exist_ok=True)
    idx_url = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
    with urllib.request.urlopen(urllib.request.Request(idx_url, headers=_hdrs(token)),
                                timeout=120) as r:
        index = json.loads(r.read())
    files = sorted(set(index["weight_map"].values()))
    if max_shards:
        files = files[:max_shards]
    log(f"[load] {len(files)} shards to fetch")

    chips = [{l: {"kda": None, "dsa": None, "moe": None, "mlp": None,
                  "attn_hc": None, "ffn_hc": None,
                  "input_ln": None, "post_ln": None}
              for l in range(cfg.n_layers)} for _ in range(d)]
    expert_host = {}
    out = {}

    t0 = time.time()
    for fi, fname in enumerate(files):
        url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
        out_path = os.path.join(workdir, fname)
        header, ds = shard_header(url, token)
        buf, total, dt = _download_shard(url, out_path, token)
        views = _Views(buf, ds)
        carve_shard(header, views, cfg, d, chips, expert_host, out)
        del buf
        log(f"  [{fi+1}/{len(files)}] {fname} {total/2**30:.2f} GiB "
            f"({dt:.0f}s, {total/dt/2**20:.0f} MB/s, "
            f"elapsed {(time.time()-t0)/60:.1f} min)")

    finalize_experts(expert_host, cfg, log)
    params_by_chip = []
    for c in range(d):
        chip = dict(chips[c])
        chip["final_ln"] = out["final_ln"]
        params_by_chip.append(chip)
    return params_by_chip, out["embed"], out["lm_head"], expert_host
