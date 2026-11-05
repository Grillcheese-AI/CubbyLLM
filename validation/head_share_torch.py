"""validation/head_share_torch.py -- order 0050, stage 1 in torch on a CUDA card (Nick's A100 session).

  decode  the trained 450M (a checkpoint's weights): one decode step (``CubbyModel.step``, the greedy token read back)
          against the head alone on a state of the step's shape, a full 32k head, and K gathered rows; batch 1 and 16,
          fp32, as serving decodes (int8 is grilly2's path, measured at home: ``head_share.py``)
  train   the head's share of a training step: forward + backward of a micro batch with the 128k head and its
          cross-entropy, against the same step through the trunk alone (its loss read off the state); time and peak
          memory. For each --ckpt's shape (the 450M and the grown 1.36B), weights at their init (shapes, not values)
  dump    held-in states for the shortlist's build, as ``head_dump.py`` writes them (fp32, TF32 off), from the
          token cache's training region: the build grows past the 106k states the home card could give

    python validation/head_share_torch.py decode --ckpt <run>/base450m_cont_final.pt
    python validation/head_share_torch.py train --ckpt <450M ckpt> --ckpt <grown ckpt> --micro 8 --seq 1024
    python validation/head_share_torch.py dump --ckpt <450M ckpt> --cache <token cache> --windows 32 --out <dir>
-> validation/logs/head_share_<mode>_cuda_<date>.json; dump -> <out>/heldin.npz

Wired: STANDALONE (an experiment; 0050's report reads it).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"


def load(ckpt: str, dev, weights: bool = True):
    import torch
    from cubbyllm.model.build import build_model
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    model = build_model(dict(ck["meta"]), dev)
    if weights:
        with torch.no_grad():
            for p, s in zip(model.parameters(), ck["params"]):
                p.copy_(s.to(p.device, torch.float32))
    meta = dict(ck["meta"])
    del ck
    return model, meta


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, reps: int, warm: int = 3) -> dict:
    import torch
    for _ in range(warm):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": round(statistics.median(ts), 3), "mean_ms": round(statistics.fmean(ts), 3), "n": reps}


def decode(a, dev) -> dict:
    import torch
    model, meta = load(a.ckpt[0], dev)               # --ckpt is a list (train takes two)
    head, d, out = model.head, meta["D"], {"meta": meta, "runs": []}
    w32 = torch.randn(32000, d, device=dev) * 0.02
    for dtype in (torch.float32,):                     # serving decodes in fp32 (bf16 autocast meets the fp32 decode ring)
        for b in (1, 16):
            amp = dtype != torch.float32
            state = model.init_state(b, dev)
            tok = torch.randint(100, 30000, (b,), device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in range(64):                                   # a prompt's worth of state
                    logits, state = model.step(tok, state)
                    tok = logits.argmax(-1)

                def one():
                    nonlocal tok, state
                    logits, state = model.step(tok, state)
                    tok = logits.argmax(-1)
                    tok.cpu()
                step = _timed(one, a.reps)
                x = torch.randn(b, 1, d, device=dev)
                h = _timed(lambda: head.logits(x, head.vocab).argmax(-1).cpu(), a.reps)
                h32 = _timed(lambda: (x @ w32.t()).argmax(-1).cpu(), a.reps)
                rows = {str(k): _timed(lambda k=k: (x @ w32[:k].t()).argmax(-1).cpu(), a.reps) for k in (256, 1024, 4096)}
            out["runs"].append({"dtype": str(dtype).split(".")[-1], "batch": b, "step": step, "head": h, "head32k": h32,
                                "shortlist_rows": rows, "head_share_of_step": round(h["p50_ms"] / step["p50_ms"], 4)})
            print(out["runs"][-1]["dtype"], b, "step", step["p50_ms"], "head", h["p50_ms"], "32k", h32["p50_ms"], flush=True)
    out["head_bytes_fp32"] = head.vocab * d * 4
    return out


def train(a, dev) -> dict:
    import torch
    import torch.nn.functional as F
    out = {"micro": a.micro, "seq": a.seq, "shapes": []}
    for ck in a.ckpt:
        model, meta = load(ck, dev, weights=False)
        params = list(model.parameters())
        micro = a.micro
        while True:                                      # a card too small for --micro: halve it, record what ran
            try:
                rec = _train_shape(model, meta, params, ck, micro, a, dev)
                break
            except torch.cuda.OutOfMemoryError:
                for p in params:
                    p.grad = None
                torch.cuda.empty_cache()
                if micro == 1:
                    raise
                micro //= 2
                print(f"out of memory: micro {micro * 2} -> {micro}", flush=True)
        out["shapes"].append(rec)
        print(rec["ckpt"], "micro", micro, "full", rec["full"]["p50_ms"], "trunk", rec["trunk"]["p50_ms"],
              "share", rec["head_share_of_step"], flush=True)
        del model, params
        torch.cuda.empty_cache()
    return out


def _train_shape(model, meta, params, ck, micro, a, dev) -> dict:
    import torch
    import torch.nn.functional as F
    if True:                                         # (one shape at one micro batch)
        x = torch.randint(100, meta["vocab"] - 1, (micro, a.seq + 1), device=dev)
        inp, y = x[:, :-1], x[:, 1:]

        def full():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.logits_from(model.features(inp))
            loss = F.cross_entropy(logits.float().reshape(-1, meta["vocab"]), y.reshape(-1))
            loss.backward()
            for p in params:
                p.grad = None

        def trunk():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.features(inp)
            h.float().pow(2).mean().backward()
            for p in params:
                p.grad = None

        def fused(chunk=None):                           # 0050 2a: the same loss, the logits never stored
            from cubbyllm.model.vocab.fused_ce import CHUNK, chunked_cross_entropy
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.features(inp)
            chunked_cross_entropy(h, model.head._codes(), y, chunk=chunk or CHUNK, dtype=torch.bfloat16).backward()
            for p in params:
                p.grad = None
        rec = {"ckpt": os.path.basename(ck), "D": meta["D"], "L": meta["L"], "vocab": meta["vocab"], "micro": micro,
               "params_M": round(sum(p.numel() for p in params) / 1e6, 1)}
        for name, fn in (("full", full), ("trunk", trunk)) + ((("fused", fused),) if a.fused else ()):
            torch.cuda.reset_peak_memory_stats()
            rec[name] = _timed(fn, a.reps, warm=2)
            rec[name]["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec["head_share_of_step"] = round(1 - rec["trunk"]["p50_ms"] / rec["full"]["p50_ms"], 4)
        rec["head_peak_gb"] = round(rec["full"]["peak_gb"] - rec["trunk"]["peak_gb"], 2)
        if a.fused:                                      # 2a's lines: >= 10 GB freed, <= 3% slower at the same micro
            rec["fused_freed_gb"] = round(rec["full"]["peak_gb"] - rec["fused"]["peak_gb"], 2)
            rec["fused_slower"] = round(rec["fused"]["p50_ms"] / rec["full"]["p50_ms"] - 1, 4)
            for c in [int(v) for v in a.chunks.split(",") if v]:   # 0050: one try at larger chunks (Cowork, 2026-11-05)
                torch.cuda.reset_peak_memory_stats()
                t = _timed(lambda: fused(c), a.reps, warm=2)
                rec.setdefault("fused_chunks", {})[str(c)] = {**t, "peak_gb": round(
                    torch.cuda.max_memory_allocated() / 2**30, 2), "slower": round(t["p50_ms"] / rec["full"]["p50_ms"] - 1, 4)}
            rec["largest_micro"] = {name: _largest(fn_of, a, meta, dev) for name, fn_of in
                                    (("full", lambda i, t: _step(model, params, meta, i, t, False)),
                                     ("fused", lambda i, t: _step(model, params, meta, i, t, True)))}
        return rec


def _step(model, params, meta, inp, y, fused: bool) -> None:
    import torch
    import torch.nn.functional as F
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.features(inp)
        logits = None if fused else model.logits_from(h)
    if fused:
        from cubbyllm.model.vocab.fused_ce import chunked_cross_entropy
        chunked_cross_entropy(h, model.head._codes(), y, dtype=torch.bfloat16).backward()
    else:
        F.cross_entropy(logits.float().reshape(-1, meta["vocab"]), y.reshape(-1)).backward()
    for p in params:
        p.grad = None


def _largest(fn, a, meta, dev) -> dict:
    """The largest micro batch (of --micros) that fits under 92% of the card, and its tokens per second."""
    import torch
    cap = 0.92 * torch.cuda.get_device_properties(0).total_memory / 2**30
    best = None
    for m in [int(v) for v in a.micros.split(",")]:
        x = torch.randint(100, meta["vocab"] - 1, (m, a.seq + 1), device=dev)
        try:
            torch.cuda.reset_peak_memory_stats()
            t = _timed(lambda: fn(x[:, :-1], x[:, 1:]), 3, warm=1)
            peak = torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if peak > cap:
            break
        best = {"micro": m, "tok_s": round(m * a.seq / (t["p50_ms"] / 1e3)), "peak_gb": round(peak, 2)}
        print("  largest micro so far:", best, flush=True)
    torch.cuda.empty_cache()
    return best or {}


def dump(a, dev) -> dict:
    import torch
    from head_dump import SHARDS, windows
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load(a.ckpt[0], dev)
    keep = {k: [] for k in ("hidden", "target", "prev", "top", "toplogit", "lse", "tlogit", "shard")}
    for si, name in enumerate(SHARDS):
        for w in windows(a.cache, name, a.split, a.windows, a.seq, a.phase):
            x = torch.as_tensor(w[:-1], device=dev).unsqueeze(0)
            y = torch.as_tensor(w[1:], device=dev)
            with torch.no_grad():
                h = model.features(x)[0]
                logits = model.logits_from(h.unsqueeze(0))[0].float()
                lse = torch.logsumexp(logits, -1)
                tv, ti = logits.topk(16, -1)
                keep["hidden"].append(h.half().cpu().numpy())
                keep["target"].append(w[1:].astype(np.int32))
                keep["prev"].append(w[:-1].astype(np.int32))
                keep["top"].append(ti.int().cpu().numpy())
                keep["toplogit"].append(tv.cpu().numpy())
                keep["lse"].append(lse.cpu().numpy())
                keep["tlogit"].append(logits.gather(1, y.unsqueeze(1))[:, 0].cpu().numpy())
            keep["shard"].append(np.full(len(w) - 1, si, dtype=np.int8))
        print(name, "done", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name or a.split}.npz")
    np.savez(path, **{k: np.concatenate(v) for k, v in keep.items()}, shards=np.array(SHARDS))
    n = sum(len(t) for t in keep["target"])
    print("->", path, n, "positions")
    return {"positions": n, "path": os.path.basename(path)}


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("decode", "train", "dump"))
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--fused", action="store_true", help="train: also the chunked cross-entropy (0050 2a) and the "
                    "largest micro batch each path fits")
    ap.add_argument("--micros", default="8,12,16,24,32,48")
    ap.add_argument("--chunks", default="", help="train --fused: also time these chunk sizes (positions a chunk)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--cache", default="")
    ap.add_argument("--split", default="heldin")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="/content/head_shortlist")
    ap.add_argument("--device", default="cuda", help="cpu: a smoke of dump only (the timings need a CUDA card)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    res = {"what": f"order 0050 stage 1 ({a.mode}, torch)", "date": _dt.date.today().isoformat(),
           "card": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           **{"decode": decode, "train": train, "dump": dump}[a.mode](a, dev)}
    path = os.path.join(HERE, "logs", f"head_share_{a.mode}_{dev.type}_{res['date']}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
