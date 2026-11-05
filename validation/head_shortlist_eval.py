"""validation/head_shortlist_eval.py -- order 0050, stage 1: the shortlist index's quality on the 13 held-out tails.

Reads ``head_dump.py``'s states (held-out for the read, held-in for the router and the successors) and the trained
head's rows from the base export, builds ``cubbyllm.model.vocab.shortlist.ShortlistIndex`` (k-means, --clusters, seed
0), trains arm (b)'s router (a linear map from the state to cluster scores, from the centroids, against the full
head's top-16 mass per cluster, held-in only), counts arm (c)'s successors (held-in windows), and reads, per arm and
K, against the full head recomputed from the same states:
  top1_in     the full head's top-1 inside the shortlist
  top10       its top-10 recall
  mass        the shortlist's share of the full head's probability
  agree       the shortlist's argmax equals the full head's (greedy agreement; the gate: >= 99.5%)
  nll_lost    held-out NLL with the mass outside the shortlist counted as lost: p(target) = its full-head probability
              when inside, else FLOOR (a target outside the shortlist is one the shortlist cannot say)
  nll_renorm  NLL of the shortlist's own softmax over the targets inside it; miss: the targets outside it
  rows        the rows scored (K plus at most one cluster, plus (c)'s successors)
The full head's NLL beside them (the gate: within +0.02 nats).

    python validation/head_shortlist_eval.py [--clusters 2048 --ks 256,1024,4096]
-> validation/logs/head_shortlist_<date>.json; the index -> $CUBBY_DATA_ROOT/head_shortlist/index_<arm>.npz

Wired: STANDALONE (an experiment; 0050's report reads it).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"

FLOOR = 1e-9


def head_rows(export: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(os.path.join(export, "model.safetensors"), framework="numpy") as f:
        return np.ascontiguousarray(f.get_tensor("lm_head.weight"), dtype=np.float32)


def train_router(idx, held: dict, epochs: int = 3, lr: float = 1e-3, seed: int = 0, most: int = 150_000) -> tuple:
    """(W, b): cluster scores = h W^T + b, from the centroids, fitted to the full head's top-16 mass per cluster, on at
    most `most` held-in positions (drawn by `seed`; the dense targets are positions x clusters)."""
    import torch
    torch.manual_seed(seed)
    sel = np.sort(np.random.default_rng(seed).choice(len(held["target"]), min(most, len(held["target"])), replace=False))
    h = torch.from_numpy(held["hidden"][sel].astype(np.float32))
    p = np.exp(held["toplogit"][sel] - held["lse"][sel][:, None])
    tgt = np.zeros((len(h), idx.n), dtype=np.float32)
    np.add.at(tgt, (np.repeat(np.arange(len(h)), p.shape[1]), idx.assign[held["top"][sel]].reshape(-1)), p.reshape(-1))
    tgt /= np.maximum(tgt.sum(1, keepdims=True), 1e-12)
    tgt = torch.from_numpy(tgt)
    w = torch.nn.Parameter(torch.from_numpy(idx.centroids.copy()))
    b = torch.nn.Parameter(torch.zeros(idx.n))
    opt = torch.optim.Adam([w, b], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        tot = 0.0
        for i in torch.randperm(len(h), generator=gen).split(1024):
            loss = -(tgt[i] * torch.log_softmax(h[i] @ w.t() + b, -1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(i)
        print(f"router epoch {ep}: {tot / len(h):.4f}", flush=True)
    return w.detach().numpy(), b.detach().numpy()


def read(idx, w: np.ndarray, out: dict, arms: list, ks: list, chunk: int = 256) -> dict:
    """Every (arm, K) line against the full head recomputed from the same states."""
    h_all, tgt, prev = out["hidden"], out["target"].astype(np.int64), out["prev"]
    acc = {(a, k): {m: 0.0 for m in ("top1_in", "top10", "mass", "agree", "nll_lost", "nll_renorm", "miss", "rows")}
           for a in arms for k in ks}
    full_nll, n = 0.0, len(tgt)
    for s in range(0, n, chunk):
        h = h_all[s:s + chunk].astype(np.float32)
        L = h @ w.T                                                     # (c, V) the full head
        m = L.max(1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(L - m).sum(1)))
        top10 = np.argpartition(-L, 10, axis=1)[:, :10]
        top1 = L.argmax(1)
        lt = L[np.arange(len(h)), tgt[s:s + chunk]]
        full_nll += float((lse - lt).sum())
        for a in arms:
            for k in ks:
                lists = idx.shortlist(h, k, a, prev[s:s + chunk])
                r = acc[(a, k)]
                for i, ids in enumerate(lists):
                    li = L[i, ids]
                    inside = np.isin(top10[i], ids).sum()
                    t = tgt[s + i]
                    hit = np.searchsorted(ids, t)
                    hit = hit < len(ids) and ids[hit] == t
                    r["top1_in"] += float(np.isin(top1[i], ids))
                    r["top10"] += inside / 10.0
                    r["mass"] += float(np.exp(li - lse[i]).sum())
                    r["agree"] += float(ids[int(li.argmax())] == top1[i])
                    r["nll_lost"] += float(lse[i] - L[i, t]) if hit else -np.log(FLOOR)
                    if hit:
                        mi = li.max()
                        r["nll_renorm"] += float(mi + np.log(np.exp(li - mi).sum()) - L[i, t])
                    else:
                        r["miss"] += 1
                    r["rows"] += len(ids)
        if (s // chunk) % 40 == 0:
            print(f"read {s + len(h)}/{n}", flush=True)
    res = {"full_nll": round(full_nll / n, 4), "positions": n, "lines": {}}
    for (a, k), r in acc.items():
        hits = n - r["miss"]
        res["lines"][f"{a}|{k}"] = {"arm": a, "k": k, "top1_in": round(r["top1_in"] / n, 5), "top10": round(r["top10"] / n, 5),
                                    "mass": round(r["mass"] / n, 5), "agree": round(r["agree"] / n, 5),
                                    "nll_lost": round(r["nll_lost"] / n, 4), "miss": round(r["miss"] / n, 5),
                                    "nll_renorm_hits": round(r["nll_renorm"] / max(1, hits), 4),
                                    "rows_mean": round(r["rows"] / n, 1)}
    return res


def held_tokens(cache: str, per: int = 2_000_000) -> np.ndarray:
    """Held-in text for the successor counts: `per` tokens from the middle of each shard's training region."""
    from head_dump import SHARDS
    out = []
    for name in SHARDS:
        arr = np.memmap(os.path.join(cache, f"{name}.u32"), dtype=np.uint32, mode="r")
        hold = int(json.load(open(os.path.join(cache, "_holdout.json"), encoding="utf-8"))[name])
        mid = (len(arr) - hold) // 2
        out.append(np.array(arr[mid:mid + per], dtype=np.int64))
    return np.concatenate(out)


def main() -> None:
    from cubbyllm.local_paths import expand
    from cubbyllm.model.vocab.shortlist import ARMS, ScreenIndex, ShortlistIndex, successors
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="$CUBBY_MODELS/base450m_cont/grilly")
    ap.add_argument("--cache", default="$CUBBY_TRAINING_DATA/token_cache_cont")
    ap.add_argument("--dumps", default="$CUBBY_DATA_ROOT/head_shortlist")
    ap.add_argument("--clusters", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--screen", type=int, default=1024, help="arm (d): context clusters over the held-in states")
    ap.add_argument("--ks", default="256,1024,4096")
    ap.add_argument("--positions", type=int, default=0, help="read only the first N positions (0: all)")
    ap.add_argument("--build", default="heldin", help="the dump the router and the screen are built from")
    ap.add_argument("--read", default="heldout", help="the dump read: heldin_val to tune, heldout once")
    ap.add_argument("--stride", type=int, default=1, help="read every n-th position (evenly across the shards)")
    ap.add_argument("--skip", type=int, default=0, help="leave out the first n positions (0050: the 2,048 held-out "
                    "positions seen while the build was shaped)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dumps = expand(a.dumps)
    w = head_rows(expand(a.export))
    held, out = dict(np.load(os.path.join(dumps, f"{a.build}.npz"))), dict(np.load(os.path.join(dumps, f"{a.read}.npz")))
    if a.positions or a.stride > 1 or a.skip:
        out = {k: (v[a.skip:][::a.stride][:a.positions or None] if k != "shards" else v) for k, v in out.items()}
    t0 = time.time()
    idx = ShortlistIndex.build(w, a.clusters, seed=0, iters=a.iters)
    built = round(time.time() - t0, 1)
    print(f"index: {a.clusters} clusters in {built} s", flush=True)
    idx.save(os.path.join(dumps, "index_clusters.npz"))
    idx.router = train_router(idx, held)
    idx.succ = successors(held_tokens(expand(a.cache)), len(w), top=32)
    idx.screen = ScreenIndex.build(held["hidden"].astype(np.float32), held["top"],
                                   np.exp(held["toplogit"] - held["lse"][:, None]), len(w), a.screen, seed=0, iters=a.iters)
    idx.save(os.path.join(dumps, "index_router_succ.npz"))
    res = read(idx, w, out, list(ARMS), [int(x) for x in a.ks.split(",")])
    rep = {"what": "order 0050 stage 1: the shortlist index's quality on the 13 held-out tails", "read": a.read, "stride": a.stride, "skip": a.skip,
           "build": a.build,
           "date": _dt.date.today().isoformat(), "clusters": a.clusters, "screen_clusters": a.screen, "iters": a.iters, "index_build_s": built,
           "cluster_sizes": {"min": int(np.diff(idx.offsets).min()), "p50": float(np.median(np.diff(idx.offsets))),
                             "max": int(np.diff(idx.offsets).max())},
           "held_in_positions": len(held["target"]), "gate": "agree >= 0.995; nll within +0.02 of full", **res}
    path = os.path.join(HERE, "logs", f"head_shortlist_{a.read}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "lines"}, indent=1))
    for k, v in rep["lines"].items():
        print(k, v)
    print("->", path)


if __name__ == "__main__":
    main()
