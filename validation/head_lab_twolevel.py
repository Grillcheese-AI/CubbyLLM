"""validation/head_lab_twolevel.py -- order 0068, route 3: a two-level head on the frozen trunk's states.

The first pass scores a frequent set of tokens (always) and one entry per cluster of the other tokens; a cluster's own
rows are scored only when its entry beats the best token so far. The entries are the only thing trained: a row and a
bias per cluster, fitted on held-in states so that the first pass gives each cluster the probability the full head
gives its tokens together. The trunk and every head row stay as trained.
  layout    the frequent set (the tokens the full head gives the most probability on held-in) and the clusters of
            the rest (spherical k-means over their rows)
  prep      per state, from one pass of the full head: each cluster's true log-sum-exp, the frequent set's, its best
  train     the entries, by the exact KL between the full head's first-pass distribution and the entries'
  evaluate  two readings of the same entries, against the full head on the same states:
            two-level  the head as its own distribution: its loss minus the full head's, its greedy token against
                       the full head's, the rows it scores
            router     the entries only decide which clusters to open (an entry within --margins of the best logit
                       so far); the token is chosen by the true logits of the rows scored: agreement, rows, and the
                       probability inside them
The dump keeps the full head's 16 best tokens per state; a greedy token is looked for among them, and a state where a
token outside them could win is counted as a disagreement.

    python validation/head_lab_twolevel.py --export <base export> --dumps <head_shortlist folder> [--f 16384 --c 4096]
-> validation/logs/head_lab_twolevel_<date><tag>.json

Wired: STANDALONE (an experiment; order 0068's report reads it). numpy, and torch on the CPU for the fit.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

__wiring__ = "STANDALONE"

TOPM = 64                                               # clusters looked at per state, best entries first


def lse(x: np.ndarray) -> np.ndarray:
    m = x.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(x - m).sum(1))


class Layout:
    """The frequent set (always scored) and the clusters of the other tokens, as one permutation of the rows."""

    def __init__(self, head: np.ndarray, tail: np.ndarray, starts: np.ndarray, vocab: int) -> None:
        self.head, self.tail, self.starts, self.vocab = np.asarray(head), np.asarray(tail), np.asarray(starts), vocab
        self.f, self.c, self.sizes = len(self.head), len(self.starts) - 1, np.diff(self.starts)
        self.cluster_of = np.full(vocab, -1, dtype=np.int64)
        self.cluster_of[self.tail] = np.repeat(np.arange(self.c), self.sizes)
        self.perm = np.concatenate([self.head, self.tail])


def build_layout(w: np.ndarray, top: np.ndarray, prob: np.ndarray, f: int, c: int, seed: int = 0, iters: int = 10) -> Layout:
    from cubbyllm.model.vocab.shortlist import kmeans
    mass = np.bincount(top.reshape(-1), weights=prob.reshape(-1), minlength=len(w))
    head = np.sort(np.argsort(-mass, kind="stable")[:f])
    tail = np.setdiff1d(np.arange(len(w)), head)
    _cent, assign = kmeans(w[tail], c, seed, iters)
    starts = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=c))])
    return Layout(head, tail[np.argsort(assign, kind="stable")], starts, len(w))


def prep(wp: np.ndarray, lay: Layout, hidden: np.ndarray, chunk: int = 256, log=print) -> dict:
    """One pass of the full head (rows in the layout's order): t (n, C), lse_h (n,), max_h (n,)."""
    n = len(hidden)
    t, lse_h, max_h = np.empty((n, lay.c), np.float32), np.empty(n, np.float32), np.empty(n, np.float32)
    for s in range(0, n, chunk):
        full = hidden[s:s + chunk].astype(np.float32) @ wp.T
        zh, zt = full[:, :lay.f], full[:, lay.f:]
        max_h[s:s + chunk], lse_h[s:s + chunk] = zh.max(1), lse(zh)
        m = zt.max(1, keepdims=True)
        t[s:s + chunk] = m + np.log(np.maximum(np.add.reduceat(np.exp(zt - m), lay.starts[:-1], axis=1), 1e-37))
        if (s // chunk) % 40 == 0:
            log(f"prep {min(s + chunk, n)}/{n}")
    return {"t": t, "lse_h": lse_h, "max_h": max_h}


def init_entries(wp: np.ndarray, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """A cluster's mean row, and the log of its size: exact when its rows are equal."""
    e = np.add.reduceat(wp[lay.f:].astype(np.float64), lay.starts[:-1], axis=0) / lay.sizes[:, None]
    return e.astype(np.float32), np.log(lay.sizes).astype(np.float32)


def train(hidden: np.ndarray, pre: dict, e0: np.ndarray, b0: np.ndarray, epochs: int = 30, lr: float = 2e-3,
          batch: int = 4096, seed: int = 0, threads: int = 6, log=print) -> tuple[np.ndarray, np.ndarray]:
    """The entries by the exact KL(full head's first pass || entries' first pass). With lse_h the frequent set's
    log-sum-exp and t the clusters' true ones: KL = LSE1 - LSEfull - sum_c q_c (z_c - t_c), q_c = exp(t_c - LSEfull)."""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    h, t, lh = (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) for x in (hidden, pre["t"], pre["lse_h"]))
    e, b = torch.nn.Parameter(torch.from_numpy(e0.copy())), torch.nn.Parameter(torch.from_numpy(b0.copy()))
    opt = torch.optim.Adam([e, b], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        tot = 0.0
        for i in torch.randperm(len(h), generator=gen).split(batch):
            zc, ti, li = h[i] @ e.t() + b, t[i], lh[i]
            full = torch.logaddexp(li, torch.logsumexp(ti, 1))
            kl = (torch.logaddexp(li, torch.logsumexp(zc, 1)) - full - (torch.exp(ti - full[:, None]) * (zc - ti)).sum(1)).mean()
            opt.zero_grad()
            kl.backward()
            opt.step()
            tot += float(kl) * len(i)
        log(f"epoch {ep}: KL {tot / len(h):.5f}")
    return e.detach().numpy(), b.detach().numpy()


def first_pass_kl(hidden: np.ndarray, pre: dict, e: np.ndarray, b: np.ndarray) -> float:
    """The mean KL the fit minimizes, in numpy (the tests and the log read it)."""
    t, lh = pre["t"].astype(np.float64), pre["lse_h"].astype(np.float64)
    zc = hidden.astype(np.float64) @ e.T.astype(np.float64) + b
    full = np.logaddexp(lh, lse(t))
    return float((np.logaddexp(lh, lse(zc)) - full - (np.exp(t - full[:, None]) * (zc - t)).sum(1)).mean())


def evaluate(d: dict, pre: dict, zc: np.ndarray, lay: Layout, margins=(0.0, 1.0, 2.0, 4.0)) -> dict:
    """Both readings of the entries' scores `zc` (n, C) against the full head on the same states."""
    t, lh, mh, zc = (np.asarray(x, dtype=np.float64) for x in (pre["t"], pre["lse_h"], pre["max_h"], zc))
    n, idx = len(lh), np.arange(len(lh))
    full, first = np.logaddexp(lh, lse(t)), np.logaddexp(lh, lse(zc))
    delta = zc - t                                                      # each entry's error on each state
    ct = lay.cluster_of[d["target"].astype(np.int64)]
    gap = first - full
    gap[ct >= 0] -= delta[idx[ct >= 0], ct[ct >= 0]]                   # the two-level loss minus the full head's
    top, tl = d["top"].astype(np.int64), d["toplogit"].astype(np.float64)
    ctop = lay.cluster_of[top]
    shift = np.where(ctop >= 0, np.take_along_axis(delta, np.maximum(ctop, 0), axis=1), 0.0)
    fpos, gpos = tl.argmax(1), (tl + shift).argmax(1)
    outside = np.maximum(np.minimum(tl.min(1)[:, None] + delta, zc).max(1), tl.min(1))   # the most a token outside
    unsafe = (tl + shift).max(1) < outside                             # the dump's 16 could score: it could win
    cf = ctop[idx, fpos]                                                # the full head's best token's cluster, or -1
    m_ = min(TOPM, lay.c)
    cand = np.argpartition(-zc, m_ - 1, axis=1)[:, :m_]
    cand = np.take_along_axis(cand, np.argsort(-np.take_along_axis(zc, cand, axis=1), axis=1), axis=1)
    rows2, capped = np.zeros(n), 0
    r = {m: {"agree": 0, "rows": 0.0, "mass": 0.0} for m in margins}
    for i in range(n):
        true_best, shifted_best = {}, {}
        for j in range(top.shape[1]):
            c = int(ctop[i, j])
            if c >= 0:
                true_best[c] = max(true_best.get(c, -np.inf), tl[i, j])
                shifted_best[c] = max(shifted_best.get(c, -np.inf), tl[i, j] + shift[i, j])
        best = mh[i]
        for k, c in enumerate(cand[i]):                                 # the two-level rule
            if zc[i, c] <= best:
                break
            rows2[i] += lay.sizes[c]
            best = max(best, shifted_best.get(int(c), -np.inf))
            capped += k == m_ - 1
        for m in margins:                                               # the router rule
            best, opened = mh[i], []
            for c in cand[i]:
                if zc[i, c] <= best - m:
                    break
                opened.append(int(c))
                best = max(best, true_best.get(int(c), -np.inf))
            r[m]["agree"] += cf[i] < 0 or int(cf[i]) in opened
            r[m]["rows"] += float(lay.sizes[np.asarray(opened, dtype=np.int64)].sum())
            inside = np.logaddexp(lh[i], lse(t[i:i + 1, opened])[0]) if opened else lh[i]
            r[m]["mass"] += float(np.exp(inside - full[i]))
    q_tail = 1.0 - np.exp(lh - full)
    w_err = (np.exp(t - full[:, None]) * np.abs(delta)).sum(1) / np.maximum(q_tail, 1e-12)
    base = lay.f + lay.c
    return {"positions": n, "rows_first_pass": base, "tail_mass": round(float(q_tail.mean()), 5),
            "best_in_frequent_set": round(float((cf < 0).mean()), 5),
            "entry_error_weighted_nats": round(float(w_err.mean()), 4),
            "lse_check_vs_dump": round(float(np.abs(full - d["lse"]).mean()), 5),
            "two_level": {"nll_full": round(float((full - d["tlogit"]).mean()), 4), "nll_gap": round(float(gap.mean()), 5),
                          "agree": round(float(((gpos == fpos) & ~unsafe).mean()), 5), "unsafe": int(unsafe.sum()),
                          "rows_mean": round(base + float(rows2.mean()), 1), "capped": int(capped)},
            "router": {str(m): {"agree": round(v["agree"] / n, 5), "rows_mean": round(base + v["rows"] / n, 1),
                                "mass_inside": round(v["mass"] / n, 5)} for m, v in r.items()}}


def cut(d: dict, skip: int, stride: int, offset: int) -> dict:
    return {k: (v if k == "shards" else v[skip:][offset::stride]) for k, v in d.items()}


def main() -> None:
    from head_shortlist_eval import head_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--build", default="heldin_w64")
    ap.add_argument("--f", type=int, default=16384, help="the frequent set's size")
    ap.add_argument("--c", type=int, default=4096, help="clusters of the other tokens")
    ap.add_argument("--train-n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--margins", default="0,1,2,4")
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--read", default="", help="heldout: the read, once (order 0068: offset 3)")
    ap.add_argument("--skip", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    margins = tuple(float(x) for x in a.margins.split(","))
    w = head_rows(a.export)
    held = np.load(os.path.join(a.dumps, f"{a.build}.npz"))
    top, prob = held["top"], np.exp(held["toplogit"] - held["lse"][:, None])
    lay = build_layout(w, top, prob, a.f, a.c)
    log(f"layout: {lay.f} frequent, {lay.c} clusters of {len(lay.tail)} (sizes min {lay.sizes.min()}, median "
        f"{int(np.median(lay.sizes))}, max {lay.sizes.max()})")
    wp = np.ascontiguousarray(w[lay.perm])
    sel = np.linspace(0, len(top) - 1, a.train_n).astype(np.int64)
    hid = held["hidden"][sel]
    pre = prep(wp, lay, hid, log=log)
    e0, b0 = init_entries(wp, lay)
    e, b = train(hid.astype(np.float32), pre, e0, b0, a.epochs, a.lr, log=log)
    rep = {"what": "order 0068 route 3: a two-level head on the frozen trunk's states", "date": _dt.date.today().isoformat(),
           "f": a.f, "c": a.c, "train_positions": int(a.train_n), "epochs": a.epochs, "lr": a.lr,
           "cluster_sizes": {"min": int(lay.sizes.min()), "median": float(np.median(lay.sizes)), "max": int(lay.sizes.max())}}
    for name, d in (("held_in", cut(dict(np.load(os.path.join(a.dumps, "heldin_val.npz"))), 0, a.val_stride, 0)),
                    ("held_out", cut(dict(np.load(os.path.join(a.dumps, f"{a.read}.npz"))), a.skip, a.stride, a.offset)
                     if a.read else None)):
        if d is None:
            continue
        p = prep(wp, lay, d["hidden"], log=log)
        rep[name] = evaluate(d, p, d["hidden"].astype(np.float32) @ e.T + b, lay, margins)
        rep[name + "_at_init"] = evaluate(d, p, d["hidden"].astype(np.float32) @ e0.T + b0, lay, margins)["two_level"]
        log(name, json.dumps(rep[name]))
    path = os.path.join(HERE, "logs", f"head_lab_twolevel_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
