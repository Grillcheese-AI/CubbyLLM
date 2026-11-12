"""validation/head_lab.py -- order 0068, route 2: the shortlist with a fallback to the full head.

0050's frozen index (screen plus successors) scores K rows exactly. Where the shortlist is unsure, the full head is
used instead. This reads, with no training of the head and no GPU:
  scan      per position: does the shortlist's best token equal the full head's; the signals a decoder has without
            the full head (the margin between the shortlist's two best scores, its best probability, its entropy,
            the gap between the two nearest context clusters); the probability inside the shortlist; both losses
  tune      on the held-in tuning split only: for each signal, the lowest fallback rate that reaches --target
            overall agreement there; the best one is frozen with its threshold (the logistic score is fitted on the
            even positions and every threshold is chosen on the odd ones)
  read      once, on held-out positions 0050 did not read (--offset): fallback rate, overall agreement with its 95%
            interval, mean rows scored (the index's pass + the shortlist + the full head where it falls back)
  --sweep   larger K, on the held-in tuning split only: what a bigger shortlist would give, with no held-out spent
Rows are rows of the head's width scored per token; a 32k head scores 32,000. Times are not computed here.

    python validation/head_lab.py --export <base export> --dumps <head_shortlist folder> [--k 4096 --target 0.997]
-> validation/logs/head_lab_fallback_<date><tag>.json

Wired: STANDALONE (an experiment; order 0068's report reads it). numpy only.
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

FLOOR = 1e-9
SIGNALS = ("margin", "p1", "logistic")
FEATURES = ("margin", "p1", "entropy", "gap")


def scan(idx, w: np.ndarray, d: dict, k: int, arm: str, chunk: int = 256, log=print) -> dict:
    """Per position, against the full head recomputed from the same states (see the module's docstring)."""
    h_all, tgt, prev = d["hidden"], d["target"].astype(np.int64), d["prev"]
    n = len(tgt)
    out = {m: np.zeros(n) for m in (*FEATURES, "mass", "nll_full", "nll_short", "rows")}
    agree, inside = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    cent = idx.screen.centroids if arm.startswith("screen") and idx.screen is not None else None
    for s in range(0, n, chunk):
        h = h_all[s:s + chunk].astype(np.float32)
        full = h @ w.T
        m = full.max(1)
        lse = m + np.log(np.exp(full - m[:, None]).sum(1))
        top1 = full.argmax(1)
        gap = np.zeros(len(h))
        if cent is not None and len(cent) > 1:
            two = np.partition(h @ cent.T, -2, axis=1)[:, -2:]
            gap = (two[:, 1] - two[:, 0]).astype(np.float64)
        for i, ids in enumerate(idx.shortlist(h, k, arm, prev[s:s + chunk])):
            li = full[i, ids].astype(np.float64)
            j = int(li.argmax())
            l2 = np.partition(li, -2)[-2] if len(li) > 1 else li[j] - 50.0
            ls = li[j] + np.log(np.exp(li - li[j]).sum())
            p = np.exp(li - ls)
            t, g = tgt[s + i], s + i
            pos = int(np.searchsorted(ids, t))
            inside[g] = pos < len(ids) and ids[pos] == t
            agree[g] = ids[j] == top1[i]
            out["margin"][g], out["p1"][g], out["gap"][g] = li[j] - l2, p[j], gap[i]
            out["entropy"][g] = float(-(p * np.log(np.maximum(p, 1e-300))).sum())
            out["mass"][g] = float(np.exp(ls - lse[i]))
            out["nll_full"][g] = float(lse[i] - full[i, t])
            out["nll_short"][g] = float(ls - full[i, t]) if inside[g] else -np.log(FLOOR)
            out["rows"][g] = len(ids)
        if (s // chunk) % 20 == 0:
            log(f"scan {min(s + chunk, n)}/{n}")
    return dict(out, agree=agree, inside=inside)


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 25, l2: float = 1e-3) -> dict:
    """A logistic score for "the shortlist disagrees" (y = 1), by Newton steps on standardized features."""
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-9)
    z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1))], axis=1)
    beta = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -30, 30)))
        grad = z.T @ (p - y) / len(y) + l2 * beta
        hess = (z * (p * (1 - p))[:, None]).T @ z / len(y) + l2 * np.eye(z.shape[1])
        beta -= np.linalg.solve(hess, grad)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist()}


def risk(signal: str, sc: dict, model: dict | None = None) -> np.ndarray:
    """Higher means "more likely to disagree": the fallback takes the positions at or above a threshold."""
    if signal == "margin":
        return -sc["margin"]
    if signal == "p1":
        return -sc["p1"]
    x = np.stack([sc[f] for f in FEATURES], axis=1)
    z = np.concatenate([(x - np.asarray(model["mu"])) / np.asarray(model["sd"]), np.ones((len(x), 1))], axis=1)
    return z @ np.asarray(model["beta"])


def pick_threshold(r: np.ndarray, agree: np.ndarray, target: float) -> tuple[float, float]:
    """(threshold, fallback rate): the fewest riskiest positions to hand to the full head so that the overall
    agreement reaches `target`. No fallback needed: (inf, 0.0)."""
    order = np.argsort(-r, kind="stable")
    caught = np.concatenate([[0], np.cumsum(~agree[order])])
    need = int(caught[-1]) - int(np.floor((1.0 - target) * len(r) + 1e-9))
    if need <= 0:
        return float("inf"), 0.0
    m = int(np.searchsorted(caught, need))
    thr = float(r[order[m - 1]])
    return thr, float((r >= thr).mean())


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(float(mid - half), 5), round(float(mid + half), 5)


def read(sc: dict, r: np.ndarray, thr: float, vocab: int, index_rows: int) -> dict:
    """The lines of one frozen rule on one scan."""
    fb = r >= thr
    n, acc = len(fb), ~fb
    overall = float((sc["agree"] | fb).mean())
    rows = index_rows + sc["rows"] + fb * float(vocab)
    return {"positions": n, "fallback_rate": round(float(fb.mean()), 5),
            "agree_overall": round(overall, 5), "agree_overall_95": wilson(overall, n),
            "agree_shortlist_alone": round(float(sc["agree"].mean()), 5),
            "agree_where_accepted": round(float(sc["agree"][acc].mean()), 5) if acc.any() else None,
            "rows_mean": round(float(rows.mean()), 1), "rows_shortlist_mean": round(float(sc["rows"].mean()), 1),
            "index_rows": index_rows,
            "mass_inside_where_accepted": round(float(sc["mass"][acc].mean()), 5) if acc.any() else None,
            "target_outside_and_accepted": round(float((~sc["inside"] & acc).mean()), 5),
            "nll_full": round(float(sc["nll_full"].mean()), 4),
            "nll_mixed_outside_lost": round(float(np.where(fb, sc["nll_full"], sc["nll_short"]).mean()), 4),
            "nll_mixed_inside_only": round(float(np.where(fb, sc["nll_full"], sc["nll_short"])[sc["inside"] | fb].mean()), 4)}


def _pair(thr: float, rate: float) -> dict:
    return {"threshold": thr if np.isfinite(thr) else None, "fallback_rate": rate}


def tune(sc: dict, target: float, targets=(0.995, 0.997, 0.999)) -> dict:
    """Fit on the even positions, choose on the odd ones; the signal with the lowest fallback rate at `target`."""
    even, odd = np.arange(len(sc["agree"])) % 2 == 0, np.arange(len(sc["agree"])) % 2 == 1
    half = {k: v[odd] for k, v in sc.items()}
    model = fit_logistic(np.stack([sc[f][even] for f in FEATURES], axis=1), (~sc["agree"][even]).astype(np.float64))
    table = {}
    for s in SIGNALS:
        r = risk(s, half, model)
        table[s] = {str(t): _pair(*pick_threshold(r, half["agree"], t)) for t in targets}
        table[s]["at_target"] = _pair(*pick_threshold(r, half["agree"], target))
    best = min(SIGNALS, key=lambda s: (table[s]["at_target"]["fallback_rate"], SIGNALS.index(s)))
    return {"signal": best, "threshold": table[best]["at_target"]["threshold"], "model": model, "table": table,
            "tuned_on": int(odd.sum()), "fitted_on": int(even.sum()), "target": target}


def sweep(idx, w, held: dict, ks: list, a, index_rows: int, log) -> None:
    """Held-in only: per K, the shortlist's agreement alone, and the fallback each signal needs for 0.995 there with
    the mean rows that makes (the index's pass + K + the full head where it falls back)."""
    rep = {"what": "order 0068 route 2: larger shortlists, on the held-in tuning split only", "arm": a.arm,
           "date": _dt.date.today().isoformat(), "positions": len(held["target"]), "index_rows": index_rows, "ks": {}}
    for k in ks:
        sc = scan(idx, w, held, k, a.arm, log=log)
        rep["ks"][str(k)] = {"agree_shortlist_alone": round(float(sc["agree"].mean()), 5),
                             "rows_shortlist_mean": round(float(sc["rows"].mean()), 1),
                             "mass_inside": round(float(sc["mass"].mean()), 5)}
        for t in [float(x) for x in a.sweep_targets.split(",")]:
            table = tune(sc, t)["table"]
            best = min(SIGNALS, key=lambda s: table[s]["at_target"]["fallback_rate"])
            f = table[best]["at_target"]["fallback_rate"]
            rep["ks"][str(k)][f"fallback_for_{t}"] = {"signal": best, "rate": round(f, 5)}
            rep["ks"][str(k)][f"rows_mean_at_{t}"] = round(index_rows + float(sc["rows"].mean()) + f * len(w), 1)
        log(k, json.dumps(rep["ks"][str(k)]))
    path = os.path.join(HERE, "logs", f"head_lab_sweep_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


def cut(d: dict, skip: int, stride: int, offset: int) -> dict:
    return {k: (v if k == "shards" else v[skip:][offset::stride]) for k, v in d.items()}


def main() -> None:
    from cubbyllm.model.vocab.shortlist import ShortlistIndex
    from head_shortlist_eval import head_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="the base export (its head's rows)")
    ap.add_argument("--dumps", required=True, help="head_dump.py's folder, with 0050's frozen index")
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--arm", default="screen+succ")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--target", type=float, default=0.997, help="the agreement asked for on the held-in tuning split")
    ap.add_argument("--tune", default="heldin_val")
    ap.add_argument("--tune-stride", type=int, default=2)
    ap.add_argument("--read", default="heldout")
    ap.add_argument("--skip", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--offset", type=int, default=1, help="0 is the set 0050 read; 1 is this order's")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sweep", default="", help="K values, comma separated: read on the held-in tuning split only "
                    "(no held-out position is touched), then stop")
    ap.add_argument("--sweep-targets", default="0.995", help="the agreements asked for there, comma separated")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    w = head_rows(a.export)
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
    index_rows = len(idx.screen.centroids) if a.arm.startswith("screen") else idx.n
    held = cut(dict(np.load(os.path.join(a.dumps, f"{a.tune}.npz"))), 0, a.tune_stride, 0)
    log(f"tune: {len(held['target'])} held-in positions")
    if a.sweep:
        return sweep(idx, w, held, [int(x) for x in a.sweep.split(",")], a, index_rows, log)
    frozen = tune(scan(idx, w, held, a.k, a.arm, log=log), a.target)
    log("frozen:", frozen["signal"], frozen["threshold"], json.dumps(frozen["table"]))
    out = cut(dict(np.load(os.path.join(a.dumps, f"{a.read}.npz"))), a.skip, a.stride, a.offset)
    log(f"read: {len(out['target'])} held-out positions (skip {a.skip}, every {a.stride}th, offset {a.offset})")
    sc = scan(idx, w, out, a.k, a.arm, log=log)
    thr = float("inf") if frozen["threshold"] is None else frozen["threshold"]
    res = read(sc, risk(frozen["signal"], sc, frozen["model"]), thr, len(w), index_rows)
    rep = {"what": "order 0068 route 2: the shortlist with a fallback to the full head", "date": _dt.date.today().isoformat(),
           "arm": a.arm, "k": a.k, "index": a.index, "tune": {"dump": a.tune, "stride": a.tune_stride, **frozen},
           "read": {"dump": a.read, "skip": a.skip, "stride": a.stride, "offset": a.offset, **res},
           "lines": {"agree_overall >= 0.995": res["agree_overall"] >= 0.995, "rows_mean <= 32000": res["rows_mean"] <= 32000,
                     "nll within +0.02 (0050's)": res["nll_mixed_outside_lost"] <= res["nll_full"] + 0.02}}
    path = os.path.join(HERE, "logs", f"head_lab_fallback_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    log(json.dumps(rep["read"], indent=1))
    log(json.dumps(rep["lines"]))
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
