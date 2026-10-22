"""k10_refuse -- order 0022 A: copy or refuse, in the model, on 0021's saved joint models (the toy, CPU).

1. **Calibration documents** (seed + 199, disjoint from the held-out ones at seed + 99, same cell mix): the reads ahead
   at each answer's first asking position (what ``spatial_refuse.decide`` sees) and the copy's pick at each piece.
2. **Fit** (theta, theta_end) on them alone: among the pairs on a grid of the observed best matches, those meeting both
   wrong-answer lines (present answered wrong <= 2%, absent answered <= 5%); the fewest present refused, then the
   lowest theta.
3. **Measure once** on the held-out documents, through the model with ``copy_refuse`` set: the answer is what the model
   says -- its argmax from HERE until EOS or NONE -- exact, present wrong, present refused, absent answered; and the
   same from the captured reads (the simulation), which must agree with the model.

    python validation/k10_refuse.py --resume validation/logs/_k10_ckpt/P_s0.pt [--threads 2]
-> validation/logs/k10_refuse_<arm>_s<seed>_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_k10_free_text as ft  # noqa: E402
import exp_k10_hypertoken as kh  # noqa: E402
import exp_k10_switch as sw  # noqa: E402
from cubbyllm.model import spatial_refuse as sr  # noqa: E402
from exp_k10_far_facts import fwd  # noqa: E402
from k10_free_docs import batch, held_cells  # noqa: E402

LINES = {"present_wrong": 0.02, "absent_answered": 0.05}
STEP = (0, 0, kh.STEP)                     # the toy's hashed places: piece i 6 i cells up z
K = 8                                      # reads ahead: the toy's answers have at most 5 pieces


def documents(a, lay, seed_offset: int):
    hr = np.random.default_rng(a.seed + seed_offset)
    return batch(hr, kh.Language(np.random.default_rng(a.seed + 1)), lay, a.held, a, held_cells(hr, a.held, a.absent))


@torch.no_grad()
def capture(model, a, docs, lay) -> list:
    """Per document: (info, answer, reads ahead (K,), the copy's picks at here + j)."""
    ids, key, plc, R, _g, spans = docs
    cp, got, picks = model.spatial.copy_path, [], []
    real = sr.decide

    def keep(b, t, e):
        got.append(b.clone())
        return real(b, t, e)

    def hk(_m, inp, o):
        picks.append((inp[4], o[1].argmax(-1)))
    cp.set_refuse(0.0, 0.0, lay.none, STEP, K)
    sr.decide, hook, rows = keep, cp.register_forward_hook(hk), []
    try:
        for b in range(0, ids.shape[0], 16):
            sl, S = slice(b, b + 16), ids.shape[1] - 1
            got.clear(), picks.clear()
            fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
            bests = torch.cat(got)
            pick = {divmod(j, S): c for ix, cs in picks for j, c in zip(ix.tolist(), cs.tolist())}
            n = ids[sl].shape[0]
            assert bests.shape[0] == n, f"one answer run per document expected, got {bests.shape[0]} for {n}"
            for i in range(n):
                here, o, info = spans[b + i]
                rows.append((info, list(o), bests[i].numpy(), [pick.get((i, here + j)) for j in range(K)]))
    finally:
        sr.decide = real
        hook.remove()
        cp.refuse = None
    return rows


def simulate(rows, theta: float, theta_end: float) -> dict:
    B = torch.tensor(np.stack([r[2] for r in rows]))
    L = sr.decide(B, theta, theta_end).tolist()
    pres = [(l, r) for l, r in zip(L, rows) if not r[0]["absent"]]
    exact = [l == len(r[1]) and r[3][:l] == r[1] for l, r in pres]
    refused = [l == 0 for l, _r in pres]
    absent = [l > 0 for l, r in zip(L, rows) if r[0]["absent"]]
    return {"exact": float(np.mean(exact)), "present_refused": float(np.mean(refused)),
            "present_wrong": float(1 - np.mean(exact) - np.mean(refused)), "absent_answered": float(np.mean(absent)),
            "n_present": len(pres), "n_absent": len(absent)}


def fit(rows) -> dict:
    """The pair meeting both lines on these rows with the fewest refused, then the lowest theta."""
    vals = np.unique(np.round(np.concatenate([r[2] for r in rows]), 3))
    grid = vals[(vals > -1) & (vals < 1)][::max(1, len(vals) // 120)]
    best = None
    for te in grid:
        for th in grid[grid >= te]:
            s = simulate(rows, float(th), float(te))
            if s["present_wrong"] <= LINES["present_wrong"] and s["absent_answered"] <= LINES["absent_answered"]:
                k = (s["present_refused"], float(th))
                if best is None or k < best[0]:
                    best = (k, float(th), float(te), s)
    if best is None:
        return {"theta": None, "theta_end": None, "calibration": None}
    return {"theta": best[1], "theta_end": best[2], "calibration": best[3]}


@torch.no_grad()
def spoken(model, a, docs, lay, theta, theta_end) -> dict:
    """The model's own answers under the contract: argmax from HERE until EOS / NONE (at most K tokens)."""
    ids, key, plc, R, _g, spans = docs
    model.spatial.copy_path.set_refuse(theta, theta_end, lay.none, STEP, K)
    ex, ref, ab = [], [], []
    try:
        for b in range(0, ids.shape[0], 16):
            sl = slice(b, b + 16)
            pred = fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1]).argmax(-1)
            for i in range(pred.shape[0]):
                here, o, info = spans[b + i]
                said = []
                for t in pred[i, here:here + K].tolist():
                    if t in (kh.EOS, lay.none):
                        said = said or [t]
                        break
                    said.append(t)
                if info["absent"]:
                    ab.append(said[:1] != [lay.none])
                else:
                    ex.append(said == list(o))
                    ref.append(said[:1] == [lay.none])
    finally:
        model.spatial.copy_path.refuse = None
    return {"exact": float(np.mean(ex)), "present_refused": float(np.mean(ref)),
            "present_wrong": float(1 - np.mean(ex) - np.mean(ref)), "absent_answered": float(np.mean(ab))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    c = ap.parse_args()
    torch.set_num_threads(c.threads)
    model, a, lay, held = sw.load(c.resume)
    cal = documents(a, lay, 199)
    t0 = time.time()
    f = fit(capture(model, a, cal, lay))
    rep = {"arm": a.arm, "seed": a.seed, "lines": LINES, "lookahead": K, "calibration_docs": a.held, **f}
    if f["theta"] is not None:
        rep["test_simulated"] = simulate(capture(model, a, held, lay), f["theta"], f["theta_end"])
        rep["test_model"] = spoken(model, a, held, lay, f["theta"], f["theta_end"])
    rep["wall_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE_DIR, "logs", f"k10_refuse_{a.arm}_s{a.seed}_{c.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
