"""k10_policies -- order 0021: on a saved joint model, what the switch could do with the read as it is.

At each asking position the copy path has its pick and the read's best-match cosine (the gate's own `best`); the
trunk has its own pick. Three host-side policies, untrained, over the held-out set, at a range of thresholds on the
best match:
- **the gate:** the model's own mix (exp_k10_free_text's exact, for reference);
- **switch:** the copy's pick where best >= thr, the trunk's below -- per piece;
- **copy or refuse:** the copy's whole answer when every piece clears thr, NONE otherwise -- the trunk never answers
  a fact (alone it gets none right: 0006's LM-alone 0.000).

Reported per policy: exact, present answered wrong, present refused, absent answered. The question it answers: is the
missing 0.93 / 3% / 5% a matter of training the gate, or of what the gate hands the answer to when it closes?

    python validation/k10_policies.py --resume <joint model> [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_k10_free_text as ft  # noqa: E402
import exp_k10_switch as sw  # noqa: E402
from exp_k10_far_facts import fwd  # noqa: E402

THRESHOLDS = (0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20)


@torch.no_grad()
def reads(model, a, held) -> list:
    """Per held-out question: (info, answer pieces, [(best, copy's pick, trunk's pick) per piece])."""
    ids, key, plc, R, _g, spans = held
    calls, rows = [], []

    def hk(_m, inp, o):
        r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
        e, rn = F.normalize(table.float()[cid], dim=-1), F.normalize(r.float(), dim=-1)
        s = (rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)).masked_fill(~ok, float("-inf"))
        calls.append((idx, s.max(-1).values, o[1].argmax(-1), inp[0].argmax(-1)))
    hook = model.spatial.copy_path.register_forward_hook(hk)
    try:
        for b in range(0, ids.shape[0], 16):
            sl, S = slice(b, b + 16), ids.shape[1] - 1
            calls.clear()
            fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
            got = {divmod(j, S): (v, c, m) for ix, best, cp, lm in calls
                   for j, v, c, m in zip(ix.tolist(), best.tolist(), cp.tolist(), lm.tolist())}
            for i in range(ids[sl].shape[0]):
                here, o, info = spans[b + i]
                rows.append((info, list(o), [got.get((i, here + j)) for j in range(len(o))]))
    finally:
        hook.remove()
    return rows


def score(rows, none, say) -> dict:
    """`say(info, pieces, reads) -> the answer's tokens`: exact / present wrong / refused / absent answered."""
    p = [(say(i, o, g), o) for i, o, g in rows if not i["absent"]]
    ab = [say(i, o, g)[0] != none for i, o, g in rows if i["absent"]]
    exact = np.mean([s == o for s, o in p]); refused = np.mean([s[0] == none for s, _o in p])
    return {"exact": round(float(exact), 4), "present_wrong": round(float(1 - exact - refused), 4),
            "present_refused": round(float(refused), 4), "absent_answered": round(float(np.mean(ab)), 4)}


def policies(rows, none) -> dict:
    out = {}
    for thr in THRESHOLDS:
        pick = lambda x: (x[1] if x[0] >= thr else x[2]) if x is not None else none          # noqa: E731
        out[f"switch@{thr:.2f}"] = score(rows, none, lambda i, o, g: [pick(x) for x in g])
        clear = lambda i, g: all(x is not None and x[0] >= thr for x in (g[:1] if i["absent"] else g))   # noqa: E731
        out[f"copy_or_refuse@{thr:.2f}"] = score(rows, none, lambda i, o, g: [x[1] for x in g] if clear(i, g) else [none])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    c = ap.parse_args()
    if c.threads:
        torch.set_num_threads(c.threads)
    model, a, lay, held = sw.load(c.resume)
    rep = {"arm": a.arm, "seed": a.seed, "gate": {k: ft.evaluate(model, a, held, lay)[k] for k in
                                                  ("exact", "present_wrong", "present_refused", "absent_not_none")},
           "policies": policies(reads(model, a, held), lay.none)}
    out = os.path.join(HERE_DIR, "logs", f"k10_policies_{a.arm}_s{a.seed}_{c.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep["gate"]), "\n" + "\n".join(f"  {k:22s} {v}" for k, v in rep["policies"].items()))


if __name__ == "__main__":
    main()
