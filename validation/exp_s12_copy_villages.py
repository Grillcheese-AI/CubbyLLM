"""exp_s12_copy_villages -- order 0035 C's measure: a checkpoint's trunk and copy on 0034's 60 villages (the cut view,
the fixed placed questions: neighbour, offset, at), in two shapes:
- **0034's** (``make_spatial_shard.render``): ``here`` asks; the answer follows "Answer:" with its space (" coal");
- **the far facts'** (``render(lead=True)``): a name's piece i at its cell + 6 i z; the last token of "Answer: " asks
  for piece 0; the answer in the token form it was written in; each answer piece asks for the next.

Per answer class (``kind``: a thing; ``name``: a villager; ``nothing``): n; the first token right and the whole answer
right (teacher-forced, as ``exp_s12_scene_files`` scores them); the copy's lambda at the token that predicts the first
piece (when it asks there); and where a missed first token went: another piece of the same name, the right word in
another token form, a neighbour's piece (a thing within two steps of the cell), "nothing", a farther thing's piece,
or another token.

    CUBBY_SCENES=<0034's folder> CB_CKPT=<checkpoint> python validation/exp_s12_copy_villages.py --tag <run> [--limit N]
-> validation/logs/exp_s12_copy_villages_<tag>_<date>.json (counts only: no name, no cell, no kind word); every
   answer, with its village, to $CUBBY_SCENES/exp_s12_copy_villages_<tag>_<date>.json (that folder stays private)

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from diag_s12_copy import Tap, _where, run  # noqa: E402
from exp_s12_scene_files import CKPT, files, load_trunk, scenes_dir, sha, trunk_docs, views  # noqa: E402
from s12_questions import PLACED  # noqa: E402

__wiring__ = "STANDALONE"
MISS = {"another piece of the answer": "another piece of the same name",
        "the answer's first piece, in its written form": "the right word, another token form",
        "a neighbour's piece": "a neighbour's piece", "a farther thing's piece": "a farther thing's piece",
        "no thing's token": "another token"}


def miss(tk, sc, q, said: int, answer) -> str:
    if q["cls"] != "nothing" and tk.decode([said]).strip() == "nothing":
        return "nothing"
    return MISS.get(_where(tk, sc, q, said, answer), "another token")


def measure(model, tap, tk, eos: int, lead: bool, limit=None) -> list[dict]:
    from exp_hypertoken_recall import answers
    d = scenes_dir()
    fixed = json.load(open(os.path.join(d, "questions.json"), encoding="utf-8"))
    rows = []
    for f in (files(d)[:limit] if limit else files(d)):
        sc = views(os.path.join(d, f))["cut"]
        qs = [q for q in fixed["villages"][f]["cut"] if q["shape"] in PLACED]
        for ids, places, roles, chunk in trunk_docs(tk, sc, qs, True, eos, lead):
            pred, seen = run(model, tap, ids, places, roles)
            for q, (j, k) in zip(chunk, answers(tk, ids)):
                ok = [bool(pred[t - 1] == ids[t]) for t in range(j, k + 1)]
                rows.append({"village": f, "shape": q["shape"], "cls": q["cls"], "answer": q["answer"], "first": ok[0],
                             "exact": all(ok), "lam": seen[j - 1][0] if j - 1 in seen else None,
                             "said": tk.decode([int(pred[t - 1]) for t in range(j, k + 1)]),
                             "miss": None if ok[0] else miss(tk, sc, q, int(pred[j - 1]), ids[j:k + 1])})
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for c in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == c]
        lams = [r["lam"] for r in rs if r["lam"] is not None]
        out[c] = {"n": len(rs), "first": sum(r["first"] for r in rs), "exact": sum(r["exact"] for r in rs),
                  "first_rate": round(sum(r["first"] for r in rs) / len(rs), 3),
                  "exact_rate": round(sum(r["exact"] for r in rs) / len(rs), 3),
                  "lambda_mean": round(float(np.mean(lams)), 4) if lams else None,
                  "lambda_over_half": sum(x > 0.5 for x in lams) if lams else None,
                  "missed_first_went_to": dict(collections.Counter(r["miss"] for r in rs if r["miss"]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="the checkpoint's name in the log (e.g. run5)")
    ap.add_argument("--limit", type=int, default=None, help="scene files (default: all)")
    ap.add_argument("--shapes", default="0034,lead", help="which shapes to read: 0034, lead or both")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    t0 = time.time()
    model, tk, meta = load_trunk()
    tap = Tap(model.spatial.copy_path)
    eos = tk.token_to_id("</s>")
    arms = {s: measure(model, tap, tk, eos, s == "lead", a.limit) for s in a.shapes.split(",")}
    d = scenes_dir()
    pub = {"what": "order 0035 C's measure: a checkpoint on 0034's villages (cut, the fixed placed questions), in 0034's "
                   "shape and in the far facts' (lead) shape, by answer class",
           "date": a.date, "tag": a.tag, "checkpoint": os.path.basename(os.environ.get("CB_CKPT") or CKPT),
           "copy": meta.get("spatial"), "villages": len(files(d)[:a.limit] if a.limit else files(d)),
           "questions_sha256": sha(os.path.join(d, "questions.json")), "seconds": round(time.time() - t0, 1),
           "shapes": {s: summary(rows) for s, rows in arms.items()}}
    path = os.path.join(HERE, "logs", f"exp_s12_copy_villages_{a.tag}_{a.date}.json")
    json.dump(pub, open(path, "w", encoding="utf-8"), indent=1)
    json.dump({"public": os.path.basename(path), "rows": arms},
              open(os.path.join(d, f"exp_s12_copy_villages_{a.tag}_{a.date}.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(pub["shapes"], indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
