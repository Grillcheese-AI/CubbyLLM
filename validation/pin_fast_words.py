"""pin_fast_words -- order 0033: the ported encoder (``cubbyllm.worlds.fast_words``) against mowm's
(``mowm/encoding/semantic_words.py``, loaded from the checkout beside this one) on the production table: both load it
with their own ``from_npz``, both encode the same texts, and every code must be equal bit for bit.

The texts: the real stack's 20 fixed questions, then ``--n`` sentences of table words drawn by seed (with unseen words
among them, so the signal-code fallback is read too).

    python validation/pin_fast_words.py [--table <npz>] [--n 2000] [--seed 0]
-> validation/logs/pin_fast_words_<date>.json

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from cubbyllm.local_paths import expand  # noqa: E402
from cubbyllm.worlds import fast_words as port  # noqa: E402

__wiring__ = "STANDALONE"

MOWM = expand("$CUBBY_GITHUB/mowm/mowm/encoding/semantic_words.py")


def main() -> int:
    from cubbyllm.worlds.fact_pool import V4_TABLE
    from stack_answers import QUESTIONS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", default=str(V4_TABLE))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    if not os.path.exists(MOWM):
        print("no mowm checkout beside this one: nothing to pin against")
        return 1
    spec = importlib.util.spec_from_file_location("mowm_semantic_words", MOWM)
    mowm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mowm)
    t0 = time.time()
    ours, theirs = port.FastWordEncoder.from_npz(a.table), mowm.FastWordEncoder.from_npz(a.table)
    load_s = round(time.time() - t0, 1)
    rng = random.Random(a.seed)
    words = sorted(ours._vec)
    texts = list(QUESTIONS) + [" ".join(rng.choice(words) if rng.random() < 0.9 else f"unseen{rng.randrange(10**6)}"
                                        for _ in range(rng.randint(1, 12))) for _ in range(a.n)]
    equal, worst = 0, 0.0
    for t in texts:
        x, y = ours.encode(t), theirs.encode(t)
        equal += bool(np.array_equal(x, y))
        worst = max(worst, float(np.abs(x - y).max()))
    out = {"what": "order 0033: cubbyllm.worlds.fast_words against mowm's FastWordEncoder on the production table",
           "table": os.path.basename(a.table), "words": len(words), "k": ours.k, "l": ours.l,
           "same_meta": ours.meta == theirs.meta, "same_idf": ours._idf == theirs._idf and ours.default_idf == theirs.default_idf,
           "texts": len(texts), "equal": equal, "max_abs_diff": worst, "load_s": load_s,
           "wall_s": round(time.time() - t0, 1), "args": {"n": a.n, "seed": a.seed}}
    path = os.path.join(HERE, "logs", f"pin_fast_words_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))
    print("->", path)
    return 0 if equal == len(texts) and out["same_meta"] and out["same_idf"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
