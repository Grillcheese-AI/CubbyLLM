"""Order 0067, the view reader's round 2 (Nick, 2026-11-13: "polarity pairs"; inverted phrasing): training rows in
minimal pairs, built in reverse like step 3's long rows (`build_long_rows.py`) and run before kept.

A pair: one long instruction (2-4 parts, English or French) and its twin, the same parts, groups and wording draws but
one part said the other way (the generator's own complement: "fewer than 48 words" / "at least 48 words", "no X" /
"at least one X", a negated "every" / "every"), each with its own split and its own program. The only thing to learn
between the two is the direction, which the first reads got wrong in a quarter of parts. A share of the checks
(``--inverted``) is said goal first or in the passive (`view_wording_inv.py`), the same atom so the same program.

Kept like a long row: each instruction's program run in the VM on its replies (half pass, half fail) against the
evaluator; a pair is kept only whole. Train split only; an instruction equal to any row of step 3's set (dev and test
among them) or sharing 5 words with IFEval or IFBench is dropped. A disagreement stops the build.

  python validation/builders/build_pair_rows.py [--pairs 1500] [--seed 7] [--inverted 0.3]
      [--out $CUBBY_DATASETS/long_rows_0067/pair_rows.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
import build_long_rows as BL  # noqa: E402
import view_wording_inv as WI  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"


def flip(part: dict) -> dict:
    """The part said the other way: a negated atom unwrapped, an atom negated."""
    return part["not"] if "not" in part else {"not": part}


def wording(share: float):
    """How a part is said: goal first / passive for a ``share`` of the parts that have such a wording."""
    def imp(p, lang, rng):
        if rng.random() < share:
            s = WI.say_inverted(p, lang, rng)
            if s is not None:
                return s
        return BL._imp(p, lang, rng)
    return imp


def build_pair(rng, vm, views, replies, phrases, bench, out_dir, share):
    k, lang = rng.randint(2, 4), "fr" if rng.random() < 0.4 else "en"
    parts = BL.draw_parts(rng, views, k)
    if parts is None:
        return None, "no parts drawn"
    groups = BL.groups_for(rng, parts)
    alone = [g[1] for g in groups if g[0] == "one"]       # a condition is said of a plain atom: only a lone part flips
    if not alone:
        return None, "no lone part to flip"
    i = rng.choice(alone)
    twin = [flip(p) if j == i else p for j, p in enumerate(parts)]
    state, imp = rng.getstate(), wording(share)
    a, why = BL.finish(rng, vm, views, replies, parts, groups, lang, phrases, bench, "train", out_dir, imp)
    if a is None:
        return None, why
    rng.setstate(state)                                   # the twin draws its wording as the first did
    b, why = BL.finish(rng, vm, views, replies, twin, groups, lang, phrases, bench, "train", out_dir, imp)
    if b is None:
        return None, "twin: " + why
    if a["instruction"] == b["instruction"]:
        return None, "the twin says the same words"
    return (a, b), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--inverted", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067"
    out = Path(a.out or ds / "pair_rows.jsonl")
    seen = {json.loads(ln)["instruction"] for ln in open(ds / "rows.jsonl", encoding="utf-8")}
    vm, bench = BL.cc.CubelangSession(), BL.BO.benchmark_grams(ifbench=True)
    train_ph, _ = BL.real_rows._phrases(9)
    rng, rows, why, replies, views = random.Random(a.seed * 1000 + 17), [], Counter(), [], []
    while len(rows) < 2 * a.pairs:
        if len(rows) % 10 == 0 or not replies:
            replies = BL.VR.pool(rng, 80)
            views = [BL.RV.views(t) for t in replies]
        pair, reason = build_pair(rng, vm, views, replies, train_ph, bench, out.parent, a.inverted)
        if pair is None:
            why[reason] += 1
            continue
        if any(r["instruction"] in seen for r in pair):
            why["an instruction already in step 3's set"] += 1
            continue
        n = len(rows) // 2
        for side, r in zip("ab", pair):
            seen.add(r["instruction"])
            rows.append(dict(r, id=f"pair{n:05d}{side}", pair=n, family=f"pair_{r['n_parts']}"))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"order": "0067", "what": "the view reader's round 2: long instructions in polarity pairs, run before kept",
           "seed": a.seed, "inverted_share": a.inverted, "rows": len(rows), "pairs": len(rows) // 2,
           "by_lang": dict(Counter(r["lang"] for r in rows)), "by_parts": dict(sorted(Counter(r["n_parts"] for r in rows).items())),
           "drop_reasons": dict(why), "vm_runs_agreeing": sum(len(r["replies"]) for r in rows), "vm_disagreements": 0,
           "benchmarks_checked": sorted(bench), "vm": BL.vm_identity.identity(),
           "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "vm"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
