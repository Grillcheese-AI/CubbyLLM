"""Order 0074, step 1: the rows, built again from a new seed (`long_rows_0074`, off-repo; counts and SHA-256 in the
repo). 0073's builder (`build_rows_0073.py`: each row drawn, worded, its folded said forms joined, and every reply
kept run in the VM against the evaluator), with one change in what is drawn: the train set holds rows of 5 to 8
parts as well.

- train: 3,000 rows of 2 to 4 parts, 1,500 minimal pairs (2 to 4 parts, 3,000 rows) and 2,000 rows of 5 to 8 parts,
  the same number of each length; dev: half of 2 to 4 parts, half of 5 to 8; test 2 to 4; test_long 5 to 8;
  test_longer 9 to 12. The held-out sets get the replies that can tell, as 0073's did;
- the pool the parts are drawn on: 80, 160 and 240 replies for 2 to 4, 5 to 8 and 9 to 12 parts;
- no instruction of 0067's or 0073's rows; no held-out row with a train row's folded parts and groups.

Each part of the plan draws from its own stream and keeps its own counts (bounds, placements), as each set did in
0073. A row's id is ``d``, the plan part's index and five digits; a pair's is ``dp``, five digits and ``a`` / ``b``.

  python validation/builders/build_rows_0074.py [--train 3000] [--train-long 2000] [--pairs 1500] [--held 300]
      [--seed 74] [--inverted 0.3] [--band 0.45 0.55] [--reserve 640] [--out $CUBBY_DATASETS/long_rows_0074]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import bench_overlap as BO  # noqa: E402
import build_rows_0073 as B  # noqa: E402
import real_rows  # noqa: E402
import rows_0073_stats as ST  # noqa: E402
import view_draw as D  # noqa: E402
import view_replies as VR  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402

__wiring__ = "STANDALONE"

SPLITS = ("train", "dev", "test", "test_long", "test_longer")
HELD = SPLITS[1:]
OLD = ("long_rows_0067", "long_rows_0073")


def plan(a, train_ph: dict, held_ph: dict) -> list[tuple]:
    """(plan part, split, rows, (fewest, most parts), phrases, each length as often), in the order built."""
    half = a.held // 2
    return [("train", "train", a.train, (2, 4), train_ph, False),
            ("train_long", "train", a.train_long, (5, 8), train_ph, True),
            ("pairs", "pairs", a.pairs, (2, 4), train_ph, False),
            ("dev", "dev", half, (2, 4), train_ph, False),
            ("dev_long", "dev", a.held - half, (5, 8), train_ph, False),
            ("test", "test", a.held, (2, 4), held_ph, False),
            ("test_long", "test_long", a.held, (5, 8), held_ph, False),
            ("test_longer", "test_longer", a.held, (9, 12), held_ph, False)]


def pool_size(most: int) -> int:
    """The replies the parts are drawn on."""
    return 240 if most > 8 else 160 if most > 4 else 80


def build(a, out: Path, texts: set, vm, bench) -> tuple[dict, Counter, dict]:
    """{plan part: rows}, why rows were dropped, seconds by plan part."""
    train_ph, held_ph = real_rows._phrases(9)             # round 6's split of the phrases (held-out keys stay held out)
    parts, why, seconds, train_keys = {}, Counter(), {}, set()
    for n_part, (name, split, n, (lo, hi), phrases, even) in enumerate(plan(a, train_ph, held_ph)):
        rng, tally, seen = random.Random(a.seed * 100 + n_part), Counter(), Counter()
        rows, kept, replies, views, reserve, drawn_at, misses, t0 = [], 0, [], [], [], -1, 0, time.time()
        while kept < n:
            if (kept % 5 == 0 and kept != drawn_at) or misses >= 10:   # a new pool every five rows, or when one
                replies = VR.pool(rng, pool_size(hi))                 # gives nothing ten times running
                views = [RV.views(t) for t in replies]
                reserve = [(t, RV.views(t)) for t in VR.pool(rng, a.reserve)] if split in HELD else []
                drawn_at, misses = kept, 0
            if split == "pairs":
                got, reason = B.build_pair(rng, vm, views, replies, phrases, bench, out, tally, a.inverted, seen)
            else:
                k = lo + kept % (hi - lo + 1) if even else rng.randint(lo, hi)
                got, reason = B.build_one(rng, vm, views, replies, k, "fr" if rng.random() < 0.4 else "en",
                                          phrases, bench, split, out, tally, reserve, seen)
            if got is not None and any(r["instruction"] in texts for r in got):
                got, reason = None, "an instruction already written (0067's or 0073's rows, or this build)"
            if got is not None and split in HELD and any(B.key(r) in train_keys for r in got):
                got, reason = None, "a train row's parts and groups"
            if got is None:
                why[f"{name}: {reason}"] += 1
                misses += 1
                continue
            misses = 0
            for side, r in zip("ab", got):
                r["id"] = f"dp{kept:05d}{side}" if split == "pairs" else f"d{n_part}{kept:05d}"
                if split == "pairs":
                    r.update(pair=kept, family=f"pair_{r['n_parts']}")
                texts.add(r["instruction"])
                D.tally_row(tally, [s["w"] for s in r["said"]])
                seen[(r["n_parts"], D.placement(r["groups"]))] += 1
                if split not in HELD:
                    train_keys.add(B.key(r))
                rows.append(r)
            kept += 1
        parts[name], seconds[name] = rows, round(time.time() - t0, 1)
        print(f"  {name}: {len(rows)} rows, {seconds[name]} s", flush=True)
    return parts, why, seconds


def old_texts(ds: Path) -> dict:
    """{earlier set: its instructions}."""
    return {name: {json.loads(ln)["instruction"] for f in ("rows.jsonl", "pair_rows.jsonl") if (ds / name / f).exists()
                   for ln in open(ds / name / f, encoding="utf-8")} for name in OLD}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--train-long", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--held", type=int, default=300)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--inverted", type=float, default=0.3)
    ap.add_argument("--band", type=float, nargs=2, default=(0.45, 0.55))
    ap.add_argument("--reserve", type=int, default=640)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = Path(a.out or ds / "long_rows_0074")
    out.mkdir(parents=True, exist_ok=True)
    old = old_texts(ds)
    if not a.out and not all(old.values()):
        raise SystemExit(f"an earlier set was not found ({[k for k, v in old.items() if not v]}): the set cannot be "
                         "checked against its instructions")
    texts = set().union(*old.values())
    vm, bench = cc.CubelangSession(), BO.benchmark_grams(ifbench=True)
    parts, why, seconds = build(a, out, texts, vm, bench)
    by_split = {s: [r for (name, split, *_) in plan(a, {}, {}) if split == s for r in parts[name]] for s in SPLITS}
    files = {"rows.jsonl": [r for s in SPLITS for r in by_split[s]], "pair_rows.jsonl": parts["pairs"]}
    for name, rows in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(ST.line(r) for r in rows)
    train = by_split["train"] + parts["pairs"]
    groups = dict(parts, **{s: by_split[s] for s in HELD}, **{"train with its pairs": train})
    shares = {name: ST.bounds(rows)["all"]["lower share"] for name, rows in groups.items()}
    man = {"order": "0074", "what": "step 1: long instructions of 2 to 12 parts with their split, each check's said "
           "form in both spellings and the join of the folded forms, run before kept; the train set holds 5 to 8 parts",
           "seed": a.seed, "pairs_said_goal_first": a.inverted, "replies_searched_beyond_the_pool": a.reserve,
           "pool": {"2 to 4": pool_size(4), "5 to 8": pool_size(8), "9 to 12": pool_size(12)},
           "rows": len(files["rows.jsonl"]) + len(parts["pairs"]),
           "by_split": {"train": len(train), **{s: len(by_split[s]) for s in HELD}}, "pairs": len(parts["pairs"]) // 2,
           "by_plan_part": {name: len(rows) for name, rows in parts.items()},
           "lower bounds' share of the bounds": shares, "band": list(a.band),
           "sets": {name: ST.summary(rows) for name, rows in groups.items()},
           "held-out rows whose composition without its values is a train row's": ST.shapes(
               [r for s in HELD for r in by_split[s]], train),
           "instructions of earlier sets kept out": {name: len(v) for name, v in old.items()},
           "drop_reasons": dict(sorted(why.items())), "seconds": seconds,
           "vm_runs_agreeing": sum(len(r["replies"]) for rows in parts.values() for r in rows), "vm_disagreements": 0,
           "benchmarks_checked": sorted(bench), "vm": vm_identity.identity(),
           "sha256": dict({name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
                          **{"the lines of rows.jsonl by split": {s: ST.digest(by_split[s]) for s in SPLITS}})}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("vm", "sets")}, indent=1, ensure_ascii=False))
    off = {name: s for name, s in shares.items() if name in (*HELD, "train with its pairs") and groups[name]
           and (s is None or not a.band[0] <= s <= a.band[1])}
    if off:
        raise SystemExit(f"the lower bounds' share is outside {a.band[0]} to {a.band[1]}: {off}; the set is not to be used")


if __name__ == "__main__":
    main()
