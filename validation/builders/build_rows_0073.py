"""Order 0073, step 3: the rows, built again from a new seed (`long_rows_0073`, off-repo; counts and SHA-256 in the
repo). Long instructions built in reverse as in order 0067 (`build_long_rows.py`, `build_pair_rows.py`), each check
carrying what its words say in both spellings (arm W as said, arm M folded: `view_said`), run before kept.

Against 0067's build (the design is in the order's Report):
- the draw balances lower and upper bounds and places the tied pairs evenly (`view_draw`); each connective is said
  before and after its partner (`view_compose`);
- a row's program is the join of its **folded said forms**. The evaluator's verdict on them is its verdict on the
  parts as drawn on every usable reply of the pool, and the VM's verdict is the evaluator's on every reply kept; a
  difference stops the build;
- dev, test and test_long rows get, after the 8 replies, the replies that can tell a wrong reading
  (`view_tell.extras`), looked for in the pool the parts were drawn on and in ``--reserve`` more replies drawn
  beside it (the draw never sees them), each run in the VM like the eight; every row records what its replies reach
  and how many of its one-reversal mutants they tell apart (train rows keep 8);
- no instruction of 0067's rows; no held-out row with a train row's folded parts and groups;
- minimal pairs (train only): one lone part said the other way, the twin drawing its wording as the first did, a
  share of the checks said goal first.

  python validation/builders/build_rows_0073.py [--train 3000] [--pairs 1500] [--held 300] [--seed 73]
      [--inverted 0.3] [--band 0.45 0.55] [--reserve 640] [--out $CUBBY_DATASETS/long_rows_0073]
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
import real_rows  # noqa: E402
import rows_0073_stats as ST  # noqa: E402
import view_compose as VCo  # noqa: E402
import view_draw as D  # noqa: E402
import view_replies as VR  # noqa: E402
import view_said as S  # noqa: E402
import view_tell as T  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

HELD = ("dev", "test", "test_long")


def key(row: dict) -> str:
    """A row's composition: its folded parts and its groups, values included."""
    return json.dumps([[s["m"] for s in row["said"]], row["groups"]], sort_keys=True, ensure_ascii=False)


def _stop(stop_dir: Path, what: str, **kept) -> None:
    (stop_dir / "_disagreement.json").write_text(json.dumps(dict(kept, what=what), ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    raise SystemExit(f"{what}; kept in _disagreement.json beside the rows")


def finish(rng, vm, views, replies, parts, groups, lang, phrases, bench, split, stop_dir, share=0.0, reserve=()):
    """A row from drawn parts and groups: worded, numbered in text order, its said forms' join run on its replies.
    ``reserve``: (reply, its views) beyond the pool the parts were drawn on, searched only for replies that can tell."""
    text, segs, order = VCo.compose(rng, parts, groups, lang, phrases, share)
    parts, groups, segs = VCo.renumber(parts, groups, segs, order)
    if split in ("train", "dev") and BO.overlaps(text, bench):
        return None, "shares 5 words with a benchmark"
    said = VCo.said_forms(parts, segs, lang)
    if said is None:
        return None, "a wording with no description"
    m = [s["m"] for s in said]
    join, gold, drawn = VJ.join(m, groups), VJ.spec(m, groups), VJ.spec(parts, groups)
    used = sorted({v for p in m for v in VC.views_of(p)})
    cand, spare = ([(t, v, VS.evaluate(gold, v)) for t, v in pairs if all(len(v[u]) <= RV.MAX_ROWS for u in used)]
                   for pairs in (list(zip(replies, views)), reserve))
    for t, v, want in cand + spare:
        if VS.evaluate(drawn, v) != want:
            _stop(stop_dir, "the folded said forms do not say what was drawn", parts=parts, said=said,
                  groups=groups, reply=t)
    yes, no = [c for c in cand if c[2]], [c for c in cand if not c[2]]
    if len(yes) < 2 or len(no) < 2:
        return None, "too few usable replies on one side"
    chosen = rng.sample(yes, min(4, len(yes))) + rng.sample(no, min(len(no), 8 - min(4, len(yes))))
    pool = cand + spare if split in HELD else []
    add, reach = T.extras(rng, m, groups, [c[1] for c in chosen], [c[1] for c in pool])
    kept = [(c, False) for c in chosen] + [(pool[j], True) for j in add]
    for (t, _, want), _ in kept:
        got = str(vm.run(VJ.program(join, m, t), fn="solve").get("result")).strip() in ("1", "1.0")
        if got != want:
            _stop(stop_dir, "the VM disagrees with the evaluator on a joined program", said=said, groups=groups,
                  reply=t, evaluator=want, vm=got)
    return {"split": split, "lang": lang, "n_parts": len(parts), "instruction": text, "segments": segs, "parts": parts,
            "said": said, "groups": [list(g) for g in groups], "join": join,
            "replies": [dict({"text": t, "held": h}, **({"extra": True} if x else {})) for (t, _, h), x in kept],
            "reach": reach, "mutants": T.told(m, groups, [c[1] for c, _ in kept])}, None


def build_one(rng, vm, views, replies, k, lang, phrases, bench, split, stop_dir, tally, reserve=(), seen=None):
    parts = D.draw_parts(rng, views, k, tally)
    if parts is None:
        return None, "no parts drawn"
    row, why = finish(rng, vm, views, replies, parts, D.groups_for(rng, parts, seen=seen), lang, phrases, bench, split,
                      stop_dir, reserve=reserve)
    return (None if row is None else (row,)), why


def flip(part: dict) -> dict:
    """The part said the other way: a negated atom unwrapped, an atom negated."""
    return part["not"] if "not" in part else {"not": part}


def build_pair(rng, vm, views, replies, phrases, bench, stop_dir, tally, share, seen=None):
    k, lang = rng.randint(2, 4), "fr" if rng.random() < 0.4 else "en"
    parts = D.draw_parts(rng, views, k, tally)
    if parts is None:
        return None, "no parts drawn"
    groups = D.groups_for(rng, parts, lone=True, seen=seen)
    alone = [g[1] for g in groups if g[0] == "one"]       # a condition is said of a plain atom: only a lone part flips
    if not alone:
        return None, "no lone part to flip"
    i = rng.choice(alone)
    twin = [flip(p) if j == i else p for j, p in enumerate(parts)]
    if S.said(twin[i], "en") is None or D.held(twin[i]):
        return None, "the twin has no description or is of a kind kept for the test"
    state = rng.getstate()
    a, why = finish(rng, vm, views, replies, parts, groups, lang, phrases, bench, "train", stop_dir, share)
    if a is None:
        return None, why
    rng.setstate(state)                                   # the twin draws its wording as the first did
    b, why = finish(rng, vm, views, replies, twin, groups, lang, phrases, bench, "train", stop_dir, share)
    if b is None:
        return None, "twin: " + why
    if a["instruction"] == b["instruction"]:
        return None, "the twin says the same words"
    return (a, b), None


def build(a, out: Path, texts: set, vm, bench) -> tuple[dict, Counter, dict]:
    """The five sets in order (train, pairs, dev, test, test_long): {name: rows}, why rows were dropped, seconds."""
    train_ph, held_ph = real_rows._phrases(9)             # round 6's split of the phrases (held-out keys stay held out)
    plan = [("train", a.train, (2, 4), train_ph), ("pairs", a.pairs, (2, 4), train_ph), ("dev", a.held, (2, 4), train_ph),
            ("test", a.held, (2, 4), held_ph), ("test_long", a.held, (5, 8), held_ph)]
    sets, why, seconds, train_keys = {}, Counter(), {}, set()
    for n_split, (split, n, (lo, hi), phrases) in enumerate(plan):
        rng, tally, seen = random.Random(a.seed * 100 + n_split), Counter(), Counter()   # each set its own stream,
        rows, kept, replies, views, reserve, drawn_at, misses, t0 = [], 0, [], [], [], -1, 0, time.time()   # its own counts
        while kept < n:
            if (kept % 5 == 0 and kept != drawn_at) or misses >= 10:   # a new pool every five rows, or when one
                replies = VR.pool(rng, 160 if hi > 4 else 80)         # gives nothing ten times running
                views = [RV.views(t) for t in replies]
                reserve = [(t, RV.views(t)) for t in VR.pool(rng, a.reserve)] if split in HELD else []
                drawn_at, misses = kept, 0
            if split == "pairs":
                got, reason = build_pair(rng, vm, views, replies, phrases, bench, out, tally, a.inverted, seen)
            else:
                got, reason = build_one(rng, vm, views, replies, rng.randint(lo, hi), "fr" if rng.random() < 0.4 else "en",
                                        phrases, bench, split, out, tally, reserve, seen)
            if got is not None and any(r["instruction"] in texts for r in got):
                got, reason = None, "an instruction already written (0067's rows or this build)"
            if got is not None and split in HELD and any(key(r) in train_keys for r in got):
                got, reason = None, "a train row's parts and groups"
            if got is None:
                why[f"{split}: {reason}"] += 1
                misses += 1
                continue
            misses = 0
            for side, r in zip("ab", got):
                r["id"] = f"dp{kept:05d}{side}" if split == "pairs" else f"d{n_split}{kept:05d}"
                if split == "pairs":
                    r.update(pair=kept, family=f"pair_{r['n_parts']}")
                texts.add(r["instruction"])
                D.tally_row(tally, [s["w"] for s in r["said"]])
                seen[(r["n_parts"], D.placement(r["groups"]))] += 1
                if split not in HELD:
                    train_keys.add(key(r))
                rows.append(r)
            kept += 1
        sets[split], seconds[split] = rows, round(time.time() - t0, 1)
        print(f"  {split}: {len(rows)} rows, {seconds[split]} s", flush=True)
    return sets, why, seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--held", type=int, default=300)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--inverted", type=float, default=0.3)
    ap.add_argument("--band", type=float, nargs=2, default=(0.45, 0.55))
    ap.add_argument("--reserve", type=int, default=640)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = Path(a.out or ds / "long_rows_0073")
    out.mkdir(parents=True, exist_ok=True)
    texts = {json.loads(ln)["instruction"] for name in ("rows.jsonl", "pair_rows.jsonl")
             if (ds / "long_rows_0067" / name).exists() for ln in open(ds / "long_rows_0067" / name, encoding="utf-8")}
    n_old = len(texts)
    if not a.out and not n_old:
        raise SystemExit("0067's rows were not found: the set cannot be checked against their instructions")
    vm, bench = cc.CubelangSession(), BO.benchmark_grams(ifbench=True)
    sets, why, seconds = build(a, out, texts, vm, bench)
    main_rows = [r for s in ("train", "dev", "test", "test_long") for r in sets[s]]
    files = {"rows.jsonl": main_rows, "pair_rows.jsonl": sets["pairs"]}
    for name, rows in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(ST.line(r) for r in rows)
    train = sets["train"] + sets["pairs"]
    groups = dict(sets, **{"train with its pairs": train})
    shares = {name: ST.bounds(rows)["all"]["lower share"] for name, rows in groups.items()}
    man = {"order": "0073", "what": "step 3: long instructions with their split, each check's said form in both "
           "spellings and the join of the folded forms, run before kept",
           "seed": a.seed, "pairs_said_goal_first": a.inverted, "replies_searched_beyond_the_pool": a.reserve,
           "rows": len(main_rows) + len(sets["pairs"]),
           "by_split": {"train": len(train), **{s: len(sets[s]) for s in HELD}}, "pairs": len(sets["pairs"]) // 2,
           "lower bounds' share of the bounds": shares, "band": list(a.band),
           "sets": {name: ST.summary(rows) for name, rows in groups.items()},
           "held-out rows whose composition without its values is a train row's": ST.shapes(
               [r for s in HELD for r in sets[s]], train),
           "instructions of 0067 kept out": n_old, "drop_reasons": dict(sorted(why.items())), "seconds": seconds,
           "vm_runs_agreeing": sum(len(r["replies"]) for rows in sets.values() for r in rows), "vm_disagreements": 0,
           "benchmarks_checked": sorted(bench), "vm": vm_identity.identity(),
           "sha256": dict({name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
                          **{"the lines of rows.jsonl by split": {s: ST.digest(sets[s]) for s in ("train", *HELD)}})}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("vm", "sets")}, indent=1, ensure_ascii=False))
    off = {name: s for name, s in shares.items() if name not in ("pairs", "train") and groups[name]
           and (s is None or not a.band[0] <= s <= a.band[1])}
    if off:
        raise SystemExit(f"the lower bounds' share is outside {a.band[0]} to {a.band[1]}: {off}; the set is not to be used")


if __name__ == "__main__":
    main()
