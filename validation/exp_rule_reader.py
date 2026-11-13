"""Order 0067, step 3: the program seat as it is reads instructions into rule plans, its slots constrained to the plan
family's shape (`cubbyllm/speech/rule_reader.py`, through grilly2's logits hook, order 0071).

Rows: `validation/builders/build_rule_rows.py`'s (seen kinds; the ``dev`` split and the ``new_wording`` split, a seeded
sample of each). Arms, run one after another on the local card:
  zero  -- the instruction alone;
  few   -- K examples first, drawn once by seed from the ``train`` split (one per kind drawn, English and French).
Scored against the row's own plan: written (a plan, not "asked"); exact (the same leaves); the measurement right
(the same relations and arguments). Line 3's agreement with a benchmark's checker over replies is not read here.

  python validation/exp_rule_reader.py --export <450M grilly export> --tokenizer <tokenizer.json> \
      --adapter <programs adapter dir> [--rows <rows.jsonl>] [--n 100] [--k 8] [--arms zero,few] [--out <json>]
      [--keep <plans.json>]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.emit.seats_450m import Cubby450mTrunk  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402
from cubbyllm.speech import rule_reader as rr  # noqa: E402

__wiring__ = "STANDALONE"


def rels(plan):
    return sorted((lf["m"], lf.get("arg")) for lf in rp.leaves(plan)) if plan else None


def examples(rows, k, seed):
    rng = random.Random(seed)
    train = [r for r in rows if r["split"] == "train" and "+" not in r["kind"]]
    kinds = rng.sample(sorted({r["kind"] for r in train}), k)
    return [(r["instruction"], r["plan"]) for r in
            (rng.choice([x for x in train if x["kind"] == kd and x["lang"] == ("en", "fr")[i % 2]]) for i, kd in enumerate(kinds))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="zero,few")
    ap.add_argument("--splits", default="dev,new_wording", help="read dev first; new wordings once, after")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", default=None, help="write the plans read, {arm: {row id: plan or null}}")
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "rule_rows_0067" / "rows.jsonl")
    rows = [json.loads(line) for line in open(rows_p, encoding="utf-8")]
    rng = random.Random(a.seed)
    sample = {s: rng.sample([r for r in rows if r["split"] == s], a.n) for s in ("dev", "new_wording")}
    sample = {s: v for s, v in sample.items() if s in a.splits.split(",")}   # the same draw whichever is read
    for s in a.splits.split(","):                        # round 4's held-out Alpaca splits: all of them, or n at most
        if s not in sample:
            pool = [r for r in rows if r["split"] == s]
            sample[s] = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    shots = examples(rows, a.k, a.seed)
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"programs": a.adapter})
    tk, eos = trunk.tokenizer, trunk.eos
    encode = lambda text: tk.encode(text).ids                       # noqa: E731
    calls = Counter()

    def gen(ids, allowed, max_new):
        ids = ids[-trunk.max_ctx:]
        calls["n"] += 1
        return trunk.generate("programs", [ids], max_new,
                              prefix_allowed_tokens_fn=lambda b, s: allowed(s.tolist()[len(ids):]))[0]

    report = {"order": "0067", "what": "the program seat as it is reads instructions into rule plans, slots constrained",
              "adapter": Path(a.adapter).name, "rows_file_lines": len(rows), "n_per_split": a.n, "k": a.k, "seed": a.seed,
              "examples_kinds": [rr.surface(p) for _, p in shots], "arms": {}}
    kept: dict = {}
    for arm in a.arms.split(","):
        t0, out = time.time(), {}
        kept[arm] = {}
        for split, rs in sample.items():
            tally, per_kind, wrong = Counter(), defaultdict(Counter), []
            for r in rs:
                got = rr.read(r["instruction"], gen, encode, eos, examples=shots if arm == "few" else ())
                kept[arm][r["id"]] = got["plan"]
                ok = got["how"] == "none" if r["plan"] is None else rr.same(got["plan"], r["plan"])
                rel_ok = rels(got["plan"]) == rels(r["plan"])
                for key, hit in (("rows", 1), ("written", got["how"] == "written"), ("exact", ok), ("measurement right", rel_ok),
                                 (f"rows {r['lang']}", 1), (f"exact {r['lang']}", ok),
                                 ("rule-free rows", r["plan"] is None),
                                 ("invented a rule", r["plan"] is None and got["how"] == "written"),
                                 ("said none where there was a rule", r["plan"] is not None and got["how"] == "none")):
                    tally[key] += int(bool(hit))
                per_kind[r["kind"]].update({"rows": 1, "exact": int(ok), "measurement right": int(rel_ok)})
                if not ok and len(wrong) < 12:
                    wrong.append({"kind": r["kind"], "want": rr.surface(r["plan"]), "got": got["surface"]})
            out[split] = {"counts": dict(tally), "per_kind": {k: dict(v) for k, v in sorted(per_kind.items())},
                          "some_misses": wrong}
            print(f"[{arm}] {split}: {dict(tally)}", flush=True)
        out["seconds"] = round(time.time() - t0, 1)
        out["seat_calls"] = calls["n"]
        calls.clear()
        report["arms"][arm] = out
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.keep:
        Path(a.keep).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(text[:3000])


if __name__ == "__main__":
    main()
