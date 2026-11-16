"""Order 0073, step 6: the second chance on descriptions, on dev only (never on test or test_long: a pattern read
there would leak into what the test read measures).

Round 2's second chance reversed one comparison at a time inside the reader's program. Here the reader writes a
description, so the reversal is of one slot of it (`view_tell.part_mutants`: a comparison negated or mirrored,
``any`` and ``none`` exchanged, ``absent`` toggled), tried with the evaluator. For every gold check of a dev row
whose description is not right (`view_split_rowscore.score`):
- **fixed by one reversal:** the reversal is recorded with the cue words of the check's own text, how the check
  was said (imperative, as a condition, inverted) and how many ``not`` arm W's gold form carries: the pattern
  table, which says which wording is read which way and how often;
- **not fixed:** the slots in which the description is not the gold's (the view, the number, a condition...).
Also counted: descriptions that are right without being the gold's form, by the slots that differ.

  python validation/exp_view_split_second_chance.py --answers <dev answers.jsonl> [--rows <rows.jsonl>] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import view_split_rowscore as R  # noqa: E402

__wiring__ = "STANDALONE"

CUES = ["at most", "at least", "fewer than", "less than", "more than", "exactly", "other than", "no", "none", "every", "all",
        "not", "only", "unless", "if there is", "au plus", "au moins", "moins de", "plus de", "exactement", "autre que",
        "aucun", "aucune", "chaque", "tous", "toutes", "pas", "ne", "seulement", "sauf si", "s'il y en a"]
_CUES = [(c, re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)")) for c in CUES]


def cue(text: str) -> str:
    """The cue words a check's text holds, as whole words."""
    low = text.lower()
    return " + ".join(c for c, pat in _CUES if pat.search(low)) or "(no cue word)"


def table(rows: list, scored: list) -> dict:
    """{"counts", "patterns", "not fixed", "right in another form"} over scored dev rows (`rows`: theirs, same order)."""
    t, patterns, unfixed, other = Counter(), defaultdict(Counter), Counter(), Counter()
    for row, s in zip(rows, scored):
        checks = {x["part"]: x for x in row["segments"] if x["kind"] == "check"}
        t["rows"] += 1
        for p in s["parts"]:
            t["gold checks"] += 1
            if p["right"]:
                t["right as written"] += 1
                if not p["same"]:
                    t["right as written, not the gold's form"] += 1
                    other[" + ".join(p["differs"])] += 1
                continue
            if not p["line"]:
                t[f"no check line: written as `{p['as']}`" if p["as"] else "no check line: no line with its span"] += 1
                continue
            if p["differs"] is None:
                t["a description that does not compile"] += 1
                continue
            seg = checks[p["part"]]
            key = f"{cue(row['instruction'][seg['start']:seg['end']])} | said: {seg.get('how')} | not: {p['nots']} | {row['lang']}"
            if p["fixed"]:
                t["fixed by one reversal"] += 1
                patterns[key][p["fix"]] += 1
            else:
                t["wrong, and no one reversal fixes it"] += 1
                unfixed[" + ".join(p["differs"])] += 1

    def ranked(c):
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"counts": dict(sorted(t.items())), "right in another form, by the slots that differ": ranked(other),
            "patterns": {k: ranked(v) for k, v in sorted(patterns.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))},
            "not fixed, by the slots that differ": ranked(unfixed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from cubbyllm import local_paths
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.bridges import vm_identity
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073" / "rows.jsonl")
    by_id = {r["id"]: r for r in (json.loads(ln) for ln in open(rows_p, encoding="utf-8"))}
    answers = [json.loads(ln) for ln in open(a.answers, encoding="utf-8")]
    off = sorted({by_id[d["id"]]["split"] for d in answers} - {"dev"})
    if off:
        raise SystemExit(f"answers of {off}: the second chance runs on dev rows only")
    arms, variants = {d["arm"] for d in answers}, {bool(d.get("own_text")) for d in answers}
    if len(arms) != 1 or len(variants) != 1 or not arms <= {"m", "w"}:
        raise SystemExit(f"the answers are not one arm (m or w) in one variant: {sorted(arms)}, own_text {sorted(variants)}")
    arm, own_text, vm = arms.pop(), variants.pop(), cc.CubelangSession()
    rows = [by_id[d["id"]] for d in answers]
    got = table(rows, [R.score(r, d, arm, own_text, vm) for r, d in zip(rows, answers)])
    out = dict({"order": "0073", "what": "the second chance on descriptions, dev only: one slot reversed at a time, the pattern table",
                "arm": arm, "own_text": own_text, "answers": Path(a.answers).name, "vm": vm_identity.identity()}, **got)
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
