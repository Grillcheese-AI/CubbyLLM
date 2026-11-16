"""Order 0073, step 6: the reader's answers (arm m or w, decoded on Colab by `exp_view_split_torch.py`) scored at
home on the shared VM. Nothing is decoded here. Each row is scored by `view_split_rowscore.score`; this file adds
the rows up into the counts of "Acceptance" (lines 1 to 7 and 10), as the order states them ("Step 6, the scoring
stated before any read"). Every rate is given with its two counts.

  python validation/exp_view_split_score.py --answers <answers.jsonl> [--splits dev] [--rows <rows.jsonl>]
      --out <report.json> [--keep <per-row counts.jsonl>]

The arm and the variant (``own_text``) are the answers' own. ``--keep``: one line a row with its id and its counts
(no text), which line 9's comparison resamples (`exp_view_split_compare.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import view_split_rowscore as R  # noqa: E402

__wiring__ = "STANDALONE"

BY_PART = ("direction", "nots", "place", "filter number", "parts", "position", "lang")
DIRECTIONS = ("lower", "upper", "exactly", "other")


def rate(a: int, b: int):
    return round(a / b, 4) if b else None


def row_counts(s: dict) -> dict:
    """One scored row's counts (what `--keep` writes and line 9 resamples)."""
    seen = [x for x in s["replies"] if x["verdict"] is not None]
    fail = [x for x in seen if not x["held"]]
    return {"id": s["id"], "split": s["split"], "n_parts": s["n_parts"], "lang": s["lang"], "replies": len(s["replies"]),
            "checked": len(seen), "agree": sum(1 for x in seen if x["verdict"] == x["held"]),
            "false passes": sum(1 for x in fail if x["verdict"]), "built to fail": len(fail),
            "gold checks": len(s["parts"]), "parts right": sum(1 for p in s["parts"] if p["right"])}


def tally(scored: list) -> dict:
    """{"counts", "by"} over the scored rows of one split."""
    t, by = Counter(), {k: {} for k in BY_PART + ("connective", "pair's place", "rows by parts", "rows by lang")}

    def bump(table: str, key, right: bool) -> None:
        cell = by[table].setdefault(str(key), [0, 0])      # [right, of]
        cell[0] += int(right)
        cell[1] += 1
    for s in scored:
        c = row_counts(s)
        t["rows"] += 1
        t["rows whole"] += int(s["whole"])
        t[f"rows that ended `{s['end']}`" if s["read"] else "rows not read"] += 1    # not read: all of it flagged
        t["rows with something flagged"] += int(s["flagged_chars"] > 0)
        t["rows with a part asked by the host"] += int(s["asked_by_host"] > 0)
        t["rows checked"] += int(s["checked"])
        t["characters in no line and not flagged"] += s["unaccounted"]
        t["characters flagged"] += s["flagged_chars"]
        t["gold segments"] += s["segments"]
        t["gold segments with span and kind"] += s["segments_right"]
        for name in ("replies", "checked", "agree", "false passes", "built to fail"):
            t[{"checked": "replies checked", "built to fail": "checked replies built to fail"}.get(name, name)] += c[name]
        why = "the VM gave no verdict" if s["checked"] else "their row is not checked"
        t[f"replies not checked: {why}"] += c["replies"] - c["checked"]
        if any(p["asked_by_reader"] for p in s["parts"]):
            t["false passes on rows where the reader wrote a gold check as asked"] += c["false passes"]
        for table, key in (("rows by parts", s["n_parts"]), ("rows by lang", s["lang"])):
            cell = by[table].setdefault(str(key), {"replies": 0, "checked": 0, "agree": 0, "false passes": 0, "built to fail": 0})
            for name in cell:
                cell[name] += c[name]
        for p in s["parts"]:
            t["gold checks"] += 1
            t["gold checks with their line"] += int(p["line"])
            t[f"gold checks written as `{p['as']}`" if p["as"] else "gold checks with no line of their span"] += 1
            t["gold checks asked by the reader"] += int(p["asked_by_reader"])
            t["gold checks asked by the host"] += int(p["asked_by_host"])
            t["parts right"] += int(p["right"])
            t["parts right without a run (the gold form)"] += int(p["same"])
            t["parts fixed by one reversal"] += int(p["fixed"])
            for k in BY_PART:
                bump(k, p[k], p["right"])
        for g in s["pairs"]:
            t["gold pairs"] += 1
            t["gold pairs right"] += int(g["right"])
            t["gold pairs right, unless and either as one"] += int(g["right_or"])
            bump("connective", g["connective"], g["right"])
            bump("pair's place", g["place"], g["right"])
        t["pairs made that are not gold"] += s["pairs_not_gold"]
        t["ids"] += s["ids"]
        t["seconds"] += s["seconds"]
    return {"counts": dict(sorted(t.items())), "by": {k: dict(sorted(v.items(), key=lambda kv: (len(kv[0]), kv[0]))) for k, v in by.items()}}


def rates(got: dict) -> dict:
    """The lines' rates from one split's counts."""
    c, by = Counter(got["counts"]), got["by"]
    side = {d: rate(*by["direction"][d]) for d in DIRECTIONS if d in by["direction"]}
    pos = by["position"]

    def span(lo: int, hi: int) -> list:
        cells = [pos[str(k)] for k in range(lo, hi + 1) if str(k) in pos]
        return [sum(x[0] for x in cells), sum(x[1] for x in cells)]
    early, late = span(1, 4), span(5, 8)
    return {"line 1: characters in no line and not flagged": c["characters in no line and not flagged"],
            "line 2: gold checks with their line": rate(c["gold checks with their line"], c["gold checks"]),
            "line 2: segments with span and kind": rate(c["gold segments with span and kind"], c["gold segments"]),
            "line 3: agreement": rate(c["agree"], c["replies checked"]),
            "line 3: false passes": rate(c["false passes"], c["replies checked"]),
            "line 3: false passes over replies built to fail": rate(c["false passes"], c["checked replies built to fail"]),
            "line 4: parts right at positions 1 to 4": rate(*early), "line 4: parts right at positions 5 to 8": rate(*late),
            "line 5: parts right": rate(c["parts right"], c["gold checks"]),
            "line 5: parts fixed by one reversal": rate(c["parts fixed by one reversal"], c["gold checks"]),
            "line 6: pairs right": rate(c["gold pairs right"], c["gold pairs"]),
            "line 6: pairs right, unless and either as one": rate(c["gold pairs right, unless and either as one"], c["gold pairs"]),
            "line 7: replies checked": rate(c["replies checked"], c["replies"]),
            "line 7: gold checks asked": rate(c["gold checks asked by the reader"] + c["gold checks asked by the host"], c["gold checks"]),
            "line 10: parts right by direction": side,
            "line 10: unweighted mean of the directions": round(sum(side.values()) / len(side), 4) if side else None,
            "seconds per instruction (where it was decoded)": rate(c["seconds"], c["rows"] - c["rows not read"]),
            "ids per answer": rate(c["ids"], c["rows"] - c["rows not read"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    from cubbyllm import local_paths
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.bridges import vm_identity
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073" / "rows.jsonl")
    rows = [json.loads(ln) for ln in open(rows_p, encoding="utf-8")]
    raw = Path(a.answers).read_bytes()
    answers = {d["id"]: d for d in map(json.loads, raw.decode("utf-8").splitlines())}
    splits = a.splits.split(",")
    mine = [d for d in answers.values() if d["split"] in splits]
    arms, variants = {d["arm"] for d in mine}, {bool(d.get("own_text")) for d in mine}
    if len(arms) != 1 or len(variants) != 1 or not arms <= {"m", "w"}:
        ap.error(f"the answers of {splits} are not one arm (m or w) in one variant: arms {sorted(arms)}, own_text {sorted(variants)}")
    arm, own_text = arms.pop(), variants.pop()
    vm = cc.CubelangSession()
    report = {"order": "0073", "what": "the reader's descriptions, compiled by the host and run on each row's replies",
              "arm": arm, "own_text": own_text, "answers": Path(a.answers).name, "answers_sha256": hashlib.sha256(raw).hexdigest(),
              "rows_sha256": hashlib.sha256(rows_p.read_bytes()).hexdigest(), "vm": vm_identity.identity(), "splits": {}}
    kept = []
    for split in splits:
        scored = [R.score(r, answers.get(r["id"]), arm, own_text, vm) for r in rows if r["split"] == split]
        got = tally(scored)
        report["splits"][split] = dict(got, rates=rates(got))
        kept += [row_counts(s) for s in scored]
        print(split, json.dumps(report["splits"][split]["rates"], indent=1), flush=True)
    if {"test", "test_long"} <= set(report["splits"]):
        a_in, a_long = (report["splits"][s]["rates"]["line 3: agreement"] for s in ("test", "test_long"))
        late, early = (report["splits"]["test_long"]["rates"][f"line 4: parts right at positions {k}"] for k in ("5 to 8", "1 to 4"))
        report["line 4"] = {"agreement on test_long over test": round(a_long / a_in, 4) if a_in and a_long is not None else None,
                            "parts right at 5 to 8 over 1 to 4, test_long": round(late / early, 4) if early and late is not None else None}
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.keep:
        Path(a.keep).write_text("".join(json.dumps(k) + "\n" for k in kept), encoding="utf-8")
    print("scored:", a.out, flush=True)


if __name__ == "__main__":
    main()
