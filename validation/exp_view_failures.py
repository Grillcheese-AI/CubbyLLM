"""Order 0067, round 2: where the view reader's false passes come from, on dev rows only (Cowork, 2026-11-14 00:53).

For each answered dev row: its parts as `exp_view_second_chance.py` classes them (right as written, fixed by one
reversal, no single reversal, span not matched), its or / if groups against the gold's (the pairs a group ties, in the
gold's part numbers through the spans), and its whole program's verdict on each reply against the reply's ``held``.

Reported:
1. **the join:** rows whose parts are all right as written and whose program still disagrees on a checked reply; of
   those, how many tie another pair than the gold (or tie one where the gold ties none, or the reverse);
2. **where the false passes come from:** the false passes of rows with a one-reversal part, a part no single reversal
   fixes, an unmatched span, a wrong join, with the overlap (each combination of those four, its rows and false passes);
3. **the false-pass rate over the replies built to fail** (``held`` false), beside the rate over all checked replies.

  python validation/exp_view_failures.py --answers <answers.jsonl> [--rows <rows.jsonl>] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import view_reader as VR  # noqa: E402
import exp_view_second_chance as SC  # noqa: E402
from exp_view_reader import verdict  # noqa: E402

__wiring__ = "STANDALONE"
_GROUP = re.compile(r"if \(([pg]\d+) == ([01])\) \{ assign (g\d+) = 1; \} if \(([pg]\d+) == 1\) \{ assign \3 = 1; \}")


def groups(program: str, names: dict | None = None) -> set:
    """The or / if groups of a program's solve: {(first, "if" | "or", second)}, a group's own members named by the
    parts they hold (a nested group by its members); ``names`` maps the program's part numbers to the gold's."""
    names = names or {}
    held: dict = {}

    def name(x: str):
        if x.startswith("p"):
            return names.get(x[1:], "?" + x[1:])
        return held.get(x, x)
    out = set()
    for a, va, g, b in _GROUP.findall(program.split("public function solve", 1)[-1]):
        member = (name(a), "if" if va == "0" else "or", name(b))
        held[g] = member
        out.add(member)
    return out


def classify(vm, r: dict, got: dict) -> tuple[Counter, dict]:
    """(the row's part classes, my part number -> the gold's), as `exp_view_second_chance` classes them."""
    gold_prog, mine = r["join"], got["program"]
    gold_span = {(s["start"], s["end"]): str(s["part"] + 1) for s in r["segments"] if s["kind"] == "check"}
    gold_calls, my_calls = SC.calls(gold_prog), SC.calls(mine)
    c, names, seen = Counter(), {}, set()
    for seg in got["segments"]:
        if seg["kind"] != "check" or seg["part"] in seen:
            continue
        seen.add(seg["part"])
        mk, gk = str(seg["part"] + 1), gold_span.get((seg["start"], seg["end"]))
        if gk is None or mk not in my_calls:
            c["span"] += 1
            continue
        names[mk] = gk
        truth = SC.results(vm, gold_prog, gk, gold_calls[gk], r["replies"])
        if SC.results(vm, mine, mk, my_calls[mk], r["replies"]) == truth:
            c["right"] += 1
        elif any(SC.results(vm, v[4], mk, my_calls[mk], r["replies"]) == truth for v in SC.variants(mine, mk)):
            c["one reversal"] += 1
        else:
            c["no single reversal"] += 1
    return c, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067" / "rows.jsonl")
    rows = {r["id"]: r for r in (json.loads(ln) for ln in open(rows_p, encoding="utf-8"))}
    vm, t, combos = cc.CubelangSession(), Counter(), Counter()
    for ln in open(a.answers, encoding="utf-8"):
        k = json.loads(ln)
        r = rows[k["id"]]
        if r["split"] != "dev":
            raise SystemExit(f"{k['id']} is not a dev row: this count runs on dev only")
        t["rows"] += 1
        got = VR.parse(k["answer"], r["instruction"]) if "error" not in k else {"error": k["error"]}
        if "error" in got:
            t["answers not parsed"] += 1
            continue
        parts, names = classify(vm, r, got)
        if parts["span"]:                              # a part not matched to the gold's: its join is not compared
            t["joins not compared (a span unmatched)"] += 1
            join_wrong = False
        else:
            join_wrong = groups(got["program"], names) != groups(r["join"], {str(i + 1): str(i + 1) for i in range(20)})
        fp = disagree = checked = fail_checked = 0
        for rep in r["replies"]:
            v, _ = verdict(vm, got["program"], rep["text"])
            if v is None:
                continue
            checked += 1
            fail_checked += int(not rep["held"])
            disagree += int(v != rep["held"])
            fp += int(v and not rep["held"])
        t["checked"] += checked
        t["checked, built to fail"] += fail_checked
        t["false passes"] += fp
        all_right = set(parts) <= {"right"}
        if all_right and disagree:
            t["1. parts all right, the program still disagrees"] += 1
            t["1. ... of which a group ties another pair"] += int(join_wrong)
        flags = tuple(f for f, on in (("one reversal", parts["one reversal"]), ("no single reversal", parts["no single reversal"]),
                                      ("span", parts["span"]), ("join", join_wrong)) if on)
        key = " + ".join(flags) or "none of the four"
        combos[f"{key}: rows"] += 1
        combos[f"{key}: false passes"] += fp
        for f in flags:
            t[f"2. false passes in rows with: {f}"] += fp
    t3 = {"3. false passes / checked": f"{t['false passes']}/{t['checked']} = {t['false passes'] / max(1, t['checked']):.4f}",
          "3. false passes / checked built to fail": f"{t['false passes']}/{t['checked, built to fail']} = "
                                                     f"{t['false passes'] / max(1, t['checked, built to fail']):.4f}"}
    out = {"order": "0067", "what": "where the view reader's false passes come from (dev rows)",
           "answers": a.answers.replace("\\", "/").split("/")[-1], "vm": vm_identity.identity(), "counts": dict(t),
           "rates": t3, "2. by combination (rows; false passes)": dict(sorted(combos.items()))}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
