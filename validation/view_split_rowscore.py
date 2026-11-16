"""Order 0073, step 6: one row's read, scored. The definitions are the order's ("Step 6, the scoring stated before
any read"); this file is their code and nothing else decides a count.

`score(row, rec, arm, own_text, vm)` -> a dict for one row of `long_rows_0073` and its answer record (None: not
read):
- ``whole``: the answer ended ``finished`` and reads strictly; otherwise its whole lines are read loosely
  (`lines_of`) and what they do not hold is flagged;
- ``checked``: whole, nothing asked by the host's accounting, nothing flagged. Only then are its replies run; every
  reply of a row that is not checked has no verdict. A line the reader wrote ``asked`` is a sentence the host asks
  about (the gold rows have about one each): it leaves the row checked, also where the gold has a check there, and
  a reply that fails only that check is then a false pass (counted apart as well: ``asked_by_reader``);
- ``parts``: one entry a gold check: its line found (``line``), what the reader wrote at its span (``as``), asked by
  the reader or by the host, ``right`` (its description compiled and, run alone, the generator's verdict on every
  reply; a description equal to the gold folded form is right without a run: ``same``), ``fixed`` (not right, and
  one reversal of one slot gives the generator's verdicts; ``fix`` names that reversal), ``differs`` (the slots in
  which the description is not the gold's), and what line 10 counts it by;
- ``pairs``: one entry a gold pair: ``right`` (the same two checks, the same connective, the same roles),
  ``right_or`` ("unless" and "either" taken as the same); ``pairs_not_gold``: pairs the reader made that are not;
- ``replies``: one entry a reply: ``held`` (the generator's verdict) and ``verdict`` (the VM's on the reader's
  program; None when the row is not checked or the VM gave none);
- ``segments`` / ``segments_right``, ``flagged_chars``, ``unaccounted`` (characters in no line and not flagged:
  0 by construction, counted all the same).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation" / "builders")]
import view_draw as D  # noqa: E402
import view_said as S  # noqa: E402
import view_tell as T  # noqa: E402

from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_account as A  # noqa: E402
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402

__wiring__ = "STANDALONE"


def lines_of(rec: dict, row: dict, arm: str, own_text: bool) -> tuple[list, bool]:
    """(the answer's lines, whether it is whole)."""
    text, end = row["instruction"], rec.get("end", "finished")
    if end == "finished":
        got = V.parse(rec["answer"], text, arm, own_text=own_text)
        if "lines" in got:
            return got["lines"], True
    lines = rec["answer"].lstrip(" ").split("\n")[1:]
    if end != "finished":                                  # the last line may have been cut
        lines = lines[:-1]
    while lines:                                           # as many whole lines from the start as read
        got = V.parse("\n" + "\n".join(lines), text, arm, strict=False)
        if "lines" in got:
            return got["lines"], False
        lines = lines[:-1]
    return [], False


def run(vm, make) -> bool | None:
    """The VM's verdict on the program `make()` builds, or None when there is none (a view too long, no result)."""
    try:
        res = vm.run(make(), fn="solve").get("result")
    except Exception:                                      # noqa: BLE001  (RV.TooLong, or the VM refusing the program)
        return None
    return None if res is None else str(res).strip() in ("1", "1.0")


def attributes(row: dict, i: int) -> dict:
    """What line 10 counts gold check `i` by."""
    said, m = row["said"][i], row["said"][i]["m"]
    numbered = T.number(m) is not None

    def num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return {"direction": D.SIDE[m["op"]] if numbered else "no number", "nots": min(2, S.marks(said["w"])),
            "place": {"n": "size", "count": "count", "at": "position"}[m["q"]] if numbered else "no number",
            "filter number": any(num(c["v"]) for c in m.get("where", [])), "position": i + 1,
            "parts": row["n_parts"], "lang": row["lang"]}


SITE = {"n": "a view's size", "count": "a count compared", "at": "a position's value"}


def change(m: dict, x: dict) -> str:
    """The one slot in which `x`, a one-reversal mutant of `m`, is not `m`: its name, before -> after."""
    if m["q"] != x["q"]:
        return f"the quantifier: {m['q']} -> {x['q']}"
    if m.get("op") != x.get("op"):
        return f"{SITE[m['q']]}: {m['op']} -> {x['op']}"
    if m.get("absent") != x.get("absent"):
        return f"absent: {m.get('absent') or 'fail'} -> {x.get('absent') or 'fail'}"
    return next(f"a filter: {a['op']} -> {b['op']}" for a, b in zip(m["where"], x["where"]) if a != b)


def differs(m: dict, gold: dict) -> list[str]:
    """The slots in which a description (folded) is not the gold's."""
    out = [k for k in ("q", "view", "op", "v", "pos", "col", "val", "absent") if m.get(k) != gold.get(k)]
    a, b = m.get("where", []), gold.get("where", [])
    if len(a) != len(b):
        return out + ["the number of conditions"]
    return out + sorted({f"a condition's {k}" for x, y in zip(a, b) for k in ("col", "op", "v") if x.get(k) != y.get(k)})


def _part(row: dict, i: int, line: dict | None, check: dict | None, views: list, vm) -> dict:
    gold = row["said"][i]["m"]
    p = dict(attributes(row, i), part=i, line=bool(line and line["kind"] == "check"), **{"as": line["kind"] if line else None},
             asked_by_reader=bool(line and line["kind"] == "asked"), asked_by_host=False, right=False, same=False, fixed=False,
             fix=None, differs=None)
    if p["line"]:
        p["asked_by_host"], m = check["asked"] is not None, check["form"]
        if m is not None:
            want = [VS.evaluate(gold, v) for v in views]
            p["same"], p["differs"] = m == gold, differs(m, gold)
            p["right"] = p["same"] or [run(vm, lambda x=x: VC.program(m, x["text"])) for x in row["replies"]] == want
            fix = None if p["right"] else next((x for x in T.part_mutants(m) if [VS.evaluate(x, v) for v in views] == want), None)
            p["fixed"], p["fix"] = fix is not None, change(m, fix) if fix else None
    return p


def _pairs(row: dict, lines: list, acc: dict, gold_span) -> tuple[list, int]:
    def norm(kind, a, b) -> tuple:                         # an "either" has no roles: its two members in order
        return (kind, *sorted([a, b])) if kind == "either" else (kind, a, b)

    def same_or(a, b) -> bool:
        return a == b or ({a[0], b[0]} <= {"unless", "either"} and set(a[1:]) == set(b[1:]))
    span = {c["part"]: (lines[c["line"]]["start"], lines[c["line"]]["end"]) for c in acc["checks"]}
    mine = [norm(g[0], span[g[1]], span[g[2]]) for g in acc["pairs"]]
    gold = [(norm(g[0], gold_span(g[1]), gold_span(g[2])), min(g[1:]) + 1) for g in row["groups"] if g[0] != "one"]
    out = [{"connective": g[0], "place": place, "parts": row["n_parts"], "right": g in mine,
            "right_or": any(same_or(g, x) for x in mine)} for g, place in gold]
    return out, sum(1 for x in mine if x not in [g for g, _ in gold])


def score(row: dict, rec: dict | None, arm: str, own_text: bool, vm) -> dict:
    text = row["instruction"]
    lines, whole = lines_of(rec, row, arm, own_text) if rec else ([], False)
    acc = A.account(lines, text)
    by_span = {(x["start"], x["end"]): k for k, x in enumerate(lines)}
    of_line = {c["line"]: c for c in acc["checks"]}
    views = [RV.views(x["text"]) for x in row["replies"]]
    checks = {s["part"]: s for s in row["segments"] if s["kind"] == "check"}

    def gold_span(i: int) -> tuple[int, int]:
        return V.span(text, checks[i]["start"], checks[i]["end"])
    parts = []
    for i in range(len(checks)):
        k = by_span.get(gold_span(i))
        parts.append(_part(row, i, lines[k] if k is not None else None, of_line.get(k), views, vm))
    pairs, extra = _pairs(row, lines, acc, gold_span)
    checked = whole and not acc["asked"] and not acc["flagged"]
    replies = []
    for x in row["replies"]:
        verdict = None
        if checked:                                        # no check line at all: nothing is checked, so it passes
            verdict = True if acc["program"] is None else run(vm, lambda x=x: VJ.program(acc["program"], acc["parts"], x["text"]))
        replies.append({"held": x["held"], "verdict": verdict})
    held = [False] * len(text)
    for a, b in [(x["start"], x["end"]) for x in lines] + [tuple(f) for f in acc["flagged"]]:
        held[a:b] = [True] * (b - a)
    rec = rec or {}
    return {"id": row["id"], "split": row["split"], "lang": row["lang"], "n_parts": row["n_parts"], "read": bool(rec),
            "whole": whole, "checked": checked, "end": rec.get("end"), "ids": rec.get("ids", 0), "seconds": rec.get("seconds", 0.0),
            "parts": parts, "pairs": pairs, "pairs_not_gold": extra, "replies": replies,
            "asked_by_host": len(acc["asked"]), "flagged_chars": sum(b - a for a, b in acc["flagged"]),
            "unaccounted": sum(1 for k, ch in enumerate(text) if not held[k] and not ch.isspace()),
            "segments": len(row["segments"]),
            "segments_right": sum(1 for s in row["segments"]
                                  if (k := by_span.get(V.span(text, s["start"], s["end"]))) is not None and lines[k]["kind"] == s["kind"])}


__all__ = ["attributes", "change", "differs", "lines_of", "run", "score"]
