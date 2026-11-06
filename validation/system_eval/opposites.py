"""validation/system_eval/opposites.py -- order 0054's addendum: question pairs that differ only by an opposite.

Word embeddings put antonyms close together, so a program can take the opposite direction and the VM certify the
wrong question: a certified wrong answer. Each pair asks the same item two ways whose golds differ, in English and
French, from Wikidata's own paired properties (no external model writes them):
  follows       "What came before X?" / "What came after X?"           P155 follows / P156 followed by (inverse)
  replaces      "Who did X replace?" / "Who replaced X?"                P1365 replaces / P1366 replaced by (before/after)
  start_end     "When did X start?" / "When did X end?"                 P580 start time / P582 end time
  founded       "When was X founded?" / "When was X dissolved?"         P571 inception / P576 dissolved (start/end)
  highest       "What is the highest point of X?" / "... lowest ..."    P610 highest point / P1589 lowest point
Not built: oldest/youngest, first/last, earliest/latest, largest/smallest. Wikidata states no such pair of properties
on one item and the stack has no superlative form, so their pairs would read coverage (all refused), not direction.

An item enters when: it has exactly one value of each property (deprecated left out), English and French names (its
own and its values'), the two golds differ (labels: neither inside the other; dates: different years, since the
grading takes a year as a date's match), and its English and French names each name it alone (``Wd.search_exact``),
so a refusal is never a namesake's. Candidates come from the Query Service in its own order, then are drawn by a hash
of the Q-id (no fame); PER_TYPE items a type.

    python validation/system_eval/opposites.py build      -> $CUBBY_DATA/system_eval/opposites.jsonl
Grading (``side_verdict``): right, opposite (the other side's gold, not this one's: a polarity error), unmatched,
refused; a pair is both_right, polarity (either side opposite), refused (both), or mixed.

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

PER_TYPE = 25
MONTHS_FR = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
MONTHS_EN = "January February March April May June July August September October November December".split()
TYPES = {
    "follows": ("P155", "P156", "item", ("What came before {x}?", "What came after {x}?"),
                ("Qu'est-ce qui précède {x} ?", "Qu'est-ce qui suit {x} ?")),
    "replaces": ("P1365", "P1366", "item", ("Who did {x} replace?", "Who replaced {x}?"),
                 ("Qui a été remplacé par {x} ?", "Qui a remplacé {x} ?")),
    "start_end": ("P580", "P582", "time", ("When did {x} start?", "When did {x} end?"),
                  ("Quelle est la date de début de {x} ?", "Quelle est la date de fin de {x} ?")),
    "founded": ("P571", "P576", "time", ("When was {x} founded?", "When was {x} dissolved?"),
                ("Quelle est la date de fondation de {x} ?", "Quelle est la date de dissolution de {x} ?")),
    "highest": ("P610", "P1589", "item", ("What is the highest point of {x}?", "What is the lowest point of {x}?"),
                ("Quel est le point culminant de {x} ?", "Quel est le point le plus bas de {x} ?")),
}


def out_path() -> str:
    from cubbyllm.local_paths import data_dir
    return str(data_dir("system_eval", "opposites.jsonl"))


def _label(e: dict, lang: str) -> str | None:
    return (((e.get("labels") or {}).get(lang) or {}).get("value")) or (((e.get("labels") or {}).get("mul") or {}).get("value"))


def _one(e: dict, pid: str) -> dict | None:
    sts = [s for s in (e.get("claims") or {}).get(pid) or [] if s.get("rank") != "deprecated"]
    best = [s for s in sts if s.get("rank") == "preferred"] or sts
    if len(best) != 1 or (best[0].get("mainsnak") or {}).get("snaktype") != "value":
        return None
    return best[0]["mainsnak"]["datavalue"]


def date_golds(dv: dict, lang: str) -> list[str] | None:
    """A date's golds: its text at its precision and its year; None below a year's precision or before year 1."""
    v = dv.get("value") or {}
    t, prec = str(v.get("time", "")), int(v.get("precision", 0))
    if prec < 9 or not t.startswith("+"):
        return None
    y, m, d = int(t[1:5]), int(t[6:8]), int(t[9:11])
    months = MONTHS_EN if lang == "en" else MONTHS_FR
    if prec >= 11 and m and d:
        full = f"{months[m - 1]} {d}, {y}" if lang == "en" else f"{d} {months[m - 1]} {y}"
    elif prec == 10 and m:
        full = f"{months[m - 1]} {y}"
    else:
        full = str(y)
    return list(dict.fromkeys([full, str(y)]))


def item_golds(e: dict, lang: str) -> list[str]:
    out = [_label(e, lang), _label(e, "en")]
    out += [a.get("value") for a in (e.get("aliases") or {}).get(lang) or []][:5]
    return [x for x in dict.fromkeys(out) if x]


def differ(ga: list[str], gb: list[str], kind: str) -> bool:
    from fact_store.audit_values import fold
    if kind == "time":
        return ga[-1] != gb[-1]                                          # the years
    a, b = fold(ga[0]), fold(gb[0])
    return bool(a and b) and a not in b and b not in a


def candidates(pa: str, pb: str, kind: str = "item") -> list[str]:
    """Items with both properties; for dates, both precise to a year or better (the Query Service's first answers
    for start and end time are geological periods, dated to 10,000 years or coarser)."""
    from retrieval_bench.r4_statements import _sparql
    prec = (f"?x p:{pa}/psv:{pa}/wikibase:timePrecision ?pa . ?x p:{pb}/psv:{pb}/wikibase:timePrecision ?pb . "
            f"FILTER(?pa >= 9 && ?pb >= 9) ") if kind == "time" else ""
    q = (f"SELECT DISTINCT ?x WHERE {{ ?x wdt:{pa} ?a ; wdt:{pb} ?b . {prec}"
         f"FILTER NOT EXISTS {{ ?x wdt:P31 wd:Q13442814 }} }} LIMIT 4000")
    rows = _sparql(q) or []
    ids = [r["x"]["value"].rsplit("/", 1)[1] for r in rows]
    return sorted(ids, key=lambda x: hashlib.sha256(f"0054opp:{x}".encode()).hexdigest())


def build_type(wd, kind_name: str, n: int = PER_TYPE) -> tuple[list[dict], dict]:
    pa, pb, kind, en, fr = TYPES[kind_name]
    rows, seen = [], {"candidates": 0, "checked": 0, "kept": 0}
    cands = candidates(pa, pb, kind)
    seen["candidates"] = len(cands)
    for i in range(0, len(cands), 20):
        if seen["kept"] >= n:
            break
        ents = wd.entities(cands[i:i + 20], props="labels|aliases|claims", langs="en|fr|mul")
        for x in cands[i:i + 20]:
            if seen["kept"] >= n or x not in ents:
                continue
            seen["checked"] += 1
            e = ents[x]
            da, db = _one(e, pa), _one(e, pb)
            names = {lg: _label(e, lg) for lg in ("en", "fr")}
            if not da or not db or not all(names.values()):
                continue
            golds = {}
            if kind == "time":
                if da.get("type") != "time" or db.get("type") != "time":
                    continue
                golds = {lg: (date_golds(da, lg), date_golds(db, lg)) for lg in ("en", "fr")}
            else:
                ia, ib = (da.get("value") or {}).get("id"), (db.get("value") or {}).get("id")
                if not ia or not ib or ia == ib:
                    continue
                ve = wd.entities([ia, ib], props="labels|aliases", langs="en|fr|mul")
                if ia not in ve or ib not in ve or not all(_label(ve[q], lg) for q in (ia, ib) for lg in ("en", "fr")):
                    continue
                golds = {lg: (item_golds(ve[ia], lg), item_golds(ve[ib], lg)) for lg in ("en", "fr")}
            if not all(a and b and differ(a, b, kind) for a, b in golds.values()):
                continue
            if any(wd.search_exact(names[lg], lg)[0] != [x] for lg in ("en", "fr")):
                continue                                                 # a name that names another item too
            seen["kept"] += 1
            for lg, tpl in (("en", en), ("fr", fr)):
                a, b = golds[lg]
                for side, (q, g, other) in (("a", (tpl[0], a, b)), ("b", (tpl[1], b, a))):
                    rows.append({"id": f"opp:{kind_name}:{x}:{lg}:{side}", "bench": "opposites",
                                 "question": q.format(x=names[lg]), "golds": g, "seen": False,
                                 "meta": {"type": kind_name, "pair": f"{kind_name}:{x}:{lg}", "lang": lg,
                                          "side": side, "item": x, "pid": pa if side == "a" else pb,
                                          "other_golds": other}})
    return rows, seen


def build(path: str | None = None) -> dict:
    from fact_store.audit_wd import Wd
    wd, rows, report = Wd(), [], {}
    for t in TYPES:
        r, seen = build_type(wd, t)
        rows += r
        report[t] = seen
        print(t, seen, flush=True)
    path = path or out_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["pairs_per_language"] = sum(s["kept"] for s in report.values() if isinstance(s, dict))
    report["rows"] = len(rows)
    return report


def side_verdict(spoken: bool, answer, golds: list[str], other: list[str], declines: bool = False) -> str:
    """right | opposite | unmatched | refused (M's decline in text, when ``declines``, counted as refused)."""
    from system_eval.grade import declined, matches
    if not spoken or not (answer or "").strip() or (declines and declined(answer)):
        return "refused"
    if matches(answer, golds, "opposites"):
        return "right"
    return "opposite" if matches(answer, other, "opposites") else "unmatched"


def pair_verdict(a: str, b: str) -> str:
    if "opposite" in (a, b):
        return "polarity"
    if a == b == "right":
        return "both_right"
    return "refused" if a == b == "refused" else "mixed"


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(json.dumps(build(), indent=1))
