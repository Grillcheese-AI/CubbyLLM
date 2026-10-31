"""sweep_name_edits -- order 0047, item 1's sweep: every transform of the host's reader (``reasoning/question_forms.py``,
``reasoning/everyday.py``) that edits the NAME span of a question, and how often each fired on R0's dev rows (the same
sample as ``r0_denominator.py``: 300 per set and language, ``retrieval_sets.common.sample``).

The transforms, by where they sit:
  forms.the            question_forms.Forms._match: a leading English "the " is cut from the span
  forms.as_of          question_forms.Forms._match: the year phrase ("in 2010") is taken out of the question first
  fr.article           everyday._fr: a French frame's "{de}"/"{en}" slot is written with its articles ("de la", "de l'",
                       "du", "des", "au", "aux", "à"), so the article is read as frame, not name
  everyday.tail        everyday.EverydayForms._match: a trailing "now", "today", "on a map", ... cut from the question
  everyday.noise       everyday.EverydayForms._match: a noise word cut from inside the span ("located", "based", ...)
  everyday.state       "<US state> state" -> "<US state>", and "state of <US state>" -> "<US state>"
  everyday.year_end    a bare year at the span's end, for a relation of the "state" aspect

Each row is matched once through a mirror of the two ``_match`` methods that records which step changed the span; the
mirror's entity is checked against ``EverydayForms.parse`` on every row (``mismatches`` in the log must be 0, or the
mirror is stale and the counts are not the reader's).

    python validation/retrieval_bench/sweep_name_edits.py [--n 300]
-> validation/logs/sweep_name_edits_<date>.json (counts and row ids only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import read_rows, sample  # noqa: E402

__wiring__ = "STANDALONE"

SETS = ("mintaka", "qald9plus", "webqsp_el", "graphq_el", "simplequestions_wd", "amber", "cronquestions")
_FR_ART = re.compile(r"(?:^|\s)(?:de la|de l'|du|des|au|aux|à)\s*\{S\}", re.I)


def edits(forms, question: str) -> tuple[list[str], str | None]:
    """(the transforms that changed the name span, the entity read), mirroring EverydayForms._match."""
    from cubbyllm.reasoning import everyday as ev
    from cubbyllm.reasoning.question_forms import AS_OF
    fired = []
    q = question
    while ev._TAIL.search(q):
        q, fired = ev._TAIL.sub("", q), fired + ["everyday.tail"]
    if AS_OF.search(q):
        fired.append("forms.as_of")
        q = " ".join(AS_OF.sub(" ", q).split())
    best, top = {}, None
    for n, rel, tense, rx, form in forms._compiled:
        if top is not None and n < top:
            break
        m = rx.match(q)
        if m and m["S"].strip():
            top = n
            best.setdefault(rel, (m["S"].strip().rstrip("?").strip(), tense, form))
    if len(best) != 1:
        return [], None
    rel, (raw, _tense, form) = next(iter(best.items()))
    ent = re.sub(r"^the\s+", "", raw, flags=re.I)
    if ent != raw:
        fired.append("forms.the")
    if _FR_ART.search(form):
        fired.append("fr.article")
    before = ent
    while ev._NOISE.search(ent):
        ent = ev._NOISE.sub("", ent)
    if ent != before:
        fired.append("everyday.noise")
    if ent.lower().endswith(" state") and ent[:-6].strip().lower() in ev._US:
        ent, fired = ent[:-6].strip(), fired + ["everyday.state"]
    if ent.lower().startswith("state of ") and ent[9:].strip().lower() in ev._US:
        ent, fired = ent[9:].strip(), fired + ["everyday.state"]
    y = ev._YEAR_END.search(ent)
    if y and ent[:y.start()].strip() and int(y.group(1)) <= ev._NOW and forms.aspect(rel) == "state":
        ent, fired = ent[:y.start()].strip(), fired + ["everyday.year_end"]
    if forms._nested is not None and forms._nested.match(ent):
        return [], None
    return (fired, ent) if ent.strip() else ([], None)


def main() -> None:
    from cubbyllm.reasoning.everyday import EverydayForms
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    forms = EverydayForms()
    count, ids, read, mism = collections.Counter(), collections.defaultdict(list), collections.Counter(), []
    rows = [r for name in SETS for lang in ("en", "fr") for r in sample(read_rows(name, split="dev", lang=lang), a.n)]
    for r in rows:
        fired, ent = edits(forms, r["question"])
        p = forms.parse(r["question"])
        if (p[0] if p else None) != ent:
            mism.append(r["id"])
        if ent is None:
            continue
        read[r["lang"]] += 1
        for f in dict.fromkeys(fired):
            count[(f, r["lang"])] += 1
            ids[f].append(r["id"])
    rep = {"what": "order 0047: the reader's transforms that edit a name span, on R0's dev sample",
           "date": _dt.date.today().isoformat(), "n_per_set_lang": a.n, "rows": len(rows), "read_by_the_table": dict(read),
           "fired": {f"{f}/{lang}": c for (f, lang), c in sorted(count.items())}, "mismatches": len(mism),
           "mismatch_ids": mism[:20], "ids": {f: v[:40] for f, v in ids.items()}}
    path = os.path.join(os.path.dirname(HERE), "logs", f"sweep_name_edits_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("rows", "read_by_the_table", "fired", "mismatches")}, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
