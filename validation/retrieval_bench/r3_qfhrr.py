"""r3_qfhrr -- order 0046, R3's qFHRR arm: the words left once the names are out, against a codebook of every property,
scored on grilly2 (``r3_qfhrr_score``, under grilly2's own interpreter; never grilly v1).

  codebook  one code per property and language: the words of its wordings in the frozen lexicon (label and aliases),
            function words out (R2's ``STOP``); a property with no wording in the row's language takes its English
            ones. Codes come from the serving word table (``cubbyllm.worlds.fast_words``, fastword v4), as R2's do.
  context   the words left in the question once the names are taken out, function words out
  read      the best property's rank-normalized margin over the next, (s1 - s2) / s1, at least ``tau`` (set on dev);
            the list is then the best 5 (``r3_arms.top``)

Wired: STANDALONE (R3's bench).
"""
from __future__ import annotations

import os

__wiring__ = "STANDALONE"

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "r3_qfhrr_score.py")
KEEP = 20


def words(text: str) -> list[str]:
    from retrieval_bench.r2_arms import STOP
    from retrieval_sets.common import fold
    return [w for w in fold(text).split() if w not in STOP]


def codebook(lx: dict, langs=("en", "fr")) -> dict[str, str]:
    """{"P17|fr": "pays ..."}: each property's wordings in a language, as the text its code is made from."""
    per: dict[str, dict[str, list]] = {lg: {} for lg in langs}
    for lg in langs:
        for wording, ps in lx["by_wording"].get(lg, {}).items():
            for p in ps:
                per[lg].setdefault(p, []).append(wording)
    out = {}
    for p in lx["datatype"]:
        for lg in langs:
            ws = words(" ".join(per[lg].get(p) or per["en"].get(p) or []))
            if ws:
                out[f"{p}|{lg}"] = " ".join(ws)
    return out


def job(rows: list[dict], lx: dict, table: str, cands: dict | None = None) -> dict:
    """The scorer's input: the codebook and each row's context (rows with no context word are left out); with `cands`
    ({row id: [P..]}, R3o) each row is scored against those properties only."""
    from retrieval_bench.r3_arms import leftover
    todo = []
    for r in rows:
        ctx = words(leftover(r["question"], r.get("spans")))
        if ctx:
            todo.append({"id": r["id"], "lang": r["lang"], "cues": " ".join(ctx),
                         **({"cands": sorted(cands.get(r["id"], ()), key=lambda p: int(p[1:]))} if cands is not None else {})})
    return {"table": table, "docs": codebook(lx), "rows": todo, "keep": KEEP}


def score(jb: dict) -> dict:
    from retrieval_bench.r2_qfhrr import score as run
    return run(jb, scorer=SCORER)


def decide(ranked: list, tau: float) -> dict:
    """``ranked``: [[property, score], ...], best first. Read only past the margin."""
    from retrieval_bench.r3_arms import top
    if len(ranked) < 2 or ranked[0][1] <= 0:
        return {"read": False, "top": [], "undecided": False}
    s1, s2 = ranked[0][1], ranked[1][1]
    margin = (s1 - s2) / s1
    if margin < tau:
        return {"read": False, "top": [], "undecided": False, "margin": round(margin, 4)}
    return {"read": True, **top({p: s for p, s in ranked}), "margin": round(margin, 4)}


__all__ = ["codebook", "decide", "job", "score", "words"]
