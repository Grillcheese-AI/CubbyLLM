"""r3_gold -- order 0046, R3's rows and their gold properties, under rules written before any R3 row was read.

Sets: SimpleQuestions-Wikidata (EN), QALD-9-plus (EN, FR), Mintaka (EN, FR); ``--n`` rows a set and language, drawn by
seed 0 (``common.sample``), as R0 draws them.

The gold item gives each arm one thing only: where the name is (its words are taken out of the question; the 450M sees
them as ``[E1: ...]``). No arm reads the item's facts. Names a set does not mark are located with the gold item's own
names (``mentions.fill``, as R0 and R1 locate them).

Gold properties:
  simplequestions_wd  the set's own property (``R`` rows, the inverse direction, are flagged ``inverse``)
  qald9plus           the SPARQL's properties without P31 / P279 (a class constraint, not the relation asked); scored
                      only when exactly one is left, else "several relations" (or "no relation"), counted apart
  mintaka             Mintaka gives answers, not properties: the properties that link a question item to an answer
                      item in today's Wikidata, either way (item -p-> answer, or answer -p-> item); a row with none is
                      "no link", counted apart. Several links are several golds: a hit is any of them in the top 5.

Wired: STANDALONE (R3's bench).
"""
from __future__ import annotations

import json
import os

__wiring__ = "STANDALONE"

SETS = (("simplequestions_wd", "en"), ("qald9plus", "en"), ("qald9plus", "fr"), ("mintaka", "en"), ("mintaka", "fr"))
CLASS = {"P31", "P279"}


def rows(split: str, n: int) -> list[dict]:
    from retrieval_sets.common import read_rows, sample
    out = []
    for name, lang in SETS:
        out += sample(read_rows(name, split=split, lang=lang), n)
    return out


class ClaimIds:
    """{item: {property: [item values]}}, fetched once per item through the source's cache, kept off-repo."""

    def __init__(self, src, path: str | None = None) -> None:
        from retrieval_sets.common import data_root
        self.src, self.path = src, path or os.path.join(data_root(), "r3_claim_ids.jsonl")
        self.c: dict[str, dict] = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    q, cl = json.loads(line)
                    self.c[q] = cl

    def get(self, qids) -> dict[str, dict]:
        from retrieval_sets.common import entities
        need = [q for q in dict.fromkeys(qids) if q and q not in self.c]
        for i in range(0, len(need), 50):
            chunk = need[i:i + 50]
            got = entities(self.src, chunk, props="claims", langs="en")
            with open(self.path, "a", encoding="utf-8") as f:
                for q in chunk:
                    cl = {p: ids for p, sts in ((got.get(q) or {}).get("claims") or {}).items() if (ids := _ids(sts))}
                    self.c[q] = cl
                    f.write(json.dumps([q, cl]) + "\n")
        return {q: self.c.get(q, {}) for q in qids}


def _ids(statements: list) -> list[str]:
    out = []
    for st in statements:
        dv = (st.get("mainsnak") or {}).get("datavalue") or {}
        if dv.get("type") == "wikibase-entityid" and (dv.get("value") or {}).get("id"):
            out.append(dv["value"]["id"])
    return out


def links(items: list[str], answers: list[str], claims: dict) -> list[str]:
    """The properties linking an item to an answer, either way."""
    out = set()
    for m in items:
        for a in answers:
            out |= {p for p, vs in (claims.get(m) or {}).items() if a in vs}
            out |= {p for p, vs in (claims.get(a) or {}).items() if m in vs}
    return sorted(out, key=lambda p: int(p[1:]))


def gold(row: dict, claims: dict | None = None) -> tuple[list[str], str]:
    """(gold properties, why): why is "gold" when the row is scored, else the reason it is not."""
    s = row["set"]
    if s == "simplequestions_wd":
        return ([row["prop"]], "gold") if row.get("prop") else ([], "no relation")
    if s == "qald9plus":
        ps = [p for p in (row.get("extra") or {}).get("props") or [] if p not in CLASS]
        return (ps, "gold") if len(ps) == 1 else ([], "several relations" if ps else "no relation")
    if s == "mintaka":
        ls = links([m["qid"] for m in row["mentions"]], row.get("answer_qids") or [], claims or {})
        return (ls, "gold") if ls else ([], "no link")
    return [], "not an R3 set"


def prepare(rows_: list[dict], src) -> dict:
    """Mentions located in place, each row's ``golds`` and ``why`` set; returns the counts."""
    from retrieval_bench.mentions import Gold, fill
    c = {"mentions": fill(rows_, Gold(src))}
    mk = [r for r in rows_ if r["set"] == "mintaka"]
    claims = ClaimIds(src).get([q for r in mk for q in [m["qid"] for m in r["mentions"]] + (r.get("answer_qids") or [])]) if mk else {}
    for r in rows_:
        r["golds"], r["why"] = gold(r, claims)
        r["spans"] = sorted(m["span"] for m in r["mentions"] if m.get("span"))
    c["why"] = {}
    for r in rows_:
        k = f"{r['set']}/{r['lang']}"
        c["why"].setdefault(k, {}).setdefault(r["why"], 0)
        c["why"][k][r["why"]] += 1
    return c


__all__ = ["CLASS", "ClaimIds", "SETS", "gold", "links", "prepare", "rows"]
