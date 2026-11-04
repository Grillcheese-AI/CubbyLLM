"""r2_b1q -- order 0046, B1q (Cowork, 2026-11-04): B1, with the question's qualifier checked as a fact in each branch.

Span accounting (rule (a), written before seed 1 existed): each word of the question serves one role -- the name, the
relation, or a cue. The name (one span: the first occurrence of the longest name the row's items go by), the relation
lexicon's EN and FR wordings of the question's property, and
function words (R2's ``STOP``) are never cues. Runs break at the name and the relation; every span of 1 to 6 words in a
run that neither starts nor ends with a function word (so "canton of Bern" stays whole and "of" alone is no cue) is
looked up as a name in R1b's segment (the VM's names, exact), giving a set of value items; of those that name an
item, only the longest are the qualifier (a cue name inside a longer one is not checked: "terre" in "Terre-Neuve-et-
Labrador"). Both refinements were set on seed 0 (dev), after its first read, before seed 1 was read.

A branch (one candidate) holds the qualifier when one of its statements has one of those items as its value, in
either direction: candidate -> value (its own statements) or value -> candidate (the value item's statements, fetched
by id through the source's cache, for cue names that name at most ``IN_MAX`` items). The check runs in the VM: each
holding (candidate, cue) pair is a fact of the branch segment, keyed ``<candidate> ~ <cue words>``, and a branch
queries its cues there.

A branch stands when it verifies the relation (B1's check) and holds the qualifier. A question whose cue words name
no item has no qualifier to check: B1's outcome, counted apart ("no qualifier"). Without a relation (an
entity-linking row with none), a candidate stands on the qualifier alone: one standing is a pick, otherwise undecided.
The outcome table is B1's (``r2_vm.decide``). No margin, nothing ranked.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json

__wiring__ = "STANDALONE"

MAX_WORDS, IN_MAX = 6, 20


def relation_wordings(pids, lx: dict, langs=("en", "fr")) -> set[str]:
    """The frozen lexicon's wordings (folded) of these properties, in both languages."""
    want = set(pids or ())
    return {w for lg in langs for w, ps in lx["by_wording"].get(lg, {}).items() if want & set(ps)}


def cue_runs(question: str, mention, pids, lx: dict) -> list[str]:
    """Every run of 1 to MAX_WORDS cue words, as folded text (rule (a)). `mention`: the name as asked, or every name
    the row's items go by (a French question asks the French label)."""
    from retrieval_bench.r2_arms import STOP, fold
    toks = fold(question).split()
    role = [None] * len(toks)
    names = [mention] if isinstance(mention, str) else list(mention)
    # the name is ONE span: the first occurrence of the longest of its names ("Dhaka in Dhaka District": the second
    # "Dhaka" is the qualifier's); a relation wording takes every occurrence
    for words in sorted((fold(n).split() for n in names if fold(n)), key=lambda w: -len(w)):
        hit = next((i for i in range(len(toks) - len(words) + 1) if toks[i:i + len(words)] == words), None)
        if hit is not None:
            role[hit:hit + len(words)] = ["name"] * len(words)
            break
    for words in sorted((w.split() for w in relation_wordings(pids, lx)), key=lambda w: -len(w)):
        n = len(words)
        for i in range(len(toks) - n + 1):
            if n and toks[i:i + n] == words and all(r is None for r in role[i:i + n]):
                role[i:i + n] = ["relation"] * n
    runs, cur = [], []                                   # runs break at the name and the relation only, so a name
    for t, r in zip(toks, role):                         # with a function word inside it ("canton of Bern") stays whole
        if r is None:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    spans = {tuple(run[i:j]) for run in runs for i in range(len(run)) for j in range(i + 1, min(len(run), i + MAX_WORDS) + 1)}
    return sorted(" ".join(sp) for sp in spans if sp[0] not in STOP and sp[-1] not in STOP)   # never a function word at an edge


def longest(named: list[str]) -> list[str]:
    """The cue names that sit inside no longer cue name ("terre neuve et labrador", not "terre"): a qualifier is the
    longest span of the question that names an item."""
    def inside(a: str, b: str) -> bool:
        x, y = a.split(), b.split()
        return len(x) < len(y) and any(y[i:i + len(x)] == x for i in range(len(y) - len(x) + 1))
    return [c for c in named if not any(inside(c, o) for o in named)]


def holds(cand: str, profile: dict, values: set[str], claims_in: dict) -> bool:
    """One of the candidate's statements has one of `values` as its value, either way."""
    out_ids = {v for ids in (profile.get("value_ids") or {}).values() for v in ids}
    if out_ids & values:
        return True
    return any(cand in ids for v in values for ids in (claims_in.get(v) or {}).values())


def write_segment(path: str, rows: list[dict], profiles: dict, items_of: dict, claims_in: dict) -> int:
    """The qualifier facts: ``<candidate> ~ <cue>`` for every holding pair, appended to B1's segment at `path`."""
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        seen = set()
        for r in rows:
            for q in r.get("named") or ():
                for cue in longest([c for c in r.get("cues") or () if items_of.get(c)]):
                    vals = set(items_of.get(cue) or ())
                    if (q, cue) not in seen and vals and q in profiles and holds(q, profiles[q], vals, claims_in):
                        seen.add((q, cue))
                        f.write(json.dumps({"key": f"{q} ~ {cue}", "text": cue, "source": f"wikidata:{q}"}, ensure_ascii=False) + "\n")
                        n += 1
    return n


def b1q(session, kpath: str, named: list[str], pid: str | None, cues: list[str], items_of: dict) -> dict:
    """B1q over the candidates that bear the name."""
    from retrieval_bench import r2_vm as V
    checked = longest([c for c in cues if items_of.get(c)])
    if not checked:
        if not pid:
            return {"outcome": "undecided", "why": "no relation and no qualifier", "k": len(named), "qualifier": "none"}
        out = V.b1(session, kpath, named, pid)
        return {**out, "qualifier": "none"}
    bs, wall = [], 0.0
    for q in named:
        rel = V.branch(session, kpath, q, pid) if pid else {"item": q, "values": ["(no relation)"], "wall_ms": 0.0}
        wall += rel["wall_ms"] or 0
        qual = False
        for c in checked:
            b = V.branch(session, kpath, q, "~ " + c)                                 # the key "<q> ~ <cue>"
            wall += b["wall_ms"] or 0
            if b["values"]:
                qual = True
                break
        bs.append({"item": q, "values": rel["values"] if qual else [], "wall_ms": rel["wall_ms"]})
    if pid:
        out = V.decide(bs)
    else:
        stand = [b["item"] for b in bs if b["values"]]
        out = {"outcome": "pick", "item": stand[0], "by": "the only one the context's facts hold"} if len(stand) == 1 else \
            {"outcome": "undecided", "items": stand or named}
    return {**out, "k": len(named), "wall_ms": round(wall, 3), "qualifier": "checked"}


__all__ = ["b1q", "cue_runs", "holds", "longest", "relation_wordings", "write_segment"]
