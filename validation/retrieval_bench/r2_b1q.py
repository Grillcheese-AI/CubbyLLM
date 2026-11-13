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

A branch stands when it verifies the relation (B1's check) and holds the qualifier. Two rules from Cowork's check of seed 0 (2026-11-04
afternoon), written before seed 1 is read:
  short names   a cue of 3 characters or fewer (a code, an abbreviation) names only the items with a label or alias
                equal to it as the question writes it, case and accents kept ("né" never names "NE")
  relation      span accounting also counts the words the relation reading used: on the fame set, the template's own
                words (``relation_words``; function words stay function words)
  located in    a branch whose own statements do not hold the qualifier walks `located in` (P131) in the VM, one hop
                a query of the branch segment, up to a country (``MAX_HOPS``); reaching a cue's item holds it
  compared      (Cowork's check of the rerun, 2026-11-04 night) two names joined by a comparison or a choice are each a
                name of their own, and neither is a cue for the other: across EN "than", "versus", "vs", "compared
                with", "or" between two names, FR "que" after "plus" / "moins" / "aussi", "par rapport à", "ou" (not
                "où") between two names, the words written with a capital on the far side from the row's name are no
                cue. From the question's own words only, never a set's gold links. ("Does California have more electoral
                votes than Rhode Island?": California is no qualifier of Rhode Island.)
A question whose cue words name
no item has no qualifier to check: B1's outcome, counted apart ("no qualifier"). Without a relation (an
entity-linking row with none), a candidate stands on the qualifier alone: one standing is a pick, otherwise undecided.
The outcome table is B1's (``r2_vm.decide``). No margin, nothing ranked.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json

__wiring__ = "STANDALONE"

MAX_WORDS, IN_MAX, SHORT, MAX_HOPS = 6, 20, 3, 8
# order 0069, B: how "located in" counts. "walk" (as built, 0046): a chain reaching the cue makes a candidate hold it.
# "veto": a candidate holds the cue only by a statement of its own; a chain reaching the cue for ANOTHER candidate
# turns a pick into undecided, and a chain never makes a pick. Set by `r2_choosing --b1q-walk`.
WALK = "veto"                                            # (0069 step D, rule 1: the veto replaces the walk)
# order 0069 step D's rules 2 and 3, each measured on dev before it is kept (`r2_choosing --b1q-rules`):
#   "namesake_cue"  a cue span whose words name more than one item cannot make a candidate stand; where only such a cue
#                   held, the row is undecided; B1q never chooses among a cue's items
#   "article"       a span may open with the article that belongs to a name ("la jolla", "the hague"): tried as a
#                   name first; the article is stripped only where the name with it is not in the index (``longest``)
#   "namesake_cue_class"  (2b) as "namesake_cue", except that a candidate holding the cue's item through a class
#                   statement (instance of, subclass of, occupation) still stands on it
RULES: set = set()
CLASS_PIDS = ("P31", "P279", "P106")
ARTICLES = {"the", "le", "la", "les", "l"}


def relation_wordings(pids, lx: dict, langs=("en", "fr")) -> set[str]:
    """The frozen lexicon's wordings (folded) of these properties, in both languages."""
    want = set(pids or ())
    return {w for lg in langs for w, ps in lx["by_wording"].get(lg, {}).items() if want & set(ps)}


def cue_runs(question: str, mention, pids, lx: dict, relation_words=()) -> list[str]:
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
    compared(question, toks, role)
    from retrieval_bench.r2_arms import STOP as _S
    for i, t in enumerate(toks):                         # the relation reading's own words (a template's), not function words
        if role[i] is None and t in set(relation_words) and t not in _S:
            role[i] = "relation"
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
    lead = ARTICLES if "article" in RULES else set()      # rule 3: a leading article, kept for a name to be tried with it
    return sorted(" ".join(sp) for sp in spans if (sp[0] not in STOP or (sp[0] in lead and len(sp) > 1)) and sp[-1] not in STOP)


def _words(question: str, case: bool, accents: bool) -> list[str]:
    import re
    import unicodedata
    s = unicodedata.normalize("NFC" if accents else "NFKD", question or "")
    if not accents:
        s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", s if case else s.lower()).split()


def _connectors(toks: list[str], cased: list[str], accented: list[str]) -> list[tuple[int, int]]:
    """(start, end) of each comparison or choice word in the folded question (rule "compared")."""
    from retrieval_bench.r2_arms import STOP

    def cap(i, step):                                    # the nearest word that is not a function word, capitalised
        while 0 <= i < len(toks) and toks[i] in STOP:
            i += step
        return 0 <= i < len(toks) and cased[i][:1].isupper()
    out = []
    for i, t in enumerate(toks):
        if t in ("than", "versus", "vs"):
            out.append((i, i + 1))
        elif toks[i:i + 2] == ["compared", "with"] or toks[i:i + 3] == ["par", "rapport", "a"]:
            out.append((i, i + (2 if t == "compared" else 3)))
        elif t == "que" and any(w in ("plus", "moins", "aussi") for w in toks[:i]):
            out.append((i, i + 1))
        elif (t == "or" or (t == "ou" and accented[i] == "ou")) and cap(i - 1, -1) and cap(i + 1, 1):
            out.append((i, i + 1))
    return out


def compared(question: str, toks: list[str], role: list) -> None:
    """Rule "compared": the capitalised words across a comparison or a choice from the name get role "compared"."""
    cased, accented = _words(question, True, False), _words(question, False, True)
    if len(cased) != len(toks) or len(accented) != len(toks) or "name" not in role:
        return                                           # words not aligned, or no name found: the rule cannot place it
    a = role.index("name")
    b = a + next((k for k, r in enumerate(role[a:]) if r != "name"), len(role) - a)
    for c, d in _connectors(toks, cased, accented):
        far = range(d, len(toks)) if b <= c else range(0, c) if a >= d else ()
        for i in far:
            if role[i] is None and cased[i][:1].isupper():
                role[i] = "compared"


def written(question: str, cue: str) -> str | None:
    """The cue as the question writes it (case and accents kept), or None."""
    import re
    import unicodedata
    from retrieval_bench.r2_arms import fold
    toks = re.findall(r"\w+", unicodedata.normalize("NFC", question or ""))
    want = cue.split()
    folded = [fold(t) for t in toks]
    for i in range(len(toks) - len(want) + 1):
        if folded[i:i + len(want)] == want:
            return " ".join(toks[i:i + len(want)])
    return None


def short_ok(question: str, cue: str, items: list[str], names: dict) -> list[str]:
    """Rule 1: a short cue keeps only the items named exactly as written; a longer cue keeps them all."""
    if len(cue) > SHORT:
        return list(items)
    w = written(question, cue)
    return [q for q in items if w and w in (names.get(q) or ())]


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


def write_segment(path: str, rows: list[dict], profiles: dict, items_of_all: dict, claims_in: dict) -> int:
    """The qualifier facts: ``<candidate> ~ <cue>`` for every holding pair, appended to B1's segment at `path`."""
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        seen = set()
        for r in rows:
            items_of = r.get("cue_items", items_of_all)
            for q in r.get("named") or ():
                for cue in longest([c for c in r.get("cues") or () if items_of.get(c)]):
                    vals = set(items_of.get(cue) or ())
                    if (q, cue) not in seen and vals and q in profiles and holds(q, profiles[q], vals, claims_in):
                        seen.add((q, cue))
                        f.write(json.dumps({"key": f"{q} ~ {cue}", "text": cue, "source": f"wikidata:{q}"}, ensure_ascii=False) + "\n")
                        n += 1
                    cls = {v for p in CLASS_PIDS for v in ((profiles.get(q) or {}).get("value_ids") or {}).get(p) or ()}
                    if (q, "class", cue) not in seen and len(vals) > 1 and cls & vals:     # rule 2b's facts
                        seen.add((q, "class", cue))
                        f.write(json.dumps({"key": f"{q} ~class {cue}", "text": cue, "source": f"wikidata:{q}"}, ensure_ascii=False) + "\n")
                        n += 1
    return n


def _walk(session, kpath: str, q: str, want: set) -> tuple[bool, float]:
    """Along `located in`, one VM hop at a time, up to a country: (a chain from `q` reaches one of `want`, wall ms)."""
    from retrieval_bench import r2_vm as V
    seen, front, wall = {q}, [q], 0.0
    for _ in range(MAX_HOPS):
        nxt = []
        for x in front:
            b = V.branch(session, kpath, x, "P131")
            wall += b["wall_ms"] or 0
            nxt += [v for v in b["values"] if v not in seen]
        if want & set(nxt):
            return True, wall
        seen |= set(nxt)
        front = nxt
        if not front:
            break
    return False, wall


def b1q(session, kpath: str, named: list[str], pid: str | None, cues: list[str], items_of: dict) -> dict:
    """B1q over the candidates that bear the name (``WALK``: how "located in" counts)."""
    from retrieval_bench import r2_vm as V
    checked = longest([c for c in cues if items_of.get(c)])
    if not checked:
        if not pid:
            return {"outcome": "undecided", "why": "no relation and no qualifier", "k": len(named), "qualifier": "none"}
        out = V.b1(session, kpath, named, pid)
        return {**out, "qualifier": "none"}
    two_b = "namesake_cue_class" in RULES
    several = {c for c in checked if len(items_of.get(c) or ()) > 1} if ("namesake_cue" in RULES or two_b) else set()
    usable = [c for c in checked if c not in several]   # rule 2: a namesake cue never makes a candidate stand
    bs, wall, held_by_several = [], 0.0, False
    for q in named:
        rel = V.branch(session, kpath, q, pid) if pid else {"item": q, "values": ["(no relation)"], "wall_ms": 0.0}
        wall += rel["wall_ms"] or 0
        qual = False
        for c in usable:
            b = V.branch(session, kpath, q, "~ " + c)                                 # the key "<q> ~ <cue>"
            wall += b["wall_ms"] or 0
            if b["values"]:
                qual = True
                break
        if not qual and two_b:                           # rule 2b: a namesake cue held through a class statement
            for c in several:
                b = V.branch(session, kpath, q, "~class " + c)
                wall += b["wall_ms"] or 0
                if b["values"]:
                    qual = True
                    break
        if not qual and several:                         # (only to say why, below: the namesake cue held)
            for c in several:
                if V.branch(session, kpath, q, "~ " + c)["values"]:
                    held_by_several = True
                    break
        want = {v for c in usable for v in items_of.get(c) or ()}
        if not qual and WALK == "walk":                  # the chain makes the candidate hold the cue
            qual, w = _walk(session, kpath, q, want)
            wall += w
        bs.append({"item": q, "values": rel["values"] if qual else [], "wall_ms": rel["wall_ms"]})
    if pid:
        out = V.decide(bs)
    else:
        stand = [b["item"] for b in bs if b["values"]]
        out = {"outcome": "pick", "item": stand[0], "by": "the only one the context's facts hold"} if len(stand) == 1 else \
            {"outcome": "undecided", "items": stand or named}
    if held_by_several and out.get("outcome") != "pick":   # rule 2: where only a namesake cue held, undecided
        out = {"outcome": "undecided", "items": named, "why": "the cue names several items"}
    if WALK == "veto" and out.get("outcome") == "pick":  # another candidate reached through a chain: no pick
        want = {v for c in usable for v in items_of.get(c) or ()}
        for q in named:
            if q != out["item"]:
                hit, w = _walk(session, kpath, q, want)
                wall += w
                if hit:
                    out = {"outcome": "undecided", "items": [out["item"], q], "why": "a chain reaches the cue for another candidate"}
                    break
    return {**out, "k": len(named), "wall_ms": round(wall, 3), "qualifier": "checked"}


__all__ = ["b1q", "cue_runs", "holds", "longest", "relation_wordings", "short_ok", "write_segment", "written"]
