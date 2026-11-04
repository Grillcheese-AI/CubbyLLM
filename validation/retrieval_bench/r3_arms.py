"""r3_arms -- order 0046, R3's reading arms that run in this interpreter: today's grammar and the frozen lexicon; the list
every arm hands over (at most 5 properties, a run of equal scores kept whole); and how a row is judged.

  grammar   today's host reader as the stack mounts it (R0's pipeline): the everyday table restates the question to
            (relation, seed), and the relation's wording becomes properties through the serving property table
            (``sources.wikidata_aliases.PropertyAliases.pids``). A "who / what / where is X" question the table reads
            as a profile names no relation: counted as "profile", apart.
  lexicon   the frozen relation lexicon (``retrieval_sets.lexicon``, SHA-256 in ``frozen.json``) over the words left
            once the names are taken out: longest wording first, left to right, in the row's language; a property
            scores the length (in words) of its longest wording found. ``lexicon_content`` is the same, skipping
            wordings made only of function words (R2's ``STOP``: "in", "by", "de", "is a"), a rule over the lexicon
            written before any R3 row was read; both are reported.

The list: properties by score, a run of equal scores taken whole while the list stays within 5; a first run longer
than 5 leaves the row "undecided" (nothing tells those properties apart). Nothing orders properties by how common
they are.

Judged: ``hit`` (a gold property in the list), ``wrong`` (a list without one), ``undecided``, ``unread``.

Wired: STANDALONE (R3's bench).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

K = 5
MAX_WORDS = 6


def leftover(question: str, spans: list) -> str:
    """The question with the names' character spans taken out."""
    out, at = [], 0
    for s, e in sorted(spans or []):
        if s >= at:
            out.append(question[at:s])
            at = e
    out.append(question[at:])
    return " ".join(out)


def top(scores: dict[str, float], k: int = K) -> dict:
    """{"top": [...], "undecided": bool} -- runs of equal scores, whole, while the list stays within k."""
    runs: dict[float, list] = {}
    for p, s in scores.items():
        runs.setdefault(s, []).append(p)
    out = []
    for s in sorted(runs, reverse=True):
        run = sorted(runs[s], key=lambda p: int(p[1:]))
        if len(out) + len(run) > k:
            return {"top": out, "undecided": not out}
        out += run
    return {"top": out, "undecided": False}


def matches(text: str, by_wording: dict, content_only: bool = False) -> list[tuple[str, int, list]]:
    """(wording, words, properties) found in `text`, longest first, left to right, never overlapping."""
    from retrieval_bench.r2_arms import STOP
    from retrieval_sets.common import fold
    toks, i, out = fold(text).split(), 0, []
    while i < len(toks):
        for n in range(min(MAX_WORDS, len(toks) - i), 0, -1):
            w = toks[i:i + n]
            key = " ".join(w)
            if key in by_wording and not (content_only and all(x in STOP for x in w)):
                out.append((key, n, list(by_wording[key])))
                i += n
                break
        else:
            i += 1
    return out


def lexicon(text: str, lang: str, lx: dict, content_only: bool = False) -> dict:
    found = matches(text, lx["by_wording"].get(lang, {}), content_only)
    scores: dict[str, float] = {}
    for _w, n, ps in found:
        for p in ps:
            scores[p] = max(scores.get(p, 0), n)
    return {"read": bool(found), **top(scores), "wordings": [w for w, _n, _p in found]}


class Grammar:
    """Today's host reader (the everyday table) and the serving property table."""

    def __init__(self) -> None:
        from cubbyllm.reasoning.everyday import EverydayForms
        from cubbyllm.sources.wikidata_aliases import property_aliases
        self.forms, self.aliases = EverydayForms(), property_aliases()
        if self.aliases is None:
            raise SystemExit("the property table is not built (standin/data/out/wikidata_properties_en_fr.json)")

    def __call__(self, question: str) -> dict:
        from cubbyllm.reasoning.ask_forms import profile_ask
        r = self.forms.restate(question)
        if r is None:
            pa = profile_ask(question)
            return {"read": False, "top": [], "undecided": False, "profile": bool(pa)}
        rel = r[1]
        ps = self.aliases.pids(rel)
        return {"read": bool(ps), **top({p: 1.0 for p in ps}), "relation": rel}


def pids_of(wording: str, aliases) -> dict:
    """A program's relation wording through the serving property table, as a list (the 450M arm's)."""
    ps = aliases.pids(wording) if wording else []
    return {"read": bool(ps), **top({p: 1.0 for p in ps})}


def judge(out: dict | None, golds: list[str]) -> str:
    if not out or not out.get("read"):
        return "unread"
    if set(out.get("top") or ()) & set(golds):
        return "hit"
    return "undecided" if out.get("undecided") else "wrong"


__all__ = ["Grammar", "judge", "leftover", "lexicon", "matches", "pids_of", "top"]
