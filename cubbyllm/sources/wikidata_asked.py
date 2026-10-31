"""cubbyllm/sources/wikidata_asked.py -- the name kept as asked at the source (order 0047, item 1;
``CB_HELDOUT_FIXES``, on by default; ``=0`` off).

0046's R0: the reader dropped a leading "The" ("The Last Dragon" -> "Last Dragon"), and the source's one-exact-hit tier
then resolved another item labelled exactly the shorter name (McDermott's *Last Dragon*, not De Mari's *The Last
Dragon*; Antiphon's *Interpretation of Dreams*, not Freud's). The loop now hands the source the name with the article
the asker wrote (``reasoning.heldout.asked_span``), and the source searches it first:

- an item labelled (or aliased) exactly as asked that carries the relation the walk needs: the candidates are those;
- items labelled exactly as asked exist, but none carries the relation: the shortened name may not win alone -- the
  question is refused as ambiguous, both kinds of hit named;
- no item bears the name as asked: the article was grammar ("the United Kingdom"), and resolution goes on with the
  name as the reader gave it, as before.

Wired: WIRED (``wikidata_resolve.WikiResolve.resolve``).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"

_FR = re.compile(r"^(?:l['’]|(?:le|la|les)\s)", re.I)


def _same(a: str, b: str) -> bool:
    k = lambda s: " ".join((s or "").replace("’", "'").casefold().split())  # noqa: E731
    return bool(a) and k(a) == k(b)


def asked_hits(src, asked: str, relations) -> tuple[str, list[dict]]:
    """("carry", hits) -- items named exactly as asked that carry the relation; ("lack", hits) -- items named as asked,
    none carrying it; ("none", []) -- no item bears the name as asked."""
    lang = "fr" if _FR.match(asked) else "en"            # a French article: the name is searched in French
    s = src._get({"action": "wbsearchentities", "search": asked, "language": lang, "uselang": lang, "limit": 20}) or {}
    hits = [h for h in s.get("search") or []
            if _same(h.get("label"), asked) or _same((h.get("match") or {}).get("text"), asked)]
    if not hits:
        return "none", []
    pids = [p for r in relations or () for p in (src.aliases.pids(r) if src.aliases is not None else [])]
    if not pids:
        return "carry", hits
    carry = [h for h in hits if any(p in src._claims(h["id"]) for p in pids)]
    return ("carry", carry) if carry else ("lack", hits)


__all__ = ["asked_hits"]
