"""cubbyllm/sources/wikidata_place.py -- a name that carries where it is ("cleveland ohio", "denver co", "texas san
antonio", "london england"), read as a name and a place qualifier, as the hdc world reads "Paris, Texas" (order 0033).

Cowork's sorting of the scoreboard's 76 "couldn't find it" refusals (2026-10-24): 26 were names with a place left on
them, and no item carries "cleveland ohio" as its label or an alias.

- **A reading** splits the words into a name and a qualifier: the last one to three words, or the first one or two
  ("texas san antonio"). The qualifier's abbreviations are spelled out: a U.S. state's postal code, "usa", "uk".
- **The name's exact hits** (label or alias, meta-pages dropped) are kept when a place they are in names the
  qualifier. The places: located in (P131, up three steps), the country (P17), the continent (P30).
- **One item across every reading:** resolved. Several: refused, named, as any shared name is. None: no resolution.
- **Nothing is ranked,** and nothing is stripped blindly: "new england usa" reads "new england", in "usa".

It runs only when no item carries the whole name (``wikidata_resolve.WikiResolve.resolve``). Every place is a claim
the source states, so this is still the source reading its own values.

Wired: WIRED (``wikidata_resolve.WikiResolve.resolve``, on by default since Nick's word on 2026-10-24;
``CB_PLACE_QUALIFIER=0`` turns it off).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize
from .wikidata_aliases import _META
from .wikidata_resolve import name_key

__wiring__ = "WIRED"

PLACE = ("P131", "P17", "P30")                           # located in, the country, the continent
UP = 3                                                   # steps up "located in": city, county, state
STATES = {"al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
          "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
          "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
          "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
          "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
          "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
          "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
          "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
          "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
          "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"}
COUNTRIES = {"usa": "united states of america", "us": "united states of america", "united states": "united states of america",
             "uk": "united kingdom"}


def readings(entity: str) -> list[tuple[str, str]]:
    """(name, qualifier) readings, the qualifier last then first: "cleveland ohio" -> ("cleveland", "ohio")."""
    w = " ".join((entity or "").split()).split(" ")
    out = [(" ".join(w[:-k]), " ".join(w[-k:])) for k in (1, 2, 3) if len(w) > k]
    return out + [(" ".join(w[k:]), " ".join(w[:k])) for k in (1, 2) if len(w) > k]


def spelled(qualifier: str) -> str:
    q = normalize(qualifier)
    return STATES.get(q) or COUNTRIES.get(q) or q


def _ids(claims: dict, pid: str) -> list[str]:
    return [i for i in (((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
                        for st in claims.get(pid, [])) if i]


def places(src, qid: str) -> set[str]:
    """The names (normalized) of the places an item is in, as its claims state them."""
    ids, frontier = set(), [qid]
    for _ in range(UP):
        up = []
        for q in frontier:
            claims = src._claims(q)
            for pid in PLACE:
                for i in _ids(claims, pid):
                    if i not in ids:
                        ids.add(i)
                        if pid == "P131":
                            up.append(i)
        frontier = up
    return {normalize(lab) for lab in src._labels_for(sorted(ids)).values()}


def resolve_place(src, entity: str) -> tuple[str, str, str] | None:
    """(qid, label, how) when exactly one item of a reading's name is in the place its qualifier names; else None
    (several: `src.last["ambiguous"]` names them)."""
    found: dict[str, tuple[str, str, str]] = {}
    for name, qual in readings(entity):
        want = spelled(qual)
        s = src._get({"action": "wbsearchentities", "search": name, "language": "en", "limit": 5})
        key = name_key(name)
        exact = [h for h in (s or {}).get("search") or []
                 if (name_key(h.get("label", "")) == key or name_key(((h.get("match") or {}).get("text") or "")) == key)
                 and not _META.search(str(h.get("description") or ""))]
        for h in exact:
            if h["id"] not in found and want in places(src, h["id"]):
                found[h["id"]] = (h.get("label") or name, h.get("description") or "", f"place qualifier: {qual}")
    if len(found) == 1:
        qid, (label, _desc, how) = next(iter(found.items()))
        return qid, label, how
    if found:
        src.last["ambiguous"] = [(q, lab, desc) for q, (lab, desc, _how) in found.items()]
    return None


__all__ = ["places", "readings", "resolve_place", "spelled"]
