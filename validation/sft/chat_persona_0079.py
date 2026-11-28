"""chat_persona_0079 -- Cubby knows only the conversation (order 0079's final run, Cowork's note of 2026-11-28).

The drafting agent sees the user's persona, so Cubby's side can say what the user never said: "ta promenade le long de
l'Erdre", "a friend from church", "your home automation projects". `leak` refuses a reply that uses a word of the
persona the conversation has not had yet:
- a name or place (a capitalised word inside the persona's text, not opening a sentence), or
- a rare word of it: at least 5 letters, not a function word, in under `RARE` of the run's personas, and in under
  `GENERIC` of the replies Cubby wrote in other conversations of the first batch (`background`, frozen in the run's
  folder): "work", "movie" or "morning" never count; "automation", "apprentices", "Erdre" do.
A word counts as had when it or a form of it (`forms`) is in the user's turns, Cubby's earlier turns or the facts.

Wired: STANDALONE (`chat_gate_0079.gate`; `chat_step_0079` hands it the persona and the run's word counts).
"""
from __future__ import annotations

import collections
import re

__wiring__ = "STANDALONE"

RARE = 0.04
GENERIC = 0.01
FIELDS = ("occupation", "work", "hobby", "line")
_WORD = re.compile(r"[A-Za-zÀ-ÿœæ][\w'’\-]*")
FUNCTION = frozenset("about above after again against their there these those which while would could should "
                     "being where when what with from into over under other others every through because before "
                     "still while among between always often never really their theirs yourself people thing things "
                     "dans avec pour leurs cette comme aussi entre depuis avant après toujours souvent".split())


def _norm(w: str) -> str:
    w = w.lower().replace("’", "'")
    w = re.sub(r"^(?:l|d|qu|j|n|s|t|m|c)'", "", w)
    for suf in ("ies", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)] + ("y" if suf == "ies" else "")
    return w


def forms(w: str) -> set[str]:
    """The word and the forms it may be an inflection of: "baking" -> bake, "knitting" -> knit, "marche" -> marcher's
    march, "LEGO-like" -> lego and like, "séries" -> série."""
    w = re.sub(r"^(?:l|d|qu|j|n|s|t|m|c)'", "", w.lower().replace("’", "'"))
    out = {w}
    for part in w.split("-"):
        out.add(part)
        for suf, add in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                         ("er", ""), ("e", ""), ("ment", "")):
            if part.endswith(suf) and len(part) - len(suf) >= 3:
                base = part[:-len(suf)] + add
                out |= {base, base[:-1]} if len(base) > 3 and base[-1:] == base[-2:-1] else {base}
    return out


def persona_text(seed: dict) -> str:
    return " ".join(str(seed.get(k) or "") for k in FIELDS)


def counts(seeds: list[dict]) -> tuple[collections.Counter, int]:
    """In how many of the run's personas each word appears (normalised), and how many personas there are."""
    c = collections.Counter()
    for s in seeds:
        c.update({_norm(w) for w in _WORD.findall(persona_text(s))})
    return c, max(1, len(seeds))


def _names(text: str) -> set[str]:
    """Capitalised words inside a sentence (not opening one): names and places."""
    out = set()
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        for i, m in enumerate(_WORD.finditer(sent)):
            w = m.group(0)
            if i and w[:1].isupper() and len(w) >= 3:
                out.add(_norm(w))
    return out


def background(replies: list[str]) -> dict:
    """The share of Cubby's replies each word appears in: what Cubby says anyway, whoever the user is."""
    c = collections.Counter()
    for r in replies:
        c.update({_norm(w) for w in _WORD.findall(r or "")})
    n = max(1, len(replies))
    return {w: k / n for w, k in c.items() if k / n >= GENERIC}


def leak(reply: str, seen: list[str], seed: dict, df: tuple[collections.Counter, int], generic: dict | None = None) -> str | None:
    """Why the reply uses what only the persona says, or None."""
    text = persona_text(seed) + " " + " ".join(str(p) for p in seed.get("places") or ())
    had = {f for t in seen for w in _WORD.findall(t or "") for f in forms(w)}
    c, n = df
    names = _names(text) | {_norm(str(p)) for p in seed.get("places") or ()}
    rare = {_norm(w) for w in _WORD.findall(text) if len(w) >= 5 and w.lower() not in FUNCTION
            and c[_norm(w)] / n < RARE and _norm(w) not in (generic or {})}
    names -= set(generic or {})
    for m in _WORD.finditer(reply):
        w = _norm(m.group(0))
        if forms(m.group(0)) & had:
            continue
        if w in names or w in rare:
            return f"knows the persona: '{m.group(0)}' (the conversation has not had it)"
    return None
