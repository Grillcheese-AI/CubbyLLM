"""cubbyllm/worlds/encyclopedia_text.py -- an OCR'd volume's text: cleaned, sentences found, headwords and names
read, the entries cut. Split out of encyclopedia.py (order 0031), verbatim; encyclopedia.py re-exports every name.

Wired: WIRED (the encyclopedia source).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


# -- entries ---------------------------------------------------------------------------------------
# a headword: one to five all-caps words at a line start or after a sentence end, then a comma
_CAPS = r"[A-Z][A-Z'’.-]+(?:[ -][A-Z][A-Z'’.-]+){0,4}"


_HEAD = re.compile(rf"(?:^|(?<=\n)|(?<=\. )|(?<=\.\n))(?P<head>{_CAPS}),\s")


# a page's running head ('GOMBERT-GOMEZ GOMBERT', 'GOSHO HEINOSUKE-GOSNOLD GOSHO HEINOSUKE') is
# the page's first-last range followed by the entry's own headword, which repeats the range's start
_RUNNING = re.compile(r"^(?P<a>.+?)-(?P<b>[A-Z'’. ]+?) (?P=a)$")


_REPEATED = re.compile(r"^(?P<a>.+?)-? (?P=a)$")                 # 'GORDON GORDON', 'PEREZ GALDOS- PEREZ GALDOS': the running head, then the entry's own


_INITIALS = re.compile(r"^(?:[A-Z]\.\s?)+$")                    # 'H. M.' is a cross-reference, not an entry


_LEADING_INITIALS = re.compile(r"^(?:[A-Z]\.\s?)+")             # 'N. Y. GOTTSCHED': a byline's state ran into the headword


# the headword's own line: [pronunciation,] given names [, pronunciation] (BIRTH-DEATH)
_YEARS = re.compile(r"^(?P<mid>(?:[^().]|Jr\.|Sr\.|St\.){0,90}?)\((?P<b>\d{4})\s*[-–]\s*(?P<d>\d{4})?\s*\)")


_PARTICLES = frozenset("von van de da del della di du la le der den ten y of the af al el ibn bin und zu".split())


# an apostrophe inside a name precedes a capital ('O'Brien', 'D'Annunzio'); inside a pronunciation
# it is a stress mark before a lowercase letter ('gu'ta', 'Ibr'ing' -- the OCR's capital I for l)
_GIVEN_TOKEN = re.compile(r"^[A-Z](?:[a-zÀ-ſ-]|['’][A-Z])*[a-zÀ-ſ]\.?$|^[A-Z]\.$")


OPENING = 1500


_TAIL = 40                                                       # a match ending within the opening's last chars is cut off ('Castle Altenb')


_LINE_HYPHEN = re.compile(r"(?<=\w)- (?=[a-z])")                 # 'Lon- don', 'Ethi- opia', 'un- til': the OCR kept the line break


_SENTENCE_END = re.compile(r"(?<=[a-z)])\. (?=[A-Z])")            # not after 'Oct.' or 'Aug.' (a digit follows)


def clean(text: str) -> str:
    return _LINE_HYPHEN.sub("", " ".join(text.split()))


def sentence_at(text: str, start: int, end: int) -> str:
    """The sentence of `text` that contains [start, end)."""
    s = 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() <= start:
            s = m.end()
        elif m.start() >= end:
            return text[s:m.start() + 1].strip()
    return text[s:].strip()


def headword(raw: str) -> str | None:
    """The entry's headword from what the pattern matched, running heads and cross-reference
    initials removed; None when it is not an entry."""
    h = " ".join(raw.split())
    if _INITIALS.match(h):
        return None
    h = _LEADING_INITIALS.sub("", h).strip()
    m = _RUNNING.match(h) or _REPEATED.match(h)
    if m:
        h = m.group("a")
    if len(h) < 2:
        return None
    return h


def surname(head: str) -> str:
    """'GONGORA Y ARGOTE' -> 'Gongora y Argote'; 'O'BRIEN' -> 'O'Brien'; 'JEAN-PAUL' -> 'Jean-Paul'."""
    out = []
    for w in head.split():
        if w.lower() in _PARTICLES:
            out.append(w.lower())
        elif "'" in w and len(w) > 2 and w.index("'") <= 2:
            i = w.index("'"); out.append(w[:i + 1].capitalize() + w[i + 1:].capitalize())
        else:
            out.append("-".join(p.capitalize() for p in w.split("-")))
    return " ".join(out)


def given_names(mid: str) -> str | None:
    """The comma part of the headword line whose tokens are all capitalised words (or
    particles); the pronunciation ('gu'ta', 'gin'is', 'go-sho') is lowercase and is skipped.
    None when no part qualifies (an OCR wreck)."""
    for part in mid.split(","):
        toks = part.split()
        if not toks or not any(_GIVEN_TOKEN.match(t) and len(t) > 2 for t in toks):
            continue
        if all(_GIVEN_TOKEN.match(t) or t.lower() in _PARTICLES for t in toks):
            toks = [t for t in toks if t.rstrip(".").lower() not in ("jr", "sr")]
            if sum(t.lower() not in _PARTICLES for t in toks) > 3:
                return None                                    # 'Ulysses Simpson Historical Collection': a caption ran into the line
            return " ".join(toks)
    return None


def entries(text: str):
    """(headword, start, body_start, end) for every entry the pattern finds -- `body_start`
    is just after the headword's comma, `end` the next entry's start. Character offsets."""
    found = []
    for m in _HEAD.finditer(text):
        h = headword(m.group("head"))
        if h:
            found.append((h, m.start("head"), m.end()))
    for i, (h, s, b) in enumerate(found):
        yield h, s, b, (found[i + 1][1] if i + 1 < len(found) else len(text))
