"""cubbyllm/sources/web_frames.py -- a page's sentences as facts: the frames that read a birth, a death, a place, the
entity's mentions and passages, dates in English and French, a plausible value, a cut-short value made whole. Split
out of web.py (order 0031), verbatim; web.py re-exports every name.

Wired: WIRED (the web source).
"""
from __future__ import annotations

import re
from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


_MNUM = {m.lower(): i for i, m in enumerate(_MONTHS.split("|"), 1)}


# day-first English dates, which the web writes and DBpedia's abstracts did not ('7 November 1867')
WEB_FRAMES_EN = [
    ("date of birth", re.compile(rf"\((?:born )?(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})\s*(?:[–-]|\)|;|,)")),
    ("date of death", re.compile(rf"\((?:\d{{1,2}} )?(?:{_MONTHS}) ?\d{{0,2}},? \d{{4}}\s*[–-]\s*(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})\)")),
    ("date of death", re.compile(rf"\bdied (?:on )?(?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})")),
    ("date of death", re.compile(rf"\bdied (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?(?:on )?(?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})")),
    # 'was born in Warsaw on November 7, 1867' (nobelprize.org, live 2026-09-24)
    ("date of birth", re.compile(rf"\bborn (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?on (?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})")),
    ("date of birth", re.compile(rf"\bborn (?:in [A-Z][\w'-]+(?:,? [A-Z][\w'-]+){{0,3}},? )?on (?P<d>\d{{1,2}}) (?P<m>{_MONTHS}) (?P<y>\d{{4}})")),
]


# 'died in Desert Hot Springs, California' -- the born-in frame's twin (wikitext.py has none for death)
WEB_PLACE_FRAMES = [
    ("place of death", re.compile(r"\bdied (?:(?:on )?[A-Z][a-z]+ \d{1,2},? \d{4},? |(?:on )?\d{1,2} [A-Z][a-z]+ \d{4},? )?(?:at (?:his|her|their) home )?in (?P<o>[A-Z][\w'-]+(?:,? [A-Z][\w'-]+){0,3})")),
]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(À-Ý])")


def _opens_with(sentence: str, entity: str) -> bool:
    """The sentence is ABOUT the entity: it opens with its name (or 'The' + name), and the name is not a
    possessive or the head of a longer name -- "Einstein's friend was born in Ulm" is not about Einstein."""
    s = sentence.lstrip(" \"“(").lower()
    e = " ".join((entity or "").lower().split())
    if not e:
        return False
    for pre in ("", "the "):
        if s.startswith(pre + e):
            rest = s[len(pre + e):]
            return not rest or not (rest[0].isalnum() or rest[:2] in ("'s", "’s") or rest[0] in "-’'")
    return False


def frame_facts(entity: str, text: str, lang: str = "en") -> list[tuple[Triple, str]]:
    """wikitext's frames, and the day-first date frames, over the sentences that OPEN with the entity."""
    from ..worlds.wikitext import read_frames
    raw: list[tuple[Triple, str]] = []
    out = raw
    for sentence in _SENT.split(" ".join((text or "").split())):
        if not _opens_with(sentence, entity):
            continue
        langs = ("fr",) if lang.startswith("fr") else ("en", "fr")
        for lg in langs:
            out.extend(read_frames(entity, sentence, lg))
        for rel, rx in WEB_FRAMES_EN:
            m = rx.search(sentence)
            if m:
                out.append((Triple(obj=f"{int(m['y']):04d}-{_MNUM[m['m'].lower()]:02d}-{int(m['d']):02d}",
                                   rel=rel, subj=entity), sentence))
        for rel, rx in WEB_PLACE_FRAMES:
            m = rx.search(sentence)
            if m and normalize(m["o"]) != normalize(entity):
                out.append((Triple(obj=m["o"], rel=rel, subj=entity), sentence))
    # a place read up to the next sentence's first word: 'born in Ashtabula County, Ohio Josephine ...' (live)
    names = set(normalize(entity).split())
    trimmed = []
    for t, sentence in raw:
        if t.rel.startswith("place of") or t.rel.startswith("located in"):
            words = t.obj.split()
            while len(words) > 1 and normalize(words[-1]) in names:
                words.pop()
            t = Triple(obj=" ".join(words).rstrip(","), rel=t.rel, subj=t.subj)
        trimmed.append((t, sentence))
    return trimmed


def mentions(entity: str, text: str) -> bool:
    """The name, or every one of its words of three letters or more ('Augusta Ada King, Countess of
    Lovelace' mentions Ada Lovelace)."""
    low = " " + " ".join(re.findall(r"\w+", (text or "").lower())) + " "
    full = " ".join(re.findall(r"\w+", (entity or "").lower()))
    words = [w for w in full.split() if len(w) >= 3]
    return bool(words) and (f" {full} " in low or all(f" {w} " in low for w in words))


_HATNOTE = re.compile(r"^(?:This article is about|For other uses|Not to be confused with)|\(disambiguation\)", re.I)


def passages(entity: str, text: str, limit: int = 2, width: int = 900, min_words: int = 12) -> list[str]:
    """The paragraphs that mention the entity and read as prose (>= `min_words` words): what the reader is
    shown. Navigation and menus are short lines -- live, 2026-09-24, the first MENTION of 'Ada Lovelace'
    on her Wikipedia page was the header beside '112 languages', and the reader read that as a birthplace."""
    out = []
    for line in (text or "").split("\n"):
        if _HATNOTE.search(line):
            continue                           # 'This article is about ... For other uses, see ...'
        if len(line.split()) >= min_words and mentions(entity, line):
            out.append(" ".join(line.split())[:width])
            if len(out) >= limit:
                break
    return out


_VALUE = {"date": re.compile(r"^(?:\d{4}(?:-\d{2}(?:-\d{2})?)?|\d{1,4} BC)$"),
          "number": re.compile(r"^[+-]?\d[\d,.\s]*$")}


_FR_MONTHS = "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"


_MNUM_FR = {m: i for i, m in enumerate(_FR_MONTHS.split("|"), 1)}


_DMY = re.compile(rf"^(?P<d>\d{{1,2}})(?:er)? (?P<m>{_MONTHS}|{_FR_MONTHS}) (?P<y>\d{{4}})$", re.I)


_MDY = re.compile(rf"^(?P<m>{_MONTHS}) (?P<d>\d{{1,2}}),? (?P<y>\d{{4}})$", re.I)


def iso_date(value: str) -> str | None:
    """'10 December 1815' / 'December 10, 1815' / '10 décembre 1815' -> '1815-12-10'; an ISO date or a
    bare year stays as it is; anything else (a range like '1815-1852') is None -- not a date."""
    v = " ".join((value or "").split())
    if _VALUE["date"].match(v):
        return v
    m = _DMY.match(v) or _MDY.match(v)
    if not m:
        return None
    mo = m["m"].lower()
    n = _MNUM.get(mo) or _MNUM_FR.get(mo)
    return f"{int(m['y']):04d}-{n:02d}-{int(m['d']):02d}" if n else None


def plausible(kind: str | None, value: str) -> bool:
    """A value of the kind the relation holds: a date for 'date of birth', a number for 'height', a name
    (letters, not opening on a digit) for 'place of birth'. Live, 2026-09-24: the reader wrote '1815-1852'
    as a date of birth and '112 languages' as a place of birth; both occur in the page, neither is one."""
    v = (value or "").strip()
    if not v or len(v) > 80:
        return False
    if kind in _VALUE:
        return bool(_VALUE[kind].match(v))
    if kind == "name":
        return bool(re.search(r"[^\W\d_]", v)) and not v[0].isdigit() and len(v.split()) <= 8
    return True


# ── the source ───────────────────────────────────────────────────────────────────────────────────────
# how a person types the relation into a search box (the property table's first wording is often 'DOB')
QUERY_WORDS = {"date of birth": "born", "place of birth": "born in", "date of death": "died", "place of death": "died in",
               "inception": "founded", "country of citizenship": "nationality", "educated at": "education",
               "spouse": "married", "author": "written by", "director": "directed by", "publication date": "published",
               "located in the administrative territorial entity": "located in", "headquarters location": "headquarters"}


def _fold(s: str) -> str:
    """normalize, accents folded: 'Móstoles' and 'Mostoles' are one spelling."""
    import unicodedata
    return normalize("".join(ch for ch in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(ch)))


def untruncate(s: str) -> str:
    """A snippet the engine cut off ('... born in Syracuse, New Yor…'), cut back to where it is whole: the
    last sentence end, else the last comma, else without its last word. Live, 2026-09-24: 'New Yor' made
    Tom Cruise's birthplace two claims and the walk refused."""
    s = (s or "").rstrip()
    if not (s.endswith("…") or s.endswith("...")):
        return s
    s = s.rstrip("….").rstrip()
    for sep in (". ", "! ", "? ", "; "):
        i = s.rfind(sep)
        if i > 20:
            return s[: i + 1]
    i = s.rfind(",")
    if i > 20:
        return s[:i]
    return s.rsplit(" ", 1)[0] if " " in s else ""


def _key(t: Triple) -> str:
    return normalize(f"{t.obj} is the {t.rel} of {t.subj}")
