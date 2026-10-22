"""standin/book_frames.py -- facts read off a history book's sentences by frames (order 0012).

A frame fires only where the sentence names the entity asked about as the subject of the frame's verb
-- "Charlemagne was born in 742", "Carthage, a city in North Africa," "Rome was founded by Romulus",
"Henry II ruled England from 1154 to 1189" -- by its full name, or by its last name where no other
capitalised word runs into it ("Bonaparte was born", never "Carlo Bonaparte was born"). A pronoun is
never followed back ("He was born in 1769" says nothing here). Labels are Wikidata's, as the
Wikidata source states them, so a book's fact and Wikidata's are one fact and one corroborates the
other: date of birth / death (a full date as ISO, a year as written: "742", "480 BC"), place of
birth / death / burial, founded by, inception, location, part of, capital, point in time (a battle's
year), reign ("1154-1189", the years a ruler ruled). What no frame says, the source does not say.

Wired: WIRED (`book_source.BookSource`).
"""
from __future__ import annotations

import re

from cubbyllm.reasoning.planner import Triple

__wiring__ = "WIRED"

_MONTHS = ("january february march april may june july august september october november december").split()
_MON = "|".join(m.capitalize() for m in _MONTHS) + r"|Jan\.|Feb\.|Aug\.|Sept?\.|Oct\.|Nov\.|Dec\."
_ERA = r"(?:\s?(?P<bc>BC|BCE|B\.C\.E?\.?)|\s?(?:AD|A\.D\.|CE)\b)?"
YEAR = rf"(?:(?:AD|A\.D\.)\s)?(?P<y>\d{{3,4}}|\d{{1,2}}(?=\s?(?:BC|BCE|B\.C)))(?!\d)(?!,\d)(?!\s?(?:years|men|soldiers|km|miles|feet|metres)){_ERA}"
DATE = (rf"(?:(?P<d1>\d{{1,2}})\s(?P<m1>{_MON})\s|(?P<m2>{_MON})\s(?P<d2>\d{{1,2}}),?\s)?{YEAR}")
_CAP = r"[A-Z][\w'’-]+"
PLACE = rf"(?P<p>{_CAP}(?:\s(?:of|de|del|di|du|on|upon|la|le)\s{_CAP}|\s{_CAP}){{0,3}})"
PERSON = rf"(?P<who>{_CAP}(?:\s(?:of|de|von|van|the)\s{_CAP}|\s{_CAP}){{0,3}})"
_NOT_PLACE = set(m.capitalize() for m in _MONTHS) | {"The", "His", "Her", "Their", "This", "That", "He", "She", "It",
                                                     "In", "On", "At", "By", "A", "An", "AD", "BC", "Christ"}


def _date(m) -> str:
    """A full date as ISO (Wikidata's form); a year as the book writes it."""
    y = int(m["y"])
    mon = m["m1"] or m["m2"]
    day = m["d1"] or m["d2"]
    if m["bc"]:
        return f"{y} BC"
    if mon and day:
        k = next(i for i, n in enumerate(_MONTHS, 1) if n.startswith(mon.rstrip(".").lower()[:3]))
        return f"{y:04d}-{k:02d}-{int(day):02d}"
    return str(y)


def _place(m) -> str | None:
    """The place a frame read, or None: a frame word first ('The'), or a word with a digit glued in ('Egypt216Lorenzo':
    a page number and a name run together by the PDF's text)."""
    p = m["p"].strip()
    return None if p.split()[0] in _NOT_PLACE or re.search(r"[A-Za-z]\d|\d[A-Za-z]", p) else p


# (relations, the frame after the subject, how each relation's value is read). `{S}` is the subject;
# `_G` the gap a subject's apposition may leave ("Charlemagne, king of the Franks, was born ...").
_G = r"(?:,[^,.;()]{1,60},)?\s"
_W = r"(?:,\s(?:who\s)?|\s)"                             # born: right after the name, a participle after a comma, a who-clause
_WD = r"(?:,\swho\s|\s)"                                  # died: right after the name or in a who-clause ("Martha, named after
                                                         # Martha Washington, died ..." is about Martha)
_C = r"(?:(?:about|around|circa|c\.|ca\.)\s)?"          # an approximate year: "born about 742" reads as 742
FRAMES = [
    (("date of birth", "place of birth"), rf"{{S}}(?:{_G}(?:was\s)?|{_W}(?:was\s)?)born\s(?:on\s|in\s)?{_C}{DATE}(?:,?\s(?:in|at)\s{PLACE})?", None),
    (("place of birth", "date of birth"), rf"{{S}}(?:{_G}(?:was\s)?|{_W}(?:was\s)?)born\s(?:in|at)\s{PLACE}(?:,?\s(?:on|in)\s{_C}{DATE})?", None),
    (("date of death", "place of death"), rf"{{S}}(?:{_G}|{_WD})(?:died|was\s(?:killed|executed|assassinated))\s(?:on\s|in\s)?{_C}{DATE}(?:,?\s(?:in|at)\s{PLACE})?", None),
    (("place of death", "date of death"), rf"{{S}}(?:{_G}|{_WD})died\s(?:in|at)\s{PLACE}(?:,?\s(?:on|in)\s{_C}{DATE})?", None),
    (("place of burial",), rf"{{S}}{_G}(?:was\s)?buried\s(?:in|at)\s{PLACE}", None),
    (("founded by", "inception"), rf"{{S}}{_G}(?:was\s)?founded\s(?:in\s{DATE}\s)?by\s{PERSON}", None),
    (("inception", "founded by"), rf"{{S}}{_G}(?:was\s)?(?:founded|established)\sin\s{DATE}(?:\sby\s{PERSON})?", None),
    (("reign",), rf"{{S}}{_G}(?:ruled|reigned)\s(?:over\s)?(?:[A-Z][\w'’-]+(?:\s[A-Z][\w'’-]+)?\s)?from\s(?P<a>\d{{3,4}})\s(?:to|until)\s(?P<b>\d{{3,4}})", None),
    (("location",), rf"{{S}},\s(?:an?)\s(?:ancient\s|small\s|large\s|walled\s|coastal\s|port\s)?(?:city|town|village|fortress|port|settlement)\s(?:of|in)\s(?:modern\s(?:day\s)?|present-day\s)?{PLACE}", None),
    (("location",), rf"{{S}}{_G}(?:is|was|lies|lay)\s(?:located|situated)\s(?:in|on)\s{PLACE}", None),
    (("part of",), rf"{{S}}{_G}(?:was|became)\s(?:a\s|now\s)?part\sof\s(?:the\s)?{PLACE}", None),
    (("head of state",), rf"{PERSON}{_G}(?:ruled|reigned)\s(?:over\s)?(?:the\s)?{{S}}\sfrom\s(?P<a>\d{{3,4}})\s(?:to|until)\s(?P<b>\d{{3,4}})", "realm"),
    (("capital",), rf"{PLACE},\s(?:the\s)?capital\sof\s(?:the\s)?{{S}}\b", "capital"),
    (("point in time",), rf"(?:the\s)?{{S}}\s(?:in\s|\(|,\s)?{DATE}", "battle"),
]
_SENT = re.compile(r"(?<=[a-z0-9)\]][.!?])\s+(?=[A-Z])")


def sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT.split(paragraph) if s.strip()]


_RUN_IN = re.compile(r"[A-Z][\w'’-]*\.?\s$")
_TITLE = re.compile(r"(?<![\w'’-])(?:King|Queen|Emperor|Empress|Saint|St\.|Pope|Sir|Lord|Lady|General|President|Prince|Princess|"
                    r"Duke|Count|Bishop|Archbishop|Caliph|Sultan|Tsar|Czar|Khan|Pharaoh|Dr\.|Mr\.|Mrs\.|Rev\.|"   # a title, or a word
                    r"The|A|An|In|On|At|By|When|After|Before|During|Then|But|And|As|While|Since|Although|Though|Yet|So|"
                    r"If|Under|From|For|With|To|There|Here|Thus|Later|Meanwhile|Also|Only)\s$")   # that opens a sentence
_GOVERNED = re.compile(r"\b(?:of|after|by|to|with|from|for|against|under|than)\s(?:the\s)?$", re.I)
_FUNCTION = {"in", "on", "at", "by", "and", "or", "from", "during", "after", "before", "until", "which", "when", "where",
             "with", "under", "as", "for", "to", "the", "a", "an", "but", "near", "who", "that", "between", "about", "of"}


def _subject(entity: str) -> list[tuple[str, bool]]:
    """(regex, a last name alone?) for each way a subject is named, in any case (a seed arrives
    lowercased): the full name; for a name of two or more words, its last word (checked by `_match` for a
    capitalised word running into it)."""
    words = entity.split()
    full = r"\s".join(re.escape(w) for w in words)
    out = [(rf"(?P<s>(?<![\w'’-])(?i:{full}))(?![\w'’-])", False)]
    if len(words) >= 2 and len(words[-1]) > 3 and words[-1].lower() not in {"great", "younger", "elder"}:
        out.append((rf"(?P<s>(?<![\w'’-])(?i:{re.escape(words[-1])}))(?![\w'’-])", True))
    return out


def _match(frame: str, entity: str, sentence: str, full_only: bool, governed: bool = False):
    """The first match whose subject the sentence writes as a name (capitalised), as the subject of the clause:
    no capitalised word running into it but a title ('Carlo Bonaparte', 'SAMUEL JOHN MILLS'; 'King Henry' is
    fine), not governed by a preposition ('a house of the Poor Clares was founded', 'named after Martha
    Washington, died'); a place it reads not after a preposition ('south of the Dead Sea, capital of') nor an
    adjective before a noun ('part of the Greek national narrative')."""
    for subj, _last in _subject(entity)[:1] if full_only else _subject(entity):
        for m in re.finditer(frame.replace("{S}", subj), sentence):
            s, before = m["s"], sentence[:m.start("s")]
            if not (s[:1].isupper() or not s[:1].isalpha()):
                continue
            if (_RUN_IN.search(before) and not _TITLE.search(before)) or (not governed and _GOVERNED.search(before)):
                continue
            if m.groupdict().get("p"):
                after = re.match(r"\s+([a-z][\w'’-]*)", sentence[m.end("p"):])
                if _GOVERNED.search(sentence[:m.start("p")]) and m.start("p") < m.start("s"):
                    continue                             # the place is a preposition's, the apposition another noun's
                if after and after.group(1) not in _FUNCTION:
                    continue                             # an adjective of the noun after it, not a place
            return m, _last
    return None


def read(sentence: str, entity: str) -> list[tuple[Triple, str, dict]]:
    """(Triple, the sentence, when) for every frame that fires on the sentence with `entity` its subject;
    `when` is the fact's {point} (an event: a birth, a founding, a battle -- the date the frame read beside
    it) or {start, end} (a reign), {} when the sentence dates nothing (the store's `times`)."""
    out, seen = [], set()
    for rels, frame, only in FRAMES:
        if only == "battle" and not entity.lower().startswith("battle of"):
            continue
        m, last = _match(frame, entity, sentence, only == "battle", governed=only == "capital") or (None, False)
        if m is None:
            continue
        g = m.groupdict()
        span = {"start": m["a"].zfill(4), "end": m["b"].zfill(4)} if g.get("a") and int(m["a"]) <= int(m["b"]) else None
        point = {"point": _date(m)} if g.get("y") else {}
        for rel in rels:
            if rel in seen:
                continue
            if rel in ("reign", "head of state"):
                o = (f"{m['a']}-{m['b']}" if rel == "reign" else g.get("who")) if span else None
            elif rel in ("founded by",):
                o = g.get("who")
            elif rel in ("date of birth", "date of death", "inception", "point in time"):
                o = _date(m) if g.get("y") else None
            else:
                o = _place(m) if g.get("p") else None
            if o and o.lower() != entity.lower():
                seen.add(rel)
                when = span or (point if rel not in ("location", "part of", "capital") else {})
                out.append((Triple(obj=o, rel=rel, subj=entity if last else " ".join(m["s"].split())), sentence, when))
    return out


CUES = {"date of birth": ["born", "birth"], "place of birth": ["born", "birthplace"], "date of death": ["died", "death"],
        "place of death": ["died", "death"], "place of burial": ["buried", "tomb"], "founded by": ["founded", "founder"],
        "inception": ["founded", "established"], "reign": ["ruled", "reigned", "reign"],
        "location": ["located", "city", "town"], "part of": ["part"], "capital": ["capital"],
        "point in time": ["battle", "fought"], "head of state": ["ruled", "reigned", "king", "ruler"]}
