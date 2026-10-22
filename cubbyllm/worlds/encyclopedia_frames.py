"""cubbyllm/worlds/encyclopedia_frames.py -- an entry's opening read as facts: dates, places and their admin types,
the person and place frames, the triggers, an entry read. Split out of encyclopedia.py (order 0031), verbatim;
encyclopedia.py re-exports every name.

Wired: WIRED (the encyclopedia source).
"""
from __future__ import annotations

import re
from cubbyllm.reasoning.planner import Triple, normalize
from .encyclopedia_text import OPENING, _TAIL, _YEARS, clean, given_names, sentence_at, surname

__wiring__ = "WIRED"


# -- frames ------------------------------------------------------------------------------------------
_MON = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")                 # 'Aug. 28, 1749', 'April 3, 1934'


_MNUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


_DATE = rf"(?P<m>{_MON})\.?\s+(?P<d>\d{{1,2}}),\s*(?P<y>\d{{4}})"


_PTOK = r"(?:St\.|Mt\.|Ft\.|[A-Z][\w'’-]+)"                       # a period only in 'St. Louis' / 'Mt. Vernon' / 'Ft. Worth'


_PJOIN = r"(?:,? | (?:del|de la|de|am|an der|upon|sur|du|des|les) )"   # 'Pinar del Rio', 'Frankfurt am Main', 'Vienna, Austria'


_PLACE_IN = rf"{_PTOK}(?:{_PJOIN}{_PTOK}){{0,3}}"                 # 'Vienna, Austria', 'St. Louis', 'Frankfurt'


# 'in the free city of Frankfurt', 'in the village of X': a named kind of place, never 'in a small village near Paris'
_PLACE_LEAD = r"(?:the )?(?:(?:free|imperial|royal|old|ancient|former|then|small|little) )?(?:(?:city|town|village|borough|province|parish|district|county|hamlet|suburb|port) of )?"


_PLACE = rf"{_PLACE_LEAD}(?P<o>{_PLACE_IN})"


_BORN = r"(?:born|bom)"                                          # 'bom' is the OCR's usual reading of 'born'


def _iso(m) -> str:
    return f"{int(m['y']):04d}-{_MNUM[m['m'][:3].lower()]:02d}-{int(m['d']):02d}"


_INITIAL_NEXT = re.compile(r" [A-Z]\.")                          # 'He died in Justus J. Schifferes Coauthor ...': a byline the OCR ran in


_ALLCAPS = re.compile(r"^[A-Z]{2,}$")                            # 'New York GRISWOLD', 'New HARBIN': the next entry's headword ran in


def _place(m) -> str:
    if _INITIAL_NEXT.match(m.string, m.end()):
        return ""
    o = m["o"].split(",")[0].strip()                             # the immediate place: 'Vienna', not 'Vienna, Austria'
    if any(_ALLCAPS.match(t) for t in o.split()):
        return ""
    return o


_ADMIN_TYPES = "county|prefecture|province|region|district|parish|department|state|governorate|oblast|canton|territory"


def _admin(m) -> str:
    o = m["o"].strip()
    if any(_ALLCAPS.match(t) for t in o.split()):
        return ""
    return f"{o} {m['kind'].capitalize()}" if m.groupdict().get("kind") else o   # Wikidata's label form: 'Fulton County'


PERSON_FRAMES = [
    ("date of birth", re.compile(rf"\b{_BORN} (?:(?:in|at) {_PLACE_LEAD}{_PLACE_IN},? )?on {_DATE}"), _iso),
    ("place of birth", re.compile(rf"\b{_BORN} (?:on {_DATE},? )?(?:in|at) {_PLACE}"), _place),
    ("date of death", re.compile(rf"\bdied (?:(?:in|at) {_PLACE_LEAD}{_PLACE_IN},? )?on {_DATE}"), _iso),
    ("place of death", re.compile(rf"\bdied (?:on {_DATE},? )?(?:in|at) {_PLACE}"), _place),
]


# a place entry states the container it NAMES AS ONE: 'in Fulton county', 'is in Shizuoka prefecture',
# 'is in Hampshire,' -- never the 'a city in Japan' opener, which is a country as often as a region
PLACE_FRAMES = [
    ("located in the administrative territorial entity",
     re.compile(rf"\b(?:in|of) (?:the )?(?P<o>[A-Z][\w'’-]+(?: [A-Z][\w'’-]+){{0,2}}) (?P<kind>{_ADMIN_TYPES})\b(?! of)"), _admin),
    ("located in the administrative territorial entity",
     re.compile(rf"\bis in (?P<o>[A-Z][\w'’-]+(?: [A-Z][\w'’-]+){{0,2}})(?=[,.])"), _admin),
]


_YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")


_DATE_OF = {"place of birth": "date of birth", "place of death": "date of death"}


# an entry is a place when its first sentence says so; a bird 'found in Fulton county' is located nowhere
_PLACE_OPENER = re.compile(r"^[^.]{0,80}?\b(?:a|an|the) (?:[a-z-]+ ){0,3}(?:city|town|village|borough|county seat|seaport|port|"
                           r"municipality|district|county|capital|commune|parish|island|region|province|suburb|hamlet|township)\b")


# what a question calls the relations the frames read (lever 4 for an offline source)
TRIGGERS = {
    "date of birth": ["born", "birth", "birth date", "birthday", "date of birth"],
    "date of death": ["died", "death", "death date", "date of death", "die"],
    "place of birth": ["born", "birthplace", "place of birth"],
    "place of death": ["died", "place of death", "death place"],
    "located in the administrative territorial entity": ["located", "located in", "district", "county", "municipality"],
}


KIND = {"date of birth": "date", "date of death": "date", "place of birth": "name", "place of death": "name",
        "located in the administrative territorial entity": "name"}


_TRIGGER_INDEX: dict[str, list[str]] = {}


def read_entry(head: str, body: str) -> tuple[str | None, bool, list[tuple[Triple, str]]]:
    """(the entry's name, is it a person, [(Triple, sentence)]) from the headword and the text
    after its comma. A person's name is given names + surname; the headword's years give the
    birth and death years, and a full date read from the opening must agree with them. A person
    whose given names cannot be read is nobody: no name, no facts. A person frame counts only
    in a sentence that is ABOUT the entry's subject -- one that opens with 'He', 'She', 'Born',
    or a word of the name -- so 'His father was born in Paris' states nothing about the son."""
    opening = clean(body[:OPENING])
    m = _YEARS.match(opening)
    years: dict[str, str] = {}
    if m:
        g = given_names(m.group("mid"))
        if not g:
            return None, True, []
        name = f"{g} {surname(head)}"
        years["date of birth"] = m.group("b")
        if m.group("d"):
            years["date of death"] = m.group("d")
        frames = PERSON_FRAMES
        subject_words = {w.lower().strip(".,'’") for w in name.split() if len(w) > 2} | {"he", "she", "born", "bom"}
        subject_words.add(opening.split(" ", 1)[0].lower().strip(".,'’()"))   # the headword's own sentence opens with the pronunciation
    else:
        name = surname(head)
        frames = PLACE_FRAMES if _PLACE_OPENER.search(opening) else []
        subject_words = set()
    head_years = dict(years)
    cut = len(opening) - _TAIL if len(body) > OPENING else len(opening) + 1
    out: list[tuple[Triple, str]] = []
    seen: set[str] = set()
    for rel, rx, obj in frames:
        if rel in seen:
            continue
        for mm in rx.finditer(opening):
            if mm.end() >= cut:
                break                                          # the opening ends mid-sentence here: whatever it read is cut off
            sentence = sentence_at(opening, mm.start(), mm.end())
            first = sentence.split(" ", 1)[0].lower().strip(".,'’()") if sentence else ""
            if subject_words and first not in subject_words:
                continue                                       # a sentence about someone else
            date_rel = _DATE_OF.get(rel, rel)
            ref = head_years.get(date_rel)
            if ref and (ys := _YEAR.findall(sentence)) and ref not in ys:
                years.pop(date_rel, None)                      # a subject sentence dated to another year: a misread, or another
                break                                          # entry's run in -- the source disagrees with itself, neither is a fact
            o = obj(mm)
            if not o:
                continue                                       # a run-in the frame's reader refused; the next match may be clean
            if normalize(o) == normalize(name):
                break
            if rel in years and o[:4] != years[rel]:
                years.pop(rel)                                 # the OCR disagrees with itself: neither is a fact
                break
            seen.add(rel)
            out.append((Triple(obj=o, rel=rel, subj=name), sentence))
            break
    for rel, y in years.items():
        if rel not in seen:
            out.append((Triple(obj=y, rel=rel, subj=name), opening[:m.end()].strip()))
    return name, bool(m), out
