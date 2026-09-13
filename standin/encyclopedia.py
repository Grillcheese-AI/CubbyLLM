"""standin/encyclopedia.py -- an offline Source over OCR'd encyclopedia volumes: entries located
by their all-caps headword, facts read by frames off the entry's opening, provenance = volume +
headword + the sentence.

Wired: WIRED (stand-in serve stack; implements `cubbyllm.reasoning.learn.Source`, the same shape
as `standin/wikitext.py`). No model, no network. The corpus (2026-09-13, Nick): the Encyclopedia
Americana 2005, 29 OCR'd text volumes, off-repo. An entry opens as

    GOETHE, gu'ta, Johann Wolfgang von (1749- 1832), the greatest of all German poets ...
    GUINNESS, Alec, gin'is (1914-2000), British actor ...
    GOSPORT, a town in Hampshire, England, ...

so the headword is the surname (or the place), the given names are the comma part whose tokens
are capitalised words (the pronunciation is the lowercase one), and the parenthesis is the
birth-death years. The OCR is rough ('bom' for 'born', 'tire' for 'the', 'un- til'): a frame
tolerates the one misread it was seen to need ('bom') and no others, a full birth date must agree
with the headword's birth year or neither is stated, and the probe (validation/
exp_r16_encyclopedia.py) cross-checks the years against the wiki world before any of this feeds
a dataset. What the frames do not say, the source does not say.
"""
from __future__ import annotations

import json
import pathlib
import re

from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

# -- entries ---------------------------------------------------------------------------------------
# a headword: one to five all-caps words at a line start or after a sentence end, then a comma
_CAPS = r"[A-Z][A-Z'’.-]+(?:[ -][A-Z][A-Z'’.-]+){0,4}"
_HEAD = re.compile(rf"(?:^|(?<=\n)|(?<=\. )|(?<=\.\n))(?P<head>{_CAPS}),\s")
# a page's running head ('GOMBERT-GOMEZ GOMBERT', 'GOSHO HEINOSUKE-GOSNOLD GOSHO HEINOSUKE') is
# the page's first-last range followed by the entry's own headword, which repeats the range's start
_RUNNING = re.compile(r"^(?P<a>.+?)-(?P<b>[A-Z'’. ]+?) (?P=a)$")
_REPEATED = re.compile(r"^(?P<a>.+?) (?P=a)$")                   # 'GORDON GORDON': the running head, then the entry's own
_INITIALS = re.compile(r"^(?:[A-Z]\.\s?)+$")                    # 'H. M.' is a cross-reference, not an entry
_LEADING_INITIALS = re.compile(r"^(?:[A-Z]\.\s?)+")             # 'N. Y. GOTTSCHED': a byline's state ran into the headword
# the headword's own line: [pronunciation,] given names [, pronunciation] (BIRTH-DEATH)
_YEARS = re.compile(r"^(?P<mid>(?:[^().]|Jr\.|Sr\.|St\.){0,90}?)\((?P<b>\d{4})\s*[-–]\s*(?P<d>\d{4})?\s*\)")
_PARTICLES = frozenset("von van de da del della di du la le der den ten y of the af al el ibn bin und zu".split())
_GIVEN_TOKEN = re.compile(r"^[A-Z][a-zÀ-ſ'’-]+\.?$|^[A-Z]\.$")
OPENING = 1500
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
for _rel, _words in TRIGGERS.items():
    for _w in _words:
        _TRIGGER_INDEX.setdefault(normalize(_w), []).append(_rel)


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
    out: list[tuple[Triple, str]] = []
    seen: set[str] = set()
    for rel, rx, obj in frames:
        if rel in seen:
            continue
        for mm in rx.finditer(opening):
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


class EncyclopediaSource:
    """`facts(entity)`: the frames read off the entry whose name is the entity (exact,
    normalized: 'johann wolfgang von goethe'; a place by its headword, 'gosport'). The entry
    index (name -> offsets) is built once per volume and cached beside it as
    `<volume>.entries.json`; the volume's text is read whole and kept while it is the one in
    use. `last` carries the provenance of the last call."""
    name = "encyclopedia"

    def __init__(self, folder: str | pathlib.Path, volumes: list[str] | None = None) -> None:
        self.folder = pathlib.Path(folder)
        paths = sorted(self.folder.glob("*.txt"), key=lambda p: self._volume_no(p.name))
        if volumes:
            paths = [p for p in paths if p.name in volumes or str(self._volume_no(p.name)) in volumes]
        self.volumes: list[tuple[str, pathlib.Path, dict[str, tuple[int, int, int]]]] = []
        self.calls = 0
        self._text: tuple[str, str] | None = None
        for p in paths:
            self.volumes.append((p.stem, p, self._index(p)))
        self.last: dict = {}

    @staticmethod
    def _volume_no(name: str) -> int:
        m = re.match(r"Volume (\d+)", name)
        return int(m.group(1)) if m else 0

    def _read(self, vol: str, p: pathlib.Path) -> str:
        if self._text is None or self._text[0] != vol:
            self._text = (vol, p.read_text(encoding="utf-8", errors="replace"))
        return self._text[1]

    INDEX_VERSION = 2                                            # bump when `entries` / `read_entry` naming changes

    def _index(self, p: pathlib.Path) -> dict[str, tuple[int, int, int]]:
        cache = p.with_suffix(f".entries.v{self.INDEX_VERSION}.json")
        if cache.exists() and cache.stat().st_mtime >= p.stat().st_mtime:
            return {k: tuple(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
        text = self._read(p.stem, p)
        idx: dict[str, tuple[int, int, int]] = {}
        for head, s, b, e in entries(text):
            name, person, _facts = read_entry(head, text[b:b + OPENING])
            if name is None:
                continue
            key = normalize(name)                                # a person by full name only; a place by its headword
            if key and key not in idx:                           # first entry under a name wins; a later one is unreachable, never merged
                idx[key] = (s, b, e)
        cache.write_text(json.dumps(idx), encoding="utf-8")
        return idx

    def entry(self, entity: str) -> tuple[str, str, str] | None:
        """(volume, headword, body) or None."""
        key = normalize(entity)
        for vol, p, idx in self.volumes:
            loc = idx.get(key)
            if loc is None:
                continue
            s, b, e = loc
            text = self._read(vol, p)
            return vol, headword(text[s:b].rstrip(", \n")) or text[s:b], text[b:e]
        return None

    def relations(self, text: str) -> list[str]:
        return list(_TRIGGER_INDEX.get(normalize(text), []))

    def kind(self, label: str) -> str | None:
        return KIND.get(normalize(label))

    def facts(self, entity: str) -> list[Triple]:
        self.last = {"entity": entity, "volume": None, "headword": None, "sentences": {}}
        ent = self.entry(entity)
        if ent is None:
            return []
        vol, head, body = ent
        name, _person, pairs = read_entry(head, body)
        self.last.update(volume=vol, headword=head, name=name)
        out = []
        for t, sentence in pairs:
            out.append(t); self.last["sentences"][f"{t.rel}|{t.obj}"] = sentence
        return out
