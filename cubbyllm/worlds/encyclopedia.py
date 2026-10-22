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

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .encyclopedia_text import (OPENING, _CAPS, _GIVEN_TOKEN, _HEAD, _INITIALS, _LEADING_INITIALS,  # noqa: F401
                                _LINE_HYPHEN, _PARTICLES, _REPEATED, _RUNNING, _SENTENCE_END, _TAIL, _YEARS, clean,
                                entries, given_names, headword, sentence_at, surname)
from .encyclopedia_frames import (KIND, PERSON_FRAMES, PLACE_FRAMES, TRIGGERS, _ADMIN_TYPES, _ALLCAPS,  # noqa: F401
                                  _BORN, _DATE, _DATE_OF, _INITIAL_NEXT, _MNUM, _MON, _PJOIN, _PLACE, _PLACE_IN,
                                  _PLACE_LEAD, _PLACE_OPENER, _PTOK, _TRIGGER_INDEX, _YEAR, _admin, _iso, _place,
                                  read_entry)

__wiring__ = "WIRED"


for _rel, _words in TRIGGERS.items():
    for _w in _words:
        _TRIGGER_INDEX.setdefault(normalize(_w), []).append(_rel)


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
