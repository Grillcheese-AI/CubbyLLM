"""lexicon — a synonym oracle for relation wording: WordNet 3.0 joined with WOLF (FR), by synset.

Wired: WIRED (2026-09-12, coverage lever 5). A second relation resolver beside the source's
(`learn.resolve_relations`): the plan says `birthplace`, the store says `place of birth`; the
plan says `lieu de naissance`, the store says `place of birth`. One synset, three wordings.

The data is `standin/data/build_lexicon.py`'s jsonl (one synset per line: en lemmas, fr
literals, gloss); the reader is stdlib. `relations(text)` returns the OTHER wordings of every
synset that has `text` as an English lemma or a French literal -- the host intersects them with
the relations the store holds and refuses more than one, exactly as with the source's labels. A
polysemous word ('position': 20 synsets) fans out to many wordings; the store's vocabulary is
what makes the answer small, and an answer that is not a single relation is a refusal, never a
guess. Exact-phrase tier only: a word's synonyms are never applied to a phrase word by word
(that is how 'country' would become 'state').
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

from ..core.protocols import Wiring
from .planner import normalize

__wiring__ = Wiring.WIRED

DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "standin" / "data" / "out" / "lexicon_en_fr.jsonl"


class Lexicon:
    name = "lexicon"

    def __init__(self, path: pathlib.Path | str | None = None) -> None:
        self.path = pathlib.Path(path or DEFAULT)
        self._by_word: dict[str, list[int]] = defaultdict(list)     # normalized wording -> synset rows
        self._rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                i = len(self._rows); self._rows.append(rec)
                for w in rec.get("en", []) + rec.get("fr", []):
                    self._by_word[normalize(w)].append(i)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def synsets(self, text: str) -> list[dict]:
        return [self._rows[i] for i in self._by_word.get(normalize(text), [])]

    def relations(self, text: str) -> list[str]:
        """Every other English wording of a NOUN synset `text` belongs to, in a stable order.
        Nouns only (2026-09-13, the ask loop): the verb 'mother' -- beget, engender, FATHER,
        sire -- reached lever 4 and 'Who is the mother of Justin Trudeau?' was rewritten to
        `father` and answered Pierre Trudeau, verified. A relation is a noun phrase; a verb
        sense of its word is another word."""
        key = normalize(text)
        out: list[str] = []
        for rec in self.synsets(text):
            if rec.get("pos", "n") != "n":
                continue
            for w in rec.get("en", []):
                if normalize(w) != key and w not in out:
                    out.append(w)
        return out

    def french(self, text: str) -> list[str]:
        out: list[str] = []
        for rec in self.synsets(text):
            for w in rec.get("fr", []):
                if w not in out:
                    out.append(w)
        return out
