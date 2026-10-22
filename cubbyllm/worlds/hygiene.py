"""standin/data/world_hygiene.py -- what a world's mount mends or drops before its facts are served (order 0015,
addenda 1 and 3).

- **Debris:** a fact whose end is a Python repr (`<class 'wikikg.relations.X'>`, `<... object at 0x...>`) or a bare `_`
  is no fact; a fact whose value is its own subject ("Simon Baron Cohen" the citizenship of Simon Baron Cohen) neither.
- **Spellings:** a known misspelled relation takes its spelling (`SP0USE_OF` -> `SPOUSE_OF`).
- **Same place:** an ancient name "located in" its modern one ("Antium" in "Anzio") is an identity, not a location:
  the pairs `same_place_pairs()` lists (`build_same_place.py`, from the books' own wording) become `SAME_PLACE_AS`
  ("Anzio is the later name of Antium").
- **Folding:** `fold(value)` -- case, spacing and diacritics off -- so "Carson Mc Cullers" and "Carson McCullers",
  "Alfred Doblin" and "Alfred Döblin" are one value when two worlds are compared.

Wired: WIRED (`wikikg.wiki_world` mounts through `hygiene`; `hdc_store.disagreements` folds with `fold`).
"""
from __future__ import annotations

import pathlib
import re
import unicodedata

__wiring__ = "WIRED"

OUT = pathlib.Path(__file__).resolve().parents[2] / "standin" / "data" / "out"
SAME_PLACE = OUT / "same_place.tsv"                  # git-ignored: built from the books (build_same_place.py)
_NOT_A_NAME = re.compile(r"<[^>]*(?:class |object at )|^_+$")
MISSPELLED = {"SP0USE_OF": "SPOUSE_OF"}              # a zero for the O


def fold(value: str) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def same_place_pairs(path=SAME_PLACE) -> set[tuple[str, str]]:
    """(ancient subject, modern place) as the wiki world writes them (camel-cased), from the built list."""
    if not pathlib.Path(path).exists():
        return set()
    return {tuple(ln.split("\t")[:2]) for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines()[1:] if "\t" in ln}


def hygiene(triples, same_place=None) -> tuple[list, dict]:
    """The triples a world serves, and the counts of what was dropped or mended."""
    pairs = same_place_pairs() if same_place is None else set(same_place)
    out, c = [], {"dropped_not_a_name": 0, "dropped_self_reference": 0, "relations_respelled": 0, "same_place": 0}
    for s, r, o in triples:
        if _NOT_A_NAME.search(str(s)) or _NOT_A_NAME.search(str(o)):
            c["dropped_not_a_name"] += 1
            continue
        if fold(s) == fold(o):
            c["dropped_self_reference"] += 1
            continue
        if r in MISSPELLED:
            r = MISSPELLED[r]; c["relations_respelled"] += 1
        if r == "LOCATED_IN" and (s, o) in pairs:
            r = "SAME_PLACE_AS"; c["same_place"] += 1
        out.append((s, r, o))
    return out, c
