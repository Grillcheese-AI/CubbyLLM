"""cubbyllm/sources/wikidata_aliases.py -- Wikidata's property labels and aliases as a local table (so serving makes
no call per wording): its inflections, the table, the loader. Split out of wikidata.py (order 0031), verbatim;
wikidata.py re-exports every name.

Wired: WIRED (the Wikidata source).
"""
from __future__ import annotations

import json
import pathlib
from cubbyllm.local_paths import data_dir
from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402

__wiring__ = "WIRED"


CACHE = data_dir("wikidata_cache")


# Wikimedia's own pages ABOUT pages: never the subject of a question (2026-09-14)
_META = __import__("re").compile(r"\bWikimedia\b.*\b(disambiguation|list|category|template|module|portal|project)\b"
                                 r"|\b(disambiguation page|list article|Wikinews article)\b", __import__("re").I)


PROPERTY_TABLE = CACHE.parent / "wikidata_properties_en_fr.json"        # standin/data/out/, built once by build_property_aliases.py


def _inflections(key: str) -> list[str]:
    """The regular inflections of a normalized wording's FIRST word (the verb of 'work for',
    'die', 'attend'): -s/-es/-d/-ed/-ing forms and the stems they come from. Exact strings
    to look up, no guessing beyond them."""
    words = key.split()
    if not words:
        return []
    w, rest = words[0], words[1:]
    forms = {w + "s", w + "es", w + "d", w + "ed", w + "ing"}
    if w.endswith("e"):
        forms.add(w[:-1] + "ing")
    if w.endswith("y"):
        forms.add(w[:-1] + "ied"); forms.add(w[:-1] + "ies")
    stems: set[str] = set()
    for suf in ("ies", "ied", "ing", "es", "ed", "s", "d"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            stem = w[:-len(suf)]
            stems.add(stem)
            if suf in ("ies", "ied"):
                stems.add(stem + "y")
            if suf in ("ing", "ed", "d"):
                stems.add(stem + "e")
    for s in stems:                                              # 'establishing' -> 'establish' -> 'established'
        forms.update({s, s + "s", s + "es", s + "d", s + "ed", s + "ing"})
    forms.discard(w)
    return sorted(" ".join([f] + rest) for f in forms)


class PropertyAliases:
    """Wikidata's property labels and aliases (EN + FR) as a LOCAL relation resolver -- the
    same answer `WikidataSource.relations()` used to fetch per wording (875 API calls for
    100 questions, exp_r14 2026-09-12), read from a table built once. `relations(text)`:
    the labels (English) of every property whose label or alias, in either language, IS
    the wording. Exact tier only, like the API version; the host still intersects with what
    the store holds and refuses more than one."""
    name = "property_aliases"
    # a property's datatype -> the value kind `plan_verify.ask_type` names ('when' -> date,
    # 'how many' -> number, 'who' -> name); the rest (external ids, urls, media) are no
    # kind a question asks for and never narrow anything
    KIND = {"time": "date", "quantity": "number", "wikibase-item": "name", "string": "name", "monolingualtext": "name"}

    def __init__(self, path: pathlib.Path | str = PROPERTY_TABLE) -> None:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.meta = {k: v for k, v in d.items() if k != "properties"}
        self._by_text: dict[str, set[str]] = {}
        self._texts_of: dict[str, set[str]] = {}         # English label -> every English wording of its property
        self._kind: dict[str, str | None] = {}           # any English wording -> its property's kind (None when they disagree)
        self._pid_of: dict[str, str] = {}                # English label -> the property id (the claim key an item carries)
        for pid, p in d["properties"].items():
            label_en = p["label"].get("en")
            if not label_en:
                continue
            self._pid_of[_normalize(label_en)] = pid
            k = self.KIND.get(p.get("datatype") or "")
            en_texts = [t for t in [label_en] + list(p["aliases"].get("en", [])) if t]
            self._texts_of.setdefault(_normalize(label_en), set()).update(en_texts)
            for lang in ("en", "fr"):
                for text in [p["label"].get(lang)] + list(p["aliases"].get(lang, [])):
                    if text:
                        self._by_text.setdefault(_normalize(text), set()).add(label_en)
            for text in en_texts:
                key = _normalize(text)
                if key in self._kind and self._kind[key] != k:
                    self._kind[key] = None               # two properties share this wording and disagree on kind: no kind
                else:
                    self._kind[key] = k

    def __len__(self) -> int:
        return len(self._by_text)

    def relations(self, text: str) -> list[str]:
        """Exact tier first; a wording the table lacks is tried as its inflections ('die' ->
        'died', 'attend' -> 'attended', 'work for' -> 'works for'), each an exact lookup."""
        key = _normalize(text)
        hit = self._by_text.get(key)
        if hit:
            return sorted(hit)
        out: set[str] = set()
        for alt in _inflections(key):
            out.update(self._by_text.get(alt, ()))
        return sorted(out)

    def wordings(self, label: str) -> list[str]:
        """Every English wording (label and aliases) of the property whose English label this
        is: 'date of birth' -> ['DOB', 'birth date', 'born', ...]. A store that holds the
        property under one of these ('birth date') holds the label's relation."""
        return sorted(self._texts_of.get(_normalize(label), ()))

    def pids(self, wording: str) -> list[str]:
        """The property ids a wording names, through its labels: 'given name' -> ['P735'],
        'born' -> ['P569', 'P19']. What an item's claims are keyed by."""
        out = []
        for label in self.relations(wording) or [wording]:
            pid = self._pid_of.get(_normalize(label))
            if pid and pid not in out:
                out.append(pid)
        return out

    def kind(self, label: str) -> str | None:
        """'date' / 'number' / 'name' for a wording whose property datatype says so; None
        when the table has no datatype for it (built before 2026-09-13), when properties
        sharing the wording disagree ('born': date of birth / place of birth), or for a
        kind no question asks for. None never narrows: the host keeps the ambiguity."""
        return self._kind.get(_normalize(label))


def property_aliases(path: pathlib.Path | str = PROPERTY_TABLE) -> "PropertyAliases | None":
    """The local table if it has been built, else None (the caller may fall back to the API)."""
    return PropertyAliases(path) if pathlib.Path(path).is_file() else None
