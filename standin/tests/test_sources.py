"""standin/sources.py -- the Wikidata Source resolves an entity or refuses; it never picks.

Pinned (exp_r11 Gemini run 2, 2026-09-12: the seed 'james young' matched the first of several
items labelled James Young and a wrong birth year was verified and spoken):
  * two or more search hits with the query as label or alias -> no facts, `last['ambiguous']` names them;
  * no hit with the query as label or alias -> no facts (the old fallback to hits[0] was a guess);
  * exactly one -> its claims, as Triples.
Run: python -m pytest standin/tests/test_sources.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sources import WikidataSource  # noqa: E402


class FakeWikidata(WikidataSource):
    def __init__(self, search, entities):
        super().__init__(cache_dir=None, sleep_s=0.0, offline=False)
        self._search, self._entities = search, entities
    def _get(self, params):
        self.calls += 1
        if params.get("action") == "wbsearchentities":
            return {"search": self._search}
        if params.get("action") == "wbgetentities":
            ids = params["ids"].split("|")
            return {"entities": {i: self._entities[i] for i in ids if i in self._entities}}
        return None


def _hit(qid, label, desc, match=None):
    return {"id": qid, "label": label, "description": desc, "match": {"type": "label", "text": match or label}}


CLAIMS = {"claims": {"P569": [{"mainsnak": {"datatype": "time", "datavalue": {"type": "time", "value": {"time": "+1800-05-11T00:00:00Z", "precision": 11}}}}]}}
LABELS = {"P569": {"labels": {"en": {"value": "date of birth"}}}}


def test_a_label_shared_by_two_items_is_refused_not_picked():
    src = FakeWikidata([_hit("Q1", "James Young", "Scottish chemist"), _hit("Q2", "James Young", "American politician from Missouri")],
                       {"Q1": CLAIMS, "Q2": CLAIMS, **LABELS})
    assert src.facts("james young") == []
    assert [q for q, _l, _d in src.last["ambiguous"]] == ["Q1", "Q2"] and src.last["qid"] is None


def test_a_search_with_no_exact_hit_is_no_resolution():
    src = FakeWikidata([_hit("Q9", "James Young Simpson", "obstetrician")], {"Q9": CLAIMS, **LABELS})
    assert src.facts("james young") == [] and src.last["unresolved"] is True


def test_one_exact_hit_resolves_and_its_claims_come_back_as_triples():
    src = FakeWikidata([_hit("Q2", "James Young", "American politician from Missouri"), _hit("Q3", "James Young Simpson", "obstetrician")],
                       {"Q2": CLAIMS, **LABELS})
    out = src.facts("james young")
    assert [(t.obj, t.rel, t.subj) for t in out] == [("1800-05-11", "date of birth", "James Young")]
    assert src.last["qid"] == "Q2" and "ambiguous" not in src.last
