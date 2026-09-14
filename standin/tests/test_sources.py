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


# ---- the three tiers of 2026-09-13 (the ask loop): linked, the question's next hop, label over alias ----
ITEM = lambda qid: {"mainsnak": {"datatype": "wikibase-item", "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}}}}  # noqa: E731
FATHER_CLAIMS = {"claims": {"P22": [ITEM("Q10")]}}                       # Bill -> father Q10 (Jim)
GIVEN = {"claims": {"P735": [ITEM("Q90")]}}                              # a given name claim
JIM = {"claims": {"P735": [ITEM("Q90")]}, "labels": {"en": {"value": "Jim Haslam"}}}   # Q10: the claim and the label
LABELS2 = {"P22": {"labels": {"en": {"value": "father"}}}, "P735": {"labels": {"en": {"value": "given name"}}},
           "Q90": {"labels": {"en": {"value": "James"}}}}


def test_the_object_of_a_served_claim_is_linked_to_its_item_so_the_next_hop_never_searches_the_name():
    """'Jim Haslam' is the label of Q10 and an alias of Q11 (Jimmy); the claim Bill -> father pointed at Q10."""
    src = FakeWikidata([_hit("Q1", "Bill Haslam", "governor")], {"Q1": FATHER_CLAIMS, "Q10": JIM, "Q11": GIVEN, **LABELS2})
    facts = src.facts("Bill Haslam")
    assert [(t.obj, t.rel) for t in facts] == [("Jim Haslam", "father")] and src.last["how"] == "exact"
    assert src.links == {"jim haslam is the father of bill haslam": ["Q10"]}     # keyed by the FACT, never by the name
    src._search = [_hit("Q10", "Jim Haslam", "businessman"), _hit("Q11", "Jimmy Haslam", "team owner", match="Jim Haslam")]
    calls = src.calls
    out = src.facts("Jim Haslam", via="Jim Haslam is the father of Bill Haslam")   # reached through the walked fact
    assert src.last["how"] == "linked" and src.last["qid"] == "Q10" and [(t.obj, t.rel) for t in out] == [("James", "given name")]
    assert src.calls == calls + 2                        # the claims and their labels: no search for an item the claim named
    # the same name typed as a SEED (no fact behind it) is not linked: the search tiers decide (2026-09-14:
    # a name-keyed link had sent 'Marie Curie' to a film named after her, the object of one of her own claims)
    src.facts("Jim Haslam")
    assert src.last["how"] == "exact + label over alias" and src.last["qid"] == "Q10"
    # two items behind one fact text: the link tier stands aside
    src._link("Jim Haslam is the father of Bill Haslam", "Q11")
    src.facts("Jim Haslam", via="Jim Haslam is the father of Bill Haslam")
    assert src.last["how"] == "exact + label over alias"


def test_among_items_sharing_a_name_the_question_s_next_hop_decides_and_a_pick_is_not_remembered():
    """Marie Curie: the physicist, a book edition and a ferry share the label; one has a date of birth."""
    hits = [_hit("Q7186", "Marie Curie", "physicist"), _hit("Q114", "Marie Curie", "book edition"), _hit("Q134", "Marie Curie", "ferry")]
    src = FakeWikidata(hits, {"Q7186": CLAIMS, "Q114": {"claims": {"P31": [ITEM("Q3331189")]}}, "Q134": {"claims": {}}, **LABELS})
    assert src.facts("Marie Curie") == [] and len(src.last["ambiguous"]) == 3        # the strict rule without the hint
    out = src.facts("Marie Curie", relations=["born"])                                # 'born' names date / place of birth
    assert [(t.obj, t.rel, t.subj) for t in out] == [("1800-05-11", "date of birth", "Marie Curie")]
    assert src.last["qid"] == "Q7186" and src.last["how"].startswith("relation")
    assert src.links == {}                                                           # a pick is never remembered; only served claims are
    # two items carrying the property, both labelled with the name: still ambiguous, still refused
    src._entities["Q114"] = CLAIMS
    assert src.facts("Marie Curie", relations=["born"]) == [] and src.last["qid"] is None


def test_the_label_the_question_used_outranks_an_alias_but_two_labels_stay_ambiguous():
    src = FakeWikidata([_hit("Q10", "Jim Haslam", "businessman"), _hit("Q11", "Jimmy Haslam", "team owner", match="Jim Haslam")],
                       {"Q10": JIM, "Q11": GIVEN, **LABELS2})
    out = src.facts("Jim Haslam", relations=["given name"])                            # both carry it: the label decides
    assert src.last["qid"] == "Q10" and src.last["how"].endswith("label over alias") and len(out) == 1
    src2 = FakeWikidata([_hit("Q1", "James Young", "chemist"), _hit("Q2", "James Young", "politician")], {"Q1": CLAIMS, "Q2": CLAIMS, **LABELS})
    assert src2.facts("James Young", relations=["born"]) == [] and src2.last["qid"] is None   # the pinned case above, unchanged
