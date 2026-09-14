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


# ---- 2026-09-14: a meta-page is never the answer; the asker decides; what askers meant is a host-side tally ----
def _fake_choices(tmp_path, search, entities):
    src = FakeWikidata(search, entities)
    src.cache = tmp_path; src.use_choices = True; src.choices = {}
    return src


def test_a_wikimedia_meta_page_is_dropped_before_the_ambiguity_is_counted():
    """Live, 2026-09-14: 'the last roman emperor' matched the legendary figure AND a disambiguation page,
    and the pair was refused. A page ABOUT pages is never the thing a question is about."""
    src = FakeWikidata([_hit("Q1", "Last Roman Emperor", "legendary figure of the Apocalypse of Pseudo-Methodius"),
                        _hit("Q2", "Last Roman Emperor", "Wikimedia disambiguation page")], {"Q1": CLAIMS, **LABELS})
    out = src.facts("Last Roman Emperor")
    assert src.last["qid"] == "Q1" and "meta-pages dropped" in src.last["how"] and len(out) == 1
    # every hit a meta-page: nothing is invented, the ambiguity stands
    src2 = FakeWikidata([_hit("Q8", "Mercury", "Wikimedia disambiguation page"), _hit("Q9", "Mercury", "Wikimedia list article")],
                        {"Q8": CLAIMS, "Q9": CLAIMS, **LABELS})
    assert src2.facts("Mercury") == [] and len(src2.last["ambiguous"]) == 2


def test_the_asker_chooses_the_item_and_the_ledger_guesses_only_when_no_context_does(tmp_path):
    """Nick, 2026-09-14: "when ambiguous it should either select A based on context or B ask the user";
    and "an algorithm that allows the vm to still be neutral while scoring most requested answers... to
    guess the context if none". Rebuilt after the model competition, which found four defects in the
    first cut: the key was the bare name (a new namesake would inherit the tally), it counted questions
    rather than askers (repeating a question stuffed the ballot), it never decayed, and it fired on a
    single observation. The ledger is host-side throughout: it chooses WHICH item to ask the source
    about, never what the VM verifies."""
    import time as _t
    hits = [_hit("Q1", "Mercury", "planet"), _hit("Q2", "Mercury", "chemical element")]
    src = _fake_choices(tmp_path, hits, {"Q1": CLAIMS, "Q2": CLAIMS, **LABELS})
    assert src.facts("Mercury") == [] and src.last["qid"] is None            # B: ambiguous, the asker is asked
    out = src.facts("Mercury", qid="Q2", asker="nick")                        # the asker answers
    assert src.last["qid"] == "Q2" and src.last["how"] == "chosen by the asker" and len(out) == 1
    key = src._key("Mercury", ["Q1", "Q2"])
    assert list(src.choices[key]) == ["nick"] and src.choices[key]["nick"]["item"] == "Q2"
    # the SAME asker said what they meant: one observation is enough, and it is named as theirs
    src.facts("Mercury", asker="nick")
    assert src.last["qid"] == "Q2" and src.last["how"] == "your earlier choice"
    # ...but it is not everyone's: another asker still gets the question
    src.facts("Mercury", asker="ada")
    assert src.last["qid"] is None and len(src.last["ambiguous"]) == 2
    # repeating a question is not a second vote
    for _ in range(9):
        src.facts("Mercury", qid="Q2", asker="nick")
    assert len(src.choices[key]) == 1
    src.facts("Mercury", asker="ada")
    assert src.last["qid"] is None                                            # one asker is not a quorum
    # three askers, past a coin flip: the ledger may bind, and says how loudly
    for who in ("ada", "bo", "cy"):
        src.facts("Mercury", qid="Q2", asker=who)
    src.facts("Mercury", asker="dee")
    assert src.last["qid"] == "Q2" and src.last["how"].startswith("asker history") and "askers" in src.last["how"]
    # a 51/49 split is not a preference
    for who in ("ed", "fi", "gus"):
        src.facts("Mercury", qid="Q1", asker=who)
    src.facts("Mercury", asker="dee")
    assert src.last["qid"] is None and len(src.last["ambiguous"]) == 2
    # what a name meant a year ago is not what it means now: those votes are dead, not faint
    old = int(_t.time()) - 400 * 86400
    src.choices[key] = {w: {"item": "Q1", "t": old} for w in ("ed", "fi", "gus", "hal")}
    src.facts("Mercury", asker="dee")
    assert src.last["qid"] is None
    # a context tier still outranks the ledger
    src.choices[key] = {w: {"item": "Q2", "t": int(_t.time())} for w in ("ada", "bo", "cy", "dee", "ed")}
    src._entities["Q1"] = {"claims": {"P31": [ITEM("Qplanet")], "P735": [ITEM("Q90")]}}
    src._entities["Q2"] = {"claims": {"P31": [ITEM("Qelement")]}}
    src._entities.update(LABELS2)
    src.facts("Mercury", relations=["given name"], asker="zoe")
    assert src.last["qid"] == "Q1" and src.last["how"].startswith("relation")
    # and the benches never see any of it
    plain = FakeWikidata(hits, {"Q1": CLAIMS, "Q2": CLAIMS, **LABELS})
    assert plain.use_choices is False and plain.preferred("Mercury", ["Q1", "Q2"]) is None
    plain.choose("Mercury", "Q1", ["Q1", "Q2"]); assert plain.choices == {}


def test_a_new_namesake_is_a_new_ambiguity_and_inherits_no_history(tmp_path):
    """The competition's Q5, unanimous across the field and absent from the first cut: auto-binding stops
    producing explicit picks, so the tally freezes; the source then gains a namesake; a ledger keyed on
    the bare name still unique-maxes the old sense, the VM certifies a true chain about the wrong item,
    and every fact-checking bench passes. The key is (name, candidate set), so a changed set has no
    history and the asker is asked again."""
    import time as _t
    two = [_hit("Q1", "Mercury", "planet"), _hit("Q2", "Mercury", "chemical element")]
    src = _fake_choices(tmp_path, two, {"Q1": CLAIMS, "Q2": CLAIMS, "Q3": CLAIMS, **LABELS})
    for who in ("ada", "bo", "cy", "dee"):
        src.facts("Mercury", qid="Q2", asker=who)
    src.facts("Mercury", asker="zoe")
    assert src.last["qid"] == "Q2"                                            # settled, on four askers
    src._search = two + [_hit("Q3", "Mercury", "Roman god")]                   # the source gains a namesake
    src.facts("Mercury", asker="zoe")
    assert src.last["qid"] is None and len(src.last["ambiguous"]) == 3         # a different ambiguity: ask
    assert src._key("Mercury", ["Q1", "Q2"]) != src._key("Mercury", ["Q1", "Q2", "Q3"])


def test_the_question_s_relations_narrow_in_order_but_only_across_different_kinds_of_thing():
    """Live, 2026-09-14: 'where is quebec city' stayed ambiguous between the city (Q2145) and an 1841
    electoral district -- BOTH carry `country`, so the union of the where-relations eliminated neither;
    the city alone carries `located in the administrative territorial entity`, which the where-ask asks
    for first. Narrowing is now lexicographic over the question's own relations. The guard: it may only
    separate candidates of DIFFERENT kinds -- two items of one kind are told apart by this only through
    an accident of which is more completely described, which is the James Young incident again."""
    CITY = {"claims": {"P31": [ITEM("Qcity")], "P17": [ITEM("Qca")], "P131": [ITEM("Qqc")]}}
    RIDING = {"claims": {"P31": [ITEM("Qriding")], "P17": [ITEM("Qca")]}}
    labels = {"P17": {"labels": {"en": {"value": "country"}}},
              "P131": {"labels": {"en": {"value": "located in the administrative territorial entity"}}},
              "Qca": {"labels": {"en": {"value": "Canada"}}}, "Qqc": {"labels": {"en": {"value": "Quebec"}}}}
    src = FakeWikidata([_hit("Q2145", "Quebec City", "capital city"), _hit("Q964", "Quebec City", "electoral district")],
                       {"Q2145": CITY, "Q964": RIDING, **labels})
    where = ["country", "located in the administrative territorial entity"]        # country first: it decides nothing
    out = src.facts("Quebec City", relations=where)
    assert src.last["qid"] == "Q2145" and src.last["how"] == "relation: located in the administrative territorial entity"
    assert ("Canada", "country") in [(t.obj, t.rel) for t in out]
    # two of ONE kind: the same relation evidence must not decide (the pinned James Young case)
    HAS = {"claims": {"P31": [ITEM("Qhuman")], "P569": [{"mainsnak": {"datatype": "time", "datavalue": {"type": "time", "value": {"time": "+1800-05-11T00:00:00Z", "precision": 11}}}}]}}
    HASNT = {"claims": {"P31": [ITEM("Qhuman")]}}
    src2 = FakeWikidata([_hit("Q1", "James Young", "chemist"), _hit("Q2", "James Young", "politician")],
                        {"Q1": HAS, "Q2": HASNT, **LABELS})
    assert src2.facts("James Young", relations=["date of birth"]) == []
    assert src2.last["qid"] is None and len(src2.last["ambiguous"]) == 2
