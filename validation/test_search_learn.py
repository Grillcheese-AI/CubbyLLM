"""Pins for search-and-learn (`cubbyllm.reasoning.learn`, 2026-09-11, coverage lever 3).

What is pinned:
  * a question the store cannot answer is refused / fails with a reason; with a
    source that holds the missing fact, the loop fetches it, gates it, stores it
    WITH provenance, walks again and the VM verifies -- and the fact is there for
    the next question with no fetch;
  * the gate: a duplicate is not re-stored, a contradicting fact is refused and
    the clash named, a non-template line is refused -- none of them reach the store;
  * a source that returns nothing useful leaves the refusal exactly as it was;
  * a question the store answers never calls the source (zero fetches);
  * the loop is bounded: one round, at most `max_entities` lookups.
Run: python -m pytest validation/test_search_learn.py -q
"""
from __future__ import annotations

import pathlib
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.index import TripleIndex  # noqa: E402
from cubbyllm.reasoning.learn import learn_and_answer, gate  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402


class LookupStore:
    """The minimum store the loop needs: texts, membership, add, a TripleIndex, lookup."""
    def __init__(self, texts):
        self.texts = []; self._seen = set(); self.index = TripleIndex(texts)
        for t in texts: self.add(t)
    def add(self, fact):
        key = " ".join(fact.split())
        if key in self._seen: return False
        self._seen.add(key); self.texts.append(key); self.index.add(key); return True
    def __contains__(self, fact):
        return " ".join(fact.split()) in self._seen
    def lookup(self, plan, hop, entity):
        return self.index.hop(plan, hop, entity)


class DictSource:
    name = "test"
    def __init__(self, by_entity): self.by = {normalize(k): v for k, v in by_entity.items()}; self.calls = []
    def facts(self, entity):
        self.calls.append(entity); return list(self.by.get(normalize(entity), []))


STORE = ["paris is the capital of france", "france is the country of citizenship of jean",
         "berlin is the capital of germany", "germany is the country of citizenship of hans"]
NO_SEARCH = lambda q, k: []


def run(q, store, known, source, **kw):
    return learn_and_answer(q, NO_SEARCH, faithful_vm([]), store=store, known=known, source=source,
                            tau_vm=0.5, **kw)


def test_a_missing_fact_is_fetched_gated_stored_with_provenance_and_verified():
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = DictSource({"marie": ["canada is the country of citizenship of marie"]})
    q = "What is the capital of the country of citizenship of Marie?"
    r = run(q, store, known, src)
    assert r.first.reason == "retrieval_exhausted" and not r.first.verified
    assert [normalize(e) for e in r.entities] == ["marie", "canada"] and r.fetched == 1   # round 2 asked about canada; the source had nothing
    assert [p.status for p in r.learned] == ["accepted"]
    p = r.accepted[0]
    assert p.source == "test" and normalize(p.entity) == "marie" and len(p.snapshot_before) == 16
    assert "canada is the country of citizenship of marie" in store
    # the store learned the edge but not Canada's capital: the second walk fails at hop 1, honestly,
    # and the round that asks about canada admits nothing, which ends the loop. A source that
    # knows both closes it in two rounds (max_entities).
    assert not r.result.verified and r.result.reason == "retrieval_exhausted" and r.result is not r.first
    src2 = DictSource({"marie": ["canada is the country of citizenship of marie"], "canada": ["ottawa is the capital of canada"]})
    store2, known2 = LookupStore(STORE), StoreRelations(STORE)
    r2 = run(q, store2, known2, src2)
    assert r2.result.verified and normalize(r2.result.answer) == "ottawa"
    assert [p.status for p in r2.learned] == ["accepted", "accepted"] and [normalize(e) for e in r2.entities] == ["marie", "canada"]
    # next time: no fetch
    src3 = DictSource({})
    r3 = run(q, store2, known2, src3)
    assert r3.result.verified and src3.calls == [] and r3.learned == []


def test_the_gate_refuses_duplicates_and_non_facts_and_records_siblings():
    store = LookupStore(STORE)
    assert gate("paris is the capital of france", store, "t", "france").status == "duplicate"
    assert gate("france has a lovely capital", store, "t", "france").status == "unparseable"
    # a sibling (same subject and relation, another object): a contradiction under the
    # functional rule (a user-taught fact), admitted-and-recorded by default (a source
    # states multi-valued relations as sets)
    c = gate("lyon is the capital of france", store, "t", "france", functional=True)
    assert c.status == "contradiction" and c.clash == "paris is the capital of france"
    s = gate("lyon is the capital of france", store, "t", "france")
    assert s.status == "accepted" and s.clash == "paris is the capital of france"


def test_a_poisoned_value_becomes_an_ambiguous_refusal_never_an_answer():
    """The gate keeps the store honest about what it holds, not about what is true;
    truth is decided at answer time: two objects for one hop is a refusal that names
    both, with provenance on each -- retire-never-delete territory, never a guess."""
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = DictSource({"marie": ["france is the country of citizenship of marie",
                                "lyon is the capital of france"]})              # the poison rides along
    q = "What is the capital of the country of citizenship of Marie?"
    r = run(q, store, known, src)
    assert [p.status for p in r.learned] == ["accepted", "accepted"]
    assert r.learned[1].clash == "paris is the capital of france"
    assert not r.result.verified and r.result.reason == "ambiguous_hop"
    assert r.result.refused["hop"] == 1 and r.result.refused["objects"] == ["lyon", "paris"]
    assert r.result.answer is None
    # and the same question with nothing poisoned verifies through the learned edge
    store2, known2 = LookupStore(STORE), StoreRelations(STORE)
    r2 = run(q, store2, known2, DictSource({"marie": ["france is the country of citizenship of marie"]}))
    assert r2.result.verified and normalize(r2.result.answer) == "paris"


def test_an_answerable_question_never_calls_the_source():
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = DictSource({"jean": ["mars is the country of citizenship of jean"]})
    r = run("What is the capital of the country of citizenship of Jean?", store, known, src)
    assert r.result.verified and normalize(r.result.answer) == "paris"
    assert src.calls == [] and r.fetched == 0


def test_unknown_relation_fetches_the_seed_and_the_loop_is_bounded():
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = DictSource({"jean": ["1950 is the date of birth of jean"]})
    r = run("What is the date of birth of Jean?", store, known, src, max_entities=1)
    assert r.first.reason == "unknown_relation"
    assert [normalize(e) for e in r.entities] == ["jean"] and r.accepted and "date of birth" in known
    assert r.result.verified and normalize(r.result.answer) == "1950"
    assert len(src.calls) == 1


# ---- lever 4 (2026-09-11): the source resolves the plan's relation words to its labels ----

class AliasSource(DictSource):
    def __init__(self, by_entity, by_wording):
        super().__init__(by_entity); self.rel = {normalize(k): v for k, v in by_wording.items()}; self.rel_calls = []
    def relations(self, text):
        self.rel_calls.append(text); return list(self.rel.get(normalize(text), []))


def test_an_unknown_relation_the_source_can_name_is_rewritten_and_walked():
    """'born' shares no word with 'date of birth': neither paraphrase tier bridges
    them. The source names the label, the host keeps the one the store holds (after
    the facts about the seed were learned), rewrites the plan, records the
    translation, and the coverage check still sees 'born' in the question."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = AliasSource({"masaki tsuji": ["1932-03-23 is the date of birth of Masaki Tsuji",
                                        "Japan is the country of citizenship of Masaki Tsuji"]},
                      {"born": ["date of birth"], "citizenship": ["country of citizenship"]})
    plan = QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1)      # the emitter's words
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.first.reason == "unknown_relation"
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23"
    assert r.aliased == [("born", "date of birth")] and r.plan.tail == "date of birth of masaki tsuji"
    assert [normalize(e) for e in r.entities] == ["masaki tsuji"]        # the facts came first, then the name


def test_a_wording_that_names_two_held_relations_is_an_ambiguity_not_a_pick():
    from cubbyllm.reasoning.planner import QuestionPlan
    store = LookupStore(STORE + ["lyon is the location of jean", "third is the ranking of jean"])
    known = StoreRelations(store.texts)
    src = AliasSource({}, {"position": ["location", "ranking", "coordinate location"]})
    plan = QuestionPlan(relations=[None], tail="position of jean", n_hop=1)
    r = run("What is the position of Jean?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "ambiguous_relation"
    assert r.result.refused == {"relation": "position", "candidates": ["location", "ranking"]}
    assert r.aliased == [] and r.result.answer is None


def test_a_wording_the_source_cannot_name_leaves_the_refusal_as_it_was():
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = AliasSource({"jean": ["1950 is the date of birth of jean"]}, {})
    plan = QuestionPlan(relations=[None], tail="birthday of jean", n_hop=1)
    r = run("What is the birthday of Jean?", store, known, src, plan=plan)
    assert r.result.reason == "unknown_relation" and r.aliased == []
    assert "1950 is the date of birth of jean" in store          # learned, honestly unreachable under that name


def test_a_structured_fact_is_split_where_the_source_says_the_relation_ends():
    """exp_r11 alias run: the wiki world reuses a relation 'date' (2 facts) and had
    never seen 'date of birth', so the learned string 'date of birth of Masaki Tsuji'
    split as 'date' | 'birth of Masaki Tsuji' -- indexed under a subject that is not
    an entity, and the vocabulary never gained 'date of birth', so the alias step had
    nothing to rewrite to. A source that hands over a Triple declares its relation
    before the string is split."""
    from cubbyllm.reasoning.planner import QuestionPlan, Triple
    base = STORE + ["1990 is the date of jean", "1991 is the date of hans"]           # 'date' is a reused relation here
    store, known = LookupStore(base), StoreRelations(base)
    src = AliasSource({"masaki tsuji": [Triple(obj="1932-03-23", rel="date of birth", subj="Masaki Tsuji")]},
                      {"born": ["date of birth"]})
    plan = QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert "date of birth" in known
    assert [f for f, _t in store.index._by_subj["masaki tsuji"]] == ["1932-03-23 is the date of birth of Masaki Tsuji"]
    assert r.aliased == [("born", "date of birth")] and r.result.verified and normalize(r.result.answer) == "1932 03 23"


# ---- lever 6 (2026-09-12): the proposer names the canonical label, the question says it in its own words

def test_a_canonical_label_the_question_names_in_its_own_words_is_covered_and_walked():
    """exp_r11 Gemini run 1: a frontier proposer plans 'date of birth' for 'when was X
    born'; the label is HELD, so lever 4 never fires, and the question has none of its
    words, so coverage refused 114 of 121 plans. The host asks the resolvers which
    wording in the question names the label; 'born' does, is recorded as the alias,
    and the plan walks."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store = LookupStore(STORE + ["1932-03-23 is the date of birth of masaki tsuji"])
    known = StoreRelations(store.texts)
    src = AliasSource({}, {"born": ["date of birth"]})
    plan = QuestionPlan(relations=[None], tail="date of birth of masaki tsuji", n_hop=1)   # the label, not the words
    r = run("On what day, month, and year was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.first.reason == "plan_does_not_cover_question"
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23"
    assert r.aliased == [("born", "date of birth")] and r.entities == []          # no fetch was needed
    assert all(normalize(c) not in ("date", "of", "birth") for c in src.rel_calls)  # frame words are never asked


def test_a_question_wording_naming_two_held_labels_is_refused_not_picked():
    from cubbyllm.reasoning.planner import QuestionPlan
    store = LookupStore(STORE + ["lyon is the location of jean", "third is the ranking of jean"])
    known = StoreRelations(store.texts)
    src = AliasSource({}, {"position": ["location", "ranking"]})
    plan = QuestionPlan(relations=[None], tail="location of jean", n_hop=1)     # the model picked one sense
    r = run("What is the position of Jean?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "ambiguous_relation"
    assert r.result.refused == {"wording": "position", "candidates": ["location", "ranking"]}


def test_a_dropped_hop_is_still_refused_after_the_wording_is_found():
    """The residual rule is untouched: 'born' names the label, but 'father' is a hop
    the plan dropped, and the store knows the word."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store = LookupStore(STORE + ["1932-03-23 is the date of birth of masaki tsuji", "kenji is the father of masaki tsuji"])
    known = StoreRelations(store.texts)
    src = AliasSource({}, {"born": ["date of birth"]})
    plan = QuestionPlan(relations=[None], tail="date of birth of masaki tsuji", n_hop=1)
    r = run("When was the father of Masaki Tsuji born?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "plan_does_not_cover_question"


def test_a_label_the_store_does_not_hold_yet_is_worded_then_fetched_then_walked():
    """exp_r11 Gemini run 3: 'inception' for 'founded' was refused for coverage and,
    because coverage is judged before the relation, never reached the fetch that
    brings the label. The wording is found first, the walk then refuses the unknown
    relation, the fetch declares it, the second walk verifies."""
    from cubbyllm.reasoning.planner import QuestionPlan, Triple
    store, known = LookupStore(STORE), StoreRelations(STORE)
    src = AliasSource({"persina nature park": [Triple(obj="1997", rel="inception", subj="Persina Nature Park")]},
                      {"founded": ["inception"], "inception": ["inception"]})
    plan = QuestionPlan(relations=[None], tail="inception of persina nature park", n_hop=1)
    r = run("In what year was Persina Nature Park founded?", store, known, src, plan=plan)
    assert r.first.reason == "plan_does_not_cover_question"
    assert r.result.verified and r.result.answer == "1997"
    assert ("founded", "inception") in r.aliased and [normalize(e) for e in r.entities] == ["persina nature park"]
