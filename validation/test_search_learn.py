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
