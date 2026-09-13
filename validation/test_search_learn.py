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


def test_an_exact_alias_outranks_the_overlap_tier_that_would_read_the_inverse_relation():
    """hdc (2026-09-13): 'administrative territorial entity' is P131's own alias, and the
    walk's overlap tier (Jaccard >= 0.6) matched it to P150 'contains administrative
    territorial entity' -- the inverse -- so a store holding both directions verified the
    child for a question asking the parent. Lever 4 now runs AHEAD of the walk for a plan
    wording the store does not hold exactly: the table's one held label wins before any
    overlap can; two held labels are the ambiguity they always were."""
    from cubbyllm.reasoning.planner import QuestionPlan
    both = STORE + ["haryana is the located in the administrative territorial entity of nurpur",
                    "nurpur ward is the contains administrative territorial entity of nurpur"]
    store, known = LookupStore(both), StoreRelations(both)
    src = AliasSource({}, {"administrative territorial entity": ["located in the administrative territorial entity"]})
    plan = QuestionPlan(relations=[None], tail="administrative territorial entity of nurpur", n_hop=1)
    r = run("Where is the administrative territorial entity of Nurpur?", store, known, src, plan=plan)
    assert r.first.verified and normalize(r.result.answer) == "haryana"
    assert r.aliased == [("administrative territorial entity", "located in the administrative territorial entity")]
    assert r.plan.tail == "located in the administrative territorial entity of nurpur"
    # the table naming both held directions: refused before the overlap tier can pick one
    src2 = AliasSource({}, {"administrative territorial entity": ["located in the administrative territorial entity",
                                                                  "contains administrative territorial entity"]})
    r2 = run("Where is the administrative territorial entity of Nurpur?", LookupStore(both), StoreRelations(both), src2, plan=plan)
    assert not r2.result.verified and r2.result.reason == "ambiguous_relation" and r2.result.answer is None
    assert sorted(r2.result.refused["candidates"]) == ["contains administrative territorial entity",
                                                       "located in the administrative territorial entity"]


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


# ---- ask-type narrowing (2026-09-13): the local property table names BOTH senses of a wording

class TypedAliasSource(AliasSource):
    """The local table's shape: a wording names every property it is an alias of, and
    each label has the value kind its datatype gives it."""
    def __init__(self, by_entity, by_wording, kinds):
        super().__init__(by_entity, by_wording); self.kinds = {normalize(k): v for k, v in kinds.items()}
    def kind(self, label):
        return self.kinds.get(normalize(label))


BORN_STORE = STORE + ["1932-03-23 is the date of birth of masaki tsuji", "nagoya is the place of birth of masaki tsuji"]
BORN_KINDS = {"date of birth": "date", "place of birth": "name"}


def test_a_when_question_keeps_the_date_valued_label_of_a_two_sense_wording_at_lever_4():
    """'born' is an alias of `date of birth` AND `place of birth` in the local table
    (the API's search had ranked one). Both are held, so lever 4 saw an ambiguity;
    the question asks WHEN, the datatypes say which label is a date, exactly one is,
    and the host translates to it -- the pick is the question's, not the model's."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(BORN_STORE), StoreRelations(BORN_STORE)
    src = TypedAliasSource({}, {"born": ["date of birth", "place of birth"]}, BORN_KINDS)
    plan = QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.first.verified                     # lever 4 runs ahead of the first walk since 2026-09-13 (hdc)
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23"
    assert r.aliased == [("born", "date of birth")]


def test_a_two_sense_wording_stays_an_ambiguity_when_the_ask_type_does_not_split_it():
    """A 'what' question names no ask type: nothing narrows, the refusal stands with both
    candidates on record. 'Where' (a place, since 2026-09-13) keeps the name-valued label."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(BORN_STORE), StoreRelations(BORN_STORE)
    src = TypedAliasSource({}, {"born": ["date of birth", "place of birth"]}, BORN_KINDS)
    plan = QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1)
    r = run("What is known about how Masaki Tsuji was born?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "ambiguous_relation"
    assert r.result.refused == {"relation": "born", "candidates": ["date of birth", "place of birth"]}
    r = run("Where was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.result.verified and normalize(r.result.answer) == "nagoya" and r.aliased == [("born", "place of birth")]
    r = run("In which city was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.result.verified and normalize(r.result.answer) == "nagoya"       # 'city' is the ask, not a dropped hop


def test_a_candidate_without_a_kind_blocks_the_narrowing():
    """A label the table cannot type might be date-valued too; choosing among those
    would be a pick, so the ambiguity stands even for a 'when' question."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(BORN_STORE), StoreRelations(BORN_STORE)
    src = TypedAliasSource({}, {"born": ["date of birth", "place of birth"]}, {"date of birth": "date"})
    plan = QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "ambiguous_relation"


def test_a_when_question_finds_its_wording_at_lever_6_when_the_table_names_two_labels():
    """Lever 6 with the local table: the plan names `date of birth`, 'born' in the
    question names both held labels; the ask type keeps the plan's, the alias is
    recorded, the plan walks. A plan naming `place of birth` for a WHEN question gets
    no alias from 'born' -- the ask type kept the other label -- and stays refused."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(BORN_STORE), StoreRelations(BORN_STORE)
    src = TypedAliasSource({}, {"born": ["date of birth", "place of birth"]}, BORN_KINDS)
    plan = QuestionPlan(relations=[None], tail="date of birth of masaki tsuji", n_hop=1)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert r.first.reason == "plan_does_not_cover_question"
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23" and r.aliased == [("born", "date of birth")]
    plan = QuestionPlan(relations=[None], tail="place of birth of masaki tsuji", n_hop=1)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=plan)
    assert not r.result.verified and r.result.reason == "plan_does_not_cover_question" and r.aliased == []


# ---- the store's own wording of a property (2026-09-13): 'birth date' for 'date of birth'

class WordedAliasSource(TypedAliasSource):
    """The local table's other answer: every wording of a label's property."""
    def __init__(self, by_entity, by_wording, kinds, by_label):
        super().__init__(by_entity, by_wording, kinds); self.by_label = {normalize(k): v for k, v in by_label.items()}
    def wordings(self, label):
        return list(self.by_label.get(normalize(label), []))


WIKI_STORE = STORE + ["1932-03-23 is the birth date of masaki tsuji", "nagoya is the birthplace of masaki tsuji"]
WIKI_TABLE = dict(by_wording={"born": ["date of birth", "place of birth"], "date of birth": ["date of birth"],
                              "birth date": ["date of birth"], "birthplace": ["place of birth"], "dob": ["date of birth"]},
                  kinds={"date of birth": "date", "place of birth": "name", "birth date": "date", "birthplace": "name", "dob": "date"},
                  by_label={"date of birth": ["date of birth", "birth date", "born", "DOB"],
                            "place of birth": ["place of birth", "birthplace", "born"]})


def test_the_store_holding_the_property_under_another_wording_is_the_target():
    """The wiki world says 'birth date'; the emitter says 'born' (lever 4 case) or the
    canonical 'date of birth' (a frontier proposer's habit); neither is held, and until
    today both died as `unknown_relation` with the fact sitting in the store. The table
    names every wording of the property, the held one is the target, the ask type splits
    'born' as before."""
    from cubbyllm.reasoning.planner import QuestionPlan
    store, known = LookupStore(WIKI_STORE), StoreRelations(WIKI_STORE)
    src = WordedAliasSource({}, **WIKI_TABLE)
    r = run("When was Masaki Tsuji born?", store, known, src, plan=QuestionPlan(relations=[None], tail="born of masaki tsuji", n_hop=1))
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23" and r.aliased == [("born", "birth date")]
    r = run("What is the date of birth of Masaki Tsuji?", store, known, src,
            plan=QuestionPlan(relations=[None], tail="date of birth of masaki tsuji", n_hop=1))
    assert r.result.verified and normalize(r.result.answer) == "1932 03 23" and r.aliased == [("date of birth", "birth date")]
    assert r.entities == []                                        # nothing was fetched: the store had it


def test_a_shared_wording_the_store_holds_is_no_evidence_of_either_property():
    """'born' is a wording of both properties; a store that happens to hold a relation
    called 'born' is not thereby holding date of birth -- the fallback names only wordings
    that belong to ONE property, so the plan stays unknown and the loop fetches."""
    from cubbyllm.reasoning.planner import QuestionPlan
    base = STORE + ["something is the born of masaki tsuji"]
    store, known = LookupStore(base), StoreRelations(base)
    src = WordedAliasSource({}, **dict(WIKI_TABLE, by_wording={"date of birth": ["date of birth"], "born": ["date of birth", "place of birth"]}))
    r = run("What is the date of birth of Masaki Tsuji?", store, known, src,
            plan=QuestionPlan(relations=[None], tail="date of birth of masaki tsuji", n_hop=1))
    assert not r.result.verified and r.aliased == [] and r.result.reason in ("unknown_relation", "retrieval_exhausted")


def test_a_held_relation_that_finds_nothing_is_walked_under_the_propertys_other_held_wording():
    """The gen-3 builder (2026-09-13): the wiki world holds 'birthplace', the encyclopedia's
    facts arrive as 'place of birth'; both are held, an entity's fact sits under one. A
    plan naming the other finds nothing -- and used to fetch, with the fact in the store.
    One translation to the property's other held wording, one more walk; two held others
    is not a translation (the ask type may still split them)."""
    from cubbyllm.reasoning.planner import QuestionPlan
    base = STORE + ["tokyo is the birthplace of hans", "nagoya is the place of birth of masaki tsuji"]
    store, known = LookupStore(base), StoreRelations(base)
    src = WordedAliasSource({}, **dict(WIKI_TABLE, by_wording=dict(WIKI_TABLE["by_wording"], **{"place of birth": ["place of birth"]})))
    r = run("What is the birthplace of Masaki Tsuji?", store, known, src,
            plan=QuestionPlan(relations=[None], tail="birthplace of masaki tsuji", n_hop=1))
    assert r.first.reason == "retrieval_exhausted"
    assert r.result.verified and normalize(r.result.answer) == "nagoya" and r.aliased == [("birthplace", "place of birth")]
    assert r.entities == []                                        # translated, not fetched
    r = run("What is the place of birth of Hans?", store, known, src,
            plan=QuestionPlan(relations=[None], tail="place of birth of hans", n_hop=1))
    assert r.result.verified and normalize(r.result.answer) == "tokyo" and r.aliased == [("place of birth", "birthplace")]
    # two translations in a row ('born' -> 'place of birth' by the ask type, then -> 'birthplace'
    # by the sibling step): the question's own word stays on record, so coverage still sees it
    r = run("Where was Hans born?", store, known, src, plan=QuestionPlan(relations=[None], tail="born of hans", n_hop=1))
    assert r.result.verified and normalize(r.result.answer) == "tokyo"
    assert r.aliased == [("born", "place of birth"), ("place of birth", "birthplace")]


def test_the_question_decides_where_an_unheld_relation_ends_and_the_entity_begins():
    """'married of anne of cleves': the relation is not held, so the last ' of ' used to
    split the tail into 'married of anne' | 'cleves' -- an unknown relation nobody could
    resolve and a seed that is nobody. The question says 'Anne of Cleves'; the split
    that names what the question names wins (2026-09-13, the gen-3 builder: 67 of 127
    'married to' questions died there)."""
    from cubbyllm.reasoning.plan_verify import split_tail
    from cubbyllm.reasoning.planner import QuestionPlan
    base = STORE + ["henry viii is the spouse of anne of cleves", "a is the spouse of anne", "b is the spouse of c"]
    store, known = LookupStore(base), StoreRelations(base)
    assert split_tail("married of anne of cleves", known, "Who is Anne of Cleves married to?") == ("married", "anne of cleves")
    assert split_tail("married of anne of cleves", known, None) == ("married of anne", "cleves")      # without the question, the old rule
    assert split_tail("spouse of anne of cleves", known, "Who is the spouse of Anne of Cleves?") == ("spouse", "anne of cleves")
    # a store whose greedy parse once held the fragment 'spouse of anne' beside a reused 'spouse':
    # the reused relation wins the held-prefix tier over a one-off, undeclared fragment; a declared
    # one-off ('date of birth', from a source's Triple) keeps its length
    frag = StoreRelations.from_keys(["spouse", "spouse of anne"], {"spouse": 5, "spouse of anne": 1})
    assert split_tail("spouse of anne of cleves", frag, None) == ("spouse", "anne of cleves")
    dec = StoreRelations.from_keys(["date", "date of birth"], {"date": 2, "date of birth": 1}); dec.declare("date of birth")
    assert split_tail("date of birth of masaki tsuji", dec, None) == ("date of birth", "masaki tsuji")
    # the question joins 'place' to 'birth' with 'of' itself: the relation is 'place of birth', as before
    assert split_tail("place of birth of jean", StoreRelations(STORE), "What is the place of birth of Jean?") == ("place of birth", "jean")
    assert split_tail("born of george national gallery of art", None, "When was George National Gallery of Art born?") == ("born", "george national gallery of art")
    src = TypedAliasSource({}, {"married": ["spouse"]}, {"spouse": "name"})
    r = run("Who is Anne of Cleves married to?", store, known, src,
            plan=QuestionPlan(relations=[None], tail="married of anne of cleves", n_hop=1))
    assert r.result.verified and normalize(r.result.answer) == "henry viii" and r.aliased == [("married", "spouse")]


def test_a_garbled_seed_is_snapped_to_the_questions_own_spelling():
    """exp_r17 (2026-09-13): the gen-2 emitter re-types 'Karl Brugmann' as 'karol burgmann'
    and 'Jean Louis Barthou' as 'jean louis bastu'; the relation is right and the plan
    dies for coverage. The question spells the entity; the closest n-gram (>= 0.85, a
    unique best) replaces the seed, on record, and the gate runs on the snapped plan. A
    seed the question contains is left alone; one far from anything in it is not snapped."""
    from cubbyllm.reasoning.learn import snap_seed
    from cubbyllm.reasoning.planner import QuestionPlan
    base = STORE + ["1849-03-16 is the birth date of karl brugmann"]
    store, known = LookupStore(base), StoreRelations(base)
    src = WordedAliasSource({}, **WIKI_TABLE)
    plan = QuestionPlan(relations=[None], tail="born of karol burgmann", n_hop=1)
    snapped, rec = snap_seed("When was Karl Brugmann born?", plan, known)
    assert rec == ("karol burgmann", "karl brugmann") and snapped.tail == "born of karl brugmann"
    r = run("When was Karl Brugmann born?", store, known, src, plan=plan)
    assert r.result.verified and normalize(r.result.answer) == "1849 03 16" and r.snapped == ("karol burgmann", "karl brugmann")
    assert snap_seed("When was Karl Brugmann born?", QuestionPlan(relations=[None], tail="born of karl brugmann", n_hop=1), known)[1] is None
    assert snap_seed("When was Karl Brugmann born?", QuestionPlan(relations=[None], tail="born of ludwig wittgenstein", n_hop=1), known)[1] is None


# ---- the latent tier (2026-09-13): a local model proposes facts, and a second source has to agree

class LatentSource(DictSource):
    name = "lfm"; latent = True


class AttestingSource(DictSource):
    name = "wikidata"


class ProvStore(LookupStore):
    """A store that keeps provenance per fact, the way `worlds.FactStore` does."""
    def __init__(self, texts):
        self.provenance = {}; super().__init__(texts)
    @staticmethod
    def _key(text):
        return " ".join(text.split())


def test_a_latent_sources_fact_is_stored_but_never_spoken_until_a_second_source_agrees():
    """LFM2.5 (the emitter's own base) holds facts in its weights; it may propose them, through the
    gate, with provenance -- but a chain that rests on an LFM-only fact is refused as `latent_only`
    with the would-be answer on record. When an attesting source later states the same fact, the
    duplicate lifts the hold and the walk speaks."""
    store, known = ProvStore(STORE), StoreRelations(STORE)
    lfm = LatentSource({"marie": ["canada is the country of citizenship of marie"]})
    q = "What is the capital of the country of citizenship of Marie?"
    r = run(q, store, known, lfm)
    assert not r.result.verified and r.result.reason == "retrieval_exhausted"     # the fact is in, the capital of canada is not
    assert store.provenance["canada is the country of citizenship of marie"] == "lfm (latent)"
    # give the store the second hop from an ordinary source, then ask again: the LFM fact is still the hold
    store.add("ottawa is the capital of canada"); known.add("ottawa is the capital of canada")
    r = run(q, store, known, lfm)
    assert not r.result.verified and r.result.reason == "latent_only"
    assert normalize(r.result.refused["answer"]) == "ottawa" and r.result.refused["latent_facts"] == ["canada is the country of citizenship of marie"]
    assert store.provenance["canada is the country of citizenship of marie"] == "lfm (latent)"
    # an attesting source states the same fact: the duplicate lifts the hold, both names on record, the answer is spoken
    r = run(q, store, known, AttestingSource({"marie": ["canada is the country of citizenship of marie"]}))
    assert r.result.verified and normalize(r.result.answer) == "ottawa"
    assert store.provenance["canada is the country of citizenship of marie"] == "lfm+wikidata"
    # a store without provenance cannot hold the tier: the latent flag is inert there (documented)
    plain, known2 = LookupStore(STORE + ["ottawa is the capital of canada"]), StoreRelations(STORE + ["ottawa is the capital of canada"])
    r = run(q, plain, known2, LatentSource({"marie": ["canada is the country of citizenship of marie"]}))
    assert r.result.verified


# ---- every step as an event (2026-09-13, Nick: "all process should be listened to at every step")

def test_every_step_of_the_loop_is_an_event_linked_to_its_parent_and_no_listener_changes_nothing():
    """A listener sees the run as a tree: question -> plan -> walk -> hop -> fact, question -> fetch ->
    gate, and a final answer; each event's parent is an earlier event of the same run. Without a
    listener the loop returns exactly the same result."""
    from cubbyllm.reasoning import events as ev
    src = DictSource({"marie": ["canada is the country of citizenship of marie", "ottawa is the capital of canada"]})
    q = "What is the capital of the country of citizenship of Marie?"
    quiet = run(q, LookupStore(STORE), StoreRelations(STORE), src)
    sink = ev.MemorySink(); ev.add_sink(sink)
    try:
        heard = run(q, LookupStore(STORE), StoreRelations(STORE), DictSource(src.by))
    finally:
        ev.remove_sink(sink)
    assert heard.result.verified and normalize(heard.result.answer) == "ottawa"
    assert (quiet.result.verified, quiet.result.answer, quiet.entities) == (heard.result.verified, heard.result.answer, heard.entities)
    kinds = [e["kind"] for e in sink]
    assert kinds[0] == "question" and kinds[1] == "plan" and kinds[-1] == "answer"
    assert {"walk", "hop", "fact", "fetch", "gate", "vm"} <= set(kinds)
    ids = {e["id"] for e in sink}
    root = sink[0]["id"]
    assert all(e["parent"] in ids for e in sink[1:]) and all(e["parent"] is None for e in sink[:1])
    gates = [e for e in sink if e["kind"] == "gate"]
    assert {g["status"] for g in gates} == {"accepted"} and all(g["provenance"] is None or "test" in g["provenance"] for g in gates)
    fetch = next(e for e in sink if e["kind"] == "fetch")
    assert fetch["parent"] == root and fetch["entity"].lower() == "marie" and fetch["n"] == 2
    final = sink[-1]
    assert final["parent"] == root and final["verified"] and normalize(final["answer"]) == "ottawa" and final["entities"] == heard.entities
    # nothing leaks between runs: a fresh sink hears only its own run
    sink2 = ev.MemorySink(); ev.add_sink(sink2)
    try:
        run("What is the capital of France?", LookupStore(STORE), StoreRelations(STORE), DictSource({}))
    finally:
        ev.remove_sink(sink2)
    assert sink2[0]["kind"] == "question" and sink2[0]["text"] == "What is the capital of France?" and sink2[-1]["kind"] == "answer"
