"""Pins for the synonym oracle (`cubbyllm.reasoning.lexicon`, 2026-09-12, lever 5) and its
place in the learn loop as a second relation resolver.

The data file is built by standin/data/build_lexicon.py from WordNet 3.0 + WOLF 1.0b4
(both off-repo); the pins that need it skip when it is absent. The loop pins use a
hand-written lexicon file so they run anywhere.
Run: python -m pytest validation/test_lexicon.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.lexicon import DEFAULT, Lexicon  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan, normalize  # noqa: E402
from test_search_learn import STORE, DictSource, LookupStore, run  # noqa: E402

needs_data = pytest.mark.skipif(not DEFAULT.exists(), reason="lexicon_en_fr.jsonl not built")


@needs_data
def test_the_built_lexicon_joins_english_lemmas_and_french_literals():
    L = Lexicon()
    assert len(L) > 100_000
    assert "place of birth" in L.relations("birthplace") and "birthplace" in L.relations("place of birth")
    assert "spouse" in L.relations("conjoint")                          # FR literal -> EN wording
    assert "country" in L.relations("pays")
    assert "conjoint" in L.french("spouse")
    assert L.relations("no such wording at all") == []


def test_the_lexicon_is_a_second_resolver_and_ambiguity_is_still_refused(tmp_path):
    lex = tmp_path / "lex.jsonl"
    lex.write_text("\n".join(json.dumps(r) for r in [
        {"id": "eng-30-00000001-n", "pos": "n", "en": ["birthplace", "place of birth"], "fr": ["lieu de naissance"], "gloss": ""},
        {"id": "eng-30-00000002-n", "pos": "n", "en": ["position", "location"], "fr": [], "gloss": ""},
        {"id": "eng-30-00000003-n", "pos": "n", "en": ["position", "ranking"], "fr": [], "gloss": ""},
    ]), encoding="utf-8")
    L = Lexicon(lex)
    base = STORE + ["lyon is the place of birth of jean", "third is the ranking of jean", "paris is the location of jean"]
    # EN synonym: 'birthplace' -> the one relation the store holds
    store, known = LookupStore(base), StoreRelations(base)
    plan = QuestionPlan(relations=[None], tail="birthplace of jean", n_hop=1)
    r = run("What is the birthplace of Jean?", store, known, DictSource({}), plan=plan, resolvers=[L])
    assert r.aliased == [("birthplace", "place of birth")] and r.result.verified and normalize(r.result.answer) == "lyon"
    # FR literal: the question in French, the plan in the question's words, the store in English
    store, known = LookupStore(base), StoreRelations(base)
    plan = QuestionPlan(relations=[None], tail="lieu de naissance of jean", n_hop=1)
    r = run("Quel est le lieu de naissance de Jean?", store, known, DictSource({}), plan=plan, resolvers=[L])
    assert r.aliased == [("lieu de naissance", "place of birth")] and r.result.verified
    # a polysemous wording naming two held relations is an ambiguity, not a pick
    store, known = LookupStore(base), StoreRelations(base)
    plan = QuestionPlan(relations=[None], tail="position of jean", n_hop=1)
    r = run("What is the position of Jean?", store, known, DictSource({}), plan=plan, resolvers=[L])
    assert r.result.reason == "ambiguous_relation" and r.result.refused["candidates"] == ["location", "ranking"]
