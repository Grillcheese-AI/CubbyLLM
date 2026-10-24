"""Everyday wording, read by the host (2026-10-24, `cubbyllm/reasoning/everyday.py`): "Who wrote X?" asks `author`, the
loop restates it in its own grammar and walks that plan; a question the table does not read goes to the model as
before; the flag off leaves the loop as it was.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning.everyday import EverydayForms, everyday_forms  # noqa: E402
from cubbyllm.reasoning.planner import Triple  # noqa: E402

FORMS = EverydayForms(everyday_forms(templates=""))      # the hand-written table alone: the tests need no dataset


def test_everyday_wording_names_the_relation():
    assert FORMS.parse("Who wrote Pride and Prejudice?")[:2] == ("Pride and Prejudice", "author")
    assert FORMS.parse("Who painted the Mona Lisa?")[:2] == ("Mona Lisa", "creator")
    assert FORMS.parse("What river flows through Cairo?")[:2] == ("Cairo", "located in or next to body of water")
    assert FORMS.parse("what timezone is denver co?")[:2] == ("denver co", "located in time zone")
    assert FORMS.parse("What currency is used in Hungary?")[:2] == ("Hungary", "currency")


def test_a_value_that_changes_over_a_term_is_left_to_the_model():
    # the world's snapshot would say "As of 2019, X" of a mayor, a team or a coach whose term had ended
    assert FORMS.parse("Who is the mayor of Montreal?") is None
    assert FORMS.parse("What team did Troy Aikman play for?") is None
    assert FORMS.parse("What college did Joe Namath attend?") is None


def test_french_articles_before_the_name():
    assert FORMS.parse("Quelle est la capitale de l'Italie ?")[:2] == ("Italie", "capital")
    assert FORMS.parse("Quelle est la capitale du Canada ?")[:2] == ("Canada", "capital")
    assert FORMS.parse("Qui a écrit Les Misérables ?")[:2] == ("Les Misérables", "author")


def test_bare_shapes_stay_the_profiles_and_unread_wording_goes_to_the_model():
    assert FORMS.parse("Where is the Globe Theatre?") is None        # names no relation: the profile's
    assert FORMS.parse("When was the Eiffel Tower built?") is None   # "built" is not in the table: the model's
    assert FORMS.parse("What is the speed of light?") is None


def test_a_hand_written_form_wins_over_a_template_that_words_another_relation_the_same(tmp_path):
    csv = tmp_path / "templates.csv"
    csv.write_text("relation,template,edited_relation,noun_template\n"
                   "capital of,What is the {relation} of {subject}?,capital,the capital of {subject}\n"
                   "commissioned by,Who {relation} {subject}?,commissioned,the entity that commissioned {subject}\n",
                   encoding="utf-8")
    f = EverydayForms(everyday_forms(templates=str(csv)))
    assert f.parse("What is the capital of France?")[:2] == ("France", "capital")          # not two relations: not None
    assert f.parse("Who commissioned the Night Watch?")[:2] == ("Night Watch", "commissioned by")


def test_two_relations_worded_alike_are_a_refusal():
    f = EverydayForms({"a": ("state", {"present": ["Who runs {S}?"]}), "b": ("state", {"present": ["Who runs {S}?"]})})
    assert f.parse("Who runs Acme?") is None


def test_restate_keeps_the_year_and_drops_the_noise():
    assert FORMS.restate("What is the capital of Spain in 2010?") == ("What is the capital of Spain in 2010?", "capital", "Spain")
    assert FORMS.restate("what is the currency in croatia right now?")[1:] == ("currency", "croatia")
    assert FORMS.restate("where is rome italy located on a map?") is None          # "Where is X located?" names no relation
    assert FORMS.restate("What is the speed of light?") is None


class Books:
    """An online source stand-in that names "wrote" as `notable work`, as Wikidata's property aliases do."""
    name = "wikidata"

    def facts(self, entity, relations=None, qid=None, asker=None, classes=False):
        if entity.lower() == "pride and prejudice":
            return [Triple(obj="Jane Austen", rel="author", subj="Pride and Prejudice")]
        return []

    def relations(self, wording):
        return {"wrote": ["notable work"], "author": ["author"]}.get(wording.lower(), [])


class NoModel:
    def emit(self, prompt, max_new_tokens=300, **kw):
        return "no plan\n"


def _loop(everyday):
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.worlds.base import FactStore
    from test_books import exact_vm
    return AskLoop(NoModel(), world=FactStore([]), source=Books(), lexicon=False, run_fn=exact_vm(), everyday=everyday)


def test_the_loop_walks_the_restated_plan():
    loop = _loop(True)
    loop._everyday = FORMS
    rec = loop.ask("Who wrote Pride and Prejudice?")
    assert rec["verified"] and rec["answer"] == "Jane Austen"
    assert rec["restated"] == "What is the author of Pride and Prejudice?" and rec["plan"] == ["author"]


def test_a_nested_question_is_left_to_the_models_plan():
    # the seed starts with a relation the table names: one hop would walk "country of citizenship of Marie" as a name
    assert FORMS.parse("What is the capital of the country of citizenship of Marie?") is None
    assert FORMS.parse("What currency is used in the capital of France?") is None
    assert FORMS.parse("Who wrote the national anthem of France?")[:2] == ("national anthem of France", "author")  # a name


def test_on_by_default_and_off_with_the_switch(monkeypatch):
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.worlds.base import FactStore
    monkeypatch.delenv("CB_EVERYDAY", raising=False)
    assert AskLoop(NoModel(), world=FactStore([]), source=Books(), lexicon=False)._everyday is not None   # Nick, 2026-10-24
    monkeypatch.setenv("CB_EVERYDAY", "0")
    assert AskLoop(NoModel(), world=FactStore([]), source=Books(), lexicon=False)._everyday is None


def test_off_the_loop_is_unchanged():
    loop = _loop(False)
    assert loop._everyday is None
    rec = loop.ask("Who wrote Pride and Prejudice?")
    assert not rec["verified"] and rec["reason"] == "no_plan" and "restated" not in rec
