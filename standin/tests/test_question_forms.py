"""Questions as people ask them (standin/question_forms.py, order 0012): `{wh} {tense} {verb form} {subject}
[in {year}]?`, read back to (entity, relation, tense).

Pinned: every form of every relation, in every tense, parses back to its own (entity, relation, tense);
the hdc table's present-tense templates come out in three tenses; a year is a constraint, not part of the
entity; two relations matching equally well refuse; the time check refuses a present-tense question on a
fact that ended, a year outside the fact's dates and a past event asked in the present, and leaves an
undated fact alone; the forms name their relations to the loop's resolvers (a wh-word narrows them).
Run: python -m pytest standin/tests/test_question_forms.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from question_forms import FORMS, Forms, TemplateEmitter, load_csv, tense_of, tensed, time_check  # noqa: E402

ENTITIES = ["Rome", "Napoleon Bonaparte", "Henry II of England", "Battle of Hastings", "Holy Roman Empire"]


def test_every_form_in_every_tense_parses_back_to_its_entity_relation_and_tense():
    f = Forms()
    bad = []
    for rel, (_kind, by) in FORMS.items():
        assert set(by) == {"past", "present", "future"}, rel
        for tense in by:
            for e in ENTITIES:
                for q in f.render(rel, e, tense):
                    if f.parse(q) != (e, rel, tense):
                        bad.append((q, f.parse(q)))
    assert bad == []
    assert f.parse("Who founded Rome?") == ("Rome", "founded by", "past")
    assert f.parse("When was the Roman Empire founded?") == ("Roman Empire", "inception", "past")   # not 'the {S}?'
    assert f.parse("Who was the ruler of England in 1160?") == ("England", "head of state", "past")
    assert f.parse("What is the founded by of Rome?") is None and f.parse("Tell me a joke") is None


def test_the_hdc_templates_come_in_three_tenses(tmp_path):
    assert tensed("Who is the {relation} of {S}?") == {"present": "Who is the {relation} of {S}?",
                                                      "past": "Who was the {relation} of {S}?",
                                                      "future": "Who will be the {relation} of {S}?"}
    assert tensed("To which X does {S} belong?")["past"] == "To which X did {S} belong?"
    csv = tmp_path / "relation_template_mapping.csv"
    csv.write_text("relation,template,edited_relation,noun_template\n"
                   "part of,To which {relation} does {subject} belong?,parent entity,the {relation} {subject} belongs to\n"
                   "employer,Who is the {relation} of {subject}?,,the {relation} of {subject}\n", encoding="utf-8")
    forms = load_csv(str(csv))
    assert "To which parent entity did {S} belong?" in forms["part of"][1]["past"] and "employer" not in forms
    f = Forms(forms)
    assert f.parse("To which parent entity will Burgundy belong?") == ("Burgundy", "part of", "future")


def test_two_relations_matching_equally_well_refuse():
    f = Forms({"a": ("state", {"present": ["What is {S} known for?"]}), "b": ("state", {"present": ["What is {S} known for?"]})})
    assert f.parse("What is Rome known for?") is None


def test_the_time_check_refuses_an_ended_fact_in_the_present_a_year_outside_and_a_past_event_now():
    reign = {"start": "1154", "end": "1189"}
    assert time_check("state", "past", None, reign, "Henry II", 2026) is None
    assert "ended in 1189" in time_check("state", "present", None, reign, "Henry II", 2026)
    assert time_check("state", "past", 1160, reign, "Henry II", 2026) is None
    assert "1200" in time_check("state", "past", 1200, reign, "Henry II", 2026)
    assert time_check("state", "present", None, {"start": "2019"}, "X", 2026) is None          # ongoing
    assert "not scheduled" in time_check("state", "future", None, reign, "Henry II", 2026)
    assert time_check("state", "present", None, {}, "Paris", 2026) is None                      # undated: now or before
    assert "scheduled" in time_check("state", "future", None, {}, "Paris", 2026)
    assert "no date" in time_check("event", "present", None, {}, "Ajaccio", 2026)              # an undated event: past only
    assert time_check("event", "past", None, {}, "Ajaccio", 2026) is None
    assert time_check("event", "past", None, {"point": "1769"}, "Ajaccio", 2026) is None
    assert "happened in 1769" in time_check("event", "present", None, {"point": "1769"}, "Ajaccio", 2026)
    assert time_check("event", "past", None, None, "753 BC", 2026) is None                       # dated by its value
    assert "-753" in time_check("event", "future", None, None, "753 BC", 2026)
    assert tense_of("Who rules England?") is None and tense_of("Who is the king?") == "present"   # auxiliaries only
    assert (tense_of("Who was the king?"), tense_of("Who will be the king?")) == ("past", "future")


def test_the_forms_name_their_relations_to_the_loops_resolvers():
    f = Forms()
    assert f.relations("ruled") == ["head of state"] and set(f.relations("born")) == {"date of birth", "place of birth"}
    assert (f.kind("date of birth"), f.kind("place of birth"), f.kind("unheard of")) == ("date", "name", None)
    assert f.relations("the") == [] and f.aspect("reign") == "state" and f.aspect("inception") == "event"
    out = TemplateEmitter(f).emit("Where was Napoleon Bonaparte born?")
    assert 'bind frame, SEED, "Napoleon Bonaparte";' in out and 'bind frame, HOP1, "born";' in out   # its own word
    assert f.wording("Who was the ruler of England in 1160?") == "ruler" and f.wording("When was the Battle of X?") is None
    assert 'HOP1, "point in time";' in TemplateEmitter(f).emit("When was the Battle of Hastings?")   # no word: the label
    assert TemplateEmitter(f).emit("Tell me a joke") == "no plan\n"
