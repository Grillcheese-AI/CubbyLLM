"""'How is B related to A?' in the ask loop (2026-09-24, the skill library): the paths between the two are
composed by the adopted rules and certified in the VM; day 1 refuses what no rule composes, the night
learns the rules from what gold says, and day 2 answers -- including a chain longer than any it was told.
Run: python -m pytest standin/tests/test_ask_relation.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import sleep as S  # noqa: E402
from cubbyllm.reasoning.skills import Library, Rule  # noqa: E402
from test_ask import FakeEmitter  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from ask import AskLoop, relation_ask  # noqa: E402

FAMILY = ["michael is the father of donald", "dorothy is the sister of michael",
          "rose is the mother of dorothy", "ann is the daughter of rose"]


def loop(skills=None, calls=None, history=None):
    return AskLoop(FakeEmitter({}), world=LookupStore(FAMILY), source=DictSource({}), lexicon=False,
                   run_fn=faithful_vm(calls if calls is not None else []), skills=skills, history_path=history)


def test_the_relation_question_has_several_wordings_and_all_read_b_of_a():
    for q in ("How is Dorothy related to Donald?", "What relation is Dorothy to Donald?",
              "What is Dorothy's relationship to Donald?", "what is the relationship of Dorothy to Donald"):
        assert relation_ask(q) == ("Donald", "Dorothy"), q
    assert relation_ask("Who is the father of Donald?") is None


def test_without_a_rule_the_loop_refuses_and_says_why():
    rec = loop().ask("How is Dorothy related to Donald?")
    assert rec["kind"] == "relation" and not rec["verified"] and rec["reason"] == "no_rule"
    assert rec["paths"] == [["father", "sister"]] and rec["answer"] is None


def test_a_rule_answers_it_through_the_vm_and_the_record_says_how():
    lib = Library()
    lib.adopt(Rule("father", "sister", "aunt"), support=2, evidence=[], night="t")
    calls = []
    rec = loop(lib, calls).ask("How is Dorothy related to Donald?")
    assert rec["verified"] and rec["answer"] == "aunt" and rec["statement"] == "Dorothy is the aunt of Donald"
    assert rec["relation_paths"][0]["steps"] == ["the sister of the father is the aunt"]
    assert rec["trace"] == FAMILY[:2] and "control" in calls
    assert rec["graph"]["nodes"]                                  # the thought graph is kept like any ask


def test_day_one_refuses_the_night_learns_day_two_answers(tmp_path):
    """Day 1: eight relation asks refused (no rule), each with the relation gold states. The night adopts
    (father, sister) -> aunt and (mother, daughter) -> sister from the two-hop asks and, by closure over the
    three-hop ones, (aunt, mother) -> grandmother and (aunt, sister) -> aunt. Day 2 answers a four-hop
    question no episode was ever about: ((father sister) (mother daughter)) = (aunt, sister) = aunt."""
    day = tmp_path / "day1.jsonl"
    rows = [
        {"question": "How is Dorothy related to Donald?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister"]], "gold": "aunt"},
        {"question": "How is Jane related to Tom?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister"]], "gold": "aunt"},
        {"question": "How is Ann related to Bea?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["mother", "daughter"]], "gold": "sister"},
        {"question": "How is Cy related to Di?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["mother", "daughter"]], "gold": "sister"},
        {"question": "How is Rose related to Donald?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister", "mother"]], "gold": "grandmother"},
        {"question": "How is Vi related to Al?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister", "mother"]], "gold": "grandmother"},
        {"question": "How is Liz related to Ed?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister", "sister"]], "gold": "aunt"},
        {"question": "How is Mo related to Jo?", "kind": "relation", "verified": False, "reason": "no_rule",
         "paths": [["father", "sister", "sister"]], "gold": "aunt"},
    ]
    day.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "sleep"
    rep = S.sleep([day], out, night="2026-09-24", trusted={"wikidata"})
    assert rep["skills"]["adopted"] == 4 and rep["skills"]["rules"] == 4
    assert rep["triage"]["skill_gaps"]["tonight"] == 8                      # the refusals were routed too
    again = S.sleep([day], out, night="2026-09-24", trusted={"wikidata"})    # a re-run changes nothing
    assert again["skills"]["adopted"] == 4 and len(Library(out / "skills.jsonl")) == 4
    assert len((out / "skills.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    rec = loop(str(out / "skills.jsonl")).ask("How is Ann related to Donald?")   # father, sister, mother, daughter
    assert rec["verified"] and rec["answer"] == "aunt", rec["relation_paths"]
    assert rec["paths"] == [["father", "sister", "mother", "daughter"]]


def test_the_loops_own_answers_never_teach_it(tmp_path):
    """A spoken relation is an episode only when an asker says it was right; unconfirmed, it is nothing."""
    rec = {"question": "How is Dorothy related to Donald?", "kind": "relation", "verified": True, "answer": "aunt",
           "paths": [["father", "sister"]]}
    day = tmp_path / "d.jsonl"
    day.write_text(json.dumps(rec) + "\n" + json.dumps(dict(rec, question="How is X related to Y?")) + "\n",
                   encoding="utf-8")
    assert S.episodes_of(S.load_day(day)) == []
    day.write_text(json.dumps(dict(rec, feedback="right")) + "\n", encoding="utf-8")
    assert [e.conclusion for e in S.episodes_of(S.load_day(day))] == ["aunt"]


# ── the asker teaches ────────────────────────────────────────────────────────────────────────────────
def test_an_asker_who_states_the_relation_turns_a_refusal_into_an_episode(tmp_path):
    hist = tmp_path / "ask_history.jsonl"
    lp = loop(history=str(hist))
    rec = lp.ask("How is Dorothy related to Donald?")
    assert rec["reason"] == "no_rule" and rec["id"]
    out = lp.feedback(rec["id"], relation="Aunt", asker="nick")
    assert out == {"ok": True, "record": rec["id"], "known": True, "verdict": None, "relation": "aunt", "episode": True}
    day = S.load_day(hist)
    assert len(day) == 1 and day[0].gold == "aunt" and day[0].asker == "nick"      # the feedback line folded in
    (e,) = S.episodes_of(day)
    assert e.relations == ["father", "sister"] and e.conclusion == "aunt" and e.source == "asker:nick"


def test_one_asker_cannot_put_a_rule_in_two_can(tmp_path):
    def night(askers, name):
        hist = tmp_path / f"{name}.jsonl"
        lp = loop(history=str(hist))
        for who in askers:
            rec = lp.ask("How is Dorothy related to Donald?")
            lp.feedback(rec["id"], relation="aunt", asker=who)
        return S.sleep([hist], tmp_path / f"sleep_{name}", night="n1", trusted={"wikidata"})["skills"]
    assert night(["mallory", "mallory", "mallory"], "one")["rules"] == 0
    assert night(["nick", "ada"], "two")["rules"] == 1


def test_a_spoken_relation_the_asker_corrects_is_a_defect(tmp_path):
    hist = tmp_path / "h.jsonl"
    lib = Library()
    lib.adopt(Rule("father", "sister", "mother"), support=2, evidence=[], night="t")    # a wrong rule got in
    lp = loop(lib, history=str(hist))
    rec = lp.ask("How is Dorothy related to Donald?")
    assert rec["verified"] and rec["answer"] == "mother"
    assert lp.feedback(rec["id"], relation="aunt", asker="nick")["verdict"] == "wrong"
    rep = S.sleep([hist], tmp_path / "s", night="n1", trusted={"wikidata"})
    assert rep["audit"]["wrong"] == 1


def test_feedback_says_what_it_cannot_take():
    import pytest
    lp = loop()
    rec = lp.ask("How is Dorothy related to Donald?")
    with pytest.raises(ValueError):
        lp.feedback(rec["id"], verdict="maybe")
    with pytest.raises(ValueError):
        lp.feedback(rec["id"])


# ── real families: the store knows nothing, the source knows both families ──────────────────────────
def test_an_empty_store_fetches_both_families_and_names_each_edge_by_sex():
    src = DictSource({
        "donald": ["michael is the father of donald", "male is the sex or gender of donald"],
        "michael": ["dorothy is the sibling of michael", "donald is the child of michael", "male is the sex or gender of michael"],
        "dorothy": ["michael is the sibling of dorothy", "female is the sex or gender of dorothy"],
    })
    lib = Library()
    lib.adopt(Rule("father", "sister", "aunt"), support=2, evidence=[], night="t")
    lp = AskLoop(FakeEmitter({}), world=LookupStore([]), source=src, lexicon=False, run_fn=faithful_vm([]), skills=lib)
    rec = lp.ask("How is Dorothy related to Donald?")
    assert rec["verified"] and rec["answer"] == "aunt", rec["relation_paths"]
    assert rec["trace"] == ["michael is the father of donald", "dorothy is the sister of michael"]  # 'sibling' named by her sex
    facts = {l["fact"] for l in rec["learned"]}
    assert "donald is the son of michael" in facts                  # Wikidata's own inverse of father, by his sex
    assert "michael is the brother of dorothy" in facts and all(l["status"] == "accepted" for l in rec["learned"])
    assert {"donald", "dorothy", "michael"} <= {e.lower() for e in rec["entities"]}


def test_family_facts_keep_the_word_when_the_sex_is_unknown():
    from cubbyllm.reasoning.skills import family_facts
    assert family_facts("ann", "child", "bo", {}) == ["bo is the child of ann"]          # no word for ann as a parent
    assert family_facts("ann", "child", "bo", {"ann": "female", "bo": "male"}) == [
        "bo is the son of ann", "ann is the mother of bo"]
    assert family_facts("ann", "spouse", "bo", {"bo": "male"}) == ["bo is the husband of ann", "ann is the spouse of bo"]


def test_the_loop_turns_a_one_way_store_round_with_an_inverse_rule():
    lib = Library()
    lib.adopt_inverse("nephew", "female", "aunt", 3, [], "n1")
    lp = AskLoop(FakeEmitter({}), world=LookupStore(["donald is the nephew of dorothy", "female is the sex or gender of dorothy"]),
                 source=DictSource({}), lexicon=False, run_fn=faithful_vm([]), skills=lib)
    rec = lp.ask("How is Dorothy related to Donald?")
    assert rec["verified"] and rec["answer"] == "aunt" and rec["direction"] == "inverse"
    assert rec["statement"] == "Dorothy is the aunt of Donald"
