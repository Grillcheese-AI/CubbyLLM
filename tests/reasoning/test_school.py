"""Pins for cubbyllm/reasoning/school.py: the curriculum starts at one operation, shrinks the try budget as a level
is mastered, opens the next level with the full budget, doubles back when it struggles; dopamine is the reward
prediction error (a surprise solve bursts, a routine one is ~0, a failure where success was expected dips); a
refusal is never punished; mistakes pair with the solve (or the shown solution) and name their first wrong step
and their kind. No model, no VM."""
from __future__ import annotations

from cubbyllm.reasoning.school import (CORRECT, LEVELS, REFUSED, WRONG, Curriculum, classify_error,
                                       first_wrong_step, level_of, mistake_rows)


def test_levels_run_from_one_operation_to_long_mixed_problems():
    assert level_of({"steps": 1}) == 0 and level_of({"steps": 2}) == 1 and level_of({"steps": 4}) == 2
    assert LEVELS[level_of({"steps": 2, "word_numbers": 1})].name == "words and units"
    assert LEVELS[level_of({"steps": 3, "distractors": 2})].name == "distractors"
    assert LEVELS[level_of({"steps": 6, "unit_constants": 1})].name == "five or six steps", "the hard end mixes"
    assert LEVELS[level_of({"steps": 9})].name == "seven to ten steps"


def test_the_budget_shrinks_with_mastery_and_the_next_level_opens_with_full_budget():
    c = Curriculum(window=8, max_tries=4)
    assert c.budget(0) == 4
    for _ in range(8):
        c.record(0, [WRONG, CORRECT])
    assert c.budget(0) == 2
    for _ in range(8):
        c.record(0, [CORRECT])
    assert c.budget(0) == 1
    for _ in range(8):
        c.record(0, [CORRECT])
    assert c.state[0].mastered and c.current == 1 and c.budget(1) == 4
    assert [e["event"] for e in c.trace] == ["budget -> 2", "budget -> 1", "mastered; next level: two steps"]


def test_a_struggling_level_gets_its_tries_back_and_a_review_of_the_level_before():
    c = Curriculum(window=8, max_tries=4, current=1)
    c.state[0].mastered = True
    c.state[1].budget = 1
    for _ in range(8):
        c.record(1, [WRONG])
    assert c.budget(1) == 2 and c.trace[-1]["event"] == "struggling; budget -> 2"
    for _ in range(8):
        c.record(1, [WRONG, WRONG])
    for _ in range(8):
        c.record(1, [WRONG, WRONG, WRONG, WRONG])
    assert c.budget(1) == 4
    batch = c.plan_batch(400)
    assert 0.3 < batch.count(0) / 400 < 0.7, "review of mastered level 0 plus extra practice on it"


def test_dopamine_is_the_prediction_error_and_a_refusal_is_never_punished():
    c = Curriculum(window=64, alpha=0.5)
    first = c.record(0, [WRONG, CORRECT])
    assert first["dopamine"] == 1.0 and first["weights"][1] == 1.0, "the first solve: a full burst"
    assert first["weights"][0] < 0, "the wrong attempt before it is pushed down"
    for _ in range(10):
        routine = c.record(0, [CORRECT])
    assert routine["dopamine"] < 0.01, "a routine solve at a mastered level: nothing left to learn"
    dip = c.record(0, [WRONG, REFUSED])
    assert dip["dopamine"] < -0.9 and dip["weights"][1] == 0.0, "the dip; the refusal weighs nothing"


def test_mistakes_pair_with_the_solve_or_the_shown_solution():
    assert first_wrong_step([12.0, 26.0], [12.0, 26.0]) is None
    assert first_wrong_step([-12.0, 26.0], [12.0, 26.0]) == 0
    c = Curriculum()
    res = c.record(0, [WRONG, CORRECT])
    rows = mistake_rows("Q", [{"program": "bad", "outcome": WRONG, "values": [-12.0]},
                              {"program": "good", "outcome": CORRECT, "values": [12.0]}], res)
    assert [r["kind"] for r in rows] == ["mistake", "solve"]
    assert rows[0]["chosen"] == "good" and rows[0]["first_wrong_step"] == 0 and rows[0]["error"] == "reversed"
    res2 = c.record(0, [WRONG, WRONG])
    rows2 = mistake_rows("Q", [{"program": "bad", "outcome": WRONG, "values": [5.0]},
                               {"program": "bad2", "outcome": WRONG, "values": [720.0]}], res2,
                         reference={"program": "ref", "values": [12.0]})
    assert [r["kind"] for r in rows2] == ["shown", "mistake", "mistake"] and rows2[0]["weight"] > 0
    assert rows2[2]["error"] == "unit conversion"


def test_mistake_kinds_like_gsm8k_cube_learn():
    right = [12.0, 26.0]
    assert classify_error(12.0, 26.0, [59, 71, 38], right) == "stopped early"
    assert classify_error(-26.0, 26.0, [59, 71, 38], right) == "reversed"
    assert classify_error(1560.0, 26.0, [59, 71, 38], right) == "unit conversion"
    assert classify_error(64.0, 26.0, [59, 71, 38], right) == "one number off"
    assert classify_error(3.0, 26.0, [59, 71, 38], right) == "other"
    c = Curriculum()
    assert c.learn(0, "reversed") and not c.learn(0, "reversed") and c.patterns == {"0|reversed": 2}
    assert c.trace[-1]["event"] == "pattern learned: reversed"


def test_the_curriculum_round_trips(tmp_path):
    c = Curriculum(window=8, max_tries=4)
    for _ in range(8):
        c.record(0, [CORRECT])
    c.learn(0, "reversed")
    c.save(tmp_path / "school.json")
    d = Curriculum.load(tmp_path / "school.json", window=8)
    assert d.budget(0) == c.budget(0) and d.patterns == c.patterns and d.state[0].expected == c.state[0].expected


def test_the_competence_boundary_decides_what_the_emitter_may_answer():
    c = Curriculum(window=8, max_tries=1)
    for _ in range(8):
        c.record(0, [CORRECT])
    assert c.competence(0) == "mastered" and c.speak_policy(0)["samples"] == 1
    assert c.competence(1) == "practising" and c.speak_policy(1) == {"competence": "practising", "samples": 8,
                                                                      "agree": 0.5, "speak": True}
    assert c.competence(5) == "not yet" and not c.speak_policy(5)["speak"], "never guessed above the frontier"
    assert not c.speak_policy(None)["speak"], "a question no level admits is not the emitter's"


def test_the_no_curriculum_control_opens_every_level_at_once():
    c = Curriculum(open_all=True, seed=3)
    levels = c.plan_batch(700)
    assert set(levels) == set(range(len(LEVELS))) and min(levels.count(i) for i in range(len(LEVELS))) > 60
    assert c.competence(6) == "practising", "the control answers at every level: the false positives it risks"


def test_an_answer_written_in_earns_nothing():
    from cubbyllm.reasoning.school import answer_written_in
    assert answer_written_in("assign s0 = $N1; sub s0, $N2;", 12, [59, 71]) == ""
    assert answer_written_in("assign s0 = 26;", 26, [59, 71, 38]) == "no number from the question"
    assert answer_written_in("assign s0 = $N1; mul s0, 0; add s0, 26;", 26, [59, 71]) == \
        "the answer written in, not computed"
    assert answer_written_in("assign s0 = $N1; add s0, 12;", 59, [59, 71]) == "", "the gold is a stated number"
