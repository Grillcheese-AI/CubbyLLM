"""test_striatum -- proposer arbitration by reward prediction error.

What is pinned:
  * dopamine is the error, not the reward: the first certification on a shape carries a delta
    of +1 (nothing was expected), the tenth carries almost none;
  * the order learns: after the hippocampus certifies a shape twice and the emitter refuses it
    twice, the hippocampus is asked first on that shape -- and the emitter still comes first on
    a shape where the reverse happened;
  * refusal is neutral: a proposer that only ever refuses stays at 0, never below -- the
    don't-know contract is not punished;
  * wrong is far below correct: one audited wrong answer outweighs four certifications, so a
    proposer that guesses ranks below one that refuses;
  * a never-seen shape ranks a proposer by its prior across shapes;
  * every delta is on the record; save/load round-trips the expectation.
Run: python -m pytest validation/test_striatum.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.reasoning.striatum import REWARD, Striatum, question_shape  # noqa: E402

Q_POSS = "Jean's country of citizenship -- what is its capital?"
Q_CANON = "What is the capital of the country of citizenship of Jean?"
Q_DATE = "On what day, month, and year was Jean born?"
PROPOSERS = ["grammar", "hippocampus", "emitter"]


def test_dopamine_is_the_error_not_the_reward():
    s = Striatum()
    d1 = s.reward(Q_POSS, "hippocampus", "certified")
    assert d1 == 1.0
    deltas = [s.reward(Q_POSS, "hippocampus", "certified") for _ in range(9)]
    assert deltas[-1] < 0.15 and all(a > b for a, b in zip(deltas, deltas[1:]))
    assert s.trace[0]["delta"] == 1.0 and s.trace[-1]["reward"] == 1.0


def test_the_order_learns_per_shape():
    s = Striatum(eps=0.0)
    assert s.order(Q_POSS, PROPOSERS) == PROPOSERS                     # nothing known: the given order
    for _ in range(2):
        s.reward(Q_POSS, "emitter", "refused"); s.reward(Q_POSS, "hippocampus", "certified")
        s.reward(Q_CANON, "hippocampus", "refused"); s.reward(Q_CANON, "emitter", "certified")
    assert s.order(Q_POSS, PROPOSERS)[0] == "hippocampus"
    assert s.order(Q_CANON, PROPOSERS)[0] == "emitter"
    assert question_shape(Q_POSS) != question_shape(Q_CANON)


def test_refusal_is_neutral_never_negative():
    s = Striatum()
    for _ in range(20):
        s.reward(Q_DATE, "grammar", "refused")
    assert s.value("grammar", question_shape(Q_DATE)) == 0.0
    assert REWARD["refused"] == 0.0


def test_wrong_outweighs_correct_so_guessing_ranks_below_refusing():
    s = Striatum(eps=0.0)
    for _ in range(4):
        s.reward(Q_DATE, "emitter", "certified")
    s.reward(Q_DATE, "emitter", "wrong", source="audit")                # one spoken wrong answer
    for _ in range(5):
        s.reward(Q_DATE, "hippocampus", "refused")
    assert s.value("emitter", question_shape(Q_DATE)) < s.value("hippocampus", question_shape(Q_DATE))
    assert s.order(Q_DATE, ["emitter", "hippocampus"])[0] == "hippocampus"
    assert REWARD["wrong"] < -1.0 and s.trace[-6]["source"] == "audit"


def test_an_unseen_shape_uses_the_proposers_prior():
    s = Striatum(eps=0.0)
    for _ in range(3):
        s.reward(Q_POSS, "hippocampus", "certified"); s.reward(Q_CANON, "hippocampus", "certified")
    assert s.order("Whose father is the mayor of Lyon?", PROPOSERS)[0] == "hippocampus"


def test_exploration_only_when_the_tonic_level_is_low():
    s = Striatum(eps=1.0, tonic_floor=0.5, seed=1)
    s.reward(Q_POSS, "emitter", "certified"); s.reward(Q_POSS, "hippocampus", "refused")
    assert s.order(Q_POSS, PROPOSERS)[0] == "emitter"                   # tonic 0.5: no exploration
    for _ in range(10):
        s.reward(Q_POSS, "emitter", "refused")                          # the best proposer went quiet
    assert s.tonic < 0.5 and s.order(Q_POSS, PROPOSERS)[0] != "emitter"  # eps=1: the runner-up is tried first


def test_save_and_load_round_trip(tmp_path):
    s = Striatum(); s.reward(Q_POSS, "hippocampus", "certified")
    p = tmp_path / "striatum.json"; s.save(p)
    s2 = Striatum.load(p)
    assert s2.expected == s.expected and s2.order(Q_POSS, PROPOSERS)[0] == "hippocampus"
