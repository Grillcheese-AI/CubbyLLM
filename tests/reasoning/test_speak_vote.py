"""Order 0078, step 5 (cubbyllm/reasoning/speak_vote.py): the speak rule on fixed tries. An answer is said only when at
least m tries land on one VM value and no other value ties it; a try that did not run never votes; a tie for first is
refused; values within the tolerance are one value; m is the smallest that keeps dev's wrong answers within the line."""
from __future__ import annotations

from cubbyllm.reasoning.speak_vote import choose_m, outcome, vote


def test_said_when_m_tries_agree_and_refused_when_not():
    tries = [72.0, 72.0, 72.0, 24.0, None, 50.0, 72.0000000001, 24.0]
    assert vote(tries, 4) == {"answer": 72.0, "votes": 4, "k": 8, "why": None}     # 72.0000000001 is 72
    assert vote(tries, 5)["answer"] is None and vote(tries, 5)["why"] == "too few agree"


def test_a_try_that_did_not_run_never_votes():
    assert vote([None, None, None, 5.0], 2)["answer"] is None
    assert vote([None] * 8, 1)["why"] == "too few agree"


def test_a_tie_for_first_is_refused_and_a_lead_is_said():
    got = vote([3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 1.0, 2.0], 3)
    assert got["answer"] is None and got["why"] == "tie"
    assert vote([3.0, 3.0, 3.0, 4.0, 4.0, 1.0, 2.0, 9.0], 3)["answer"] == 3.0
    assert vote([3.0, 3.0, 4.0, 1.0], 1)["answer"] == 3.0                          # m = 1: the plurality, if alone


def test_m_is_the_smallest_within_the_wrong_line():
    dev = [([10.0] * 6 + [11.0, 11.0], 10.0)] * 98 + [([5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0], 4.0)] * 2
    got = choose_m(dev, 8, max_wrong=0.01)
    assert got["by m"][1]["wrong"] == 2 and got["by m"][2]["wrong"] == 2          # five leads with two, wrongly
    assert got["by m"][3] == {"right": 98, "wrong": 0, "refused": 2, "wrong share": 0.0}
    assert got["m"] == 3
    loose = choose_m(dev, 8, max_wrong=0.02)
    assert loose["m"] == 1                                                         # 2 of 100 is within 2%
    assert outcome([10.004] * 3, 10.0, 3) == "right" and outcome([9.0] * 3, 10.0, 3) == "wrong"


def test_no_m_within_the_line_is_said_plainly():
    dev = [([1.0] * 8, 2.0)] * 10
    assert choose_m(dev, 8)["m"] is None
