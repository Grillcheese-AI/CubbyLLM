"""frame a claim -> name its test -> run it -> THE RESULT is the verdict.

Nick, 2026-09-15: *"same way with other problems outside the game, it can frame
an hypothesis test it against the VM or a sandbox if code then the result is
what is wrong or not."* One loop, three verifiers; the world one is the only
that can answer "not yet", and its silence is eventually an answer."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hypothesis import (CONFIRMED, OPEN, REFUSED, Hypotheses, Hypothesis,  # noqa: E402
                        runner_verifier, world_verifier)


def test_a_claim_is_open_until_something_settles_it():
    h = Hypothesis(claim="x", test="t", verifier="world")
    assert h.open and h.state == OPEN and h.settled_at is None
    assert h.settle(True, "because", 7) == []            # no if_true: nothing earned
    assert not h.open and h.state == CONFIRMED and h.settled_at == 7


def test_the_world_may_say_not_yet_and_its_silence_is_eventually_an_answer():
    """The pellet case. Nothing happens; after `patience` steps, nothing
    happening IS the finding."""
    state = {"moved": False}
    led = Hypotheses({"world": world_verifier(state)})
    led.frame(Hypothesis(claim="it might move", test="watch it", verifier="world",
                         if_true="moving is its way", if_false="staying put is its way",
                         payload={"holds": lambda env: True if env["moved"] else None},
                         made_at=0, patience=5))
    for now in range(1, 5):
        assert led.sweep(now) == [], "not settled yet: he keeps watching"
        assert led.open
    earned = led.sweep(5)
    assert earned == ["staying put is its way"], "the world declined to confirm it, and that is a finding"
    assert led.settled(REFUSED) and not led.open


def test_the_world_can_also_say_yes():
    state = {"moved": False}
    led = Hypotheses({"world": world_verifier(state)})
    led.frame(Hypothesis(claim="it might move", test="watch it", verifier="world",
                         if_true="moving is its way", if_false="staying put is its way",
                         payload={"holds": lambda env: True if env["moved"] else None},
                         made_at=0, patience=5))
    assert led.sweep(1) == []
    state["moved"] = True
    assert led.sweep(2) == ["moving is its way"]
    assert led.settled(CONFIRMED)[0].verdict_why == "the world did it"


def test_the_same_claim_is_not_asked_twice():
    led = Hypotheses({"world": world_verifier({})})
    h = Hypothesis(claim="same", test="t", verifier="world", payload={"holds": lambda e: None})
    assert led.frame(h) is h
    assert led.frame(Hypothesis(claim="same", test="t", verifier="world")) is None, \
        "he does not get to re-ask a question the world already answered"


def test_code_is_settled_by_running_it_not_by_reading_it():
    led = Hypotheses({"runner": runner_verifier(timeout=20)})
    led.frame(Hypothesis(claim="this adds up", test="run it", verifier="runner",
                         if_true="right is the shape of this code",
                         if_false="wrong is the shape of this code",
                         payload={"code": "print(2 + 2)", "expected": "4"}))
    assert led.sweep(1) == ["right is the shape of this code"]

    led2 = Hypotheses({"runner": runner_verifier(timeout=20)})
    led2.frame(Hypothesis(claim="this adds up too", test="run it", verifier="runner",
                          if_false="wrong is the shape of this code",
                          payload={"code": "print(2 + 2)", "expected": "5"}))
    assert led2.sweep(1) == ["wrong is the shape of this code"]
    assert "printed '4', expected '5'" in led2.settled(REFUSED)[0].verdict_why


def test_code_that_crashes_or_hangs_is_a_refusal_not_a_crash():
    led = Hypotheses({"runner": runner_verifier(timeout=20)})
    led.frame(Hypothesis(claim="boom", test="run it", verifier="runner", if_false="it does not run",
                         payload={"code": "raise ValueError('nope')", "expected": "anything"}))
    assert led.sweep(1) == ["it does not run"]

    slow = Hypotheses({"runner": runner_verifier(timeout=1.0)})
    slow.frame(Hypothesis(claim="forever", test="run it", verifier="runner", if_false="it does not finish",
                          payload={"code": "import time\nwhile True: time.sleep(0.05)", "expected": "x"}))
    assert slow.sweep(1) == ["it does not finish"]
    assert "did not finish" in slow.settled(REFUSED)[0].verdict_why


def test_a_broken_verifier_is_not_a_verdict():
    """A verifier that throws must leave the claim OPEN. Deciding a question
    because the instrument fell over is the worst possible answer."""
    def boom(h, now):
        raise RuntimeError("instrument down")
    led = Hypotheses({"world": boom})
    led.frame(Hypothesis(claim="c", test="t", verifier="world", if_false="f"))
    assert led.sweep(1) == [] and led.open, "still an open question"


def test_stats_say_what_was_asked_and_what_it_bought():
    led = Hypotheses({"runner": runner_verifier(timeout=20), "world": world_verifier({})})
    led.frame(Hypothesis(claim="a", test="t", verifier="runner", if_true="A",
                         payload={"code": "print(1)", "expected": "1"}))
    led.frame(Hypothesis(claim="b", test="t", verifier="world",
                         payload={"holds": lambda e: None}, patience=0))
    led.sweep(1)
    s = led.stats()
    assert s["total"] == 2 and s["open"] == 1 and s["earned"] == 1
    assert s["by_verifier"]["runner"][CONFIRMED] == 1
