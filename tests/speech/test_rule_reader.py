"""Order 0067, step 3: the reader. With the slots constrained, a seat that writes a plan's surface form gets that plan
back; a seat that picks any allowed token still gives a plan `rule_plan.check` accepts, or "asked"; never a plan the
family refuses. (A stand-in seat over characters; the 450M is measured by validation/exp_rule_reader.py.)"""
from __future__ import annotations

import random

from cubbyllm.speech import rule_plan as rp
from cubbyllm.speech import rule_reader as rr

EOS = 0
encode = lambda text: [ord(c) for c in text]                      # noqa: E731


def oracle(target: str):
    """A seat that writes `target` (the text after "Rule:") as far as the slots allow."""
    def gen(ids, allowed, max_new):
        done = "".join(map(chr, ids)).rsplit("Rule:", 1)[1]
        rest, out = target[len(done):], []
        for i in range(max_new):
            ok = allowed(out)
            want = ord(rest[i]) if i < len(rest) else None
            t = want if want in ok else (EOS if EOS in ok else ok[0])
            out.append(t)
            if t == EOS:
                break
        return out
    return gen


def wild(seed: int):
    rng = random.Random(seed)

    def gen(ids, allowed, max_new):
        out = []
        for _ in range(max_new):
            t = rng.choice(allowed(out))
            out.append(t)
            if t == EOS:
                break
        return out
    return gen


PLANS = [{"m": "word count", "op": "<", "v": 50},
         {"and": [{"m": "case", "op": "==", "v": "lower"}, {"m": "language", "op": "==", "v": "en"}]},
         {"m": "count of the word", "arg": "river", "op": ">=", "v": 2},
         {"and": [{"m": "starts with", "arg": '"', "op": "==", "v": "yes"}, {"m": "ends with", "arg": '"', "op": "==", "v": "yes"}]},
         {"m": "opening", "op": "has", "v": "in short"}]
INSTRUCTION = 'Say river twice, in short, wrapped in " marks, under 50 words, in lowercase English.'


def test_a_seat_writing_the_surface_gets_the_plan_back():
    for plan in PLANS:
        got = rr.read(INSTRUCTION, oracle(" " + rr.surface(plan) + "\n"), encode, EOS)
        assert got["how"] == "written" and rr.same(got["plan"], plan), (plan, got)


def test_asked_is_a_result():
    got = rr.read(INSTRUCTION, oracle(" asked\n"), encode, EOS)
    assert got["how"] == "asked" and got["plan"] is None


def test_any_allowed_choice_is_a_plan_the_family_accepts():
    for seed in range(40):
        got = rr.read(INSTRUCTION, wild(seed), encode, EOS)
        assert got["how"] in ("written", "asked")
        if got["how"] == "written":
            assert rp.check(got["plan"]) is None and len(rp.leaves(got["plan"])) <= rr.MAX_LEAVES


def test_examples_are_shown_in_the_surface_form():
    seen = []

    def gen(ids, allowed, max_new):
        seen.append("".join(map(chr, ids)))
        return [ord("a")] if ord("a") in allowed([]) else [EOS]
    rr.read("Be brief.", gen, encode, EOS, examples=[("Use fewer than 9 words.", PLANS[0])])
    assert "Instruction: Use fewer than 9 words.\nRule: word count < 50\n\n" in seen[0]
    assert rr.same(PLANS[1], {"and": list(reversed(PLANS[1]["and"]))}) and not rr.same(PLANS[0], PLANS[2])
