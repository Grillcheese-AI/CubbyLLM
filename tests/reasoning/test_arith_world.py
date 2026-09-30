"""Pins for cubbyllm/reasoning/arith_world.py: arithmetic as a world the emitter asks. Conversion facts come from
the world as $K slots (never from the model's weights); solved procedures are recalled by what the question asks,
whatever its numbers, and only once attested. No model, no VM."""
from __future__ import annotations

from cubbyllm.reasoning.arith_world import ArithmeticWorld, conversions, procedure_words
from cubbyllm.reasoning.slots import extract


def test_a_conversion_is_offered_only_when_the_question_names_both_units():
    q = "Tom runs 3 hours a day. How many minutes does he run in a week?"
    got = [f.text for f in conversions(q)]
    assert got == ["60 minutes per hour", "24 hours per day", "7 days per week"], \
        "every conversion the units allow; the emitter picks (an unused $K is no error)"
    assert conversions("A quarter of the 12 pies were eaten.") == [], "'quarter' is a coin only next to cents"
    assert [f.value for f in conversions("It costs 45 cents; he pays with quarters.")] == [25.0]


def test_the_world_constants_become_K_slots_the_emitter_must_use():
    w = ArithmeticWorld()
    q = "Tom runs 3 hours a day. How many minutes does he run in a week?"
    t = extract(q, constants=w.constants(q))
    assert [(s.id, s.filled) for s in t.spans] == [("$N1", "3"), ("$K1", "60"), ("$K2", "24"), ("$K3", "7")]
    assert t.annotate().endswith("[K1 = 60 minutes per hour]; [K2 = 24 hours per day]; [K3 = 7 days per week]")
    prog = "assign s0 = $N1; mul s0, $K1; mul s0, $K3;"
    assert t.check(prog).ok and t.fill(prog) == "assign s0 = 3; mul s0, 60; mul s0, 7;"
    assert t.check("assign s0 = $N1; mul s0, 60;").copied == ["60"], "the world had it: writing it is a copy"
    t2 = extract("A 60 minute class ends; how many hours in 120 minutes?", constants=w.constants("hours minutes"))
    assert [s.id for s in t2.spans] == ["$N1", "$N2"], "a constant the question already states is not repeated"


def test_procedures_are_recalled_by_what_they_ask_not_their_numbers_and_only_once_attested():
    w = ArithmeticWorld()
    a = w.remember("Sam has [N1: 12] apples and buys [N2: 5] more. How many apples does he have?",
                   "assign s0 = $N1; add s0, $N2;", attested=False, source="curriculum")
    w.remember("A train travels [N1: 60] miles per hour for [N2: 3] hours. How far does it go?",
               "assign s0 = $N1; mul s0, $N2;", attested=True, source="gold")
    q = "Sam has 40 apples and buys 17 more. How many apples does he have?"
    assert procedure_words(q) == procedure_words("Sam has 3 apples and buys nine more. How many apples does he have?")
    assert w.recall(q) == [], "latent: not recalled until the VM hit a known answer"
    assert w.recall(q, latent=True)[0][0] == a.id
    w.attest(a.id, "gold")
    (pid, prog, sim), *_ = w.propose(q, n_slots=2)
    assert pid == a.id and prog == "assign s0 = $N1; add s0, $N2;" and sim > 0.9
    assert all(p != a.id for p, _, _ in w.propose(q, n_slots=1)), "needs two numbers; the question has one"


def test_the_world_round_trips(tmp_path):
    w = ArithmeticWorld()
    w.remember("Sam has [N1: 12] apples and buys [N2: 5] more.", "assign s0 = $N1; add s0, $N2;", attested=True,
               source="gold")
    w.save(tmp_path / "arith.jsonl")
    w2 = ArithmeticWorld.load(tmp_path / "arith.jsonl")
    assert w2.recall("Sam has 3 apples and buys 4 more.") == w.recall("Sam has 3 apples and buys 4 more.")
