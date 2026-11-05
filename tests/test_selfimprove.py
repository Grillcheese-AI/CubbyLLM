"""Order 0049 (``cubbyllm/selfimprove/``): on a toy stream the gap finder flags a planted gap and not a shuffled family;
the mistake memory keeps negatives until a right attempt; the router; the gate refuses an adapter that lowers the
battery. Pinned against the claims, not the implementation."""
from __future__ import annotations

import random

from cubbyllm.selfimprove import mistakes, router, signature
from cubbyllm.selfimprove.gaps import GapFinder, Rule
from cubbyllm.selfimprove.ledger import Ledger

PROG = '''use vsa;
program CotPlan implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, SEED, "$E1";
        bind frame, HOP1, "%s";
        return recover(frame, SEED);
    }
}'''


def test_a_signature_names_the_skill_not_the_answer():
    a = signature.of(PROG % "Birth", "en")
    b = signature.of(PROG % "birth", "en")
    assert a == b and a.relations == ("birth",) and a.hops == 1 and not a.inverse and a.task == "lookup"
    assert signature.of(PROG % "birth", "fr").key() != a.key()
    ar = signature.of("        create s0 : quantity;\n        assign s0 = $N1;\n        div s0, $N2;\n        add s0, $N3;\n")
    assert ar.task == "arithmetic" and ar.arith == 2


def _toy_stream(seed=0, n=600):
    """Three families: 'planted' (the base fails; learnable), 'shuffled' (fails at chance; its golds carry no rule),
    'easy' (the base is right)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        fam = rng.choice(("planted", "shuffled", "easy"))
        right = {"planted": rng.random() < 0.2, "shuffled": rng.random() < 0.5, "easy": rng.random() < 0.97}[fam]
        out.append((fam, f"{fam}-{i}", right))
    return out


def test_the_gap_finder_flags_a_planted_gap_and_never_the_shuffled_family():
    led, gf = Ledger(window=400), GapFinder(Rule(min_failures=20, min_gain=0.05))
    learnable = {"planted": 0.3, "shuffled": 0.0, "easy": 0.2}            # what a probe adapter lowers held loss by
    probes = []

    def probe(key, n_train, n_held):
        probes.append(key)
        return learnable[key]
    flagged = []
    for i, (fam, req, ok) in enumerate(_toy_stream()):
        led.add(fam, "right" if ok else "caught", gold_right=ok, request=req)
        if (i + 1) % 100 == 0:
            flagged += gf.night(led, (i + 1) // 100, probe)
    assert flagged == ["planted"] and "shuffled" in probes and "easy" not in probes   # easy never fails 20 times
    assert gf.gaps["planted"]["night"] == 1


def test_a_probe_is_not_repeated_every_night():
    led, gf = Ledger(), GapFinder(Rule(min_failures=1, recheck=3))
    led.add("noise", "refused")
    calls = []
    for night in range(1, 7):
        gf.night(led, night, lambda k, a, b: calls.append(night) or 0.0)
    assert calls == [1, 4]


def test_the_mistake_memory_keeps_negatives_until_a_right_attempt():
    mm = mistakes.MistakeMemory(tries=8)
    mm.attempt("p1", "g", "wrong A", False, "no value")
    mm.attempt("p1", "g", "wrong A", False, "no value")                    # the same mistake again: a repeat
    mm.attempt("p1", "g", "wrong B", False, "a namesake")
    assert mm.open("p1") and mm.negatives("p1") == [("wrong A", "no value"), ("wrong B", "a namesake")]
    mm.attempt("p1", "g", "right", True)
    mm.attempt("p1", "g", "wrong C", False)                               # solved: no more attempts recorded
    assert not mm.open("p1") and mm.right_set("g") == [("p1", "right")]
    assert mm.pairs("g") == [("p1", "wrong A", "right"), ("p1", "wrong B", "right")]
    mm.attempt("p2", "h", "x", False)
    mm.attempt("p2", "h", "y", True)
    assert mm.pairs(exclude="g") == [("p2", "x", "y")] and mm.pairs(exclude="h") == mm.pairs("g")
    s = mm.stats("g")
    assert s == {"problems": 1, "solved": 1, "repeats": 1, "attempts_until_right": 4.0, "negatives": 2}
    for i in range(8):
        mm.attempt("p3", "g", f"w{i}", False)
    assert not mm.open("p3") and mm.right_set("g") == [("p1", "right")]


def test_the_router_sends_a_gap_to_its_adapter_and_everything_else_to_the_base():
    r = router.Router(latent_after=3)
    assert r.route("k1", "q1") == router.BASE
    r.promote("k1", "adapter-k1")
    assert r.route("k1", "q2") == "adapter-k1" and r.route("k2", "q3") == router.BASE
    for i in range(4):
        r.route("k2", f"q{4 + i}")
    assert r.adapters["k1"]["state"] == "latent" and r.loaded() == []
    assert r.route("k1", "q9") == "adapter-k1" and r.loaded() == ["adapter-k1"]   # used again: loaded again
    assert [d["adapter"] for d in r.decisions][:3] == [router.BASE, "adapter-k1", router.BASE]


def test_the_gate_refuses_an_adapter_that_lowers_the_battery():
    g = router.Gate(battery_drop=0.01)
    base = {"E8": 0.80, "emitter": 0.90}
    ok, why = g.judge(0.40, 0.70, base, {"E8": 0.795, "emitter": 0.91})
    assert ok
    ok, why = g.judge(0.40, 0.70, base, {"E8": 0.78, "emitter": 0.91})
    assert not ok and "E8" in why
    assert not g.judge(0.40, 0.40, base, base)[0]                          # no rise on its own gap
    assert not g.judge(0.40, 0.70, base, {"E8": 0.80})[0]                  # a battery line not read
