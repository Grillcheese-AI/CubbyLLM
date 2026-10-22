"""Pins for standin/data/build_program_first.py (H-E18): the sampler's programs are v12e's dialect and the
VM computes what the sampler says; the host checks refuse an uncovered number, a stated step, a missing
seed or a leaked hop; the round trips keep a question only when the blind answer or the emitter's program
agrees; chains never reuse a dataset question. Fakes for the writer, the solver and the emitter; the real
VM where it is built (skipped otherwise). Run: python -m pytest standin/tests/test_build_program_first.py -q
-p no:hypothesispytest"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.append(p)

import build_program_first as b  # noqa: E402
from build_program_first import Operand as O, Step  # noqa: E402

JULIE = [Step(O("lit", 12), "*", O("lit", 2)), Step(O("lit", 12), "+", O("reg", 0)),
         Step(O("lit", 120), "-", O("reg", 1)), Step(O("reg", 2), "/", O("lit", 2))]
JULIE_V12E = """        create s0 : quantity;   # step 0: 12 * 2 = 24
        assign s0 = 12;
        mul s0, 2;
        create s1 : quantity;   # step 1: 12 + 24 = 36
        assign s1 = 12;
        add s1, s0;
        create s2 : quantity;   # step 2: 120 - 36 = 84
        assign s2 = 120;
        sub s2, s1;
        create s3 : quantity;   # step 3: 84 / 2 = 42
        assign s3 = 0;
        add s3, s2;
        div s3, 2;
        sum s3;
        query s3;
        return s3;
"""
NATALIA_V12E = ('# Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. '
                'How many clips did Natalia sell altogether in April and May?\nprogram GSM0 implements ISolver {\n'
                '    type Input = str;\n    type Output = quantity;\n    @external\n'
                '    public function parse(raw: str): Input { return raw; }\n\n'
                '    public pure function verify(input: Input, output: Output): bool { return true; }\n\n\n\n'
                '    @external\n    public function solve(input: Input): Output {\n'
                '        create s0 : quantity;   # step 0: 48 / 2 = 24\n        assign s0 = 48;\n        div s0, 2;\n'
                '        create s1 : quantity;   # step 1: 48 + 24 = 72\n        assign s1 = 48;\n        add s1, s0;\n'
                '        sum s1;\n        query s1;\n        return s1;\n    }\n}\n')


def _vm_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        cc.find_cubelang_exe()
    except Exception:
        pytest.skip("cubelang not built on this machine")


# ── the sampler and the dialect ─────────────────────────────────────────────
def test_the_dialect_is_v12e_byte_for_byte():
    natalia = [Step(O("lit", 48), "/", O("lit", 2)), Step(O("lit", 48), "+", O("reg", 0))]
    q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
         "How many clips did Natalia sell altogether in April and May?")
    assert b.arith_program(natalia, q, name="GSM0") == NATALIA_V12E
    assert JULIE_V12E in b.arith_program(JULIE)
    assert b.values(JULIE) == [24, 36, 84, 42] and b.inputs_of(JULIE) == [12, 2, 120]
    assert b.arith_shape(b.arith_program(JULIE)) == "L*L|L+r0|L-r1|r2/L"
    assert b.arith_shape(NATALIA_V12E) == "L/L|L+r0"


def test_sampled_programs_are_valid_dags():
    s = b.ArithSampler(0)
    for _ in range(300):
        steps = s.sample()
        vals = b.values(steps)
        assert 2 <= len(steps) <= 10 and all(0 < v <= b.MAX_VALUE for v in vals)
        used = {o.v for st in steps for o in (st.a, st.b) if o.kind == "reg"}
        assert used == set(range(len(steps) - 1)), "every step but the last is used"
        assert all(o.v < k for k, st in enumerate(steps) for o in (st.a, st.b) if o.kind == "reg")
        assert not any(b.trivial(st, vals) for st in steps)
    shapes = {b.arith_shape(b.arith_program(ArithSampler_one(seed))) for seed in range(60)}
    assert len(shapes) > 30, "the sampler is not one shape"


def ArithSampler_one(seed):
    return b.ArithSampler(seed).sample()


def test_values_refuse_inexact_division():
    with pytest.raises(ValueError):
        b.values([Step(O("lit", 7), "/", O("lit", 2))])


def test_the_vm_computes_what_the_sampler_says():
    _vm_or_skip()
    s = b.ArithSampler(1)
    for _ in range(12):
        steps = s.sample()
        ok, res, err = b.vm_run(b.arith_program(steps, "q?"), "solve")
        assert ok and b.same_value(res, b.values(steps)[-1]), (err, res)


def test_the_writer_sees_names_never_computed_values():
    t = b.steps_text(JULIE)
    assert t.splitlines()[0] == "start with 12, multiply by 2 -> A"
    assert "12 plus A -> B" in t and "C divided by 2 -> D" in t and t.endswith("the question asks for D")
    assert "24" not in t and "36" not in t and "84" not in t and "42" not in t
    assert b.arith_user("a farm", JULIE).endswith("\nProblem:")


def test_clean_question_keeps_the_problem_and_drops_what_a_small_writer_adds():
    """Smoke 2026-09-29 (LFM2.5-2.6B, 24 questions): 13 had no '?' after cleaning -- a stray </think>, the
    computation echoed back, or a solution appended."""
    raw = ("</think>\nSituation: a farm\nA farmer packs 12 eggs into each of 2 boxes. He buys 120 more. "
           "How many eggs does he have?\nA = 12 * 2 = 24\nAnswer: 144\n</s>")
    assert b._clean_question(raw) == ("A farmer packs 12 eggs into each of 2 boxes. He buys 120 more. "
                                      "How many eggs does he have?")
    assert b._clean_question("<think>let me see</think>Problem: Tom has 3 apples. How many? Answer: 3") == \
        "Tom has 3 apples. How many?"
    echo = "situation: a zoo\nstart with 10, add 7 -> A\nA plus 3 -> B\nthe question asks for B"
    assert "?" not in b._clean_question(echo)                # nothing but the echo: rejected as no_question


# ── the host checks ─────────────────────────────────────────────────────────
def test_arith_question_checks():
    ok = ("Julie is reading a 120-page book. Yesterday she read 12 pages and today twice as many. "
          "If she reads half of the remaining pages tomorrow, how many pages will she read?")
    assert b.check_arith_question(ok, JULIE) is None                          # 2 as 'twice' / 'half'
    assert b.check_arith_question(ok.replace("120-page", "long"), JULIE) == "uncovered_number"
    assert b.check_arith_question(ok.replace("today twice as many", "today 24 pages"), JULIE) == "states_value"
    assert b.check_arith_question(ok.replace("?", "."), JULIE) == "no_question"
    assert b.covered(12, "a dozen eggs") and b.covered(3, "three cats") and not b.covered(7, "a week")


def test_plan_question_checks():
    hidden = ["Iridomyrmex", "Taxxon", "Books/Organism"]
    seed = "iridomyrmex bigi"
    assert b.check_plan_question("Which component belongs to the kind of thing that is the parent group of Iridomyrmex bigi?",
                                 seed, hidden) is None
    assert b.check_plan_question("What is the component of Taxxon?", seed, hidden) == "no_seed"
    assert b.check_plan_question("Taxxon is the kind of the genus of iridomyrmex bigi; what is its component?",
                                 seed, hidden) == "states_entity"


# ── the round trips ─────────────────────────────────────────────────────────
class FakeLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, system, user, max_tokens=400, temperature=None):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


class FakeEmitter:
    def __init__(self, programs):
        self.programs, self.calls = list(programs), 0

    def emit(self, prompt, **kw):
        self.calls += 1
        return self.programs.pop(0) if self.programs else ""


def fake_vm(values):
    it = iter(values)

    def run(program, fn):
        return True, next(it), None
    return run


def test_answer_round_trip_is_the_blind_value():
    c = b.Checks(modes=("answer",), solver=FakeLLM(["Work... Answer: 42", "Answer: 41"]))
    assert c.arith("q?", 42) == ["answer"] and c.arith("q?", 42) == []


def test_emitter_round_trip_is_the_vm_on_the_emitters_program():
    em = FakeEmitter(["program A {}", "```cubelang\nprogram B {}\n```"])
    c = b.Checks(modes=("emitter",), emitter=em, samples=3, vm=fake_vm([41, 42]))
    assert c.arith("q?", 42) == ["emitter_sampled"] and em.calls == 2          # greedy missed, the 2nd sample hit
    both = b.Checks(modes=("answer", "emitter"), solver=FakeLLM(["Answer: 7"]), emitter=FakeEmitter(["p"]),
                    samples=1, vm=fake_vm([42]))
    assert both.arith("q?", 42) == ["emitter"], "either keeps; the record says which"


def test_plan_round_trips():
    seed, rels = "Grand Saconnex", ["administrative territorial entity", "legislative body"]
    vocab = ["administrative territorial entity", "capital", "legislative body"]
    solver = FakeLLM(['{"seed": "Grand Saconnex", "hops": ["administrative territorial entity", "legislative body"]}',
                      '{"seed": "Grand Saconnex", "hops": ["capital", "legislative body"]}'])
    c = b.Checks(modes=("answer",), solver=solver)
    assert c.plan("q?", seed, rels, vocab) == ["answer"] and c.plan("q?", seed, rels, vocab) == []
    assert json.dumps(vocab) in solver.calls[0][0], "the solver chooses from the store's relation list"
    em = FakeEmitter([b.cot_plan("grand saconnex", ["territorial entity", "legislative body"])])
    assert b.Checks(modes=("emitter",), emitter=em, samples=1).plan("q?", seed, rels, vocab) == ["emitter"]


def test_cot_plan_is_v12e_shape():
    assert b.cot_plan("Grand Saconnex", ["Administrative territorial entity", "legislative body"]) == (
        'use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n'
        '        create frame: number;\n        bind frame, SEED, "grand saconnex";\n'
        '        bind frame, HOP1, "administrative territorial entity";\n        bind frame, HOP2, "legislative body";\n'
        '        return recover(frame, SEED);\n    }\n}\n')
    assert b.parse_cot_plan(b.cot_plan("x y", ["a", "b c"])) == ("x y", ["a", "b c"])


# ── the builders, end to end on fakes ───────────────────────────────────────
def test_build_arith_keeps_only_checked_questions_and_writes_v12e_records():
    s = b.ArithSampler(5)
    first = s.sample()
    inputs = ", ".join(str(v) for v in b.inputs_of(first))

    class Writer:
        name = "fake-writer"

        def __init__(self):
            self.n = 0

        def chat(self, system, user, max_tokens=400, temperature=None):
            self.n += 1
            import re
            nums = ", ".join(re.findall(r"\b\d+\b", user.split("Computation:", 1)[1]))
            return (f"Numbers {nums} are involved. What is the total?" if self.n % 2
                    else "There are no numbers in this story at all, so what is the total?")

    class Solver:
        def chat(self, system, user, max_tokens=400, temperature=None):
            return "Answer: 1"
    checks = b.Checks(modes=("answer",), solver=Solver())
    checks.arith = lambda q, v: ["answer"]                       # the round trip itself is pinned above
    recs, stats = b.build_arith(6, Writer(), checks, variants=2, seed=5,
                                vm=lambda p, fn: (True, b.values(_last_steps(p))[-1], None), log=lambda *a: None)
    assert stats["programs"] == 6 and stats["written"] == 12
    assert stats["kept"] == len(recs) and stats.get("rejected:uncovered_number", 0) >= 5
    r = recs[0]
    assert r["task"] == "arithmetic" and r["subtype"].startswith("pf:steps=") and r["source"] == "program_first/arith"
    assert r["program"].startswith("# " + r["prompt"]) and r["split"] == "train" and r["pf"]["writer"] == "fake-writer"
    assert all(b.bucket(x["pf"]["shape"]) != 0 for x in recs), "bucket 0 is the held-out eval's"
    sys.path.append(str(ROOT / "validation"))
    from emitter_data import slot_record
    slotted = slot_record(r)
    assert slotted is not None and "$N" in slotted["program"], "the slot converter takes it unchanged"


def _last_steps(program):
    """Re-derive the steps' values from a program's comments (the fake VM's answer)."""
    import re
    vals = [int(m.group(1)) for m in re.finditer(r"= (\d+)\n", program)]
    return [Step(O("lit", vals[-1]), "+", O("lit", 0))] if vals else []


def test_build_chain_never_reuses_a_dataset_question_and_writes_plan_and_chain_records():
    facts = ["Iridomyrmex is the parent taxon of iridomyrmex bigi", "Iridomyrmex is the parent taxon of iridomyrmex alpha",
             "Taxxon is the instance of Iridomyrmex", "Taxxon is the instance of Formica",
             "Organism is the component of Taxxon", "Organism is the component of Plantae",
             "Formica is the parent taxon of formica rufa", "Formica is the parent taxon of formica fusca"]
    exclude = {b.question_body("What is the parent taxon of iridomyrmex bigi?")}
    sampler = b.PathSampler(facts, exclude=exclude, seed=3)
    for _ in range(50):
        got = sampler.sample()
        if got:
            seed, rels, triples, fs = got
            assert b.question_body(b.canonical_question(seed, rels)) not in exclude
            assert all(t.obj != seed for t in triples)

    class Writer:
        name = "fake-writer"

        def chat(self, system, user, max_tokens=400, temperature=None):
            canon = user.split(": ", 1)[1]
            seed = canon.rsplit(" of ", 1)[1].rstrip("?")
            return f"Tell me the thing asked about {seed}?\nWhat is it for something else?"
    checks = b.Checks(modes=("answer",))
    checks.plan = lambda q, seed, rels, vocab: ["answer"]
    recs, stats = b.build_chain(3, sampler, Writer(), checks, variants=2, heldout=False,
                                vm=lambda p, fn: (True, p.rsplit('"', 2)[-2] if fn == "solve" else _last_bind(p), None),
                                log=lambda *a: None)
    assert stats["paths"] >= 1 and stats["kept"] * 2 == len(recs)
    assert stats.get("rejected:no_seed", 0) >= 1, "the second wording never names the seed"
    plans = [r for r in recs if r["task"] == "plan"]
    chains = [r for r in recs if r["task"] == "chain"]
    assert len(plans) == len(chains) and plans[0]["program"].startswith("use vsa;\n\nprogram CotPlan")
    assert "\nFacts:\n- " in chains[0]["prompt"] and chains[0]["program"].startswith("use vsa;\n\nprogram CotChain")


def _last_bind(program):
    """The object bound for the recovered role of the program's LAST hop function (a fake VM's answer)."""
    import re
    fn = program.split("public function hop_")[-1]
    role = re.search(r"return recover\(frame, (\w+)\)", fn).group(1)
    return re.search(rf'bind frame, {role}, "([^"]*)"', fn).group(1)


def test_mix_adds_train_records_only(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({"id": "a", "task": "arithmetic", "split": "train"}) + "\n", encoding="utf-8")
    pf = tmp_path / "pf.jsonl"
    pf.write_text("\n".join(json.dumps(r) for r in [{"id": "b", "task": "plan", "split": "train"},
                                                    {"id": "c", "task": "plan", "split": "val"},
                                                    {"id": "a", "task": "arithmetic", "split": "train"}]) + "\n",
                  encoding="utf-8")
    out = tmp_path / "mix.jsonl"
    b.main(["mix", "--base", str(base), "--pf", str(pf), "--out", str(out)])
    ids = [json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert ids == ["a", "b"]
    man = json.loads((tmp_path / "mix.manifest.json").read_text(encoding="utf-8"))
    assert man["n_base"] == 1 and man["added"] == {"plan": 1}
