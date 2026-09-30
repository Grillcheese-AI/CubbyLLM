"""The harness in front of the serving brain's doors (H-E14 v0.1): gate 1 by construction -- a brain shaped like
CubbyBrain (worlds of FactStores, an emitter, the VM client, sense/route/turn) answers byte-identically wired and
unwired, while every read, program and write inside a turn is a ledger row under the frame's budget.
Run: python -m pytest standin/tests/test_harness_wiring.py -q -p no:hypothesispytest"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import vault  # noqa: E402
from ledger import Ledger  # noqa: E402
from harness import Harness, Preflight, ToolRegistry, current_frame, sign_config  # noqa: E402
from harness_wiring import GatedStore, GatedWorlds, boot_serve, derive_packs, uninstall_vm_door, wire  # noqa: E402
from worlds import FactStore, route_world  # noqa: E402
from cubbyllm.bridges.possibility import same_subject_clash  # noqa: E402
from cubbyllm.reasoning.planner import parse_fact  # noqa: E402
import cubbyllm.bridges.cubelang_client as cc  # noqa: E402

GEO = ["Paris is the capital of France", "Berlin is the capital of Germany", "Rome is the capital of Italy"]
MUSIC = ["Thriller is the album of Michael Jackson", "Abbey Road is the album of The Beatles"]
OWNER, HK = b"owner", b"harness"


class FakeEmitter:
    def __init__(self):
        self.calls = []

    def emit(self, prompt, context="programs", max_new_tokens=450, **kw):
        self.calls.append({"prompt": prompt, "context": context, "max_new_tokens": max_new_tokens})
        if context == "talk":
            return "Hi there."
        first = next((l[2:] for l in prompt.splitlines() if l.startswith("- ")), None)
        t = parse_fact(first) if first else None
        return f'fn solve() {{ return "{t.obj}" }}' if t else "fn solve() { }"


class FakeCubbyBrain:
    """CubbyBrain's shape: worlds, an emitter, the chat state, sense -> route -> turn, the VM through cubelang_client."""

    def __init__(self):
        self.worlds = {"facts": FactStore(GEO, name="facts"), "music": FactStore(MUSIC, name="music")}
        self.emitter = FakeEmitter()
        self.chat = SimpleNamespace(state={"cortisol": 0.10, "dopamine": 0.5})
        self.route_tau = 0.3
        self.trace_log = []

    def sense(self, text):
        self.chat.state["cortisol"] = round(self.chat.state["cortisol"] + 0.01, 3)   # the ODE moves on every input
        self.trace_log.append(("sense", text))

    def route(self, text):
        if text.lower().startswith("remember "):
            return {"cortex": "memory", "fact": text[9:].strip(), "world": "facts"}
        if text.lower().startswith("help"):
            return {"cortex": "help"}
        world, score = route_world(self.worlds, text)
        # the hormone-modulated threshold: the SAME sense must precede the route, wired or not
        tau = self.route_tau + self.chat.state["cortisol"]
        if text.endswith("?"):                            # a question needs facts: reasoning, flat fallback gated by tau
            return {"cortex": "reasoning", "world": world, "score": score, "allow_flat": score >= tau}
        return {"cortex": "talk", "score": score}

    def turn(self, user_text, feedback=None, route=None, sensed=False):
        if not sensed:
            self.sense(user_text)
        route = route or self.route(user_text)
        cortex = route["cortex"]
        if cortex == "talk":
            reply = self.emitter.emit(user_text, context="talk", max_new_tokens=64)
            asked = cc.run_program_proto("fn think() { }", fn="think", args=[user_text])     # the ASK exit
            return {"reply": reply, "kind": "chat", "route": route, "ask": asked.get("ok")}
        if cortex == "help":
            cc.run_program_proto("fn think() { }", fn="think", args=[user_text])
            return {"reply": "I can answer questions about what I know.", "kind": "help", "route": route}
        if cortex == "memory":
            world = self.worlds[route["world"]]
            accepted = world.add(route["fact"])
            return {"reply": "Got it." if accepted else "I already know that.", "kind": "learn",
                    "learn": {"accepted": accepted, "fact": route["fact"]}, "route": route}
        world = self.worlds[route["world"]]
        hits = world(user_text, 3)
        facts = [f for _s, f in hits]
        m = re.match(r"given that (.+?), (what .+)$", user_text, re.I)   # a contradicting context: refuse, never resolve it
        if m and same_subject_clash(m.group(1), facts):
            return {"reply": "That clashes with what I know.", "dont_know": True, "kind": "task",
                    "task": {"speak_ok": False, "clash": True}, "route": route}
        program = self.emitter.emit(user_text + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts), max_new_tokens=450)
        out = cc.run_program_proto(program, fn="solve")
        answer = out.get("result")
        # the ground check: the answer is the object of an offered fact about the asked subject, in the asked relation
        grounded = answer is not None and any((t := parse_fact(f)) is not None and t.obj == answer
                                              and t.subj.lower() in user_text.lower()
                                              and t.rel.lower() in user_text.lower() for f in facts)
        return {"reply": f"It is {answer}." if grounded else "The facts don't say.", "dont_know": not grounded,
                "kind": "task", "task": {"speak_ok": grounded, "vm_answer": answer, "budget": out.get("budget")}, "route": route}


def fake_vm(program_source, fn="solve", args=None, exe=None, timeout=30.0, answers=None, knowledge_path=None,
            branch=None, budget=None):
    fake_vm.calls.append({"fn": fn, "budget": budget, "knowledge_path": knowledge_path, "program": program_source})
    if fn == "think":
        return {"ok": True, "result": "spoken"}
    if 'return "' in program_source:
        return {"ok": True, "result": program_source.split('return "')[1].split('"')[0], "budget": budget}
    return {"ok": True, "result": None, "budget": budget}


fake_vm.calls = []

PACK = {"boot": {"families": {
    "fact": {"bar": 1.0, "items": [{"ask": "What is the capital of France?", "expect": "spoken", "value": "Paris"}]},
    "contradiction": {"bar": 1.0, "defect": "contradicting_context", "items": [{"ask": "What is the capital of Atlantis?", "expect": "refused"}]},
    "absent": {"bar": 1.0, "defect": "absent_role", "items": [{"ask": "What is the river of France?", "expect": "refused"}]},
    "wrong_role": {"bar": 1.0, "defect": "wrong_role", "items": [{"ask": "What is the capital of Atlantis?", "expect": "not_wrong", "wrong": "Paris"}]}}}}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    monkeypatch.setattr(cc, "run_program_proto", fake_vm)
    fake_vm.calls = []
    led = Ledger(tmp_path / "ledger.db", vm_build="fake-vm")
    cfg = sign_config({"thresholds": {"tau_vm": 0.22}, "budget": {"reads": 64, "vm_runs": 8, "vm_hops": 24, "gen_tokens": 600, "wall_s": 30.0},
                       "vm_modules": {"memory": "write"}, "vm_budget": {"max_ops": 100000, "max_queries": 64}}, OWNER)
    brain = FakeCubbyBrain()
    reg = ToolRegistry(led)
    h = Harness(cfg, led, brain, reg, preflight=Preflight(PACK), vm_selftest=lambda: True, owner_key=OWNER, harness_key=HK)
    wire(brain, h, max_ops=100000, max_queries=64)
    yield h, brain, led
    uninstall_vm_door()


TURNS = ["What is the capital of Germany?", "hello there", "remember Lyon is the city of France", "help",
         "Whose album is Thriller?", "What is the capital of Atlantis?", "What is the capital of Germany?"]


def test_wired_replies_are_byte_identical_to_the_unwired_brain(wired):
    h, brain, led = wired
    control = FakeCubbyBrain()
    h.boot()                                                           # the boot pack's asks move the ODE, as they would live
    brain.chat.state = dict(control.chat.state)                        # same draw from here on
    n0 = len(brain.trace_log)
    for t in TURNS:
        a = h.turn(t)
        b = control.turn(t)
        assert a["reply"] == b["reply"], t
        assert brain.chat.state == control.chat.state, "the same sense precedes the same route, wired or not"
    assert brain.trace_log[n0:] == control.trace_log, "sensed exactly once per turn, in the brain's own order"
    kinds = [h.turns[f"t{i:06d}"]["row"]["status"] for i in range(1, len(TURNS) + 1)]
    assert kinds == ["answered", "answered", "answered", "answered", "refused", "refused", "answered"]


def test_every_door_inside_a_turn_is_a_row_under_the_frames_budget(wired):
    h, brain, led = wired
    h.boot()
    rec = h.turn("What is the capital of Germany?")
    assert rec["status"] == "answered" and rec["reply"] == "It is Berlin."
    calls = [led.get(x) for x in h.turns[rec["turn_id"]]["row"]["calls"]]
    names = [c["name"] for c in calls]
    assert names == ["store.read", "vm.run"], "the walk's read, then the program; routing was outside the frame"
    vm = json.loads(calls[1]["program"])
    assert vm["harness_set"] == ["budget"]
    assert fake_vm.calls[-1]["budget"] == {"max_jumps": 24, "max_wall_ms": pytest.approx(30000, abs=2000), "max_ops": 100000, "max_queries": 64}
    assert rec["task"]["budget"]["max_jumps"] == 24, "the program ran under the frame's budget"
    spent = h.turns[rec["turn_id"]]["row"]["budget"]["spent"]
    assert spent["reads"] == 1 and spent["vm_runs"] == 1 and spent["gen_tokens"] >= 1
    chat = h.turn("hello there")
    chat_calls = [led.get(x)["name"] for x in h.turns[chat["turn_id"]]["row"]["calls"]]
    assert chat_calls == ["vm.run"], "a chat turn reads no store; its ASK runs on the VM"
    learned = h.turn("remember Lyon is the city of France")
    w = [led.get(x) for x in h.turns[learned["turn_id"]]["row"]["calls"]]
    assert [c["name"] for c in w] == ["store.write"] and learned["status"] == "answered"
    assert "Lyon is the city of France" in brain.worlds["facts"].raw
    assert json.loads(w[0]["input"])["fact"]["world"] == "facts" and json.loads(w[0]["input"])["fact"]["status"] == "asserted"


def test_outside_a_turn_every_door_passes_through(wired):
    h, brain, led = wired
    assert current_frame() is None
    assert isinstance(brain.worlds, GatedWorlds) and isinstance(brain.worlds["facts"], GatedStore)
    assert brain.worlds["facts"]("capital of Italy", 1)[0][1] == "Rome is the capital of Italy"
    assert cc.run_program_proto("fn solve() { }", fn="solve") == {"ok": True, "result": None, "budget": None}
    assert led.count() == 0, "no frame, no row"
    assert "Rome is the capital of Italy" in brain.worlds["facts"] and len(brain.worlds["facts"]) == 3
    assert brain.worlds["facts"].lookup == brain.worlds["facts"].raw.lookup


def test_budgets_bind_as_a_declared_outcome_never_a_wrong_answer(wired):
    h, brain, led = wired
    h.boot()
    for line, want in (("gen_tokens", 0), ("vm_runs", 0), ("reads", 0)):
        h.config["budget"] = {**h.config["budget"], line: want}
        rec = h.turn("What is the capital of Germany?")
        assert rec["status"] == "exhausted" and rec["reply"] is None and rec["reason"] == line, line
        assert not rec["spoken"]
        h.config["budget"].pop(line)
    h.config["budget"]["gen_tokens"] = 10                              # a cap, not a stop: max_new_tokens is clipped
    rec = h.turn("What is the capital of Germany?")
    assert rec["status"] == "answered" and brain.emitter.calls[-1]["max_new_tokens"] == 10


def test_a_program_may_not_choose_its_budget_or_use_an_undeclared_module(wired):
    h, brain, led = wired
    h.boot()
    real = brain.emitter.emit

    def sneaky(prompt, **kw):
        return real(prompt, **kw).replace("fn solve", "use sensors\nfn solve")   # `use` of an undeclared module
    brain.emitter.emit = sneaky
    rec = h.turn("What is the capital of Germany?")
    assert rec["status"] == "refused" and rec["reason"].startswith("use sensors")
    brain.emitter.emit = real
    frame = h.frame_for("task", "q", "probe")
    out = h.registry.call(frame, "vm.run", program_source="fn solve() { }", budget={"max_jumps": 10 ** 6})
    assert fake_vm.calls[-1]["budget"]["max_jumps"] == 24, "the caller's budget never reaches the VM"


def test_mounting_later_and_the_oracle_reach_through_the_gate(wired):
    from cubbyllm.bridges.possibility import StorePossibility
    h, brain, led = wired
    brain.worlds["wiki"] = FactStore(["Seine is the river of Paris"], name="wiki")
    assert isinstance(brain.worlds["wiki"], GatedStore)
    o = StorePossibility(brain.worlds)
    for w in brain.worlds.values():
        w.oracle = o
    assert brain.worlds["wiki"].raw.oracle is o and o.possible("Seine is the river of Paris").possible


def test_uninstall_restores_the_client(wired):
    h, brain, led = wired
    uninstall_vm_door()
    assert cc.run_program_proto is fake_vm
    assert getattr(cc, "_harness_door", None) is None


def test_boot_serve_bootstraps_a_dev_config_from_the_brains_own_store(tmp_path, monkeypatch):
    """`serve_api --harness` bare: a config signed here, a boot pack derived from the store, the three defects in it."""
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    monkeypatch.setattr(cc, "run_program_proto", fake_vm)
    brain = FakeCubbyBrain()
    led = Ledger(tmp_path / "ledger.db", vm_build="fake-vm")
    try:
        packs = derive_packs(brain)
        fam = packs["boot"]["families"]
        assert [it["ask"] for it in fam["fact"]["items"]] == ["What is the capital of France?", "What is the capital of Germany?",
                                                              "What is the capital of Italy?"]
        assert fam["absent_role"]["items"][0]["ask"] == "What is the anthem of France?"
        assert fam["wrong_role"]["items"][0] == {"ask": "What is the capital of Atlantis?", "expect": "not_wrong", "wrong": "Paris"}
        h = boot_serve(brain, None, ledger=led)
        assert h.serving and brain.harness is h and h.preflight.missing_defects("boot") == []
        assert h.config["signature"] and h.config["thresholds"]["route_tau"] == 0.3 and h.config["budget"]["wall_s"] == 60.0
        assert [r["verdict"] for r in led.rows("boot") if r["fn"] is None] == ["serving"]
        rec = h.turn("What is the capital of Italy?")
        assert rec["reply"] == "It is Rome." and rec["spoken"] and rec["status"] == "answered"
        assert h.replay(rec["turn_id"])["calls"][0]["name"] == "store.read"
    finally:
        uninstall_vm_door()
