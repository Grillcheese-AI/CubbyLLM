"""Pins for the harness v0.1 (docs/research/2026-09-28-harness-v0.md + the round-3 amendments, H-E14): boot refuses
on any failed check and never moves a threshold; two keys; store generations instead of a lock; deny-by-default at
the registry with effect classes; typed postconditions and child intersection; budgets (gen_tokens, the VM mapping)
end a frame as a declared outcome; speaking is ledger-first and at-most-once; audit replay apart from regeneration;
quarantine and promote; worlds as namespaces; rotating gate packs with the real defects; the live-emitter cap.
No model, no VM, no GPU: a fake brain. Run: python -m pytest standin/tests/test_harness.py -q -p no:hypothesispytest"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import vault  # noqa: E402
from ledger import Ledger  # noqa: E402
from harness import (GRANTS, NEVER_LIVE, Budget, Exhausted, Fact, Generations, Harness, Preflight, Refused,  # noqa: E402
                     RefuseToServe, TaskFrame, Tool, ToolRegistry, current_frame, file_sha, sign_config)

OWNER, HARNESS_KEY = b"owner-key", b"harness-key"


class FakeBrain:
    """Routes like the stand-in brain, answers from a store file (one generation), records its model outputs."""

    def __init__(self, registry, store_path: str | None = None):
        self.registry = registry
        self.facts = {"when was x": "1683"}
        self.state = {"cortisol": 0.2, "dopamine": 0.5}
        self.store_path = store_path
        if store_path:
            self.use_generation({"history": store_path})

    # -- generations ------------------------------------------------------------------------------------
    def use_generation(self, stores: dict) -> None:
        self.store_path = stores["history"]
        self.facts = dict(json.loads(pathlib.Path(self.store_path).read_text(encoding="utf-8")).get("facts", {}))

    def view(self, stores: dict) -> "FakeBrain":
        v = FakeBrain(self.registry, stores["history"])
        v.state = dict(self.state)
        return v

    # -- hormones ---------------------------------------------------------------------------------------
    def hormones(self) -> dict:
        return dict(self.state)

    def set_hormones(self, h: dict) -> None:
        self.state = dict(h)

    # -- the turn ---------------------------------------------------------------------------------------
    def route(self, text):
        if text.startswith("remember"):
            return {"cortex": "memory"}
        if text.endswith("?"):
            return {"cortex": "reasoning"}
        return {"cortex": "talk"}

    def turn(self, text, feedback=None, frame=None):
        if frame is None:                                # the preflight's plain call
            return self._answer(text)
        if frame.kind == "task":
            got = self.registry.call(frame, "history.lookup", ask=text)
            if got is None:
                return {"reply": "The facts don't say.", "dont_know": True, "kind": "task", "model_outputs": ["<none>"]}
            return {"reply": f"It was {got}.", "kind": "task", "verdicts": ["verified"], "model_outputs": [f"It was {got}."]}
        if frame.kind == "learn":
            self.registry.call(frame, "store.write", fact=Fact(text, "history"))
            return {"reply": "Noted.", "kind": "learn", "learned": True, "model_outputs": ["Noted."]}
        return {"reply": "Hi.", "kind": "chat", "model_outputs": ["Hi."]}

    def _answer(self, text):
        v = self.facts.get(text.lower().rstrip("?").strip())
        return {"reply": f"It was {v}." if v else "The facts don't say.", "dont_know": v is None, "kind": "task"}

    def replay(self, text, outputs, hormones=None, generation=None):
        """The deterministic host over the recorded model outputs: the reply is the output, verbatim."""
        return {"reply": outputs[0] if outputs and outputs[0] != "<none>" else "The facts don't say."}


PACK = {"boot": {"families": {
    "fact": {"bar": 1.0, "items": [{"ask": "when was x?", "expect": "spoken", "value": "1683"}]},
    "contradiction": {"bar": 1.0, "defect": "contradicting_context", "items": [{"ask": "when was y?", "expect": "refused"}]},
    "absent": {"bar": 1.0, "defect": "absent_role", "items": [{"ask": "when was y?", "expect": "refused"}]},
    "wrong_role": {"bar": 1.0, "defect": "wrong_role", "items": [{"ask": "when was y?", "expect": "not_wrong", "wrong": "1492"}]}}},
    "regression": {"families": {"fact": {"bar": 1.0, "items": [{"ask": "when was x?", "expect": "spoken", "value": "1683"}]}}},
    "sentinel": {"families": {"held": {"bar": 1.0, "items": [{"ask": "who was z?", "expect": "not_wrong", "wrong": "nobody"}]}}}}


def make(tmp_path, monkeypatch, pack=None, tamper=None, generations=True, speaker=None, cfg_extra=None):
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    led = Ledger(tmp_path / "ledger.db", vm_build="fake-vm")
    store = tmp_path / "history.json"
    store.write_text(json.dumps({"facts": {"when was x": "1683"}}), encoding="utf-8")
    adapter = tmp_path / "talk.safetensors"
    adapter.write_bytes(b"\x00" * 64)
    live = tmp_path / "emitter_lora.safetensors"
    live.write_bytes(b"\x01" * 32)
    cfg = {"thresholds": {"tau_vm": 0.2202}, "budget": {"tool_calls": 3, "fetches": 1},
           "adapters": {"talk": {"path": str(adapter), "sha256": file_sha(adapter)}},
           "emitters": {"live": {"name": "cubby450m", "params_m": 450, "path": str(live), "sha256": file_sha(live)},
                        "teacher": {"name": "standin-2.6b", "params_m": 2600}},
           "max_live_params_m": 600,
           "hormones": {"narrow": [{"axis": "cortisol", "above": 0.8, "drop": ["fetch"]}]},
           "vm_modules": {"memory": "write", "sensors": "read"}}
    gens = None
    if generations:
        gens = Generations(tmp_path / "gens", HARNESS_KEY)
        man = gens.seed({"history": str(store)})
        store_path = gens.paths(man)["history"]
    else:
        cfg["stores"] = {"history": {"path": str(store), "sha256": file_sha(store)}}
        store_path = str(store)
    cfg.update(cfg_extra or {})
    cfg = sign_config(cfg, OWNER)
    if tamper:
        tamper(cfg, pathlib.Path(store_path), adapter, gens)
    reg = ToolRegistry(led)
    brain = FakeBrain(reg, store_path)
    reg.declare(Tool("history.lookup", "read", frozenset({"read_history"}),
                     fn=lambda ask: brain.facts.get(ask.lower().rstrip("?").strip()), spends="reads"))
    writes = []
    reg.declare(Tool("store.write", "write", frozenset({"write_store"}), fn=lambda fact: writes.append(fact) or True))
    reg.declare(Tool("forge.tool", "propose", frozenset({"forge", "vm"}), fn=lambda spec: "program"))
    reg.declare(Tool("learn.fetch", "external", frozenset({"fetch"}), fn=lambda entity: ["a fact"], spends="fetches"))
    reg.declare(Tool("teach", "propose", frozenset({"teacher"}), fn=lambda q: "a teacher's program"))
    reg.promote_tool(lambda fact: writes.append(fact) or True)
    pre = Preflight(pack or PACK)

    def night_fn(dst=None, n=None):
        if dst is None:
            return {"consolidated": 1}
        p = dst / "history.json"
        p.write_text(json.dumps({"facts": {"when was x": "1683", "when was w": "1815"}}), encoding="utf-8")
        return {"consolidated": 1, "stores": {"history": str(p)}}
    h = Harness(cfg, led, brain, reg, preflight=pre, vm_selftest=lambda: True, night_fn=night_fn,
                owner_key=OWNER, harness_key=HARNESS_KEY, generations=gens, speaker=speaker)
    return h, led, reg, writes


# ── boot ────────────────────────────────────────────────────────────────────
def test_boot_serves_when_every_check_passes_and_pins_the_generation(tmp_path, monkeypatch):
    h, led, *_ = make(tmp_path, monkeypatch)
    row = h.boot()
    assert h.serving and all(c["ok"] for c in row["checks"])
    assert [c["check"] for c in row["checks"]] == ["config.signature", "generation", "vm.selftest", "adapter.talk",
                                                   "emitter.live", "ledger.write", "preflight.defects", "preflight"]
    assert h.generation == 1 and row["generation"] == 1 and "gen_0001" in h.stores["history"]


@pytest.mark.parametrize("what", ["store", "adapter", "threshold", "vm", "preflight", "manifest_key", "emitter_cap", "defects"])
def test_boot_refuses_on_each_injected_fault_and_never_moves_a_threshold(tmp_path, monkeypatch, what):
    def tamper(cfg, store, adapter, gens):
        if what == "store":
            store.write_text('{"facts": {}}', encoding="utf-8")          # the generation's file changed under its hash
        elif what == "adapter":
            adapter.write_bytes(b"\x01" * 64)
        elif what == "threshold":
            cfg["thresholds"]["tau_vm"] = 0.05                             # edited after the owner signed
        elif what == "manifest_key":                                       # a manifest the OWNER signed is not the harness's
            man = gens.current()
            gens.key = OWNER
            gens.promote(man["generation"], gens.paths(man), "", {})
            gens.key = HARNESS_KEY
        elif what == "emitter_cap":
            cfg["emitters"]["live"]["params_m"] = 2600                     # the stand-in on the serve path
            cfg.update(sign_config({k: v for k, v in cfg.items() if k != "signature"}, OWNER))
    pack = None
    if what == "defects":
        pack = {"boot": {"families": {"fact": PACK["boot"]["families"]["fact"]}}}   # no absent role, no contradiction, no wrong role
    h, led, *_ = make(tmp_path, monkeypatch, tamper=tamper, pack=pack)
    if what == "vm":
        h.vm_selftest = lambda: False
    if what == "preflight":
        h.brain.facts.clear()
    with pytest.raises(RefuseToServe) as e:
        h.boot()
    assert not h.serving
    name = {"store": "generation", "adapter": "adapter.talk", "threshold": "config.signature", "vm": "vm.selftest",
            "preflight": "preflight", "manifest_key": "generation", "emitter_cap": "emitter.live", "defects": "preflight.defects"}[what]
    assert name in str(e.value)
    assert h.config["thresholds"]["tau_vm"] == (0.05 if what == "threshold" else 0.2202), "boot never rewrites a threshold"
    rows = [r for r in led.rows("boot") if not r["ok"]]
    assert rows and any(name in (r["verdict"] or "") for r in rows), "the refusal is a ledger row naming the check"


def test_v0_shape_still_boots_from_owner_signed_store_hashes(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch, generations=False)
    row = h.boot()
    assert h.serving and h.generation is None and "store.history" in [c["check"] for c in row["checks"]]
    assert h.night() == {"consolidated": 1}                              # in place, ledgered
    assert [r["fn"] for r in h.ledger.rows("night")] == ["begin", "end"]


def test_not_booted_never_serves(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    with pytest.raises(RefuseToServe):
        h.turn("hello")


# ── the registry ────────────────────────────────────────────────────────────
def test_deny_by_default_effect_classes_never_live_and_external(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    chat = h.frame_for("chat", "hi", "t1")
    with pytest.raises(Refused) as e:
        reg.call(chat, "history.lookup", ask="when was x?")
    assert "read_history" in e.value.reason
    with pytest.raises(Refused):
        reg.call(chat, "store.write", fact=Fact("x", "history"))           # a chat turn never writes
    task = h.frame_for("task", "q", "t2")
    with pytest.raises(Refused) as e:
        reg.call(task, "forge.tool", spec="s")                             # never in a live turn
    assert "never in a live turn" in e.value.reason
    with pytest.raises(Refused):
        reg.call(task, "teach", q="?")                                     # the teacher is a night tool
    fact = h.frame_for("fact", "q", "t2b")
    fact.allowed.add("fetch")                                              # even granted, an external effect is off for fact/chat
    with pytest.raises(Refused) as e:
        reg.call(fact, "learn.fetch", entity="x")
    assert "external effect off" in e.value.reason
    night = h.frame_for("night", "n", "t3")
    assert reg.call(night, "forge.tool", spec="s")["proposal"].startswith("p")
    with pytest.raises(Refused):
        reg.call(task, "no.such.tool")
    assert writes == []
    refused = len([r for r in led.rows("tool") if not r["ok"]])
    assert refused == 6, "every refusal is a ledger row"


def test_deny_probe_500_none_executed_all_ledgered(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    probes = []
    forbidden = [("chat", "store.write", {"fact": Fact("x", "history")}), ("chat", "history.lookup", {"ask": "q?"}),
                 ("task", "forge.tool", {"spec": "s"}), ("task", "teach", {"q": "?"}), ("fact", "learn.fetch", {"entity": "e"}),
                 ("learn", "learn.fetch", {"entity": "e"}), ("help", "store.write", {"fact": Fact("x", "history")}),
                 ("task", "promote", {"proposal": "p000001", "world": "history"}), ("chat", "no.such.tool", {}),
                 ("plugin", "store.write", {"fact": Fact("x", "history")})]
    while len(probes) < 500:
        probes.extend(forbidden)
    rep = h.deny_probe(probes[:500])
    assert rep == {"n": 500, "executed": 0, "ledgered": 500} and writes == []


def test_budgets_end_a_frame_as_a_declared_outcome_and_map_onto_the_vm(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    f = h.frame_for("task", "q", "t9")
    f.budget.reads = 3
    for _ in range(3):
        reg.call(f, "history.lookup", ask="when was x?")
    with pytest.raises(Exhausted) as e:
        reg.call(f, "history.lookup", ask="when was x?")
    assert str(e.value) == "reads" and f.budget.bound == ["reads"]
    g = h.frame_for("task", "q", "t10")
    reg.call(g, "learn.fetch", entity="x")
    with pytest.raises(Exhausted) as e:
        reg.call(g, "learn.fetch", entity="y")                             # the second fetch, over its own line (config: 1)
    assert str(e.value) == "fetches"
    with pytest.raises(Exhausted):                                         # depth: a chain of sub-goals past the budget
        c = g
        for _ in range(5):
            c = c.child("task", "sub")
    b = Budget(vm_hops=10, wall_s=100.0)
    b.spend("vm_hops", 4)
    vm = b.vm(max_ops=50_000, max_queries=32)
    assert vm["max_jumps"] == 6 and vm["max_ops"] == 50_000 and vm["max_queries"] == 32 and 0 < vm["max_wall_ms"] <= 100_000
    assert b.remaining("gen_tokens") == 2048 and Budget().fetches == 2   # refuse -> fetch -> a second walk is two by design


def test_child_permissions_are_an_intersection_and_a_tree(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    h.boot()
    task = h.frame_for("task", "q", "t1")
    learn = task.child("learn", "sub")
    assert learn.allowed == (task.allowed & GRANTS["learn"]) and "write_store" not in learn.allowed
    assert learn.parent is task and learn.budget is task.budget and learn.post == "learned"
    night = task.child("night", "n")
    assert not (night.allowed & NEVER_LIVE), "a child of a live frame never gets a night grant"


def test_harness_sets_a_programs_budget_and_knowledge_never_the_caller(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    seen = {}
    reg.declare(Tool("vm.run", "read", frozenset({"vm"}), fn=lambda **kw: seen.update(kw) or {"ok": True, "result": []},
                     spends="vm_runs", bind=lambda f, a: {"budget": f.budget.vm(), "knowledge_path": "gen/knowledge.jsonl"}))
    f = h.frame_for("task", "q", "t1")
    got = reg.call(f, "vm.run", program_source="fn solve() {}", budget={"max_jumps": 9999}, knowledge_path="/anything")
    assert seen["budget"]["max_jumps"] == f.budget.vm_hops and seen["knowledge_path"] == "gen/knowledge.jsonl"
    assert got == {"ok": True, "result": []}
    row = json.loads(led.get(f.calls[-1])["program"])
    assert row["harness_set"] == ["budget", "knowledge_path"]
    assert "empty" in led.get(f.calls[-1])["verdict"], "an empty result is a return, not a success"
    with pytest.raises(Refused) as e:                                      # a program that `use`s a write module in a fact turn
        reg.call(h.frame_for("fact", "q", "t2"), "vm.run", program_source="use memory\nfn solve() {}", uses=["memory"])
    assert "write module" in e.value.reason
    with pytest.raises(Refused):
        reg.call(f, "vm.run", program_source="use unknown", uses=["unknown"])


def test_quarantine_then_promote_is_the_one_path_to_a_store(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    night = h.frame_for("night", "n", "n1")
    night.budget.tool_calls = 10
    prop = reg.call(night, "forge.tool", spec="s")
    assert prop["value"] == "program" and reg.quarantine[prop["proposal"]]["promoted"] is None
    assert writes == []                                                    # a proposal is not a fact
    with pytest.raises(Refused):
        reg.call(h.frame_for("learn", "l", "t1"), "promote", proposal=prop["proposal"], world="history")   # live: never
    assert reg.call(night, "promote", proposal=prop["proposal"], world="history", status="derived") is True
    assert writes[-1].text == "program" and writes[-1].world == "history" and writes[-1].status == "derived"
    assert reg.quarantine[prop["proposal"]]["promoted"] == "history"
    with pytest.raises(Refused):
        reg.call(night, "promote", proposal=prop["proposal"], world="history")      # once
    with pytest.raises(Refused) as e:
        reg.call(night, "promote", proposal="p999999", world="history")
    assert "no such proposal" in e.value.reason


def test_worlds_are_namespaces(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    learn = h.frame_for("learn", "l", "t1")
    assert reg.call(learn, "store.write", fact=Fact("Paris is the capital of France", "history", "asserted"))
    with pytest.raises(Refused) as e:
        reg.call(learn, "store.write", fact=Fact("Lyon is the capital of France", "history", "counterfactual"))
    assert "never shares a namespace" in e.value.reason
    assert reg.call(learn, "store.write", fact=Fact("Lyon is the capital of France", "branch:t1", "counterfactual"))
    learn.worlds = {"history"}
    with pytest.raises(Refused) as e:
        reg.call(learn, "store.write", fact=Fact("x", "wiki", "asserted"))
    assert "not this frame's to write" in e.value.reason
    with pytest.raises(ValueError):
        Fact("x", "history", "guessed")
    assert [w.world for w in writes] == ["history", "branch:t1"]


# ── the turn: typed postconditions, the row, speaking ───────────────────────
def test_turn_hashes_the_reply_into_the_ledger_with_everything_that_bent_it(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    rec = h.turn("when was x?")
    assert rec["reply"] == "It was 1683." and rec["status"] == "answered" and rec["frame"] and rec["spoken"]
    rep = h.replay(rec["turn_id"])
    row = rep["row"]
    assert rep["found"] and rep["verified"] and row["reply_sha256"] == led.get(rec["ledger"])["got"]
    assert [c["name"] for c in rep["calls"]] == ["history.lookup"] and row["verdicts"] == ["verified"]
    assert row["post"] == "answer" and row["generation"] == 1 and row["hormones"] == {"cortisol": 0.2, "dopamine": 0.5}
    assert row["model"]["adapters"]["talk"] == h.config["adapters"]["talk"]["sha256"] and row["model"]["emitter"]
    assert row["budget"]["spent"] == {"reads": 1} and row["budget"]["bound"] == []
    assert led.get(rec["ledger"])["input"] == "when was x?", "the person's words go through the vault and come back"
    raw = led.db.execute("SELECT input FROM certifications WHERE hash=?", (rec["ledger"],)).fetchone()[0]
    assert "when was x" not in raw, "never in the clear on disk"
    absent = h.turn("when was y?")
    assert absent["status"] == "refused" and "don't say" in absent["reply"] and absent["spoken"]
    chat = h.turn("hello")
    assert chat["status"] == "answered" and h.replay(chat["turn_id"])["calls"] == [] and chat["reply"] == "Hi."
    learned = h.turn("remember that z")
    assert learned["status"] == "answered" and writes[-1].text == "remember that z" and writes[-1].world == "history"


def test_a_postcondition_is_typed_at_creation_and_judged_by_the_host(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    f = h.frame_for("task", "q", "t1")
    assert f.post == "answer"
    assert f.judge({"reply": "It was 1683.", "verdicts": ["verified"]}) == "answered"
    assert f.judge({"reply": "It was 1683."}) == "failed", "a spoken task answer without a verdict fails its postcondition"
    assert f.judge({"reply": "The facts don't say.", "dont_know": True}) == "refused"
    assert f.judge({"reply": None, "kind": "ask", "candidates": ["a", "b"]}) == "asked"
    assert h.frame_for("learn", "l", "t2").judge({"reply": "Noted.", "learn": {"accepted": False}}) == "refused"
    assert h.frame_for("chat", "c", "t3").judge({"reply": "Hi."}) == "answered"
    # through the turn: a brain that speaks a task answer with no verdict is not spoken
    real = h.brain.turn
    h.brain.turn = lambda text, feedback=None, frame=None: {"reply": "It was 1492.", "kind": "task"} if frame else real(text)
    rec = h.turn("when was x?")
    assert rec["status"] == "failed" and rec["reply"] is None and not rec["spoken"]


def test_a_turn_that_overruns_its_budget_is_exhausted_not_wrong(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    h.config["budget"] = {"reads": 0}
    rec = h.turn("when was x?")
    assert rec["status"] == "exhausted" and rec["reply"] is None and rec["reason"] == "reads" and rec["budget_bound"] == ["reads"]
    assert led.get(rec["ledger"])["verdict"] == "exhausted" and not rec["spoken"]


def test_speaking_is_ledger_first_and_at_most_once(tmp_path, monkeypatch):
    said = []
    h, led, reg, _ = make(tmp_path, monkeypatch, speaker=lambda rec: said.append(rec["reply"]))
    h.boot()
    rec = h.turn("when was x?")
    rows = led.rows("speak", name=rec["turn_id"])
    assert [r["fn"] for r in rows] == ["intent", "spoken"] and said == ["It was 1683."]
    assert rows[0]["got"] == rows[1]["got"] == led.get(rec["ledger"])["got"]
    assert h._speak(rec["turn_id"], rec) is False and said == ["It was 1683."], "never twice"
    order = [r[0] for r in led.db.execute("SELECT fn FROM certifications WHERE kind IN ('turn','speak') ORDER BY rowid").fetchall()]
    assert order[-3:] == ["task", "intent", "spoken"], "the row, the intent, the bytes, the spoken row"


def test_a_kill_between_intent_and_spoken_is_resolved_at_boot_and_never_re_spoken(tmp_path, monkeypatch):
    said = []

    def crashing(rec):
        if rec["reply"] == "It was 1683.":
            raise ConnectionResetError("client went away")
        said.append(rec["reply"])
    h, led, reg, _ = make(tmp_path, monkeypatch, speaker=crashing)
    h.boot()
    with pytest.raises(ConnectionResetError):
        h.turn("when was x?")
    assert [r["fn"] for r in led.rows("speak")] == ["intent"]           # the orphan
    # the process restarts on the same ledger
    h2, led2, *_ = make(tmp_path, monkeypatch, speaker=lambda rec: said.append(rec["reply"]))
    h2.ledger = led
    row = h2.boot()
    assert row["orphan_intents"] == 1 and h2.interrupted[0]["turn"] == "t000001"
    assert [r["fn"] for r in led.rows("speak")] == ["intent", "interrupted"]
    assert h2._speak("t000001", {"reply": "It was 1683."}) is False and said == []
    rec = h2.turn("hello")                                                # serving goes on; new turns speak
    assert rec["spoken"] and said == ["Hi."]
    assert h2.boot()["orphan_intents"] == 0                               # resolved once


# ── replay: audit apart from regeneration ───────────────────────────────────
def test_audit_replay_reproduces_every_reply_hash_and_regeneration_is_measured_apart(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    ids = [h.turn(t)["turn_id"] for t in ["when was x?", "hello", "when was y?", "remember that q"] * 5]
    audits = [h.replay(i, mode="audit") for i in ids]
    assert all(a["audit"] is True for a in audits) and len(audits) == 20
    assert led.rows("model", name=ids[0])[0]["got"] == led.get(h.turns[ids[0]]["hash"])["got"]
    h.brain.facts["when was x"] = "1684"                                   # the engine drifted: regeneration differs, audit does not
    assert h.replay(ids[0], mode="regen")["regen"] is False
    assert h.replay(ids[0], mode="audit")["audit"] is True
    assert h.replay("t999999")["found"] is False


# ── hormones: narrow, never widen; same spoken values at the extremes ───────
def test_hormones_narrow_a_grant_never_widen_it(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    h.boot()
    calm = h.frame_for("task", "q", "t1", hormones={"cortisol": 0.1})
    stressed = h.frame_for("task", "q", "t2", hormones={"cortisol": 0.95})
    assert "fetch" in calm.allowed and "fetch" not in stressed.allowed and stressed.allowed < calm.allowed
    h.config["hormones"]["narrow"].append({"axis": "dopamine", "above": 0.9, "drop": ["nothing_here"]})
    assert h.frame_for("chat", "c", "t3", hormones={"dopamine": 1.0}).allowed == GRANTS["chat"]
    rep = h.hormone_gate([{"cortisol": 1.0, "dopamine": 0.0}, {"cortisol": 0.0, "dopamine": 1.0}])
    assert rep["same_spoken_values"] and rep["never_wider"] and rep["n_extremes"] == 2
    assert h.brain.state == {"cortisol": 0.2, "dopamine": 0.5}, "the gate restores the state it found"


# ── the night: generations, no lock ─────────────────────────────────────────
def test_the_night_writes_the_next_generation_beside_the_current_and_swaps_after_the_pack(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    before = h.turn("when was w?")
    assert before["status"] == "refused" and before["generation"] == 1
    old_path = h.stores["history"]
    rep = h.night()
    assert rep["promoted"] and rep["generation"] == 2 and h.generation == 2
    assert "gen_0002" in h.stores["history"] and pathlib.Path(old_path).exists(), "g is never touched"
    man = h.generations.current()
    assert man["generation"] == 2 and h.generations.verify(man) == (True, "ok") and man["gates"]["passed"]
    after = h.turn("when was w?")
    assert after["reply"] == "It was 1815." and after["generation"] == 2
    kinds = [(r["fn"], r["ok"]) for r in led.rows("night")]
    assert kinds == [("begin", True), ("end", True)]


def test_a_generation_that_fails_the_pack_is_rejected_and_the_current_keeps_serving(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()

    def bad_night(dst, n):
        p = dst / "history.json"
        p.write_text(json.dumps({"facts": {"when was y": "1492"}}), encoding="utf-8")   # x lost, the sentinel's wrong value in
        return {"stores": {"history": str(p)}}
    h.night_fn = bad_night
    rep = h.night()
    assert rep["promoted"] is False and rep["generation"] == 2 and h.generation == 1
    assert set(rep["failed"]) >= {"fact", "wrong_role"}
    assert h.generations.current()["generation"] == 1 and (h.generations.root / "gen_0002" / "history.json").exists()
    assert h.turn("when was x?")["reply"] == "It was 1683."
    assert h.turn("when was y?")["status"] == "refused"
    assert [r["ok"] for r in led.rows("night", fn="end")] == [False]


def test_turns_keep_serving_while_the_night_writes(tmp_path, monkeypatch):
    import threading
    h, led, *_ = make(tmp_path, monkeypatch)
    h.boot()
    started, release = threading.Event(), threading.Event()
    real_night = h.night_fn

    def slow_night(dst, n):
        started.set()
        release.wait(5)
        return real_night(dst, n)
    h.night_fn = slow_night
    t = threading.Thread(target=h.night)
    t.start()
    started.wait(5)
    rec = h.turn("when was x?")                                            # arrives while the night writes g+1
    assert rec["status"] == "answered" and rec["generation"] == 1, "no lock: the day serves from g"
    release.set()
    t.join(5)
    assert h.generation == 2 and h.turn("when was w?")["generation"] == 2


# ── gate packs rotate; sentinels are never harvested ────────────────────────
def test_gate_packs_rotate_and_sentinels_stay_out_of_the_harvest(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    h.boot()
    pre = h.preflight
    assert pre.missing_defects("boot") == [] and pre.missing_defects("regression") == list(("contradicting_context", "absent_role", "wrong_role"))
    assert pre.run(h._preflight_answer, "regression")["passed"]
    assert pre.sentinel_asks() == {"who was z?"}
    harvest = [{"question": "who was z?", "program": "..."}, {"question": "when was x?", "program": "..."}]
    assert [r["question"] for r in pre.exclude_from_harvest(harvest)] == ["when was x?"]
    assert Preflight({"families": {}}).packs == {"boot": {"families": {}}}     # v0's single-pack form


def test_current_frame_is_the_turns_and_none_outside(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    h.boot()
    seen = {}
    real = h.brain.turn

    def spy(text, feedback=None, frame=None):
        seen["frame"] = current_frame()
        return real(text, feedback, frame)
    h.brain.turn = spy
    h.turn("hello")
    assert seen["frame"] is not None and seen["frame"].kind == "chat" and current_frame() is None
