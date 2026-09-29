"""Pins for the harness v0 (docs/research/2026-09-28-harness-v0.md, H-E14): boot refuses on any failed check and
never moves a threshold; deny-by-default at the registry; budgets end a frame as a declared outcome; every call and
every reply is a ledger row that replays; the night holds the store lock. No model, no VM, no GPU: a fake brain.
Run: python -m pytest standin/tests/test_harness.py -q"""
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
from harness import (Budget, Exhausted, Harness, Preflight, Refused, RefuseToServe, TaskFrame, Tool,  # noqa: E402
                     ToolRegistry, file_sha, sign_config)


class FakeBrain:
    """Routes like the stand-in brain and answers through the registry when a frame is given."""

    def __init__(self, registry):
        self.registry = registry
        self.facts = {"when was x": "1683"}

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
                return {"reply": "The facts don't say.", "dont_know": True, "kind": "task"}
            return {"reply": f"It was {got}.", "kind": "task", "verdicts": ["verified"]}
        if frame.kind == "learn":
            self.registry.call(frame, "store.write", fact=text)
            return {"reply": "Noted.", "kind": "learn"}
        return {"reply": "Hi.", "kind": "chat"}

    def _answer(self, text):
        v = self.facts.get(text.lower().rstrip("?").strip())
        return {"reply": f"It was {v}." if v else "The facts don't say.", "dont_know": v is None, "kind": "task"}


def make(tmp_path, monkeypatch, pack=None, tamper=None):
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    led = Ledger(tmp_path / "ledger.db", vm_build="fake-vm")
    store = tmp_path / "history.jsonl"
    store.write_text('{"e": 1}\n', encoding="utf-8")
    adapter = tmp_path / "talk.safetensors"
    adapter.write_bytes(b"\x00" * 64)
    cfg = {"thresholds": {"tau_vm": 0.2202}, "budget": {"tool_calls": 3, "fetches": 1},
           "stores": {"history": {"path": str(store), "sha256": file_sha(store)}},
           "adapters": {"talk": {"path": str(adapter), "sha256": file_sha(adapter)}}}
    cfg = sign_config(cfg, b"config-key")
    if tamper:
        tamper(cfg, store, adapter)
    reg = ToolRegistry(led)
    brain = FakeBrain(reg)
    reg.declare(Tool("history.lookup", "read", frozenset({"read_history"}),
                     fn=lambda ask: brain.facts.get(ask.lower().rstrip("?").strip())))
    writes = []
    reg.declare(Tool("store.write", "write", frozenset({"write_store"}), fn=lambda fact: writes.append(fact) or True))
    reg.declare(Tool("forge.tool", "propose", frozenset({"forge", "vm"}), fn=lambda spec: "program"))
    reg.declare(Tool("learn.fetch", "propose", frozenset({"fetch"}), fn=lambda entity: ["a fact"], spends="fetches"))
    pre = Preflight(pack or {"families": {
        "fact": {"bar": 1.0, "items": [{"ask": "when was x?", "expect": "spoken", "value": "1683"}]},
        "absent": {"bar": 1.0, "items": [{"ask": "when was y?", "expect": "refused"}]},
        "sentinel": {"bar": 1.0, "items": [{"ask": "when was y?", "expect": "not_wrong", "wrong": "1492"}]}}})
    h = Harness(cfg, led, brain, reg, preflight=pre, vm_selftest=lambda: True,
                night_fn=lambda: {"consolidated": 1}, config_key=b"config-key")
    return h, led, reg, writes


def test_boot_serves_when_every_check_passes_and_ledgers_it(tmp_path, monkeypatch):
    h, led, *_ = make(tmp_path, monkeypatch)
    row = h.boot()
    assert h.serving and all(c["ok"] for c in row["checks"])
    assert [c["check"] for c in row["checks"]] == ["config.signature", "store.history", "vm.selftest", "adapter.talk", "ledger.write", "preflight"]
    assert led.count(ok=True) >= 2                       # the probe row and the boot row


@pytest.mark.parametrize("what", ["store", "adapter", "threshold", "vm", "preflight"])
def test_boot_refuses_on_each_injected_fault_and_never_moves_a_threshold(tmp_path, monkeypatch, what):
    def tamper(cfg, store, adapter):
        if what == "store":
            store.write_text('{"e": 2}\n', encoding="utf-8")          # the file changed under its hash
        elif what == "adapter":
            adapter.write_bytes(b"\x01" * 64)
        elif what == "threshold":
            cfg["thresholds"]["tau_vm"] = 0.05                         # edited after signing
    h, led, *_ = make(tmp_path, monkeypatch, tamper=tamper)
    if what == "vm":
        h.vm_selftest = lambda: False
    if what == "preflight":
        h.brain.facts.clear()                                          # the fact family now fails its bar
    with pytest.raises(RefuseToServe) as e:
        h.boot()
    assert not h.serving
    name = {"store": "store.history", "adapter": "adapter.talk", "threshold": "config.signature",
            "vm": "vm.selftest", "preflight": "preflight"}[what]
    assert name in str(e.value)
    assert h.config["thresholds"]["tau_vm"] == (0.05 if what == "threshold" else 0.2202), "boot never rewrites a threshold"
    rows = [led.get(r[0]) for r in led.db.execute("SELECT hash FROM certifications WHERE kind='boot' AND ok=0").fetchall()]
    assert rows and any(name in (r["verdict"] or "") for r in rows), "the refusal is a ledger row naming the check"


def test_deny_by_default_effect_classes_and_never_live(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    chat = h.frame_for("chat", "hi", "t1")
    with pytest.raises(Refused) as e:
        reg.call(chat, "history.lookup", ask="when was x?")
    assert "read_history" in e.value.reason
    with pytest.raises(Refused):
        reg.call(chat, "store.write", fact="x")                        # a chat turn never writes
    task = h.frame_for("task", "q", "t2")
    with pytest.raises(Refused) as e:
        reg.call(task, "forge.tool", spec="s")                         # never in a live turn
    night = h.frame_for("night", "n", "t3")
    night.allowed |= {"forge"}                                         # the night's grant is applied by night(), not frame_for
    assert reg.call(night, "forge.tool", spec="s") == "program"
    with pytest.raises(Refused):
        reg.call(task, "no.such.tool")
    assert writes == []
    refused = led.db.execute("SELECT COUNT(*) FROM certifications WHERE kind='tool' AND ok=0").fetchone()[0]
    assert refused == 4, "every refusal is a ledger row"


def test_budgets_end_a_frame_as_a_declared_outcome(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    f = h.frame_for("task", "q", "t9")
    for _ in range(3):
        reg.call(f, "history.lookup", ask="when was x?")
    with pytest.raises(Exhausted) as e:
        reg.call(f, "history.lookup", ask="when was x?")
    assert str(e.value) == "tool_calls"
    g = h.frame_for("task", "q", "t10")
    reg.call(g, "learn.fetch", entity="x")
    with pytest.raises(Exhausted) as e:
        reg.call(g, "learn.fetch", entity="y")                         # the second fetch, over its own line
    assert str(e.value) == "fetches"
    with pytest.raises(Exhausted):                                     # depth: a chain of sub-goals past the budget
        c = g
        for _ in range(5):
            c = c.child("task", "sub")


def test_turn_hashes_the_reply_into_the_ledger_and_replays(tmp_path, monkeypatch):
    h, led, reg, writes = make(tmp_path, monkeypatch)
    h.boot()
    rec = h.turn("when was x?")
    assert rec["reply"] == "It was 1683." and rec["status"] == "answered" and rec["frame"]
    rep = h.replay(rec["turn_id"])
    assert rep["found"] and rep["verified"] and rep["row"]["reply_sha256"] == led.get(rec["ledger"])["got"]
    assert [c["name"] for c in rep["calls"]] == ["history.lookup"] and rep["row"]["verdicts"] == ["verified"]
    assert rep["row"]["budget"]["spent"]["tool_calls"] == 1
    assert led.get(rec["ledger"])["input"] == "when was x?", "the person's words go through the vault and come back"
    raw = led.db.execute("SELECT input FROM certifications WHERE hash=?", (rec["ledger"],)).fetchone()[0]
    assert "when was x" not in raw, "never in the clear on disk"
    absent = h.turn("when was y?")
    assert absent["status"] == "refused" and "don't say" in absent["reply"]
    chat = h.turn("hello")
    assert chat["status"] == "answered" and h.replay(chat["turn_id"])["calls"] == []
    learned = h.turn("remember that z")
    assert learned["status"] == "answered" and writes == ["remember that z"]


def test_a_turn_that_overruns_its_budget_is_exhausted_not_wrong(tmp_path, monkeypatch):
    h, led, reg, _ = make(tmp_path, monkeypatch)
    h.boot()
    h.config["budget"] = {"tool_calls": 0}
    rec = h.turn("when was x?")
    assert rec["status"] == "exhausted" and rec["reply"] is None and rec["reason"] == "tool_calls"
    assert led.get(rec["ledger"])["verdict"] == "exhausted"


def test_the_night_holds_the_store_lock_and_a_turn_meanwhile_is_refused(tmp_path, monkeypatch):
    import threading
    h, led, *_ = make(tmp_path, monkeypatch)
    h.boot()
    started, release = threading.Event(), threading.Event()
    seen = {}

    def slow_night():
        started.set()
        release.wait(5)
        return {"consolidated": 1}
    h.night_fn = slow_night
    t = threading.Thread(target=lambda: seen.setdefault("night", h.night()))
    t.start()
    started.wait(5)
    rec = h.turn("when was x?")                                        # arrives while the night writes
    assert rec["kind"] == "refused" and "consolidating" in rec["reason"]
    release.set()
    t.join(5)
    assert seen["night"] == {"consolidated": 1}
    assert h.turn("when was x?")["status"] == "answered"                # and serves again after
    kinds = [r[0] for r in led.db.execute("SELECT fn FROM certifications WHERE kind='night' ORDER BY created_utc").fetchall()]
    assert kinds == ["begin", "end"]


def test_not_booted_never_serves(tmp_path, monkeypatch):
    h, *_ = make(tmp_path, monkeypatch)
    with pytest.raises(RefuseToServe):
        h.turn("hello")
