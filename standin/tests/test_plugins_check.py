"""Order 0028 review 2: a plugin the router woke loads only when it claims the question.
- a source whose ``covers`` says no is not mounted and fetches nothing, and it is not counted as woken; one that says
  yes is built once and mounted;
- the talk adapter wakes only on a turn the thalamus says needs no facts;
- the alerts plugin no longer claims "what causes freezing rain?" and still claims a warning question;
- the board's "which plugin would wake?" names what declined and why; the ask record keeps it for the audit audience
  only.
Run: python -m pytest standin/tests/test_plugins_check.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

import plugins_guard as guard  # noqa: E402
from plugins_index import discover  # noqa: E402
from plugins_load import Loader  # noqa: E402

SOURCE = """class S:
    def __init__(self, ctx):
        self.ctx = ctx
        ctx.save("built", str(int(ctx.load("built") or b"0") + 1).encode())

    def covers(self, q):
        return 0.9 if "tide" in q else 0.0

    def answer(self, q):
        self.ctx.save("asked", q.encode())
        return ["the tide is high at noon"]
"""


@pytest.fixture(autouse=True)
def lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBBY_PLUGINS_LOCK", str(tmp_path / "lock.json"))


def tides(root: Path) -> Path:
    d = root / "tides"
    d.mkdir(parents=True)
    (d / "PLUGIN.md").write_text("---\nname: tides\ndescription: tides and the sea, high and low water times\n"
                                 "examples: [when is high tide at the harbour, what are the tide times]\n"
                                 "kinds: [source]\nentry: {source: \"s.py:S\"}\n---\n", encoding="utf-8")
    (d / "s.py").write_text(SOURCE, encoding="utf-8")
    return d


def wakes(name):
    """A router that wakes ``name`` on every question: the check is what is tested."""
    from types import SimpleNamespace
    return SimpleNamespace(tau=0.5, wake=lambda q, disabled=(): [(name, 1.0)], scores=lambda q: {name: 1.0})


def built(loader, p):
    c = guard.Ctx(p, loader.data_root)
    return int(c.load("built") or b"0"), c.load("asked")


def test_a_source_that_does_not_claim_the_question_is_not_mounted(tmp_path):
    tides(tmp_path)
    idx = discover([tmp_path])
    guard.approve(idx["tides"])
    loader = Loader(idx, data_root=tmp_path / "data", log=lambda *_: None)
    loader.router = wakes("tides")
    assert loader.wake("what are the sea times for the harbour") == []          # the router woke it; covers said no
    assert "does not claim" in loader.declined["tides"] and "tides" not in loader.loaded
    assert not loader.knows.by_name and built(loader, idx["tides"]) == (1, None)   # built to ask, not mounted or asked
    assert loader.wake("when is high tide at the harbour") == ["tides"]
    assert "tides" in loader.loaded and built(loader, idx["tides"])[0] == 2     # built once for this wake, then mounted
    assert loader.wake("when is high tide at the harbour") == ["tides"] and built(loader, idx["tides"])[0] == 2


class Trunk:
    def __init__(self):
        self.roles = []

    def add_role(self, role, d, base=None):
        self.roles.append(role)

    def remove_role(self, role):
        self.roles.remove(role)


def test_the_talk_adapter_wakes_only_on_a_turn_without_facts(tmp_path):
    shutil.copytree(ROOT / "plugins" / "talk", tmp_path / "talk")
    idx = discover([tmp_path])
    guard.approve(idx["talk"])
    from serve import CubbyBrain
    thalamus = CubbyBrain.needs_facts.__get__(CubbyBrain.__new__(CubbyBrain))
    trunk = Trunk()
    loader = Loader(idx, data_root=tmp_path / "data", trunk=trunk, thalamus=thalamus, log=lambda *_: None)
    loader.router = wakes("talk")
    assert loader.wake("who wrote Pride and Prejudice?") == [] and "needs facts" in loader.declined["talk"]
    assert trunk.roles == []
    assert loader.wake("hi Cubby, how are you?") == ["talk"] and trunk.roles == ["talk"]
    no_thalamus = Loader(idx, data_root=tmp_path / "d2", trunk=Trunk(), log=lambda *_: None)
    no_thalamus.router = wakes("talk")
    assert no_thalamus.wake("hi Cubby, how are you?") == [] and "no thalamus" in no_thalamus.declined["talk"]


def test_the_alerts_plugin_claims_warnings_not_weather_in_general():
    import importlib.util
    spec = importlib.util.spec_from_file_location("alerts_src", ROOT / "plugins" / "envcanada-alerts" / "source.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = m.Alerts(None)
    for q in ("what causes freezing rain?", "how much snow did Montreal get in 1998?", "qu'est-ce qui cause la pluie verglaçante ?"):
        assert a.covers(q) == 0.0, q
    for q in ("is there a storm warning in Halifax?", "y a-t-il un avertissement de neige à Montréal ?",
              "is there heavy snow in Ottawa right now?"):
        assert a.covers(q) > 0, q


def test_the_board_and_the_record_say_what_declined(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from audience import for_user
    from plugins_serve import board_get, on_question
    tides(tmp_path)
    idx = discover([tmp_path])
    guard.approve(idx["tides"])
    brain = SimpleNamespace(worlds={}, plugins=None)
    brain.plugins = Loader(idx, brain=brain, data_root=tmp_path / "data", log=lambda *_: None)
    brain.plugins.router = wakes("tides")
    got = board_get(brain, "/board/plugins/route", "q=what+are+the+sea+times", "audit")
    assert got["would_wake"] == [] and "does not claim" in got["declined"]["tides"]
    rec = {"id": "r1", "answer": None}
    on_question(brain, "what are the sea times for the harbour")
    on_question(brain, "what are the sea times for the harbour", rec)
    assert "plugin:tides" in rec["plugins_declined"] and "plugins_declined" not in for_user(rec)
