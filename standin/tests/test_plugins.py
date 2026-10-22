"""Order 0028's plugins, each acceptance line on synthetic folders (no network, no model):
- startup reads 50 plugins' front matter alone: fast, no code imported;
- a file changed after approval blocks the plugin; urllib or subprocess fails the review; ``ctx.fetch`` refuses a
  host the plugin did not declare;
- a tool loads on a question that needs it and checks its arguments;
- an adapter of another layout or base is refused with its reason; loading and unloading one leaves the other roles'
  weights bit for bit;
- a world plugin answers a question naming its work with the work's facts, refuses what it does not hold, and
  unloading it leaves the other worlds as they were;
- an internal plugin is absent from the public description.
Run: python -m pytest standin/tests/test_plugins.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import plugins_guard as guard  # noqa: E402
from plugins_index import discover  # noqa: E402
from plugins_load import Loader  # noqa: E402


@pytest.fixture(autouse=True)
def lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBBY_PLUGINS_LOCK", str(tmp_path / "lock" / "plugins.lock.json"))


def plugin(root: Path, name: str, kinds="tool", extra="", files=None, audience="public", network="[]") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "PLUGIN.md").write_text(f"---\nname: {name}\ndescription: {name} does a thing for testing\nkinds: [{kinds}]\n"
                                 f"audience: {audience}\npermissions: {{network: {network}}}\n{extra}---\n\nbody\n",
                                 encoding="utf-8")
    for f, text in (files or {}).items():
        (d / f).write_text(text, encoding="utf-8")
    return d


def units_dir(root: Path) -> Path:
    d = root / "units"
    shutil.copytree(ROOT / "plugins" / "units", d)
    return d


def test_startup_reads_front_matter_only(tmp_path):
    for i in range(50):
        plugin(tmp_path, f"dummy-{i:02d}", extra='entry: {tool: "tool.py:T"}\n',
               files={"tool.py": "raise SystemExit('imported at startup')\n", "payload.bin": "x" * 200_000})
    t0 = time.perf_counter()
    idx = discover([tmp_path])
    took = time.perf_counter() - t0
    assert len(idx) == 50 and not idx.errors
    assert not any(m.startswith("cubby_plugin_dummy") for m in sys.modules)
    assert took < 1.0, f"{took:.3f} s"                         # the report measures it against the 200 ms line


def test_a_changed_file_blocks_and_bad_imports_fail_review(tmp_path):
    d = units_dir(tmp_path)
    p = discover([tmp_path])["units"]
    assert guard.status(p) == "needs approval" and guard.check(p) == []
    guard.approve(p)
    assert guard.status(p) == "ok"
    (d / "tool.py").write_text((d / "tool.py").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert guard.status(p) == "blocked: tool.py changed"
    loader = Loader(discover([tmp_path]), data_root=tmp_path / "data")
    assert not loader.load("units") and loader.status("units") == "blocked: tool.py changed"
    for i, bad in enumerate(("import urllib.request\n", "from subprocess import run\n", "import os\nos.system('dir')\n")):
        q = plugin(tmp_path / "bad", f"bad-{i}", files={"tool.py": bad})
        assert guard.check(discover([q.parent])[q.name]), bad


def test_ctx_fetch_reaches_only_declared_hosts(tmp_path):
    d = plugin(tmp_path, "fetcher", kinds="source", network="[ok.example]")
    calls = []
    fake = lambda url, timeout=20: (calls.append(url), (200, b"ok") if url.endswith("/data") else (404, b""))[1]  # noqa: E731
    ctx = guard.Ctx(discover([d.parent])["fetcher"], tmp_path / "data", fetch=fake)
    assert ctx.fetch("https://ok.example/data") == (200, b"ok")
    with pytest.raises(guard.Refused):
        ctx.fetch("https://evil.example/data")
    assert not any("evil" in c for c in calls)


def test_a_tool_wakes_on_its_question_and_checks_its_arguments(tmp_path):
    units_dir(tmp_path)
    guard.approve(discover([tmp_path])["units"])
    loader = Loader(discover([tmp_path]), data_root=tmp_path / "data")
    assert loader.wake("how many kilometres is 5 miles?") == ["units"]
    assert loader.call("units", {"value": 5, "from": "mi", "to": "km"}) == {"value": 8.04672, "unit": "km"}
    assert loader.call("units", {"value": 100, "from": "fahrenheit", "to": "celsius"})["value"] == pytest.approx(37.777778)
    with pytest.raises(ValueError):
        loader.call("units", {"value": "five", "from": "mi", "to": "km"})
    assert loader.wake("who wrote Hamlet?") == []


# -- adapters on the 450M trunk, its loaded state faked: the claim is about the weights it moves -------------------
class _W:
    def __init__(self, a):
        self.a = np.array(a, dtype=np.float32)

    def copy_(self, x):
        self.a[...] = x


class _Mod:
    def __init__(self, n):
        self.lora_A, self.lora_B = type("A", (), {"weight": _W(np.zeros(n))})(), type("B", (), {"weight": _W(np.zeros(n))})()


def _adapter(d: Path, kind: str, seed: int, rank=4):
    from safetensors.numpy import save_file
    d.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(seed)
    (d / f"{kind}_lora.json").write_text(json.dumps({"rank": rank, "alpha": 8.0, "targets": ["q", "v"]}), encoding="utf-8")
    save_file({f"{p}.lora_{x}.weight": g.standard_normal(6).astype(np.float32) for p in ("l0.q", "l0.v") for x in "AB"},
              str(d / f"{kind}_lora.safetensors"))


def _trunk(tmp_path):
    from contextlib import nullcontext
    from safetensors.numpy import load_file
    from emitter import Cubby450mTrunk
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"base weights")
    (tmp_path / "tok.json").write_text("{}", encoding="utf-8")
    _adapter(tmp_path / "emitter", "emitter", 1)
    t = Cubby450mTrunk(str(export), str(tmp_path / "tok.json"), {"programs": str(tmp_path / "emitter")})
    t._model, t._wrapped = object(), {p: _Mod(6) for p in ("l0.q", "l0.v")}
    t._grilly = type("G", (), {"no_grad": staticmethod(nullcontext), "from_numpy": staticmethod(lambda x: x)})()
    w = load_file(str(tmp_path / "emitter" / "emitter_lora.safetensors"))
    t._weights["programs"] = {p: (w[f"{p}.lora_A.weight"], w[f"{p}.lora_B.weight"]) for p in t._wrapped}
    return t


def test_an_adapter_of_another_layout_or_base_is_refused_and_unloading_restores_the_role(tmp_path):
    t = _trunk(tmp_path)
    t.use("programs")
    before = {p: (m.lora_A.weight.a.copy(), m.lora_B.weight.a.copy()) for p, m in t._wrapped.items()}
    _adapter(tmp_path / "wide", "talk", 2, rank=8)
    with pytest.raises(ValueError, match="layout"):
        t.add_role("wide", str(tmp_path / "wide"))
    _adapter(tmp_path / "talk", "talk", 3)
    with pytest.raises(ValueError, match="another base"):
        t.add_role("talk", str(tmp_path / "talk"), base="0" * 64)
    assert "talk" not in t.adapter_dirs and "wide" not in t.adapter_dirs
    t.add_role("talk", str(tmp_path / "talk"), base=t.base_hash())
    t.use("talk")
    assert any(not np.array_equal(m.lora_A.weight.a, before[p][0]) for p, m in t._wrapped.items())
    t.remove_role("talk")
    t.use("programs")
    for p, m in t._wrapped.items():
        assert np.array_equal(m.lora_A.weight.a, before[p][0]) and np.array_equal(m.lora_B.weight.a, before[p][1])


PLAY = ("ACT I. Verona.\n\nThen Juliet speaks to the nurse, and later Romeo answers. Juliet's mother, Lady Capulet, "
        "calls her, and Juliet goes. Romeo's father, Montague, is angry, so Romeo leaves. Lady Capulet waits while "
        "Montague argues and Juliet listens.\n\n") * 6


def test_a_world_plugin_answers_its_work_refuses_the_rest_and_unloads_cleanly(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "1513.txt").write_text(PLAY, encoding="utf-8")
    (pack / "1513_metadata.json").write_text(json.dumps({"title": "The Test Lovers"}), encoding="utf-8")
    monkeypatch.setenv("CUBBY_TEST_PLAYS", str(pack))
    plugin(tmp_path / "pl", "plays", kinds="world", extra='world: {route: fiction, data_env: CUBBY_TEST_PLAYS, glob: "[0-9]*.txt"}\n')
    units_dir(tmp_path / "pl")
    idx = discover([tmp_path / "pl"])
    for n in idx.names():
        guard.approve(idx[n])
    loader = Loader(idx, data_root=tmp_path / "data")
    assert loader.load("plays") and loader.load("units")
    got = loader.ask("In The Test Lovers, who is Juliet's mother?")
    assert got["world"] == "plugin:plays" and any("Lady Capulet" in f for f in got["facts"])
    assert loader.ask("In The Test Lovers, who is Juliet's husband?")["facts"] == []       # not held: refused
    assert loader.ask("Who is Juliet's mother?")["world"] is None                          # no work named: not covered
    loader.unload("plays")
    assert "plugin:plays" not in loader.knows.by_name and "units" in loader.tools


def test_an_internal_plugin_is_absent_from_the_public_description(tmp_path):
    plugin(tmp_path, "secret-feed", kinds="source", audience="internal")
    units_dir(tmp_path)
    loader = Loader(discover([tmp_path]), data_root=tmp_path / "data")
    assert [p["name"] for p in loader.describe("user")] == ["units"]
    assert {p["name"] for p in loader.describe("audit")} == {"units", "secret-feed"}
