"""standin/plugins_load.py -- plugins loaded when a question needs them, like skills (order 0028).

``Loader(index, ...)`` holds the front-matter index (``plugins_index``) and a router over the plugins' descriptions and
examples (``plugins_route``). ``wake(question)`` loads, once, every plugin the question scores at or above ``tau`` for
(and every pinned one), as its kinds say:

- ``source``: its class (``entry.source = "file.py:Class"``) built with a ``Ctx`` (``plugins_guard``); one that
  ``covers`` and ``answer``s (``worlds.Knows``) is mounted in the loader's own ``Worlds`` router, so its facts come
  back through ``ask`` like any askable world's -- knowledge, never state -- tagged ``plugin:<name>``; one with
  ``facts(entity)`` (``learn.Source``) is kept in ``sources``;
- ``tool``: its class with a JSON ``schema`` and ``run(args) -> dict``; ``call(name, args)`` checks the arguments first;
- ``world``: a data pack ingested once into ``<data_root>/plugins/<name>/`` by 0015's route (``world.route``: ``fiction``
  -- in-fiction facts, answering only when the question names the work -- or ``facts``, a JSON-lines file of facts
  mounted through ``CubbyBrain.mount`` as ``plugin:<name>``);
- ``adapter``: a LoRA role added to the 450M base (``Cubby450mTrunk.add_role``), refused with its reason when its layout or
  base hash differs; unloaded after ``idle_min`` minutes unused (``sweep``);
- ``panel``: a card for the board (``panel.json``: title, chart, and the loaded source's method that feeds it).

Nothing loads unless its files are as Nick approved them (``plugins_guard.status``) and, for code, it passes
``check``. A plugin never writes a reply: what it returns goes through the same gates as any fact. Disabled plugins
never wake; pinned ones load at once and stay.

Wired: WIRED (``serve_api.py --plugins``; the board's Plugins view).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

__wiring__ = "WIRED"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plugins_guard as guard  # noqa: E402
from plugins_route import TAU_ENCODER, TAU_OVERLAP, Router  # noqa: E402
from plugins_world import FictionWorld  # noqa: E402,F401 -- the fiction world's own module
from worlds import FactStore, Worlds  # noqa: E402

CODE_KINDS = ("source", "tool")


class _Mount:
    def __init__(self, name, worlds) -> None:
        self.name, self._w = name, worlds

    def worlds(self):
        return self._w


def _import(plugin, spec: str):
    """``file.py:Class`` from the plugin's folder, as a module of its own name."""
    file, _, cls = spec.partition(":")
    path = (plugin.folder / file).resolve()
    if plugin.folder.resolve() not in path.parents:
        raise ValueError(f"{spec} is outside the plugin's folder")
    mod_name = f"cubby_plugin_{plugin.name.replace('-', '_')}_{path.stem}"
    sp = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return getattr(mod, cls)


class Loader:
    def __init__(self, index, brain=None, data_root=None, enc=None, tau: float | None = None, idle_min: float = 30.0,
                 log=print, fetch=None, trunk=None) -> None:
        self.index, self.brain, self.log, self.fetch, self.trunk = index, brain, log, fetch, trunk
        self.data_root = Path(data_root or os.environ.get("CUBBY_PLUGINS_DATA") or
                              Path(__file__).resolve().parent / "data" / "out")
        self.router = Router(index, enc=enc, tau=tau if tau is not None else (TAU_ENCODER if enc is not None else TAU_OVERLAP))
        self.idle_s = float(idle_min) * 60
        self.knows = Worlds(tau=0.5)
        self.loaded: dict[str, dict] = {}                    # name -> {kinds: objects, at, used}
        self.refused: dict[str, str] = {}                    # name -> why its last load failed
        self.sources, self.tools, self.panels = {}, {}, {}
        st = self._state()
        self.pinned, self.disabled = set(st.get("pinned", [])), set(st.get("disabled", []))
        for name in sorted(self.pinned - self.disabled):
            if name in index:
                self.load(name)

    # -- state the board changes: pinned and disabled, kept beside the lock (local) ---------------------------------
    def _state_path(self) -> Path:
        return guard.lock_path().with_name("plugins.state.json")

    def _state(self) -> dict:
        try:
            return json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self._state_path().parent.mkdir(parents=True, exist_ok=True)
        self._state_path().write_text(json.dumps({"pinned": sorted(self.pinned), "disabled": sorted(self.disabled)}),
                                      encoding="utf-8")

    def pin(self, name: str, on: bool = True) -> None:
        (self.pinned.add if on else self.pinned.discard)(name)
        self._save()
        if on and name not in self.disabled:
            self.load(name)

    def disable(self, name: str, on: bool = True) -> None:
        (self.disabled.add if on else self.disabled.discard)(name)
        self._save()
        if on:
            self.unload(name)

    # -- waking and loading -----------------------------------------------------------------------------------------
    def wake(self, question: str) -> list[str]:
        """Load what the question needs; the names of the plugins it woke (loaded now or already)."""
        woke = []
        for name, _s in self.router.wake(question, disabled=self.disabled):
            if self.load(name):
                woke.append(name)
        for name in woke:
            self.loaded[name]["used"] = time.time()
        self.sweep()
        return woke

    def status(self, name: str) -> str:
        if name in self.disabled:
            return "disabled"
        g = guard.status(self.index[name])
        if g != "ok":
            return g
        if name in self.refused:
            return "refused: " + self.refused[name]
        if name in self.loaded:
            return "pinned" if name in self.pinned else "loaded"
        return "idle"

    def load(self, name: str) -> bool:
        if name in self.loaded:
            return True
        if name in self.disabled or name not in self.index:
            return False
        p = self.index[name]
        why = guard.status(p)
        if why == "ok" and any(k in CODE_KINDS for k in p.kinds):
            bad = guard.check(p)
            why = "check: " + bad[0] if bad else "ok"
        if why != "ok":
            self.refused[name] = why
            return False
        ctx = guard.Ctx(p, self.data_root, self.log, self.fetch)
        got = {}
        try:
            for kind in p.kinds:
                got[kind] = getattr(self, f"_load_{kind}")(p, ctx)
        except Exception as e:                               # noqa: BLE001 -- a plugin that fails stays out, with why
            self.refused[name] = f"{type(e).__name__}: {str(e)[:200]}"
            self._drop(name, got)
            return False
        self.refused.pop(name, None)
        self.loaded[name] = {"kinds": got, "at": time.time(), "used": time.time()}
        self.log(f"plugin {p.id} loaded ({', '.join(p.kinds)})")
        return True

    def _load_source(self, p, ctx):
        obj = _import(p, p.entry["source"])(ctx)
        if hasattr(obj, "covers") and hasattr(obj, "answer"):
            obj.name, obj.domain = p.id, getattr(obj, "domain", p.description)
            self.knows.mount(obj)
        if hasattr(obj, "facts"):
            self.sources[p.name] = obj
        return obj

    def _load_tool(self, p, ctx):
        obj = _import(p, p.entry["tool"])(ctx)
        if not callable(getattr(obj, "run", None)) or not isinstance(getattr(obj, "schema", None), dict):
            raise TypeError("a tool needs run(args) and a JSON schema")
        self.tools[p.name] = obj
        return obj

    def _load_world(self, p, ctx):
        w = p.extra.get("world") or {}
        if w.get("route") == "fiction":
            from types import SimpleNamespace
            import sqlite3
            import worlds_adapters as wa
            from worlds_fiction import ingest
            from worlds_ingest import SCHEMA
            data = Path(os.environ.get(w.get("data_env") or "", "") or ctx.data_dir / "pack")
            files = sorted(f for f in data.glob(w.get("glob") or "*.txt") if f.is_file()) if data.is_dir() else []
            reg = sqlite3.connect(ctx.data_dir / "registry.sqlite")
            reg.executescript(SCHEMA)
            ingest(SimpleNamespace(path=str(data)), p.id, wa.Context(ctx.data_dir, reg, None, lambda *_: None), files=files)
            db = ctx.data_dir / "fiction" / "fiction.sqlite"
            if not db.exists():
                raise FileNotFoundError(f"no pack to read (set {w.get('data_env')})")
            world = FictionWorld(p.id, p.description, db)
            self.knows.mount(world)
            return world
        if w.get("route") == "facts":
            store = FactStore([json.loads(l)["fact"] for l in open(p.folder / w["file"], encoding="utf-8") if l.strip()],
                              name=p.id)
            if self.brain is not None:
                self.brain.mount(_Mount(p.id, {p.id: store}))
            return store
        raise ValueError(f"world route {w.get('route')!r}: fiction or facts")

    def _load_adapter(self, p, ctx):
        if self.trunk is None:
            raise RuntimeError("no 450M base mounted to take an adapter")
        a = p.extra.get("adapter") or {}
        d = os.environ.get(a.get("dir_env") or "") or str(p.folder)
        self.trunk.add_role(a.get("role") or p.name, d, (p.requires or {}).get("base"))
        return a.get("role") or p.name

    def _load_panel(self, p, ctx):
        panel = json.loads((p.folder / p.entry.get("panel", "panel.json")).read_text(encoding="utf-8"))
        self.panels[p.name] = panel
        return panel

    def _drop(self, name, got) -> None:
        for kind, obj in got.items():
            if kind in ("source", "world") and getattr(obj, "name", None) in self.knows.by_name:
                del self.knows.by_name[obj.name]
            if kind == "world" and self.brain is not None and f"plugin:{name}" in getattr(self.brain, "worlds", {}):
                del self.brain.worlds[f"plugin:{name}"]
            if kind == "adapter" and self.trunk is not None:
                self.trunk.remove_role(obj)
        self.sources.pop(name, None), self.tools.pop(name, None), self.panels.pop(name, None)

    def unload(self, name: str) -> None:
        got = self.loaded.pop(name, None)
        if got is not None:
            self._drop(name, got["kinds"])

    def sweep(self, now: float | None = None) -> list[str]:
        """Adapters unused for ``idle_min`` minutes, unloaded (worlds and sources stay until restart)."""
        now = now or time.time()
        idle = [n for n, g in self.loaded.items() if "adapter" in g["kinds"] and n not in self.pinned
                and now - g["used"] > self.idle_s]
        for n in idle:
            self.unload(n)
        return idle

    # -- using what is loaded ---------------------------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """The loaded askable worlds' facts for the question (``Worlds.ask``): {world, score, facts}."""
        return self.knows.ask(question)

    def call(self, name: str, args: dict) -> dict:
        tool = self.tools[name]
        for k in tool.schema.get("required", []):
            if k not in args:
                raise ValueError(f"{name}: missing {k!r}")
        types = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for k, spec in (tool.schema.get("properties") or {}).items():
            if k in args and spec.get("type") in types and not isinstance(args[k], types[spec["type"]]):
                raise ValueError(f"{name}: {k!r} must be a {spec['type']}")
        self.loaded[name]["used"] = time.time()
        return tool.run(args)

    def describe(self, who: str = "user") -> list[dict]:
        """Every plugin for the board; the public audience never sees an internal one."""
        out = []
        for name in self.index.names():
            p = self.index[name]
            if who != "audit" and p.audience == "internal":
                continue
            g = self.loaded.get(name)
            out.append({"name": name, "version": p.version, "kinds": p.kinds, "audience": p.audience,
                        "status": self.status(name), "description": p.description, "licence": p.licence,
                        "permissions": p.permissions, "last_used": g and time.strftime("%Y-%m-%dT%H:%M", time.localtime(g["used"]))})
        return out
