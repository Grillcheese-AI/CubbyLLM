"""standin/plugins_serve.py -- plugins in the chat server and on the board (order 0028).

- ``mount(brain, dirs, idle_min)``: the index of ``dirs`` (front matter only) and a ``Loader`` that routes with the
  brain's encoder (its fact store's) and adds adapters to its 450M trunk when one is mounted; ``serve_api.py
  --plugins DIR[,DIR]`` calls it (off when not given: the server answers as before).
- ``on_question(brain, text, rec)``: wakes what the question needs. A reality plugin's facts (a source, a fact world)
  are one of the ask loop's sources: each goes to the loop's store through ``learn.gate`` -- parsed, deduplicated,
  judged by the store's oracle -- with the functional rule (a fact that contradicts one the store holds is refused, never
  stored, and written to the ledger) and the provenance ``plugin:<name>``; the loop then answers from its store, under
  copy-or-refuse, and the audit chain names the plugin. A fiction plugin's facts never enter the reality store (0015's
  isolation): they answer inside their own world, recorded for the audit view. The record keeps the plugins woken and
  the gate's verdicts (``plugins``, ``plugin_facts``: internal fields ``audience.for_user`` drops). Plugins never write
  the reply.
- ``board_get(brain, route, query, who)`` / ``board_post(brain, route, who)``: the board's Plugins view -- every plugin
  with its kinds, status, permissions and audience (the public audience never sees an internal one), the panels of
  loaded plugins, "which plugin would wake?" for a question, and Approve / Pin / Disable (audit audience only).

Wired: WIRED (``serve_api.py --plugins``, ``board_api``).
"""
from __future__ import annotations

import urllib.parse

__wiring__ = "WIRED"


def _trunk(brain):
    from emitter import Cubby450mTrunk
    em = getattr(brain, "emitter", None)
    for obj in (em, getattr(em, "programs", None), getattr(em, "talk", None), getattr(getattr(brain, "chat", None), "talker", None)):
        t = getattr(obj, "trunk", None)
        if isinstance(t, Cubby450mTrunk):
            return t
    return None


def mount(brain, dirs: str, idle_min: float = 30.0, log=print):
    from plugins_index import discover
    from plugins_load import Loader
    idx = discover(dirs)
    enc = next((getattr(w, "enc", None) for w in (getattr(brain, "worlds", {}) or {}).values() if getattr(w, "enc", None)), None)
    loader = Loader(idx, brain=brain, enc=enc, idle_min=idle_min, log=log, trunk=_trunk(brain))
    brain.plugins = loader
    for name, why in sorted(idx.errors.items()):
        log(f"  plugin folder {name} not read: {why}")
    return loader


def admit(brain, world_name: str, facts: list[str]) -> list[dict]:
    """A plugin's facts into the ask loop's store through the learn gate, functional: [{fact, status, clash}]."""
    from cubbyllm.reasoning.learn import gate
    from cubbyllm.reasoning.planner import parse_fact
    loop = getattr(brain, "ask_loop", None)
    store = getattr(loop, "world", None)
    if store is None or not hasattr(store, "add"):
        return [{"fact": f, "status": "no store"} for f in facts]
    prov, fkey = getattr(store, "provenance", None), getattr(store, "_key", None) or (lambda x: " ".join(x.split()))
    out = []
    for f in facts:
        t = parse_fact(f)
        p = gate(f, store, world_name, t.subj if t else "", functional=True)
        if p.status == "accepted":
            store.add(f)
            if prov is not None:
                prov[fkey(f)] = world_name
        elif p.status == "contradiction":
            _ledger(brain).record(program="plugin-gate", name=world_name, kind="plugin_fact", fn=None, input=f,
                                  expected=p.clash, got=None, ok=False, verdict="contradiction: refused, not stored")
        out.append({"fact": f, "status": p.status, "clash": p.clash})
    return out


def _ledger(brain):
    if getattr(brain, "ledger", None) is None:
        from ledger import Ledger
        brain.ledger = Ledger()
    return brain.ledger


def on_question(brain, text: str, rec: dict | None = None) -> list[str]:
    """Before the loop answers (``rec`` None): wake the plugins and admit their reality facts. After (``rec`` given): note
    in the record what woke and what the gate said."""
    loader = getattr(brain, "plugins", None)
    if loader is None:
        return []
    woke = loader.wake(text)
    if not woke:
        return woke
    if rec is None:
        got = loader.ask(text) if loader.knows.by_name else {}
        fiction = isinstance(loader.knows.by_name.get(got.get("world")), _fiction())
        loader.last_admitted = {"world": got.get("world"), "fiction": fiction, "facts": got.get("facts") or [],
                                "gate": [] if fiction or not got.get("facts") else admit(brain, got["world"], got["facts"])}
        return woke
    rec["plugins"] = [f"plugin:{n}" for n in woke]
    last = getattr(loader, "last_admitted", None)
    if last and last.get("facts"):
        rec["plugin_facts"] = last
    return woke


def _fiction():
    from plugins_load import FictionWorld
    return FictionWorld


def internal_names(brain) -> set[str]:
    loader = getattr(brain, "plugins", None)
    if loader is None:
        return set()
    return {n for n in loader.index.names() if loader.index[n].audience == "internal"}


def board_get(brain, route: str, query: str, who: str) -> dict | None:
    loader = getattr(brain, "plugins", None)
    if route == "/board/plugins":
        if loader is None:
            return {"built": False, "message": "no plugins mounted (start the server with --plugins)", "plugins": []}
        visible = {p["name"] for p in loader.describe(who)}
        panels = {}
        for name, panel in loader.panels.items():
            if name not in visible:
                continue
            src = loader.sources.get(name) or (loader.loaded.get(name) or {}).get("kinds", {}).get("source")
            call = panel.get("call", "").split(".", 1)[-1]
            try:
                panels[name] = {**panel, "data": getattr(src, call)() if src is not None and call else None}
            except Exception as e:                       # noqa: BLE001 -- a panel that fails says so
                panels[name] = {**panel, "data": None, "error": f"{type(e).__name__}"}
        return {"built": True, "tau": loader.router.tau, "plugins": loader.describe(who), "panels": panels}
    if route == "/board/plugins/review":                 # what Approve shows first: the audit audience only
        name = (urllib.parse.parse_qs(query).get("name") or [""])[0]
        if who != "audit" or loader is None or name not in loader.index:
            return {"name": name, "review": None}
        import plugins_guard as guard
        return {"name": name, "review": guard.review(loader.index[name])}
    if route == "/board/plugins/doc":
        name = (urllib.parse.parse_qs(query).get("name") or [""])[0]
        if loader is None or name not in loader.index or (who != "audit" and loader.index[name].audience == "internal"):
            return {"name": name, "doc": None}
        return {"name": name, "doc": (loader.index[name].folder / "PLUGIN.md").read_text(encoding="utf-8")[:8000]}
    if route == "/board/plugins/route":
        if loader is None:
            return {"scores": {}, "tau": None}
        q = (urllib.parse.parse_qs(query).get("q") or [""])[0]
        hidden = internal_names(brain) if who != "audit" else set()
        scores = {n: round(s, 3) for n, s in loader.router.scores(q).items() if n not in hidden}
        return {"question": q, "tau": loader.router.tau, "scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
                "would_wake": [n for n, s in scores.items() if s >= loader.router.tau and n not in loader.disabled]}
    return None


def board_post(brain, route: str, who: str, body: dict | None = None) -> tuple[int, dict]:
    """``/board/plugins/<name>/<approve|pin|unpin|disable|enable>``: the audit audience only. Approve needs the digest of
    the review the page showed (``/board/plugins/review``): what is approved is what was read."""
    if who != "audit":
        return 403, {"error": "audit only"}
    loader = getattr(brain, "plugins", None)
    parts = route.strip("/").split("/")
    if loader is None or len(parts) != 4 or parts[3] not in ("approve", "pin", "unpin", "disable", "enable"):
        return 404, {"error": "unknown plugin action"}
    name, act = parts[2], parts[3]
    if name not in loader.index:
        return 404, {"error": f"no plugin {name}"}
    if act == "approve":
        import plugins_guard as guard
        bad = guard.check(loader.index[name])
        if bad:
            return 409, {"error": "its code fails review", "why": bad}
        digest = (body or {}).get("digest")
        if not digest:
            return 409, {"error": "show the review first (GET /board/plugins/review?name=...)"}
        try:
            guard.approve(loader.index[name], expect=digest)
        except guard.Refused as e:
            return 409, {"error": str(e)}
        loader.refused.pop(name, None)
    elif act in ("pin", "unpin"):
        loader.pin(name, act == "pin")
    else:
        loader.disable(name, act == "disable")
    return 200, {"name": name, "status": loader.status(name)}
