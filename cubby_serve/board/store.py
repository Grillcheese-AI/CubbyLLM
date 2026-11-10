"""cubby_serve/board/store.py -- the board's Knowledge section: the hot Wikidata store (order 0055), searched,
read, drawn as a graph and queried, for the audit audience only.

    GET /board/store                       the store's state: its snapshot hash, counts, sizes, the dump it was built from
    GET /board/store/search?q=<name>       every entity the name may mean (exact, accent-folded, then a bounded typo),
                                           in id order, with its label, classes and description (never ordered by fame)
    GET /board/store/brief?ids=Q42,Q35     each id's label, classes, description, hot or cold
    GET /board/store/entity/<id>?as_of=    one entity, its statements by property, labelled; with as_of (YYYY[-MM[-DD]])
                                           each says whether it holds then. A redirected id is followed and said.
    GET /board/store/graph/<id>?limit=     its neighbours: the entities its statements point at and those pointing at it
    GET /board/store/who?p=P31&v=Q5        the entities with that statement (main values; deprecated ones marked)
    GET /board/store/path?from=&to=&context=1   the shortest chain of statements between two entities, at most 3
                                           steps, each with its statement's rank and qualifiers; context: a few more
                                           neighbours of each node
    GET /board/store/resolve?id=           the id a redirect leads to
Every answer says how long it took (`ms`).

The store's answers name the dataset and carry every source: the public audience (0016) gets only a notice. The store
is asked through `store_client` (a pipe to `wdstore serve`): this module opens no socket.

Wired: WIRED (`cubby_serve/board/api.py`, `/board/store`).
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import parse_qs

from . import store_graph as sg
from .store_client import from_env

__wiring__ = "WIRED"

ID = re.compile(r"^[QPL][1-9][0-9]*$")
LIMIT = 60


def _value_text(v, labels) -> str:
    if isinstance(v, str):
        return {"somevalue": "(some value: unknown)", "novalue": "(no value)"}.get(v, v)
    kind, x = next(iter(v.items()))
    if kind == "entity":
        return f"{labels.get(x, x)} ({x})"
    if kind == "mono":
        return f"{x[0]} @{x[1]}"
    if kind == "typed":
        return x[0]
    return str(x)


def _status(client) -> dict:
    snap = client.ask({"op": "snapshot"})
    man = {}
    try:
        with open(os.path.join(client.store, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        pass
    release = next((i.get("release") for i in man.get("inputs", []) if isinstance(i, dict) and i.get("release")), None) or {}
    return {"section": "store", "configured": True, "snapshot": snap.get("snapshot"), "error": snap.get("error"),
            "dump_date": man.get("dump_date"), "counts": man.get("counts", {}), "bytes": man.get("bytes", {}),
            "bytes_per_statement": man.get("bytes_per_statement", {}), "files": len(man.get("files", [])),
            "release": {k: release.get(k) for k in ("name", "date", "size", "md5", "zenodo", "licence")} if release else None,
            "params": man.get("params", {})}


def _search(client, text: str) -> dict:
    got = client.ask({"op": "name", "text": text})
    if "too_many" in got:
        return {"query": text, "too_many": got["too_many"], "hits": []}
    seen, hits = set(), []
    for h in got.get("hits", []):
        if h["id"] not in seen:
            seen.add(h["id"])
            hits.append({"id": h["id"], "how": h["how"], "lang": h["lang"], "matched": h["key"]})
    b = sg.briefs(client, [h["id"] for h in hits])
    for h in hits:
        x = b.get(h["id"]) or {}
        h.update({"label": x.get("label"), "description": x.get("description"), "types": x.get("types", []),
                  "statements": x.get("statements"), "cold": x.get("cold")})
    return {"query": text, "hits": hits, "error": got.get("error")}


def _entity(client, eid: str, as_of: str | None) -> dict:
    req = {"op": "get", "id": eid}
    if as_of:
        req["as_of"] = as_of
    e = client.ask(req)
    if "error" in e or not e.get("found"):
        return {"id": eid, "found": False, "error": e.get("error")}
    ids = set()
    for s in e["statements"]:
        ids.add(f"P{s['pid']}")
        for v in [s["value"]] + [q[1] for q in s["hot_qualifiers"] + s["cold_qualifiers"]]:
            if isinstance(v, dict) and "entity" in v:
                ids.add(v["entity"])
        ids.update(f"P{q[0]}" for q in s["hot_qualifiers"] + s["cold_qualifiers"])
    labels = sg.labels(client, ids)
    groups: dict[int, dict] = {}
    for s in e["statements"]:
        g = groups.setdefault(s["pid"], {"pid": f"P{s['pid']}", "label": labels.get(f"P{s['pid']}"), "statements": []})
        g["statements"].append({
            "value": _value_text(s["value"], labels), "raw": s["value"], "rank": sg.RANKS[s["rank"]],
            "best": s["best"], "holds": s.get("holds"), "references": len(s["references"]), "guid": s["guid"],
            "qualifiers": [{"pid": f"P{q[0]}", "label": labels.get(f"P{q[0]}"), "value": _value_text(q[1], labels), "time": i < len(s["hot_qualifiers"])}
                           for i, q in enumerate(s["hot_qualifiers"] + s["cold_qualifiers"])]})
    names = [{"lang": n[0], "kind": ["label", "alias", "description"][n[1]], "text": n[2]} for n in e["names"]]
    types = [{"id": s["value"]["entity"], "label": labels.get(s["value"]["entity"])} for s in e["statements"]
             if s["pid"] == 31 and s["best"] and isinstance(s["value"], dict) and "entity" in s["value"]]
    out_first = sorted(groups.values(), key=lambda g: (not any(isinstance(s["raw"], dict) and "entity" in s["raw"] for s in g["statements"]), int(g["pid"][1:])))
    return {"id": e["id"], "found": True, "cold": e["cold"], "redirected_from": e.get("redirected_from"), "as_of": e.get("as_of"),
            "revision": e["revision"], "modified": e["modified"], "datatype": e.get("datatype"), "names": names, "types": types,
            "label": labels.get(e["id"]) or next((n["text"] for n in names if n["kind"] == "label"), None),
            "statements_total": len(e["statements"]), "residency": "mmap zero-copy",
            "sitelinks": e["sitelinks"], "groups": sorted(groups.values(), key=lambda g: int(g["pid"][1:])),
            "key": [g["pid"] for g in out_first]}


def _route(client, sub: str, q: dict, lim: int) -> dict:
    arg = sub.split("/", 1)[1] if "/" in sub else ""
    if sub == "":
        return _status(client)
    if sub == "search":
        return _search(client, q.get("q", "").strip()) if q.get("q", "").strip() else {"error": "q: a name"}
    if sub == "brief":
        ids = [i for i in q.get("ids", "").split(",") if ID.match(i)][:100]
        return {"briefs": sg.briefs(client, ids)} if ids else {"error": "ids: Q42,Q35"}
    if ID.match(arg) and sub.startswith("entity/"):
        return _entity(client, arg, q.get("as_of") or None)
    if ID.match(arg) and sub.startswith("graph/"):
        return sg.graph(client, arg, lim)
    if sub == "who" and ID.match(q.get("p", "")) and q["p"][0] == "P" and ID.match(q.get("v", "")):
        return sg.who(client, q["p"], q["v"], lim)
    if sub == "path" and ID.match(q.get("from", "")) and ID.match(q.get("to", "")):
        return sg.path(client, q["from"], q["to"], context=q.get("context") == "1")
    if sub == "resolve" and ID.match(q.get("id", "")):
        return client.ask({"op": "resolve", "id": q["id"]})
    return {"error": "unknown store path or argument"}


def board_store(route: str, query: str, who: str, client=None) -> dict:
    if who != "audit":
        return {"section": "store", "audit_only": True,
                "message": "The knowledge store names its dataset and sources: it is shown in the audit view only."}
    client = client or from_env()
    if client is None:
        return {"section": "store", "configured": False,
                "message": "No store configured: set CUBBY_WD_STORE (the store's folder) and CUBBY_WDSTORE_EXE (wdstore)."}
    q = {k: v[0] for k, v in parse_qs(query).items()}
    lim = max(1, min(int(q["limit"]) if q.get("limit", "").isdigit() else LIMIT, 500))
    t0 = time.perf_counter()
    out = _route(client, route[len("/board/store"):].strip("/"), q, lim)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out
