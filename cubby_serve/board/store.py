"""cubby_serve/board/store.py -- the board's Knowledge section: the hot Wikidata store (order 0055), searched,
read, drawn as a graph and queried, for the audit audience only.

    GET /board/store                       the store's state: its snapshot hash, counts, sizes, the dump it was built from
    GET /board/store/search?q=<name>       every entity the name may mean (exact, accent-folded, then a bounded typo),
                                           in id order, with its label and description (never ordered by fame)
    GET /board/store/entity/<id>?as_of=    one entity, its statements by property, labelled; with as_of (YYYY[-MM[-DD]])
                                           each says whether it holds then. A redirected id is followed and said.
    GET /board/store/graph/<id>?limit=     its neighbours: the entities its statements point at and those pointing at it
    GET /board/store/who?p=P31&v=Q5        the entities with that statement (main values; deprecated ones marked)
    GET /board/store/path?from=&to=        the shortest chain of statements between two entities, at most 3 steps
    GET /board/store/resolve?id=           the id a redirect leads to

The store's answers name the dataset and carry every source: the public audience (0016) gets only a notice. The store
is asked through `store_client` (a pipe to `wdstore serve`): this module opens no socket.

Wired: WIRED (`cubby_serve/board/api.py`, `/board/store`).
"""
from __future__ import annotations

import json
import os
import re
from collections import deque
from urllib.parse import parse_qs

from .store_client import from_env

__wiring__ = "WIRED"

ID = re.compile(r"^[QPL][1-9][0-9]*$")
LIMIT, PATH_BUDGET, SEARCH_DESCRIBED = 60, 1500, 25


def _labels(client, ids) -> dict[str, str]:
    """Each id's label: English, else `mul`, else French; ids with none are left out."""
    out, want = {}, sorted({i for i in ids if i})
    for lang in ("en", "mul", "fr"):
        if not want:
            break
        got = client.ask({"op": "labels", "ids": want, "lang": lang})
        if "error" in got:
            break
        out.update(got)
        want = [i for i in want if i not in got]
    return out


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
    labels = _labels(client, [h["id"] for h in hits])
    for h in hits[:SEARCH_DESCRIBED]:
        e = client.ask({"op": "get", "id": h["id"]})
        h["description"] = next((n[2] for n in e.get("names", []) if n[1] == 2 and n[0] == "en"), None)
        h["statements"] = len(e.get("statements", []))
    for h in hits:
        h["label"] = labels.get(h["id"])
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
    labels = _labels(client, ids)
    groups: dict[int, dict] = {}
    for s in e["statements"]:
        g = groups.setdefault(s["pid"], {"pid": f"P{s['pid']}", "label": labels.get(f"P{s['pid']}"), "statements": []})
        g["statements"].append({
            "value": _value_text(s["value"], labels), "raw": s["value"], "rank": ["deprecated", "normal", "preferred"][s["rank"]],
            "best": s["best"], "holds": s.get("holds"), "references": len(s["references"]), "guid": s["guid"],
            "qualifiers": [{"pid": f"P{q[0]}", "label": labels.get(f"P{q[0]}"), "value": _value_text(q[1], labels), "time": i < len(s["hot_qualifiers"])}
                           for i, q in enumerate(s["hot_qualifiers"] + s["cold_qualifiers"])]})
    names = [{"lang": n[0], "kind": ["label", "alias", "description"][n[1]], "text": n[2]} for n in e["names"]]
    return {"id": e["id"], "found": True, "cold": e["cold"], "redirected_from": e.get("redirected_from"), "as_of": e.get("as_of"),
            "revision": e["revision"], "modified": e["modified"], "datatype": e.get("datatype"), "names": names,
            "label": labels.get(e["id"]) or next((n["text"] for n in names if n["kind"] == "label"), None),
            "sitelinks": e["sitelinks"], "groups": sorted(groups.values(), key=lambda g: int(g["pid"][1:]))}


def _out_edges(client, eid: str) -> list[tuple[str, int]]:
    """(target id, pid) for each statement of `eid` whose main value is an entity, deprecated ones left out."""
    e = client.ask({"op": "get", "id": eid})
    return [(s["value"]["entity"], s["pid"]) for s in e.get("statements", [])
            if isinstance(s["value"], dict) and "entity" in s["value"] and s["rank"] != 0]


def _in_edges(client, eid: str, limit: int = 1000) -> list[tuple[str, int]]:
    """(source id, pid) for the first `limit` statements pointing at `eid` (an item like "human" has millions)."""
    got = client.ask({"op": "pointing_at", "value": eid, "limit": limit})
    return [(h[1], h[0]) for h in got.get("hits", []) if h[2] != 0]


def _in_total(client, eid: str) -> int:
    return int(client.ask({"op": "pointing_at", "value": eid, "limit": 0}).get("total", 0))


def _graph(client, eid: str, limit: int) -> dict:
    to = client.ask({"op": "resolve", "id": eid}).get("id", eid)
    out, inn = _out_edges(client, to), _in_edges(client, to, limit)
    edges = [{"from": to, "to": t, "pid": f"P{p}"} for t, p in out[:limit]]
    edges += [{"from": s, "to": to, "pid": f"P{p}"} for s, p in inn[:limit]]
    nodes = sorted({to} | {x["from"] for x in edges} | {x["to"] for x in edges})
    labels = _labels(client, nodes + sorted({x["pid"] for x in edges}))
    for x in edges:
        x["label"] = labels.get(x["pid"])
    return {"center": to, "nodes": [{"id": n, "label": labels.get(n)} for n in nodes], "edges": edges,
            "more": {"out": max(0, len(out) - limit), "in": max(0, _in_total(client, to) - len(edges) + len(out[:limit]))}}


def _who(client, pid: str, value: str, limit: int) -> dict:
    got = client.ask({"op": "who", "pid": int(pid[1:]), "value": value, "limit": limit})
    hits = got.get("hits", [])
    labels = _labels(client, [h[0] for h in hits[:limit]] + [pid, value])
    return {"p": pid, "v": value, "p_label": labels.get(pid), "v_label": labels.get(value), "total": got.get("total", len(hits)),
            "hits": [{"id": h[0], "label": labels.get(h[0]), "rank": ["deprecated", "normal", "preferred"][h[1]], "best": h[2]}
                     for h in hits[:limit]], "error": got.get("error")}


def _path(client, a: str, b: str, depth: int = 3, budget: int = PATH_BUDGET) -> dict:
    """Breadth-first from both ends (statements out of `a`'s side, statements into `b`'s side), `budget` reads at most."""
    if a == b:
        return {"from": a, "to": b, "steps": []}
    prev, nxt = {a: None}, {b: None}           # id -> (neighbour, pid) towards a / towards b
    qa, qb, reads = deque([(a, 0)]), deque([(b, 0)]), 0
    meet = None
    while (qa or qb) and meet is None and reads < budget:
        # at most (depth + 1) // 2 steps out of `a`, depth // 2 into `b`: together, depth
        for q, side, other, grow, cap in ((qa, prev, nxt, _out_edges, (depth + 1) // 2), (qb, nxt, prev, _in_edges, depth // 2)):
            if not q or meet is not None:
                continue
            node, d = q.popleft()
            if d >= cap:
                continue
            reads += 1
            for t, p in grow(client, node):          # a side's neighbours, the first 1000 (pointing_at's limit)
                if t not in side:
                    side[t] = (node, p)
                    if t in other:
                        meet = t
                        break
                    q.append((t, d + 1))
    if meet is None:
        return {"from": a, "to": b, "steps": None, "reads": reads, "message": f"no chain of at most {depth} statements found"}
    left, x = [], meet
    while prev[x] is not None:
        y, p = prev[x]; left.append((y, p, x)); x = y
    right, x = [], meet
    while nxt[x] is not None:
        y, p = nxt[x]; right.append((x, p, y)); x = y
    steps = list(reversed(left)) + right
    labels = _labels(client, [s[0] for s in steps] + [s[2] for s in steps] + [f"P{s[1]}" for s in steps])
    return {"from": a, "to": b, "reads": reads, "steps": [{"from": f, "from_label": labels.get(f), "pid": f"P{p}",
            "label": labels.get(f"P{p}"), "to": t, "to_label": labels.get(t)} for f, p, t in steps]}


def board_store(route: str, query: str, who: str, client=None) -> dict:
    if who != "audit":
        return {"section": "store", "audit_only": True,
                "message": "The knowledge store names its dataset and sources: it is shown in the audit view only."}
    client = client or from_env()
    if client is None:
        return {"section": "store", "configured": False,
                "message": "No store configured: set CUBBY_WD_STORE (the store's folder) and CUBBY_WDSTORE_EXE (wdstore)."}
    sub = route[len("/board/store"):].strip("/")
    q = {k: v[0] for k, v in parse_qs(query).items()}
    lim = max(1, min(int(q["limit"]) if q.get("limit", "").isdigit() else LIMIT, 500))
    arg = sub.split("/", 1)[1] if "/" in sub else ""
    if sub == "":
        return _status(client)
    if sub == "search":
        return _search(client, q.get("q", "").strip()) if q.get("q", "").strip() else {"error": "q: a name"}
    if ID.match(arg) and sub.startswith("entity/"):
        return _entity(client, arg, q.get("as_of") or None)
    if ID.match(arg) and sub.startswith("graph/"):
        return _graph(client, arg, lim)
    if sub == "who" and ID.match(q.get("p", "")) and q["p"][0] == "P" and ID.match(q.get("v", "")):
        return _who(client, q["p"], q["v"], lim)
    if sub == "path" and ID.match(q.get("from", "")) and ID.match(q.get("to", "")):
        return _path(client, q["from"], q["to"])
    if sub == "resolve" and ID.match(q.get("id", "")):
        return client.ask({"op": "resolve", "id": q["id"]})
    return {"error": "unknown store path or argument"}
