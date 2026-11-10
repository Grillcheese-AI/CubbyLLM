"""cubby_serve/board/store_graph.py -- the Knowledge section's graph, who-has and path answers (order 0055's store),
for `store.py`'s routes. Every entity named in an answer comes with its brief (label, classes, hot or cold).

Wired: WIRED (`cubby_serve/board/store.py`).
"""
from __future__ import annotations

from collections import deque

__wiring__ = "WIRED"

RANKS = ["deprecated", "normal", "preferred"]
PATH_BUDGET, CONTEXT_PER_NODE = 1500, 3


def labels(client, ids) -> dict[str, str]:
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


def briefs(client, ids) -> dict[str, dict]:
    """Per id: label, description, classes (each with its label), hot or cold, statement count."""
    ids = sorted({i for i in ids if i})
    got = client.ask({"op": "brief", "ids": ids, "lang": "en"}) if ids else {}
    if "error" in got:
        return {}
    names = labels(client, {t for b in got.values() for t in b.get("types", [])})
    for b in got.values():
        b["types"] = [{"id": t, "label": names.get(t)} for t in b.get("types", [])]
    return got


def out_edges(client, eid: str) -> list[tuple[str, int]]:
    """(target id, pid) for each statement of `eid` whose main value is an entity, deprecated ones left out."""
    e = client.ask({"op": "get", "id": eid})
    return [(s["value"]["entity"], s["pid"]) for s in e.get("statements", [])
            if isinstance(s["value"], dict) and "entity" in s["value"] and s["rank"] != 0]


def in_edges(client, eid: str, limit: int = 1000) -> tuple[int, list[tuple[str, int]]]:
    """(how many in all, (source id, pid) for the first `limit`): an item like "human" has millions."""
    got = client.ask({"op": "pointing_at", "value": eid, "limit": limit})
    return int(got.get("total", 0)), [(h[1], h[0]) for h in got.get("hits", []) if h[2] != 0]


def _nodes(client, ids, path=()) -> list[dict]:
    b = briefs(client, ids)
    return [{"id": i, "label": (b.get(i) or {}).get("label"), "types": (b.get(i) or {}).get("types", []),
             "cold": (b.get(i) or {}).get("cold"), "on_path": i in path} for i in sorted(set(ids))]


def graph(client, eid: str, limit: int) -> dict:
    to = client.ask({"op": "resolve", "id": eid}).get("id", eid)
    out = out_edges(client, to)
    total_in, inn = in_edges(client, to, limit)
    edges = [{"from": to, "to": t, "pid": f"P{p}"} for t, p in out[:limit]] + [{"from": s, "to": to, "pid": f"P{p}"} for s, p in inn]
    names = labels(client, {x["pid"] for x in edges})
    for x in edges:
        x["label"] = names.get(x["pid"])
    ids = {to} | {x["from"] for x in edges} | {x["to"] for x in edges}
    return {"center": to, "nodes": _nodes(client, ids), "edges": edges,
            "more": {"out": max(0, len(out) - limit), "in": max(0, total_in - len(inn))}}


def who(client, pid: str, value: str, limit: int) -> dict:
    got = client.ask({"op": "who", "pid": int(pid[1:]), "value": value, "limit": limit})
    hits = got.get("hits", [])
    b = briefs(client, [h[0] for h in hits])
    names = labels(client, [pid, value])
    return {"p": pid, "v": value, "p_label": names.get(pid), "v_label": names.get(value), "total": got.get("total", len(hits)),
            "hits": [{"id": h[0], "label": (b.get(h[0]) or {}).get("label"), "types": (b.get(h[0]) or {}).get("types", []),
                      "rank": RANKS[h[1]], "best": h[2]} for h in hits], "error": got.get("error")}


def _step_statement(client, f: str, pid: int, t: str, names: dict) -> dict:
    """The statement a step stands for: its rank and its qualifiers, labelled."""
    e = client.ask({"op": "get", "id": f})
    s = next((s for s in e.get("statements", []) if s["pid"] == pid and s["value"] == {"entity": t}), None)
    if s is None:
        return {"rank": None, "qualifiers": []}
    quals = s["hot_qualifiers"] + s["cold_qualifiers"]
    more = labels(client, {f"P{q[0]}" for q in quals} | {q[1]["entity"] for q in quals if isinstance(q[1], dict) and "entity" in q[1]})
    names.update(more)

    def text(v):
        if isinstance(v, str):
            return v
        k, x = next(iter(v.items()))
        return f"{names.get(x, x)} ({x})" if k == "entity" else (x[0] if isinstance(x, list) else str(x))
    return {"rank": RANKS[s["rank"]], "references": len(s["references"]),
            "qualifiers": [{"pid": f"P{q[0]}", "label": names.get(f"P{q[0]}"), "value": text(q[1])} for q in quals]}


def path(client, a: str, b: str, depth: int = 3, budget: int = PATH_BUDGET, context: bool = False) -> dict:
    """The shortest chain of at most `depth` statements: breadth-first from both ends (statements out of `a`'s side,
    statements into `b`'s side), `budget` entity reads at most. With `context`, a few more neighbours of each node."""
    if a == b:
        return {"from": a, "to": b, "steps": [], "nodes": _nodes(client, [a], [a]), "edges": []}
    prev, nxt = {a: None}, {b: None}
    qa, qb, reads, meet = deque([(a, 0)]), deque([(b, 0)]), 0, None
    while (qa or qb) and meet is None and reads < budget:
        # at most (depth + 1) // 2 steps out of `a`, depth // 2 into `b`: together, depth
        for q, side, other, grow, cap in ((qa, prev, nxt, out_edges, (depth + 1) // 2),
                                          (qb, nxt, prev, lambda c, n: in_edges(c, n)[1], depth // 2)):
            if not q or meet is not None:
                continue
            node, d = q.popleft()
            if d >= cap:
                continue
            reads += 1
            for t, p in grow(client, node):
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
    on = [steps[0][0]] + [s[2] for s in steps]
    names = labels(client, {f"P{s[1]}" for s in steps})
    edges = [{"from": f, "to": t, "pid": f"P{p}", "label": names.get(f"P{p}"), "on_path": True} for f, p, t in steps]
    if context:
        for n in on:
            extra = [(t, p) for t, p in out_edges(client, n) if t not in on][:CONTEXT_PER_NODE]
            more = labels(client, {f"P{p}" for _t, p in extra})
            edges += [{"from": n, "to": t, "pid": f"P{p}", "label": more.get(f"P{p}"), "on_path": False} for t, p in extra]
    nodes = _nodes(client, {x["from"] for x in edges} | {x["to"] for x in edges}, on)
    by = {n["id"]: n for n in nodes}
    return {"from": a, "to": b, "reads": reads, "nodes": nodes, "edges": edges,
            "steps": [{"from": f, "from_label": by[f]["label"], "from_types": by[f]["types"], "pid": f"P{p}", "label": names.get(f"P{p}"),
                       "to": t, "to_label": by[t]["label"], "to_types": by[t]["types"], **_step_statement(client, f, p, t, names)}
                      for f, p, t in steps]}
