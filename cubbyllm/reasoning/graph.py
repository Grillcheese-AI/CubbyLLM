"""The thought graph: one question's run as typed nodes and typed edges (graph-of-thought, 2026-09-24).

Wired: WIRED (standin/ask.py attaches one to every ask record it keeps).

The events already form a tree: every step names its parent. A tree says how each step came
about and nothing else. This reads the same events as a graph of THOUGHTS and adds the relations
between them that the tree cannot hold:

    derived_from     child -> parent: how every step came about (the tree itself)
    supports         fact -> walk | branch: a certified answer rests on this fact
    verifies         vm -> walk: the VM certified the chain
    alternative_of   branch -- branch: two paths through the same ambiguous hop
    agrees_with      branch -- branch: both certified, the same answer
    contradicts      branch -- branch: both certified, different answers (the reason it refused)
    learned_into     gate -> fact: a fact a source handed over, admitted, then walked
    retries          walk -> walk: the same question walked again after the loop learned
    concludes        walk -> answer: the walk the final verdict came from

Nothing here decides anything. It is a record of what the loop did -- derived from the events,
never edited -- for the ask history, the panel and the sleep cycle to read. A fetch that handed
over 150 facts keeps the ones the walk used and collapses the rest into one counted node, so a
record stays a record and not a dump.
"""
from __future__ import annotations

from itertools import combinations

from ..core.protocols import Wiring
from . import events as ev

__wiring__ = Wiring.WIRED

SCHEMA = 1
GATE_KEEP = 12          # accepted gates kept per fetch beyond the ones a walk used


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def verdict(e: dict) -> str | None:
    k = e.get("kind")
    if k in ("walk", "answer"):
        if e.get("verified"):
            return "ok"
        return {"latent_only": "held", "profile": "profile", "self": "profile"}.get(e.get("reason"), "refused")
    if k == "vm":
        return "ok" if e.get("verified") else "refused"
    if k == "branch":
        return {"verified": "ok", "pruned": "gap"}.get(e.get("status"), "refused")
    if k == "gate":
        return "ok" if (e.get("status") == "accepted" or e.get("lifted")) else (None if e.get("status") == "duplicate" else "refused")
    return None


def label(e: dict) -> str:
    k = e.get("kind")
    if k == "question":
        return e.get("text") or ""
    if k == "plan":
        return " > ".join(e.get("relations") or []) + (f" . {e['seed']}" if e.get("seed") else "")
    if k == "walk":
        n = e.get("n_branches")
        tail = f" ({n} branches)" if n else ""
        return (f"verified -> {e.get('answer')}" if e.get("verified") else f"refused . {e.get('reason')}") + tail
    if k == "branch":
        path = " > ".join(str(v.get("object")) for v in (e.get("via") or []))
        st = e.get("status")
        end = (f"-> {e.get('answer')}" if st == "verified" else
               f"stalled at {e.get('stalled')}" if st == "pruned" else f"{st} -> {e.get('answer')}")
        return f"{path} {end}"
    if k == "hop":
        return f"hop {e.get('hop')} . {e.get('fact') or e.get('query')}"
    if k == "fact":
        return e.get("text") or ""
    if k == "fetch":
        return f"{e.get('source')} <- {e.get('entity')} ({e.get('n')})"
    if k == "gate":
        return f"{e.get('status')} . {e.get('fact')}"
    if k == "gates":
        return ", ".join(f"{v} {s}" for s, v in sorted(e.get("counts", {}).items()))
    if k == "vm":
        return "VM " + ("verified" if e.get("verified") else "not verified")
    if k == "answer":
        return f"answer: {e.get('answer')}" if e.get("verified") else f"refused: {e.get('reason')}"
    if k == "alias":
        return ", ".join(f"{a}->{b}" for a, b in (e.get("pairs") or []))
    if k == "latent":
        return f"held: {e.get('answer')}"
    return k or ""


# the fields a node keeps beyond id/kind/label/verdict: enough to read the thought, not the whole event
_KEEP = {"hop": ("hop", "how", "similarity", "provenance"), "branch": ("via", "status", "answer", "stalled", "similarity"),
         "walk": ("reason", "answer", "refused"), "answer": ("reason", "answer"), "fact": ("subj", "rel", "obj"),
         "fetch": ("source", "entity", "n"), "gate": ("status", "provenance", "clash"), "gates": ("counts",),
         "plan": ("seed", "relations", "n_hop", "how"), "vm": ("verified", "repairs")}


def _descendants(events: list[dict], roots: set[int]) -> list[dict]:
    keep = set(roots)
    out = []
    for e in sorted(events, key=lambda x: x["id"]):          # a parent is always emitted before its child
        if e["id"] in keep or e.get("parent") in keep:
            keep.add(e["id"])
            out.append(e)
    return out


def build(events: list[dict], question: str | None = None) -> dict:
    """The thought graph of the question roots in `events` (the ones asking `question`, when given
    and present; else every root). Pure: the same events give the same graph."""
    roots = [e for e in events if e.get("kind") == "question" and e.get("parent") is None]
    if question is not None:
        mine = [e for e in roots if _norm(e.get("text")) == _norm(question)]
        roots = mine or roots
    evs = _descendants(events, {e["id"] for e in roots})
    kids: dict[int, list[dict]] = {}
    for e in evs:
        kids.setdefault(e.get("parent"), []).append(e)

    def under(eid: int, kind: str) -> list[dict]:
        out, stack = [], list(kids.get(eid, []))
        while stack:
            c = stack.pop()
            if c["kind"] == kind:
                out.append(c)
            stack.extend(kids.get(c["id"], []))
        return sorted(out, key=lambda x: x["id"])

    walked = {_norm(e.get("text")) for e in evs if e["kind"] == "fact"}
    # collapse the gates a fetch handed over: keep the walked ones and a few accepted, count the rest
    drop: set[int] = set()
    extra: list[dict] = []
    for f in (e for e in evs if e["kind"] == "fetch"):
        gates = [g for g in kids.get(f["id"], []) if g["kind"] == "gate"]
        kept_ok = 0
        counts: dict[str, int] = {}
        for g in gates:
            if _norm(g.get("fact")) in walked:
                continue
            if g.get("status") == "accepted" and kept_ok < GATE_KEEP:
                kept_ok += 1
                continue
            drop.add(g["id"])
            counts[g.get("status") or "?"] = counts.get(g.get("status") or "?", 0) + 1
        if counts:
            extra.append({"id": f"g{f['id']}", "kind": "gates", "parent": f["id"], "counts": counts})

    nodes, edges = [], []
    for e in [e for e in evs if e["id"] not in drop] + extra:
        n = {"id": e["id"], "kind": e["kind"], "label": label(e), "verdict": verdict(e)}
        for k in _KEEP.get(e["kind"], ()):
            if e.get(k) is not None:
                n[k] = e[k]
        nodes.append(n)
        if e.get("parent") is not None:
            edges.append({"src": e["id"], "dst": e["parent"], "type": "derived_from"})

    def edge(a, b, t):
        edges.append({"src": a, "dst": b, "type": t})

    facts_by_text: dict[str, list[int]] = {}
    for e in evs:
        if e["kind"] == "fact":
            facts_by_text.setdefault(_norm(e.get("text")), []).append(e["id"])
    for r in roots:
        walks = under(r["id"], "walk")
        for a, b in zip(walks, walks[1:]):
            edge(b["id"], a["id"], "retries")
        answers = [a for a in kids.get(r["id"], []) if a["kind"] == "answer"]
        if walks and answers:
            edge(walks[-1]["id"], answers[-1]["id"], "concludes")
        for w in walks:
            prefix = [f for h in kids.get(w["id"], []) if h["kind"] == "hop" for f in kids.get(h["id"], []) if f["kind"] == "fact"]
            branches = [b for b in kids.get(w["id"], []) if b["kind"] == "branch"]
            for v in (c for c in kids.get(w["id"], []) if c["kind"] == "vm" and c.get("verified")):
                edge(v["id"], w["id"], "verifies")
            for b in branches:
                if b.get("status") == "verified":
                    for f in prefix + under(b["id"], "fact"):
                        edge(f["id"], b["id"], "supports")
            if w.get("verified"):
                for f in prefix + [f for b in branches for f in under(b["id"], "fact")]:
                    edge(f["id"], w["id"], "supports")
            for a, b in combinations(branches, 2):
                edge(a["id"], b["id"], "alternative_of")
                if a.get("status") == b.get("status") == "verified":
                    same = _norm(a.get("answer")) == _norm(b.get("answer"))
                    edge(a["id"], b["id"], "agrees_with" if same else "contradicts")
    for g in (e for e in evs if e["kind"] == "gate" and e["id"] not in drop and e.get("status") == "accepted"):
        for fid in facts_by_text.get(_norm(g.get("fact")), []):
            edge(g["id"], fid, "learned_into")

    final = next((a for r in reversed(roots) for a in reversed(kids.get(r["id"], [])) if a["kind"] == "answer"), None)
    types: dict[str, int] = {}
    for x in edges:
        types[x["type"]] = types.get(x["type"], 0) + 1
    return {"schema": SCHEMA, "roots": [r["id"] for r in roots], "nodes": nodes, "edges": edges,
            "summary": {"verdict": verdict(final) if final else None, "answer": (final or {}).get("answer"),
                        "reason": (final or {}).get("reason"), "nodes": len(nodes), "edges": types,
                        "branches": sum(1 for n in nodes if n["kind"] == "branch")}}


class Collector:
    """Every event emitted while it is open, for `graph()`. A sink like any other: the loop's
    behaviour is the same with or without it."""

    def __init__(self) -> None:
        self.events = ev.MemorySink()

    def __enter__(self) -> "Collector":
        ev.add_sink(self.events)
        return self

    def __exit__(self, *exc) -> None:
        ev.remove_sink(self.events)

    def graph(self, question: str | None = None) -> dict:
        return build(list(self.events), question)
