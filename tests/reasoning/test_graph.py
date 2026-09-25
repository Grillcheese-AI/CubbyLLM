"""The thought graph (2026-09-24): events in, typed nodes and edges out; derived, never edited."""
import json

from cubbyllm.reasoning import events as ev
from cubbyllm.reasoning.graph import GATE_KEEP, SCHEMA, Collector, build
from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.pipeline import answer

from .test_branching import BE, FR, Q2, echo_vm, no_search


def run(q, facts):
    with Collector() as col:
        qid = ev.emit("question", text=q)
        pid = ev.emit("plan", qid, relations=["country of citizenship", "continent"], seed="marie")
        res = answer(q, no_search, echo_vm(), tau_vm=0.5, tau_ret=0.2, lookup=TripleIndex(facts).hop)
        ev.emit_walk(pid, res)
        ev.emit_answer(qid, res)
    return res, col.graph(q)


def types(g):
    out = {}
    for e in g["edges"]:
        out.setdefault(e["type"], []).append((e["src"], e["dst"]))
    return out


def test_disagreeing_branches_are_a_contradiction_in_the_graph():
    res, g = run(Q2, [FR, BE, "europe is the continent of france", "atlantis is the continent of belgium"])
    assert res.reason == "ambiguous_hop" and g["schema"] == SCHEMA
    t = types(g)
    br = [n for n in g["nodes"] if n["kind"] == "branch"]
    assert len(br) == 2 and {n["verdict"] for n in br} == {"ok"}
    assert len(t["contradicts"]) == 1 and len(t["alternative_of"]) == 1 and "agrees_with" not in t
    # each certified branch rests on its own facts; the refused walk rests on none
    walk = next(n for n in g["nodes"] if n["kind"] == "walk")
    assert all(d != walk["id"] for _s, d in t["supports"])
    assert {d for _s, d in t["supports"]} == {n["id"] for n in br}
    assert g["summary"]["verdict"] == "refused" and g["summary"]["branches"] == 2
    assert t["concludes"] and json.dumps(g)


def test_agreeing_branches_support_the_answer_they_share():
    res, g = run(Q2, [FR, BE, "europe is the continent of france", "europe is the continent of belgium"])
    assert res.verified
    t = types(g)
    walk = next(n for n in g["nodes"] if n["kind"] == "walk")
    assert len(t["agrees_with"]) == 1 and "contradicts" not in t
    # the answer rests on ALL four facts, not only the branch it shows
    assert len([1 for _s, d in t["supports"] if d == walk["id"]]) == 4
    assert g["summary"] == {**g["summary"], "verdict": "ok", "answer": "europe"}


def test_a_fetch_keeps_the_facts_walked_and_counts_the_rest():
    evs = [{"id": 1, "kind": "question", "parent": None, "text": "q"},
           {"id": 2, "kind": "fetch", "parent": 1, "source": "s", "entity": "x", "n": 40}]
    evs += [{"id": 10 + i, "kind": "gate", "parent": 2, "fact": f"v{i} is the r of x", "status": "accepted"} for i in range(30)]
    evs += [{"id": 50 + i, "kind": "gate", "parent": 2, "fact": f"w{i}", "status": "unparseable"} for i in range(10)]
    evs += [{"id": 70, "kind": "plan", "parent": 1}, {"id": 71, "kind": "walk", "parent": 70, "verified": False, "reason": "x"},
            {"id": 72, "kind": "walk", "parent": 70, "verified": True, "answer": "v29"},
            {"id": 73, "kind": "hop", "parent": 72, "hop": 0, "fact": "v29 is the r of x"},
            {"id": 74, "kind": "fact", "parent": 73, "text": "v29 is the r of x"},
            {"id": 75, "kind": "answer", "parent": 1, "verified": True, "answer": "v29"}]
    g = build(evs)
    gates = [n for n in g["nodes"] if n["kind"] == "gate"]
    assert len(gates) == GATE_KEEP + 1 and any(n["id"] == 39 for n in gates)      # the walked one is always kept
    summ = next(n for n in g["nodes"] if n["kind"] == "gates")
    assert summ["counts"] == {"accepted": 30 - GATE_KEEP - 1, "unparseable": 10}
    t = types(g)
    assert t["learned_into"] == [(39, 74)] and t["retries"] == [(72, 71)] and t["concludes"] == [(72, 75)]


def test_other_questions_in_the_stream_stay_out():
    evs = [{"id": 1, "kind": "question", "parent": None, "text": "mine"},
           {"id": 2, "kind": "question", "parent": None, "text": "theirs"},
           {"id": 3, "kind": "answer", "parent": 2, "verified": False, "reason": "no_plan"},
           {"id": 4, "kind": "answer", "parent": 1, "verified": False, "reason": "no_plan"}]
    g = build(evs, "Mine")
    assert g["roots"] == [1] and {n["id"] for n in g["nodes"]} == {1, 4}


def test_a_sink_is_removed_by_identity_not_by_what_it_heard():
    outer, inner = ev.MemorySink(), ev.MemorySink()
    ev.add_sink(outer); ev.add_sink(inner)
    try:
        ev.emit("question", text="q")
        assert outer == inner                    # same content: list equality would confuse them
        ev.remove_sink(inner)
        ev.emit("question", text="q2")
        assert len(outer) == 2 and len(inner) == 1
    finally:
        ev.remove_sink(outer); ev.remove_sink(inner)
    assert not ev.listening() or all(s is not outer and s is not inner for s in ev._sinks)
