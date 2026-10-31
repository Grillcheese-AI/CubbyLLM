"""r2_vm -- order 0046, arm B1 ("branch, check, decide"): one VM branch per item the name gives, no model, no new
opcode.

The test segment holds each candidate's values of the relation under the key "<item> <property>" (a knowledge
``.jsonl``, one fact a value, its source the item: ``wikidata:<qid>``), written once for a run and cached by the VM's
run-proto process. For each candidate the host opens a branch on the wire (its own clone of the store) and runs the one
fixed program, ``QUERY "<item> <property>"``. The branch verifies when its QUERY returns chunks; the branch report says
what it queried and how many hits it had. Then the graph over the branches that verify decides -- never a rank:

  none verify                 refuse
  one                         that item (pick)
  several, the same values    say the value once (no item is picked)
  several, different values   answer each, at most three; more -> ASK

Wired: STANDALONE (R2's bench; the client is ``cubbyllm.bridges.cubelang_client``).
"""
from __future__ import annotations

import json

__wiring__ = "STANDALONE"

EACH = 3
PROGRAM = """
program Hop implements ISolve {
    @external
    public function solve(key: str): number {
        query key;
        create hits : number;
        pop hits;
        return hits;
    }
}
"""


def write_segment(path: str, profiles: dict, pids) -> int:
    """The test segment: every profile's values of `pids`, one fact a value. Returns the facts written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for q, p in profiles.items():
            for pid in pids:
                for v in dict.fromkeys((p.get("values") or {}).get(pid) or ()):
                    f.write(json.dumps({"key": f"{q} {pid}", "text": v, "source": f"wikidata:{q}"}, ensure_ascii=False) + "\n")
                    n += 1
    return n


def branch(session, kpath: str, qid: str, pid: str) -> dict:
    r = session.run(PROGRAM, "solve", [f"{qid} {pid}"], knowledge_path=kpath, branch={"id": qid})
    chunks = json.loads(r["result"] or "[]")
    rep = r.get("report") or {}
    return {"item": qid, "values": sorted({c["text"] for c in chunks}), "sources": sorted({c["source"] for c in chunks}),
            "queried": rep.get("queried"), "wall_ms": rep.get("wall_ms")}


def decide(branches: list[dict]) -> dict:
    ok = [b for b in branches if b["values"]]
    if not ok:
        return {"outcome": "refuse", "why": "no branch verifies", "branches": len(branches)}
    if len(ok) == 1:
        return {"outcome": "pick", "item": ok[0]["item"], "by": "the only branch that verifies"}
    if len({tuple(b["values"]) for b in ok}) == 1:
        return {"outcome": "once", "value": ok[0]["values"], "items": [b["item"] for b in ok]}
    if len(ok) <= EACH:
        return {"outcome": "each", "items": [b["item"] for b in ok], "values": {b["item"]: b["values"] for b in ok}}
    return {"outcome": "ask", "items": [b["item"] for b in ok]}


def b1(session, kpath: str, named: list[str], pid: str) -> dict:
    """B1 over the candidates that bear the name (``r2_arms.bearing``), for one relation."""
    bs = [branch(session, kpath, q, pid) for q in named]
    out = decide(bs)
    out["wall_ms"] = sum(b["wall_ms"] or 0 for b in bs)
    return out


__all__ = ["b1", "decide", "write_segment"]
