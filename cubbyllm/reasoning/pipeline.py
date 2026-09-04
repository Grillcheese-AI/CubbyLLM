"""The chain-of-thought walk: retrieve, verify in the VM, answer or refuse.

Retrieval walks the chain; the VM holds it, verifies it, and reads the
answer out (spec section 3). `verified=True` is impossible with any hop
below tau_vm — asserted at the single return site that sets it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .planner import (QuestionPlan, Triple, accepts, normalize, parse_fact,
                      parse_question)
from .programs import build_chain_program

__wiring__ = Wiring.WIRED


@dataclass
class HopTrace:
    query: str
    fact: str
    triple: Triple | None
    ret_score: float
    symbol: str | None = None
    similarity: float | None = None
    # how the fact was found: "lookup" (the triple index; ret_score is 1.0, no
    # threshold applied) or "search" (cosine top-k above tau_ret)
    source: str = "search"


@dataclass
class CoTResult:
    answer: str | None
    verified: bool
    trace: list[HopTrace] = field(default_factory=list)
    repairs_used: int = 0
    reason: str | None = None
    # The exact CubeLang source built for `trace` (the LAST attempt that
    # reached build_chain_program) -- set whenever a program actually got
    # built, i.e. reason in (None, "vm_verify_failed"); stays None for
    # "unparseable"/"retrieval_exhausted", where no program was ever built.
    source: str | None = None
    # Verify-stage repair DPO pairs: one entry per fact the walk banned and
    # retried, {"hop": int, "rejected_fact": str, "replacement_fact": str |
    # None}. `replacement_fact` is the fact the NEXT attempt picked for that
    # same hop position (None if that attempt's walk never reached the hop,
    # e.g. retrieval_exhausted before getting there). At most one entry is
    # possible today (the walk allows a single ban-and-retry round), but the
    # shape is a list so a future multi-repair budget doesn't need a format
    # change.
    repairs: list[dict] = field(default_factory=list)


_accept = accepts     # the acceptance test moved to the planner (2026-09-04); this name stays for the validation scripts


def _walk(plan: QuestionPlan, retrieve, tau_ret: float, top_k: int,
          budget: list[int], trace: list[HopTrace],
          banned: set[str], lookup=None) -> list[Triple] | None:
    """Pick one accepted triple per hop; None when the budget dies.
    Bounded by `budget` alone (the spec's 3-per-question repair budget);
    `banned` holds facts a failed VM verify blacklisted.

    `lookup(plan, hop, entity) -> [(fact, triple)]` (a `TripleIndex.hop`) is
    tried FIRST at every hop: an exact answer to the same acceptance test
    search would apply, with no threshold and no k. Search runs only for a hop
    the index misses. Ambiguity (0.8% of harvest hops) is broken by the
    search's own ranking of the query, then by fact text — deterministic."""
    triples: list[Triple] = []
    entity: str | None = None
    hop = 0
    question_tail = plan.tail
    while hop < plan.n_hop:
        if hop == 0:
            query = f"what is the {question_tail}"
        else:
            query = f"{entity} {plan.relations[hop]}"
        found = None
        if lookup is not None:
            cands = [(f, t) for f, t in lookup(plan, hop, entity) if f not in banned]
            if len(cands) > 1:
                rank = {f: float(sc) for sc, f in retrieve(query, max(top_k, len(cands)))}
                cands.sort(key=lambda ft: (-rank.get(ft[0], -1.0), ft[0]))
            if cands:
                fact, t = cands[0]
                found = (1.0, fact, t, "lookup")
        while found is None:
            for score, fact in retrieve(query, top_k):
                if score < tau_ret or fact in banned:
                    continue
                t = parse_fact(fact)
                if t is not None and accepts(plan, hop, entity, t):
                    found = (score, fact, t, "search")
                    break
            if found is None:
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                seen = " ".join(h.fact for h in trace)
                query = f"{query} {seen}" if seen else f"{query} {question_tail}"
        score, fact, t, source = found
        trace.append(HopTrace(query=query, fact=fact, triple=t,
                              ret_score=score, source=source))
        triples.append(t)
        entity = t.obj
        hop += 1
    return triples


def answer(question: str, retrieve, run_fn, tau_vm: float, tau_ret: float,
           top_k: int = 3, max_repairs: int = 1, lookup=None) -> CoTResult:
    """`lookup`: a `TripleIndex.hop`-shaped callable; when given, every hop is
    looked up before it is searched (see `_walk`)."""
    # max_repairs 3 -> 1 (2026-09-03): on the 800-question harvest every failure burned all three
    # repairs with zero hops verified and no verified chain ever needed more than one; budget 1
    # reproduces 517 verified / 513 correct / control 528/528 exactly at 45.9 ms vs 120.9 ms per
    # question (validation/logs/exp_m3_cot_pipeline_rb1.json vs _v3cf.json).
    plan = parse_question(question)
    if plan is None:
        return CoTResult(answer=None, verified=False, reason="unparseable")

    budget = [max_repairs]
    banned: set[str] = set()
    last_trace: list[HopTrace] = []
    repairs: list[dict] = []
    pending_repair: dict | None = None
    # up to two walk+verify rounds (spec 4.4: one verify-stage repair pass)
    for _attempt in range(2):
        trace: list[HopTrace] = []
        triples = _walk(plan, retrieve, tau_ret, top_k, budget, trace, banned, lookup=lookup)
        used = max_repairs - budget[0]

        # Close out a pending repair from the PREVIOUS attempt's ban: the
        # replacement is whatever fact THIS attempt's walk landed on at that
        # same hop position (None if the walk didn't get that far).
        if pending_repair is not None:
            hop_i = pending_repair["hop"]
            if hop_i < len(trace):
                pending_repair["replacement_fact"] = trace[hop_i].fact
            repairs.append(pending_repair)
            pending_repair = None

        if triples is None:
            return CoTResult(answer=None, verified=False,
                             trace=trace or last_trace, repairs_used=used,
                             reason="retrieval_exhausted", repairs=repairs)

        display_rels = [(t.rel if i == 0 else plan.relations[i]) or t.rel
                        for i, t in enumerate(triples)]
        source, fns = build_chain_program(triples, display_rels)
        ok = True
        for i, fn in enumerate(fns[:-1]):                # hop functions
            out = run_fn(source, fn)
            trace[i].symbol = out.get("result")
            trace[i].similarity = out.get("similarity")
            if (trace[i].similarity is None or trace[i].similarity < tau_vm
                    or normalize(trace[i].symbol or "")
                    != normalize(triples[i].obj)):
                ok = False
        ctrl = run_fn(source, fns[-1])                   # control
        ctrl_result = ctrl.get("result")
        ctrl_sim = ctrl.get("similarity")
        # Control role must stay below tau_vm. The real VM's cosine cleanup always
        # returns the nearest symbol from a populated frame — an absent role returns
        # (noise_symbol, low_similarity), never None. Violation if: high similarity
        # (≥tau_vm) OR a symbol without verifiable similarity (unbound frame edge case).
        if (ctrl_sim is not None and ctrl_sim >= tau_vm) or (ctrl_result is not None and ctrl_sim is None):
            ok = False

        if ok:
            # the claimed-answer invariant, at the only verified=True site
            assert all(h.similarity is not None and h.similarity >= tau_vm
                       for h in trace)
            return CoTResult(answer=triples[-1].obj, verified=True,
                             trace=trace, repairs_used=used, source=source,
                             repairs=repairs)

        # verify failed: blacklist the weakest hop's fact and retry once
        # (only if another attempt will actually run and budget remains)
        last_trace = trace
        if _attempt == 0:
            if budget[0] <= 0:
                # No budget left: can't retry, so break
                break
            weakest = min(range(len(trace)),
                          key=lambda i: trace[i].similarity or -1.0)
            banned.add(trace[weakest].fact)
            pending_repair = {"hop": weakest, "rejected_fact": trace[weakest].fact,
                              "replacement_fact": None}
            budget[0] -= 1

    return CoTResult(answer=None, verified=False, trace=last_trace,
                     repairs_used=max_repairs - budget[0],
                     reason="vm_verify_failed", source=source, repairs=repairs)
