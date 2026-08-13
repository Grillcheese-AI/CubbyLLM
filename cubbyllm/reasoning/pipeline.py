"""The chain-of-thought walk: retrieve, verify in the VM, answer or refuse.

Retrieval walks the chain; the VM holds it, verifies it, and reads the
answer out (spec section 3). `verified=True` is impossible with any hop
below tau_vm — asserted at the single return site that sets it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .planner import (QuestionPlan, Triple, normalize, parse_fact,
                      parse_question, relation_matches)
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


@dataclass
class CoTResult:
    answer: str | None
    verified: bool
    trace: list[HopTrace] = field(default_factory=list)
    repairs_used: int = 0
    reason: str | None = None


def _accept(plan: QuestionPlan, hop: int, entity: str | None,
            t: Triple) -> bool:
    """Does this parsed fact serve hop `hop` of the plan?"""
    if hop == 0:
        # tail hop: the fact's "rel of subj" must reproduce the question tail
        return normalize(f"{t.rel} of {t.subj}") == normalize(plan.tail)
    expected = plan.relations[hop]
    assert expected is not None
    return (relation_matches(expected, t.rel)
            and normalize(t.subj) == normalize(entity or ""))


def _walk(plan: QuestionPlan, retrieve, tau_ret: float, top_k: int,
          budget: list[int], trace: list[HopTrace],
          banned: set[str]) -> list[Triple] | None:
    """Pick one accepted triple per hop; None when the budget dies.
    Bounded by `budget` alone (the spec's 3-per-question repair budget);
    `banned` holds facts a failed VM verify blacklisted."""
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
        while found is None:
            for score, fact in retrieve(query, top_k):
                if score < tau_ret or fact in banned:
                    continue
                t = parse_fact(fact)
                if t is not None and _accept(plan, hop, entity, t):
                    found = (score, fact, t)
                    break
            if found is None:
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                seen = " ".join(h.fact for h in trace)
                query = f"{query} {seen}" if seen else f"{query} {question_tail}"
        score, fact, t = found
        trace.append(HopTrace(query=query, fact=fact, triple=t,
                              ret_score=score))
        triples.append(t)
        entity = t.obj
        hop += 1
    return triples


def answer(question: str, retrieve, run_fn, tau_vm: float, tau_ret: float,
           top_k: int = 3, max_repairs: int = 3) -> CoTResult:
    plan = parse_question(question)
    if plan is None:
        return CoTResult(answer=None, verified=False, reason="unparseable")

    budget = [max_repairs]
    banned: set[str] = set()
    last_trace: list[HopTrace] = []
    # up to two walk+verify rounds (spec 4.4: one verify-stage repair pass)
    for _attempt in range(2):
        trace: list[HopTrace] = []
        triples = _walk(plan, retrieve, tau_ret, top_k, budget, trace, banned)
        used = max_repairs - budget[0]
        if triples is None:
            return CoTResult(answer=None, verified=False,
                             trace=trace or last_trace, repairs_used=used,
                             reason="retrieval_exhausted")

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
        ctrl_sim = ctrl.get("similarity")
        if ctrl.get("result") is not None or (ctrl_sim is not None and ctrl_sim >= tau_vm):
            ok = False

        if ok:
            # the claimed-answer invariant, at the only verified=True site
            assert all(h.similarity is not None and h.similarity >= tau_vm
                       for h in trace)
            return CoTResult(answer=triples[-1].obj, verified=True,
                             trace=trace, repairs_used=used)

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
            budget[0] -= 1

    return CoTResult(answer=None, verified=False, trace=last_trace,
                     repairs_used=max_repairs - budget[0],
                     reason="vm_verify_failed")
