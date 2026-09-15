"""The chain-of-thought walk: retrieve, verify in the VM, answer or refuse.

Retrieval walks the chain; the VM holds it, verifies it, and reads the
answer out (spec section 3). `verified=True` is impossible with any hop
below tau_vm — asserted at the single return site that sets it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.protocols import Wiring
from . import simlog
from .plan_verify import answer_type_mismatch, verify_plan
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
    # threshold applied), "lookup_paraphrase" (hop 0 served by the index's
    # paraphrase tier: subject exact, relation by relation_matches -- 2026-09-11,
    # lever 1) or "search" (cosine top-k above tau_ret)
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
    # Plan-time refusal (2026-09-11, plan_verify): set when `answer(..., known=)`
    # refused the plan BEFORE any walk -- reason is then "unknown_relation" or
    # "plan_does_not_cover_question" and this carries the relations the store
    # does not hold. None whenever a walk ran (or no `known` was given) -- except
    # "ambiguous_hop" (2026-09-11): the walk found several facts with different
    # objects for one hop and refused to pick; this then carries the hop, the
    # candidate objects and the facts, for an ASK.
    refused: dict | None = None


_accept = accepts     # the acceptance test moved to the planner (2026-09-04); this name stays for the validation scripts


def _exact_tail(plan: QuestionPlan, t: Triple) -> bool:
    """Hop 0's exact tier: the fact's 'rel of subj' reproduces the question tail."""
    return normalize(f"{t.rel} of {t.subj}") == normalize(plan.tail)


class _Ambiguous(Exception):
    """The walk found several facts with different objects for one hop (see `_walk`)."""
    def __init__(self, hop: int, objects: list[str], facts: list[str]) -> None:
        super().__init__(f"hop {hop}: {len(objects)} candidate objects")
        self.hop, self.objects, self.facts = hop, objects, facts


_AS_OF = __import__("re").compile(r"\b(?:in|as of|during|by|back in)\s+(1[0-9]{3}|2[0-9]{3})\b", __import__("re").I)


def _covering(cands, times, year: int):
    """The candidates whose stated time COVERS `year` -- a point in it, or a span around it.

    This is the honest half of the multi-value problem and the one Nick logged as the "as of"
    residue: "who was governor in 2015" and "what was the population in 1608" are not ambiguous
    questions at all. They name a year, the source dated its values, and exactly one of them
    answers. Nothing is being picked here -- the QUESTION picks, and the store can finally tell
    the values apart. Several survivors or none is still a refusal."""
    out = []
    for f, t in cands:
        w = times.get(" ".join(str(f).split())) or times.get(normalize(str(f))) or {}
        pt, st, en = w.get("point"), w.get("start"), w.get("end")
        if pt and pt[:4] == str(year):
            out.append((f, t)); continue
        if st or en:
            y0 = int(st[:4]) if st else None
            y1 = int(en[:4]) if en else None
            # an OPEN span (a start, no end) covers its start year only -- counting it forward
            # claims an end nobody stated. Same rule the date index uses (ask.py, 2026-09-14).
            if y0 is not None and y1 is not None and y0 <= year <= y1:
                out.append((f, t))
            elif y0 is not None and y1 is None and y0 == year:
                out.append((f, t))
            elif y0 is None and y1 is not None and y1 == year:
                out.append((f, t))
    return out


def _latest_dated(cands, times):
    """The one candidate the SOURCE dated later than every other, or None.

    None when nothing is dated, when fewer than all of them are, or when the latest date is
    shared -- in every one of those the values are not separated by anything the source said,
    and a refusal is the result. "Later" is `point`, else `start`: a value with a start and no
    end is a value still standing, and a value dated at a point is that point's value."""
    if not times:
        return None
    stamped = []
    for f, t in cands:
        w = times.get(" ".join(str(f).split())) or times.get(normalize(str(f))) or {}
        when = w.get("point") or w.get("start")
        if not when:
            return None                                   # one undated candidate and the set is unordered
        stamped.append((when, f, t))
    if len(stamped) < 2:
        return None
    stamped.sort(key=lambda q: q[0], reverse=True)
    if stamped[0][0] == stamped[1][0]:
        return None                                       # a tie in time is not an order
    return (stamped[0][1], stamped[0][2])


def _walk(plan: QuestionPlan, retrieve, tau_ret: float, top_k: int,
          budget: list[int], trace: list[HopTrace],
          banned: set[str], lookup=None, times: dict | None = None,
          as_of: int | None = None) -> list[Triple] | None:
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
            if hop == 0 and len(cands) > 1:
                # the exact tier outranks the paraphrase tier at hop 0: a fact that
                # reproduces the tail is never displaced by one that only matches it
                exact = [ft for ft in cands if _exact_tail(plan, ft[1])]
                cands = exact or cands
            objects = sorted({normalize(t.obj) for _f, t in cands})
            if len(objects) > 1:
                # AMBIGUOUS (2026-09-11, exp_r11): several stored facts serve this hop with
                # DIFFERENT objects -- a multi-valued relation ('population' over the years,
                # three citizenships) or a conflict between sources. Picking one by rank
                # spoke a wrong population for 'as of 2022'. The honest outcome is a
                # refusal that names the candidates (ASK territory), never a guess.
                #
                # ONE thing may break the tie, and only one: a time the SOURCE stated and
                # recorded beside the fact. The neutral-prior competition (2026-09-14, Q5)
                # named this exact call site as where invariant 4 dies -- "picking among
                # claims is a claim decision ... only qualifiers carried with the fact may
                # order it, and if nothing does, refuse". So: the latest dated value wins
                # and says so, and an asker tally may NEVER be consulted here, however
                # convenient. Ranking, popularity and 'what people usually mean' stay on
                # the referent side of the line where they belong.
                picked = None
                if as_of is not None and times:
                    covering = _covering(cands, times, as_of)
                    # exactly one value answers the year the QUESTION named: that is not a pick,
                    # it is the question doing its job. None or several and we are back to a
                    # refusal, which names the candidates as it always did.
                    if len({normalize(t.obj) for _f, t in covering}) == 1:
                        picked = covering[0]
                if picked is None:
                    picked = _latest_dated(cands, times)
                if picked is None:
                    raise _Ambiguous(hop, objects, [f for f, _t in cands])
                cands = [picked]
            if len(cands) > 1:
                rank = {f: float(sc) for sc, f in retrieve(query, max(top_k, len(cands)))}
                cands.sort(key=lambda ft: (-rank.get(ft[0], -1.0), ft[0]))
            if cands:
                fact, t = cands[0]
                source = "lookup" if (hop or _exact_tail(plan, t)) else "lookup_paraphrase"
                found = (1.0, fact, t, source)
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
           top_k: int = 3, max_repairs: int = 1, lookup=None, known=None,
           plan: QuestionPlan | None = None, aliases: dict[str, list[str]] | None = None,
           times: dict | None = None, chunk: int = 0) -> CoTResult:
    """`lookup`: a `TripleIndex.hop`-shaped callable; when given, every hop is
    looked up before it is searched (see `_walk`).

    `aliases`: {store relation (normalized): [the plan's original words]} for a plan
    the host REWROTE into the store's wording (learn.py's alias step, lever 4) --
    the disposer's coverage check then accepts the original words in the question.

    `known`: a `plan_verify.KnownRelations` (the store's relation vocabulary --
    `StoreRelations(store)` host-side, or `VMRelations` to let the VM's QUERY
    answer). When given, the plan is DISPOSED OF before any walk: a plan that
    does not reconstruct the question, or that asks for a relation the store
    does not hold, is refused with that reason and zero retrieval/VM cost.
    exp_r6 (2026-09-11): refuses 86/92 of the harvest's misparsed chains and
    0/517 verified ones. None keeps the pre-disposer behaviour exactly.

    `times`: the store's `{fact: {start, end, point}}` -- the times the SOURCE stated, beside
    the facts. The only thing permitted to break an ambiguous hop: several values for one
    relation are separated by a date the source gave, or by nothing, and nothing means refuse.
    See `_latest_dated`. None keeps the pre-2026-09-14 behaviour (every such hop refuses).

    `plan`: a PROPOSED plan (the emitter's, exp_r7) used instead of the
    grammar's parse. Model proposes, host disposes: it goes through `known`
    exactly like a grammar plan, and the walk and the VM treat it identically.
    None = the grammar parses the question, as before."""
    # max_repairs 3 -> 1 (2026-09-03): on the 800-question harvest every failure burned all three
    # repairs with zero hops verified and no verified chain ever needed more than one; budget 1
    # reproduces 517 verified / 513 correct / control 528/528 exactly at 45.9 ms vs 120.9 ms per
    # question (validation/logs/exp_m3_cot_pipeline_rb1.json vs _v3cf.json).
    if plan is None:
        plan = parse_question(question)
    if plan is None:
        return CoTResult(answer=None, verified=False, reason="unparseable")
    m_asof = _AS_OF.search(question or "")
    as_of = int(m_asof.group(1)) if m_asof else None     # the year the QUESTION named, if it named one
    if known is not None:
        # The coverage check reads the question WITHOUT its year phrase, because that is the
        # question the plan was made for. 2026-09-14, live: with the year left in, a perfectly
        # good plan (['position held'], seed 'bill haslam') was refused as
        # plan_does_not_cover_question over the words "in 2015" -- which no relation covers and
        # none should, because a year is a constraint on the answer and not part of the chain.
        # The year is not discarded: `as_of` above carries it to the hop that needs it.
        covered = " ".join(_AS_OF.sub(" ", question).split()) if as_of is not None else question
        verdict = verify_plan(covered, plan, known, aliases)
        if not verdict.ok:
            return CoTResult(answer=None, verified=False, reason=verdict.reason,
                             refused={"covers": verdict.covers,
                                      "unknown_relations": list(verdict.unknown_relations),
                                      "tail_relation": verdict.tail_relation,
                                      "paraphrased": list(verdict.paraphrased)})

    budget = [max_repairs]
    banned: set[str] = set()
    last_trace: list[HopTrace] = []
    # The FIRST attempt's verify clauses, kept across the retry. A verify failure
    # bans the weakest hop's fact; if the store has no replacement the retry's
    # walk returns None and the honest reason is still "the VM rejected the
    # binding", not "retrieval ran out". Reporting the retry's symptom is how
    # exp_r29 got 226 refusals all labelled `retrieval_exhausted` over a
    # retrieval that had found the right fact every time.
    verify_clauses: list[str] = []
    repairs: list[dict] = []
    pending_repair: dict | None = None
    # up to two walk+verify rounds (spec 4.4: one verify-stage repair pass)
    for _attempt in range(2):
        trace: list[HopTrace] = []
        try:
            triples = _walk(plan, retrieve, tau_ret, top_k, budget, trace, banned, lookup=lookup,
                            times=times, as_of=as_of)
        except _Ambiguous as amb:
            return CoTResult(answer=None, verified=False, trace=trace, repairs_used=max_repairs - budget[0],
                             reason="ambiguous_hop", repairs=repairs,
                             refused={"hop": amb.hop, "objects": amb.objects, "facts": amb.facts})
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
                             reason=("vm_verify_failed" if verify_clauses
                                     else "retrieval_exhausted"),
                             refused=({"clauses": verify_clauses, "tau_vm": tau_vm,
                                       "then": "retrieval_exhausted"}
                                      if verify_clauses else None),
                             repairs=repairs)

        display_rels = [(t.rel if i == 0 else plan.relations[i]) or t.rel
                        for i, t in enumerate(triples)]
        # `chunk` is the number of hops that share a frame; 0 keeps the shipped
        # whole-chain shape. The CALLER owns tau: under chunking the frame holds
        # `chunk` bindings rather than `n_hop`, so the applicable threshold is
        # tau(min(chunk, n_hop)), not tau(n_hop). Passing chunk without moving
        # tau measures nothing (exp_r28, WO-2.6).
        source, fns = build_chain_program(triples, display_rels, chunk=chunk)
        ok = True
        # WHICH clause failed, not just that one did. Without this the refusal
        # that follows names a symptom: a binding rejected below tau bans its
        # fact, the retry finds nothing else, and the caller is told
        # `retrieval_exhausted` about a retrieval that worked perfectly
        # (exp_r29, WO-2.6 -- 226 refusals, every one of them misattributed).
        failed: list[str] = []
        for i, fn in enumerate(fns[:-1]):                # hop functions
            out = run_fn(source, fn)
            trace[i].symbol = out.get("result")
            trace[i].similarity = out.get("similarity")
            if trace[i].similarity is None:
                ok = False; failed.append(f"hop{i}:no_similarity")
            elif trace[i].similarity < tau_vm:
                ok = False
                failed.append(f"hop{i}:below_tau({trace[i].similarity:.4f}<{tau_vm:.4f})")
            elif normalize(trace[i].symbol or "") != normalize(triples[i].obj):
                ok = False; failed.append(f"hop{i}:symbol_mismatch")
        ctrl = run_fn(source, fns[-1])                   # control
        ctrl_result = ctrl.get("result")
        ctrl_sim = ctrl.get("similarity")
        # Control role must stay below tau_vm. The real VM's cosine cleanup always
        # returns the nearest symbol from a populated frame — an absent role returns
        # (noise_symbol, low_similarity), never None. Violation if: high similarity
        # (≥tau_vm) OR a symbol without verifiable similarity (unbound frame edge case).
        if (ctrl_sim is not None and ctrl_sim >= tau_vm) or (ctrl_result is not None and ctrl_sim is None):
            ok = False; failed.append("control:not_below_tau")

        if ok:
            # the claimed-answer invariant, at the only verified=True site
            assert all(h.similarity is not None and h.similarity >= tau_vm
                       for h in trace)
            # the typed answer class (2026-09-12): a certified chain whose value is a kind
            # the question did not ask for -- a name for "in what year", a date for "who" --
            # answers a different question; refused with both kinds named, never spoken
            mismatch = answer_type_mismatch(question, triples[-1].obj)
            if mismatch is not None:
                return CoTResult(answer=None, verified=False, trace=trace, repairs_used=used,
                                 source=source, repairs=repairs, reason="answer_type_mismatch",
                                 refused={"asked": mismatch[0], "got": mismatch[1], "value": triples[-1].obj,
                                          "facts": [t.fact for t in trace]})
            # WO-0.4: keep the similarity of every ACCEPTED binding. This is
            # the only site that speaks, so it is the only site whose
            # distribution matters. No-op unless CUBBY_SIMLOG is set, and it
            # never raises -- see simlog.record.
            simlog.record(question, triples[-1].obj, tau_vm, trace, ctrl_sim)
            return CoTResult(answer=triples[-1].obj, verified=True,
                             trace=trace, repairs_used=used, source=source,
                             repairs=repairs)

        # verify failed: blacklist the weakest hop's fact and retry once
        # (only if another attempt will actually run and budget remains)
        last_trace = trace
        verify_clauses = verify_clauses or failed
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
                     reason="vm_verify_failed", source=source, repairs=repairs,
                     refused={"clauses": verify_clauses or failed, "tau_vm": tau_vm})
