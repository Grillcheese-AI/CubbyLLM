"""exp_r8_decomposition — GoT-1's first slice on the questions no flat plan reaches.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Arm C (the 37 questions the grammar cannot parse) splits in two:

  19 one-hop questions in unseen frames -- "What is X a participant of?",
     "Which languages spoken, written or signed by X?", "What award did X
     receive?". Not a GoT case: the emitter already plans them (exp_r3:
     PARTICIPANT / AWARD / NOMINATION accepted) and exp_r7 only failed to find
     the seed entity. exp_r7's residual seed rule (this same day) and gen 2's
     explicit SEED cover them; they are re-walked by exp_r7, not here.

  18 nested questions -- "Which list includes the list that includes the
     component of the instance of X?" -- a relative clause wrapping a chain.
     The grammar declared depth-2 recursion out of scope; the emitter emits
     [LIST, LIST] and drops the inner chain, or nothing well-formed. No flat
     plan covers the question, so covers() v2 refuses every one. THIS is the
     GoT case: a goal DECOMPOSITION -- an inner Question node whose verified
     answer becomes the seed of an outer one, `depends_on` between them, the
     host composing, the VM certifying each part, and bounded alternatives
     for the part the surface form leaves ambiguous.

This script is the CEILING for that mechanism on our substrate, with the
decomposition ORACLE (split at "that includes"; the emitter's proposal is what
GoT-1 trains). Per nested question:

    inner  = "What is the <inner chain>?"  -> the grammar parses it ->
             answer(..., known=) -> the VM verifies a chain -> E, or refuse
    outer  = bounded alternatives over the ambiguous surface form
             "which list includes the list that includes E":
               k=1  plan [None]          tail "list of E"
               k=2  plan [None, "list"]  tail "list of E"
             each disposed of, walked, VM-certified; a composite is verified
             when inner AND one outer verify. If BOTH outers verify with
             different answers the host does not pick: that is ASK territory
             (the don't-know contract), counted as `ambiguous`.

First run (2026-09-11): inner verified 17/18, outer 0/17 on BOTH forward readings.
So the outer fact is not "L is the list of E" with E as subject -- the store
holds E some other way. v2 prints E's neighbourhood (forward and backward
edges) before guessing, and adds the BACKWARD reading: "L includes E" as the
inverse of "E is the <cls> of L", i.e. the fact with E as OBJECT, composed
through the index (1 and 2 hops) and then VM-certified like a forward chain
(bind every hop, verify each, control below the floor). Ambiguous backward
frontiers (>1 candidate) are counted, never chosen.

Second run: E='Books/Organism' has the forward edge ('list', 'Living beings') --
the outer fact IS in the store, one hop, and 'Living beings' IS the gold. The
fwd1 plan was REFUSED by the disposer (`plan_does_not_cover_question`): in
"Which LIST includes the list that includes E" the class noun 'list' stayed in
the residual and read as a dropped 'list' hop. A disposer bug, now fixed
(plan_verify._WHICH_CLASS strips "which <class> includes/contains/has ..." as
it strips "which <class> is ..."; 'includes'/'contains' are joint words).

What that means for the 18: with the fix, the FLAT three-hop plan
[instance, component, list] covers the whole question -- "that includes" is a
joint, like "contained within the" on arm B. So this family is a chain with a
joint the grammar lacks, not a genuinely nested sub-question, and the
decomposition is a ceiling for the general mechanism rather than the only
route. The emitter's [LIST, LIST] still drops the inner chain and is still
refused, correctly. GoT-1's decomposition stays the answer for questions whose
inner part is not a chain the walk can take.

Reported: inner verified, outer verified (by k), composite verified / correct
vs gold, ambiguous, VM calls -- against the 3x-VM-calls kill rule of the GoT
plan and the flat baseline (0 of 18, by construction).

  python validation/exp_r8_decomposition.py --resident
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys, time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NESTED = re.compile(r"^\s*which\s+(?P<cls>[a-z]+)\s+includes\s+the\s+(?P=cls)\s+that\s+includes\s+the\s+(?P<inner>.+?)\s*\?\s*$", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=str(LOGS / "cot_harvest_v3cf.jsonl"))
    ap.add_argument("--taus", default=str(LOGS / "exp_m3_cot_pipeline_lookup_vp_res.json"))
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import exp_m3_cot_pipeline as m
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning import TripleIndex, answer, parse_question
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations

    recs = [json.loads(l) for l in open(a.harvest, encoding="utf-8")]
    nested = [(r, NESTED.match(r["question"])) for r in recs if r.get("reason") == "unparseable"]
    nested = [(r, mm) for r, mm in nested if mm]
    taus = json.loads(pathlib.Path(a.taus).read_text(encoding="utf-8"))["calibration"]
    floors = {int(k): float(v) for k, v in taus["tau_vm_floors"].items()}
    fallback = float(taus["fallback_tau_vm"]); tau_ret = float(taus["tau_ret"])

    sw = m._load_semantic_words(); enc = sw.FastWordEncoder.from_npz(m.V4_TABLE)
    _q, _a, _h, _c, store = m.load_sample(800, seed=0)
    retrieve = m.make_retriever(store, enc)
    index = TripleIndex(store); known = StoreRelations(store)
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = [0]
    def run_fn(source, fn):
        calls[0] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    def walk(q, plan):
        return answer(q, retrieve, run_fn, tau_vm=floors.get(plan.n_hop, fallback), tau_ret=tau_ret,
                      top_k=3, max_repairs=1, lookup=index.hop, known=known, plan=plan)
    log(f"nested questions: {len(nested)} of {sum(1 for r in recs if r.get('reason') == 'unparseable')} unparseable | "
        f"store {len(store)} facts, {len(known)} relations | VM {'resident' if session else 'spawn'}")
    log(f"'list' is a store relation: {'list' in known}\n")

    c = Counter(); out = []
    for r, mm in nested:
        q = r["question"]; cls = mm.group("cls").lower(); inner_text = mm.group("inner")
        gold = r.get("gold_answer")
        inner_q = f"What is the {inner_text}?"
        iplan = parse_question(inner_q)
        rec = {"question": q, "inner_question": inner_q, "gold": gold, "gold_n_hop": r.get("n_hop")}
        if iplan is None:
            c["inner_unparseable"] += 1; rec["outcome"] = "inner_unparseable"; out.append(rec); continue
        ir = walk(inner_q, iplan)
        rec["inner"] = {"reason": ir.reason, "verified": ir.verified, "answer": ir.answer, "n_hop": iplan.n_hop}
        if not ir.verified:
            c["inner_failed"] += 1; c[f"inner:{ir.reason}"] += 1; rec["outcome"] = "inner_failed"; out.append(rec); continue
        c["inner_verified"] += 1
        E = ir.answer
        # what the store holds around E, both directions -- printed before anything is guessed
        fwd = [(t.rel, t.obj) for _f, t in index._by_subj.get(normalize(E), [])]
        bwd = [(t.rel, t.subj) for _f, t in index.by_object(E)]
        rec["neighbourhood"] = {"E": E, "as_subject": fwd[:8], "as_object": bwd[:8]}
        c["E_has_forward_edges"] += int(bool(fwd)); c["E_has_backward_edges"] += int(bool(bwd))
        outers = {"fwd1": QuestionPlan(relations=[None], tail=f"{cls} of {E}", n_hop=1),
                  "fwd2": QuestionPlan(relations=[None, cls], tail=f"{cls} of {E}", n_hop=2)}
        verified = {}
        for k, plan in outers.items():
            outer_q = f"Which {cls} includes the {cls} that includes {E}?"
            orr = walk(outer_q, plan)
            rec[f"outer_{k}"] = {"reason": orr.reason, "verified": orr.verified, "answer": orr.answer}
            if orr.verified:
                verified[k] = orr.answer
        # the BACKWARD reading: "L includes E" is the inverse of "E is the <cls> of L" -- the fact may
        # sit with E as OBJECT. Hop 1: subjects of (rel~cls, obj=E); hop 2: the same from those.
        # Host-composed through the index, then the composite chain is VM-certified exactly as the
        # forward walk certifies one (build_chain_program + verify every hop + control).
        from cubbyllm.reasoning import build_chain_program
        from cubbyllm.reasoning.planner import relation_matches, Triple
        def back(entity):
            return [(f, t) for f, t in index.by_object(entity) if relation_matches(cls, t.rel)]
        for depth in (1, 2):
            frontier = [([], E)]
            for _ in range(depth):
                nxt = []
                for chain, ent in frontier:
                    for f, t in back(ent):
                        nxt.append((chain + [t], t.subj))
                frontier = nxt
            rec[f"outer_bwd{depth}_candidates"] = len(frontier)
            if len(frontier) == 1:
                chain, L = frontier[0]
                # certify: bind each hop's SUBJECT (what this direction recovers) and run the VM's hop + control checks
                triples = [Triple(obj=t.subj, rel=t.rel, subj=t.obj) for t in chain]
                source, fns = build_chain_program(triples, [t.rel for t in triples])
                ok = True; sims = []
                for i, fn in enumerate(fns[:-1]):
                    o = run_fn(source, fn); sims.append(o.get("similarity"))
                    tau = floors.get(len(triples), fallback)
                    if o.get("similarity") is None or o["similarity"] < tau or normalize(o.get("result") or "") != normalize(triples[i].obj):
                        ok = False
                ctl = run_fn(source, fns[-1])
                if ctl.get("similarity") is not None and ctl["similarity"] >= floors.get(len(triples), fallback):
                    ok = False
                rec[f"outer_bwd{depth}"] = {"verified": ok, "answer": L if ok else None, "similarities": sims,
                                            "chain": [f"{t.obj} <- {t.rel} <- {t.subj}" for t in chain]}
                if ok:
                    verified[f"bwd{depth}"] = L
            elif len(frontier) > 1:
                c[f"bwd{depth}_ambiguous"] += 1
        if not verified:
            c["outer_failed"] += 1; rec["outcome"] = "outer_failed"; out.append(rec); continue
        answers = {normalize(v) for v in verified.values()}
        if len(answers) > 1:
            c["ambiguous_ask"] += 1; rec["outcome"] = "ambiguous"; out.append(rec); continue
        ans = next(iter(verified.values()))
        ok = gold is not None and normalize(ans) == normalize(gold)
        c["composite_verified"] += 1; c["composite_correct" if ok else "composite_wrong"] += 1
        c[f"outer_k={'+'.join(str(k) for k in sorted(verified))}"] += 1
        rec.update({"outcome": "verified", "answer": ans, "correct": ok, "outer_k": sorted(verified)})
        out.append(rec)

    n = len(nested)
    log("GoT-1 slice — host-composed decomposition, VM-certified per part, bounded outer alternatives")
    log(f"  nested questions                 {n:4d}   (flat plans: 0 verified, by construction)")
    log(f"  inner chain verified             {c['inner_verified']:4d}")
    for k in sorted(k for k in c if k.startswith("inner:")):
        log(f"    {k:30s} {c[k]}")
    log(f"  E has forward edges / backward   {c['E_has_forward_edges']:4d} / {c['E_has_backward_edges']}")
    log(f"  outer failed (fwd1, fwd2, bwd1, bwd2) {c['outer_failed']:4d}   bwd ambiguous: 1-hop {c['bwd1_ambiguous']}, 2-hop {c['bwd2_ambiguous']}")
    log(f"  ambiguous (both k, disagree)     {c['ambiguous_ask']:4d}   -> ASK, never a guess")
    log(f"  COMPOSITE VERIFIED               {c['composite_verified']:4d}   correct {c['composite_correct']}  wrong {c['composite_wrong']}")
    for k in sorted(k for k in c if k.startswith("outer_k=")):
        log(f"    {k:30s} {c[k]}")
    log(f"  VM calls {calls[0]}  ({calls[0]/max(1,n):.1f} per question; the flat walk spends ~{3} on a 3-hop verify)")
    for rec in out[:6]:
        log(f"    {rec['outcome']:16s} {rec['question'][:70]!r} -> {rec.get('answer')!r} (gold {rec['gold']!r})")
        nb = rec.get("neighbourhood")
        if nb:
            log(f"        E={nb['E']!r}  as subject: {nb['as_subject'][:4]}  as object: {nb['as_object'][:4]}")
    if session:
        session.close()
    wall = time.perf_counter() - t0
    (LOGS / f"exp_r8_decomposition{a.tag}.json").write_text(json.dumps({"n": n, "counts": dict(c), "vm_calls": calls[0],
        "records": out, "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r8_decomposition{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.1f}s | wrote exp_r8_decomposition{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
