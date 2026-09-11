"""exp_r6_plan_verify — does disposing of the PLAN catch the misparses the walk cannot?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Motivation: the plan never crosses into the VM (see plan_verify.py). This
measures the host-side disposer on the 800-question harvest, where the
GOLD hop count is known (`n_hop` in the record comes from the corpus, and
`exp_m3_exhaustion_decomp` classifies 92 failures as `misparsed_chain`
when parsed n_hop != gold n_hop).

Three questions, each a number:

  Q1  covers()        — do grammar plans reconstruct their question? 517/517
                        on the verified chains (the grammar is deterministic
                        on the text, so this check is emitter-facing). covers
                        v2 (2026-09-11, joint-agnostic + hop-complete) also
                        refuses 2 misparsed inversion frames v1 let through
                        ("To which ethnic group does the ethnic group of the
                        country of X belong?" parsed as 1-hop with a compound
                        entity) -- a feature, not a regression.
  Q2  unknown_relation — as a MISPARSE DETECTOR: how many of the 92 misparsed
                        chains does it reject, and how many of the 517
                        verified chains does it wrongly reject (precision)?
  Q4  residue         — of the failures that survive lookup-first, how many does
                        the disposer refuse at plan time, and how many remain
                        with a plan that verifies? The remainder is the only
                        part a better retriever or emitter can still win.
  Q3  segment()       — vocabulary-guided re-segmentation: for each question,
                        unique / ambiguous / none; when unique, does its
                        n_hop equal the GOLD n_hop? Split by misparsed vs not.
                        This is the repair the disposer can propose.

DEAD REPAIR PROBES (2026-09-11, observed vocab, recorded so nobody re-runs them)
  relation-led segmentation over ARBITRARY joints (body must start with a
  known relation, then any 1-4 words ending in "the", recurse): fixes 10/92
  misparses but yields 14 confidently-wrong unique readings, because a known
  relation can be a PREFIX of the true one ('award' of 'award received by').
  Greedy-prefix over a vocabulary is unsafe by construction. Dead.
  ' of the '-only segmentation (segment() below): 1/92. Reported in Q3.

STORE CAVEAT: --observed builds the relation vocabulary from the 966 facts
that surfaced in the logs. The relation vocabulary saturates much faster
than the fact set (few relations, many facts), so this is far less
confounded than exp_r5's entity check — but the size is printed and
`--real` exists for the same reason it does there.

  python validation/exp_r6_plan_verify.py --observed
  python validation/exp_r6_plan_verify.py --real
  python validation/exp_r6_plan_verify.py --real --vm      # membership answered by the VM's QUERY;
                                                          # asserts agreement with the host reference
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, types
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_W = ("from ..core.protocols import Wiring", 'class Wiring:\n    WIRED = "WIRED"\n    STANDALONE = "STANDALONE"')
_PLN = ("from .planner import QuestionPlan, normalize, parse_fact",
        "from planner_real import QuestionPlan, normalize, parse_fact")


def _load(rel, name, subs):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for a, b in subs:
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__dict__["__name__"] = name
    m.__dict__["__file__"] = str(ROOT / rel)      # find_cubelang_exe locates the sibling repo from it
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true")
    g.add_argument("--observed", action="store_true")
    ap.add_argument("--vm", action="store_true",
                    help="answer 'is this relation known' IN THE VM (QUERY over the vocabulary as knowledge) "
                         "and assert it agrees with the host-side StoreRelations on every relation asked")
    ap.add_argument("--exe", default=None, help="cubelang binary (else $CUBELANG_EXE / sibling build)")
    a = ap.parse_args()
    if not (a.real or a.observed):
        a.observed = True
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    pl = _load("cubbyllm/reasoning/planner.py", "planner_real", [_W])
    pv = _load("cubbyllm/reasoning/plan_verify.py", "plan_verify_real", [_W, _PLN])

    v3 = [json.loads(l) for l in (LOGS / "cot_harvest_v3cf.jsonl").open(encoding="utf-8")]
    decomp = json.loads((LOGS / "exp_m3_exhaustion_decomp.json").read_text(encoding="utf-8"))
    kind = {x["question"]: x["kind"] for x in decomp["details"]}

    if a.real:
        import exp_m3_cot_pipeline as m
        *_r, store = m.load_sample(800, seed=0)
        mode = "real"
    else:
        lk = [json.loads(l) for l in (LOGS / "cot_harvest_lookup.jsonl").open(encoding="utf-8")]
        s = set()
        for src in (v3, lk):
            for r in src:
                for h in (r.get("trace") or []):
                    if h.get("fact"): s.add(h["fact"])
                for hop in (r.get("candidates_topk") or []):
                    for _sc, txt in hop: s.add(txt)
        store = sorted(s); mode = "observed"
    host = pv.StoreRelations(store)
    log(f"mode {mode}: {len(store)} facts, {host.n_parsed} parsed, {len(host)} distinct relations")
    known = host
    if a.vm:
        cc = _load("cubbyllm/bridges/cubelang_client.py", "cubelang_client_real", [_W])
        kn = LOGS / f"exp_r6_vocab_{mode}.jsonl"
        pv.write_vocab_jsonl(store, kn)
        vm = pv.VMRelations(knowledge=kn, program=ROOT / "cubbyllm" / "bridges" / "programs" / "plan_verify.cube",
                            exe=a.exe, run=cc.run_program)

        class Both:                      # every membership question goes to BOTH; disagreement is fatal
            disagreements: list[tuple[str, bool, bool]] = []
            def __contains__(self, rel):
                h, v = rel in host, rel in vm
                if h != v:
                    self.disagreements.append((rel, h, v))
                return v                 # the VM's answer drives the numbers
        known = Both()
        log(f"  --vm: {cc.find_cubelang_exe(a.exe)}  knowledge {kn.name} ({len(host)} relations)")

    def outcome(r):
        if r.get("verified"): return "verified"
        k = kind.get(r["question"])
        return k or (r.get("reason") or "other")

    parsed = [r for r in v3 if r.get("parsed")]
    log(f"questions parsed by the grammar: {len(parsed)} of {len(v3)}\n")

    # ---- Q1 / Q2 -------------------------------------------------------------
    cov = Counter(); rej = Counter(); tot = Counter(); ex_unknown = []; ex_nocov = []
    for r in parsed:
        q = r["question"]; o = outcome(r)
        plan = pl.parse_question(q)
        v = pv.verify_plan(q, plan, known)
        tot[o] += 1
        if v.covers: cov[o] += 1
        elif len(ex_nocov) < 5: ex_nocov.append((q, pv.canonical_body(plan)))
        if v.unknown_relations:
            rej[o] += 1
            if len(ex_unknown) < 8: ex_unknown.append((o, q, v.unknown_relations))

    log("Q1  covers(): grammar plans that reconstruct their question")
    for o in sorted(tot, key=lambda k: -tot[k]):
        log(f"    {o:20s} {cov[o]:4d} / {tot[o]:4d}  ({cov[o]/tot[o]:.0%})")
    for q, body in ex_nocov:
        log(f"      !! {q!r}\n         body {body!r}")

    log("\nQ2  unknown_relation as a misparse detector (rejected / total)")
    for o in sorted(tot, key=lambda k: -tot[k]):
        log(f"    {o:20s} {rej[o]:4d} / {tot[o]:4d}  ({rej[o]/tot[o]:.0%})")
    tp = rej.get("misparsed_chain", 0); fp = rej.get("verified", 0)
    n_mis = tot.get("misparsed_chain", 0); n_ver = tot.get("verified", 0)
    log(f"    -> catches {tp}/{n_mis} misparses; wrongly rejects {fp}/{n_ver} verified chains"
        f"  (precision vs verified {tp/(tp+fp):.2f})" if tp + fp else "    -> rejects nothing")
    for o, q, u in ex_unknown:
        log(f"      [{o}] {q!r}\n         unknown {u}")

    # ---- Q3 ------------------------------------------------------------------
    seg_all = {}
    for strict in (False, True):
        log(f"\nQ3{'b' if strict else 'a'}  segment({'joint_in_entity=False' if strict else 'exhaustive'}): "
            "vocabulary-guided re-segmentation of the chain body")
        seg = Counter(); agree = Counter(); fixed = []; broke = []
        for r in parsed:
            q = r["question"]; o = outcome(r); gold = r.get("n_hop")
            plan = pl.parse_question(q)
            mm = pl._Q.match(pl.normalize_causal(q))       # only the chain frame has a body
            if not mm:
                seg[(o, "no_chain_frame")] += 1; continue
            plans = pv.segment(mm.group("body"), known, joint_in_entity=not strict)
            bucket = "none" if not plans else ("unique" if len(plans) == 1 else "ambiguous")
            seg[(o, bucket)] += 1
            if bucket == "unique" and gold is not None:
                hit = plans[0].n_hop == gold
                agree[(o, hit)] += 1
                if o == "misparsed_chain" and hit and len(fixed) < 6:
                    fixed.append((q, plan.n_hop, plans[0].n_hop))
                if o == "verified" and not hit and len(broke) < 4:
                    broke.append((q, plan.n_hop, plans[0].n_hop, gold))
        for o in sorted(tot, key=lambda k: -tot[k]):
            row = "  ".join(f"{b}={seg[(o, b)]}" for b in ("unique", "ambiguous", "none", "no_chain_frame"))
            log(f"    {o:20s} {row}   unique&gold_nhop: {agree[(o, True)]} right / {agree[(o, False)]} wrong")
        log(f"    misparsed chains whose UNIQUE segmentation has the gold hop count: "
            f"{agree[('misparsed_chain', True)]} / {n_mis}")
        for q, was, now in fixed:
            log(f"      fixed  {q!r}  grammar n_hop {was} -> segment {now}")
        for q, was, now, gold in broke:
            log(f"      BROKE  {q!r}  grammar {was} -> segment {now}, gold {gold}")
        seg_all["strict" if strict else "exhaustive"] = (dict(seg), dict(agree))

    # ---- Q4: what survives the disposer? --------------------------------------
    # The failures that survive lookup-first (cot_harvest_lookup.jsonl, not verified),
    # split by whether verify_plan refuses them. What it does NOT refuse is the honest
    # residue -- the only part a better retriever or emitter could still win.
    lk_recs = [json.loads(l) for l in (LOGS / "cot_harvest_lookup.jsonl").open(encoding="utf-8")]
    post = {r["question"]: r for r in lk_recs}
    surviving = [r for r in v3 if r["question"] in post and not post[r["question"]].get("verified")]
    q4 = Counter(); residue_ex = []
    for r in surviving:
        o = outcome(r); plan = pl.parse_question(r["question"])
        if plan is None:
            q4[(o, "unparseable")] += 1; continue
        v = pv.verify_plan(r["question"], plan, known)
        q4[(o, "refused" if not v.ok else "residue")] += 1
        if v.ok and len(residue_ex) < 6:
            residue_ex.append((o, r["question"]))
    n_ref = sum(c for (o, b), c in q4.items() if b == "refused")
    n_res = sum(c for (o, b), c in q4.items() if b == "residue")
    log(f"\nQ4  after the disposer: {len(surviving)} failures survive lookup-first")
    log(f"    refused at plan time (no walk, no VM)  : {n_ref:4d}  ({n_ref/max(1,len(surviving)):.0%})")
    log(f"    residue (plan verifies, walk still fails): {n_res:4d}  ({n_res/max(1,len(surviving)):.0%})")
    for o in sorted({o for o, _ in q4}, key=lambda o: -sum(c for (oo, _), c in q4.items() if oo == o)):
        log(f"      {o:20s} refused={q4[(o,'refused')]:3d}  residue={q4[(o,'residue')]:3d}  unparseable={q4[(o,'unparseable')]}")
    for o, q in residue_ex:
        log(f"      residue e.g. [{o}] {q!r}")

    if a.vm:
        log(f"\nVM vs host: {vm.n_calls} QUERY calls (distinct relations), "
            f"{len(known.disagreements)} disagreements")
        for rel, h, v in known.disagreements[:10]:
            log(f"    DISAGREE {rel!r}: host={h} vm={v}")
        assert not known.disagreements, "the VM and the host-side reference disagree -- see log"
    wall = time.perf_counter() - t0
    out = {"mode": mode, "vm": bool(a.vm), "vm_calls": (vm.n_calls if a.vm else 0), "store": len(store), "relations": len(host), "parsed": len(parsed),
           "covers": dict(cov), "rejected": dict(rej), "total": dict(tot),
           "segment": {m: {f"{o}|{b}": v for (o, b), v in sg.items()} for m, (sg, _a) in seg_all.items()},
           "unique_gold_agree": {m: {f"{o}|{h}": v for (o, h), v in ag.items()} for m, (_s, ag) in seg_all.items()},
           "after_disposer": {"surviving": len(surviving), "refused": n_ref, "residue": n_res,
                              "by_outcome": {f"{o}|{b}": c for (o, b), c in q4.items()}},
           "wall_s": round(wall, 2)}
    tag = "" if mode == "real" else "_observed"
    (LOGS / f"exp_r6_plan_verify{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (LOGS / f"exp_r6_plan_verify{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.2f}s")


if __name__ == "__main__":
    main()
