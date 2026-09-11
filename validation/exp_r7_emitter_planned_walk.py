"""exp_r7_emitter_planned_walk — the emitter's plan, the grammar's walk, the VM's verdict.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

exp_r3 (2026-09-11) showed emitter_v8e proposing plans on the 92 questions the
grammar misparses and the 37 it cannot parse: 45/92 with the gold hop count,
18/92 accepted by the disposer with the gold hop count. Those questions were
NEVER in v8e's training data (checked: 0/92 and 0/37 in the v8e chain records;
the emitter was trained on the grammar's 517 verified chains, ×3). So they are
plans from outside the grammar's basin. A plan is not an answer. This script
walks them.

For every well-formed emitted program in the exp_r3 json:

    roles H1_REL1, H2_REL2, ...  ->  QuestionPlan(relations=[None, REL2, ...],
                                                  tail="REL1 of <seed entity>")
    answer(question, plan=that, known=StoreRelations(store), lookup=index, ...)

The seed entity is the question text after the LAST "<rel1> of " (normalized),
falling back to the last " of "-segment of the grammar's own tail when the
grammar parses the question at all. Neither is the emitter's doing: the
emitted program carries objects, not the seed, and the seed is the one thing
the question states verbatim.

Same store, same index, same tau floors and tau_ret as the lookup_vp_res
harvest (read from its json), the VM resident. The disposer runs on the
emitter's plan exactly as on the grammar's. What comes out per arm:

    built      a plan could be formed (well-formed program + a seed entity)
    refused    the disposer refused it (unknown relation / does not cover)
    walked     the walk ran
    verified   the VM verified a chain -- a claimed answer
    correct    ... and it equals the gold answer

On arms B and C the grammar's verified count is 0 by construction. Every
verified answer here is a question the pipeline could not answer before the
emitter proposed the plan -- with the VM, not the emitter, saying so. The
verified chains are written out as harvest records (`cot_harvest_r7.jsonl`):
the first training data that did not come from the grammar.

FIRST RUN (2026-09-11, covers v1) AND WHAT IT EXPOSED
-----------------------------------------------------
    A  built 93  refused 6   walked 87  verified 84  correct 83   (training set: sanity)
    B  built 79  refused 70  walked  9  verified  7  correct  3   verified_wrong 4
    C  built  0  (18 malformed, 19 no seed entity: nested relative clauses)

Two faults in the DISPOSER, not the emitter, both fixed in plan_verify.covers v2:
  * 65 of the 70 B refusals were `plan_does_not_cover_question` because v1
    required the grammar's literal " of the " body to be a substring of the
    question -- so it could only accept plans inside the grammar's basin. 13
    of those plans had the gold hop count and were refused for the joint
    ("contained within the", "received by the") alone.
  * the 4 verified-WRONG answers were 1-hop plans that had DROPPED the outer
    hop ("source that describes the country of X" planned as "country of X");
    v1's 1-hop rule (relation and entity both present) cannot see a dropped
    hop, the walk verified the binding faithfully, and the answer was wrong.
  v2 is joint-agnostic (relations in order, then the entity, anything
  between) and hop-complete (what is left of the question after removing them
  must contain no word any store relation is made of). Re-predicted on the
  same emitted plans: B accepts 16, all with the gold hop count, and refuses
  every dropped-hop plan; 22 gold-hop-count plans stay refused because the
  emitter's role names paraphrase the question's relation ('award' for
  'award received', 'history of' for 'history of the topic') -- an
  emitter-side loss, left refused on purpose.

  python validation/exp_r7_emitter_planned_walk.py \
      [--r3 validation/logs/exp_r3_emitter_floor_emitter_v8e.Q4_K_M.json]
      [--taus validation/logs/exp_m3_cot_pipeline_lookup_vp_res.json] [--resident]
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys, time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROLE = re.compile(r"^H(?P<hop>\d+)_(?P<rel>.+)$")


def rel_text(role: str) -> str:
    m = ROLE.match(role)
    return m.group("rel").lower().replace("_", " ").strip() if m else ""


# frame words a 1-hop inversion spends around the entity; never part of it
_SEED_FRAME = frozenset("""
what which where who whom is was are were the a an of to in on at by for that does do did
have has had belong belongs use uses compete competes receive receives get gets
""".split())


def seed_entity(question: str, rels: list[str], grammar_plan, normalize) -> str | None:
    """The seed entity of a question given the emitted relations (gen 1 plans carry
    no SEED; gen 2's CotPlan does and bypasses this).
      1. text after the LAST '<rel1> of ' -- the chain frame;
      2. else the last ' of '-segment of the grammar's tail;
      3. else (2026-09-11, arm C) the RESIDUAL: remove the relations and the frame
         words from the question; what is left, contiguous, is the entity --
         'What is erik bergvall a participant of?' -> 'erik bergvall',
         'Which languages spoken, written or signed by Dwight Loomis?' -> 'dwight loomis'.
         The same residual idea covers() v2 uses to detect a dropped hop."""
    q = normalize(question)
    key = f"{normalize(rels[0])} of "
    i = q.rfind(key)
    if i >= 0 and q[i + len(key):].strip():
        return q[i + len(key):].strip()
    if grammar_plan is not None and " of " in grammar_plan.tail:
        return normalize(grammar_plan.tail.rsplit(" of ", 1)[1])
    body = q
    for r in sorted((normalize(r) for r in rels), key=len, reverse=True):
        body = body.replace(r, " ", 1)
    words = [w for w in body.split() if w not in _SEED_FRAME]
    return " ".join(words) if words else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3", default=str(LOGS / "exp_r3_emitter_floor_emitter_v8e.Q4_K_M.json"))
    ap.add_argument("--taus", default=str(LOGS / "exp_m3_cot_pipeline_lookup_vp_res.json"))
    ap.add_argument("--harvest", default=str(LOGS / "cot_harvest_v3cf.jsonl"), help="gold answers")
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude", default=None, help="gen2_exclusions.json -- drop trained questions from arms B/C")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import exp_m3_cot_pipeline as m
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning import TripleIndex, answer, parse_question
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from cubbyllm.reasoning.plan_verify import StoreRelations

    r3 = json.loads(pathlib.Path(a.r3).read_text(encoding="utf-8"))
    taus = json.loads(pathlib.Path(a.taus).read_text(encoding="utf-8"))["calibration"]
    floors = {int(k): float(v) for k, v in taus["tau_vm_floors"].items()}
    fallback = float(taus["fallback_tau_vm"]); tau_ret = float(taus["tau_ret"])
    gold = {r["question"]: r for r in (json.loads(l) for l in open(a.harvest, encoding="utf-8"))}

    sw = m._load_semantic_words(); enc = sw.FastWordEncoder.from_npz(m.V4_TABLE)
    questions, answers, hops, chains, store = m.load_sample(800, seed=0)
    retrieve = m.make_retriever(store, enc)
    index = TripleIndex(store); known = StoreRelations(store)
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    def run_fn(source, fn):
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    log(f"emitter plans from {pathlib.Path(a.r3).name} | store {len(store)} facts, {len(known)} relations | "
        f"tau_vm floors {floors} fallback {fallback:.4f} | tau_ret {tau_ret:.4f} | VM {'resident' if session else 'spawn'}")

    excluded = set()
    if a.exclude:
        excluded = set(json.loads(pathlib.Path(a.exclude).read_text(encoding="utf-8"))["questions"])
        log(f"excluding {len(excluded)} trained questions from arms B/C ({a.exclude})")
    out_recs = []; per_arm = {}
    for arm in ("A_control", "B_misparsed", "C_unparseable"):
        c = Counter(); ex = []
        rows = [x for x in r3["rows"] if x["arm"] == arm and not (arm != "A_control" and x["question"] in excluded)]
        for x in rows:
            q = x["question"]; c["n"] += 1
            if not x.get("well_formed"):
                c["malformed"] += 1; continue
            rels = [rel_text(r) for r in x["roles"]]
            gplan = parse_question(q)
            ent = normalize(x["seed"]) if x.get("seed") else seed_entity(q, rels, gplan, normalize)
            if ent is None:
                c["no_seed_entity"] += 1; continue
            plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {ent}", n_hop=len(rels))
            c["built"] += 1
            res = answer(q, retrieve, run_fn, tau_vm=floors.get(plan.n_hop, fallback), tau_ret=tau_ret,
                         top_k=3, max_repairs=1, lookup=index.hop, known=known, plan=plan)
            if res.refused is not None:
                c["refused"] += 1; c[f"refused:{res.reason}"] += 1; continue
            c["walked"] += 1
            g = gold.get(q, {}).get("gold_answer")
            if res.verified:
                c["verified"] += 1
                ok = g is not None and normalize(res.answer) == normalize(g)
                c["correct" if ok else "verified_wrong"] += 1
                out_recs.append({"question": q, "arm": arm, "plan": {"relations": plan.relations, "tail": plan.tail,
                                 "n_hop": plan.n_hop}, "program_source": res.source, "answer": res.answer,
                                 "gold_answer": g, "correct": ok,
                                 "trace": [{"fact": h.fact, "symbol": h.symbol, "similarity": h.similarity,
                                            "source": h.source} for h in res.trace],
                                 "emitted_roles": x["roles"], "gold_n_hop": x.get("gold_n_hop"),
                                 "provenance": "emitter_v8e plan -> grammar walk -> VM verify (exp_r7)"})
                if len(ex) < 6:
                    ex.append((q, res.answer, g, ok))
            else:
                c[f"walk:{res.reason}"] += 1
        per_arm[arm] = dict(c)
        log(f"\n{arm}: n={c['n']}  malformed={c['malformed']}  no_seed={c['no_seed_entity']}  built={c['built']}  "
            f"refused={c['refused']}  walked={c['walked']}  VERIFIED={c['verified']}  correct={c['correct']}  "
            f"verified_wrong={c['verified_wrong']}")
        for k in sorted(k for k in c if k.startswith("refused:") or k.startswith("walk:")):
            log(f"    {k:32s} {c[k]}")
        for q, ans, g, ok in ex:
            log(f"    {'OK ' if ok else 'BAD'} {q[:90]!r} -> {ans!r} (gold {g!r})")

    b, cu = per_arm["B_misparsed"], per_arm["C_unparseable"]
    log("\n" + "=" * 76)
    log(f"questions the grammar could not answer, now VM-verified: "
        f"{b.get('verified',0) + cu.get('verified',0)} (correct {b.get('correct',0) + cu.get('correct',0)}) "
        f"of {b.get('n',0) + cu.get('n',0)}   [B {b.get('verified',0)}/{b.get('n',0)}, C {cu.get('verified',0)}/{cu.get('n',0)}]")
    log(f"arm A (the emitter's training questions, sanity): verified {per_arm['A_control'].get('verified',0)}"
        f"/{per_arm['A_control'].get('n',0)}, correct {per_arm['A_control'].get('correct',0)}")
    log("=" * 76)
    if session:
        log(f"resident VM: {session.n_requests} requests"); session.close()
    wall = time.perf_counter() - t0
    tag = a.tag
    (LOGS / f"cot_harvest_r7{tag}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_recs) + ("\n" if out_recs else ""), encoding="utf-8")
    (LOGS / f"exp_r7_emitter_planned_walk{tag}.json").write_text(json.dumps({"per_arm": per_arm, "taus": taus, "r3": a.r3,
        "n_harvest_records": len(out_recs), "wall_s": round(wall, 1)}, indent=1), encoding="utf-8")
    (LOGS / f"exp_r7_emitter_planned_walk{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.1f}s | wrote cot_harvest_r7{tag}.jsonl ({len(out_recs)} verified records)")


if __name__ == "__main__":
    main()
