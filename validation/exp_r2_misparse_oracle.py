"""exp_r2_misparse_oracle — are the 92 misparses a PARSE failure or a RETRIEVAL failure?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Why this exists
---------------
`exp_m3_exhaustion_decomp.py` classified 246 exhausted walks, and 92 of them
(37%) came back `misparsed_chain` — the grammar parsed a different hop count
than the gold chain has. Under lookup-first those 92 are 100% still open
(`exp_r1_gate_diagnostics.py`), which makes them the single largest surviving
failure class.

They were never backward-probed, and that was deliberate, not an oversight.
The decomposition guards its probe with

    if in_store and plan.n_hop == len(chains[i]):   # the hop index only aligns
                                                    # with the gold chain when
                                                    # the parse is right

and then *defines* `misparsed_chain` as exactly the case where that guard is
false. So `misparsed_chain` <=> the probe was skipped, by construction.

The guard's reasoning is correct — with a wrong hop count, `gold[hop]` is the
wrong fact — but it leaves the decisive question unanswered:

    WOULD THE WALK HAVE SUCCEEDED IF THE PARSE HAD BEEN RIGHT?

That is what decides whether the program emitter has a real target here.

  - If yes: the facts were retrievable and the parse was the binding
    constraint. Getting the frame right is worth up to 92 of the 195 surviving
    failures, and that IS the emitter's job.
  - If no: retrieval would have failed anyway. The 92 are retrieval failures
    wearing a parser's label, they belong with the backward-retrieval fix, and
    training an emitter for them buys nothing.

Design
------
Take the parse out of the loop entirely. For each misparsed question, use the
GOLD chain as an oracle and replay the walk the pipeline *would* have made,
reconstructing each hop's query exactly as `pipeline._walk` does:

    hop 0 : "what is the {rel} of {subj}"     (the canonical tail)
    hop k : "{entity} {rel}"                   (entity = previous hop's object)

Three arms, all on the same rebuilt store and encoder, same seed:

  A  forward-oracle   the gold chain's own forward queries, unbounded scan.
                      Serves the hop above tau_ret?
  B  backward         query from the gold fact's OBJECT side (the probe the
                      decomposition runs for not_retrieved/far_below), now run
                      on the misparses too.
  C  split-probe      would a cheaper chain-splitter have got the hop count
                      right? Measured, not assumed: relations themselves
                      contain "of" ("country of citizenship"), which is why
                      the grammar splits on " of the " in the first place.

No GPU. No training. Reuses exp_m3_cot_pipeline's own loaders so the store,
encoder, sample and seed are identical to the harvest run.

  python validation/exp_r2_misparse_oracle.py
      -> validation/logs/exp_r2_misparse_oracle.{json,log}
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import exp_m3_cot_pipeline as m  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question  # noqa: E402

DECOMP = ROOT / "validation" / "logs" / "exp_m3_exhaustion_decomp.json"
OUT_JSON = ROOT / "validation" / "logs" / "exp_r2_misparse_oracle.json"
OUT_LOG = ROOT / "validation" / "logs" / "exp_r2_misparse_oracle.log"
TOP_K = 50


def canonical_queries(gold_triples):
    """The queries pipeline._walk would issue given a correct parse.
    hop 0 takes the canonical tail 'R of E'; later hops are '{entity} {rel}'."""
    qs = []
    for i, t in enumerate(gold_triples):
        if i == 0:
            qs.append(f"what is the {t.rel} of {t.subj}")
        else:
            qs.append(f"{gold_triples[i - 1].obj} {t.rel}")
    return qs


def main() -> None:
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    if not DECOMP.exists():
        sys.exit(f"missing {DECOMP} — run exp_m3_exhaustion_decomp.py first")
    decomp = json.loads(DECOMP.read_text(encoding="utf-8"))
    tau_ret = float(decomp["tau_ret"])
    misparsed = {d["question"] for d in decomp["details"] if d["kind"] == "misparsed_chain"}
    log(f"decomp {DECOMP.name}: {decomp['n_exhausted']} exhausted, "
        f"{len(misparsed)} misparsed_chain, tau_ret {tau_ret:.4f}, git_rev {decomp['git_rev'][:12]}")

    sw = m._load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(m.V4_TABLE)
    questions, answers, hops, chains, store = m.load_sample(800, seed=0)
    retrieve = m.make_retriever(store, enc)
    log(f"eval sample rebuilt: {len(questions)} questions, {len(store)} store facts (seed 0, as the harvest run)")

    rows = []
    armA = Counter()   # forward-oracle, per question
    armB = Counter()   # backward, per question
    hopA = Counter()   # forward-oracle, per hop
    hopB = Counter()
    split_fix = Counter()

    for i, q in enumerate(questions):
        if q not in misparsed:
            continue
        gold_facts = [store[j] for j in chains[i]]
        gold_triples = [parse_fact(f) for f in gold_facts]
        if any(t is None for t in gold_triples):
            armA["gold_unparseable"] += 1
            continue
        plan = parse_question(q)
        queries = canonical_queries(gold_triples)

        # ---- arm A: forward query, oracle chain, unbounded scan
        per_hop = []
        for h, (query, gf, gt) in enumerate(zip(queries, gold_facts, gold_triples)):
            gnorm = normalize(gf)
            hit = None
            for score, fact in retrieve(query, TOP_K):
                if normalize(fact) == gnorm:
                    hit = float(score)
                    break
            served = hit is not None and hit >= tau_ret
            per_hop.append({"hop": h, "query": query, "score": hit, "served": served})
            hopA["served" if served else ("below_tau" if hit is not None else "not_retrieved")] += 1
        a_ok = all(p["served"] for p in per_hop)
        armA["all_hops_served" if a_ok else "at_least_one_hop_fails"] += 1

        # ---- arm B: backward (object-side), every gold fact in the chain
        per_hop_bw = []
        for h, (gf, gt) in enumerate(zip(gold_facts, gold_triples)):
            gnorm = normalize(gf)
            bw = None
            for score, fact in retrieve(gt.obj, TOP_K):
                if normalize(fact) == gnorm:
                    bw = float(score)
                    break
            ok = bw is not None and bw >= tau_ret
            per_hop_bw.append({"hop": h, "score": bw, "served": ok})
            hopB["served" if ok else ("below_tau" if bw is not None else "not_retrieved")] += 1
        b_ok = all(p["served"] for p in per_hop_bw)
        armB["all_hops_served" if b_ok else "at_least_one_hop_fails"] += 1

        # ---- arm C: would a different chain split have got the hop count right?
        gold_n = len(chains[i])
        body = q.strip().rstrip("?").strip()
        for name, pat in (("of_the", r"\s+of\s+the\s+"),
                          ("of_any", r"\s+of\s+"),
                          ("of_the_or_in", r"\s+of\s+the\s+|\s+contained\s+within\s+")):
            n = len(re.split(pat, body, flags=re.I))
            split_fix[f"{name}:{'RIGHT' if n == gold_n else 'wrong'}"] += 1

        rows.append({"question": q, "parsed_n_hop": plan.n_hop if plan else None,
                     "gold_n_hop": gold_n, "gold_answer": answers[i],
                     "forward_oracle_ok": a_ok, "backward_ok": b_ok,
                     "forward_hops": per_hop, "backward_hops": per_hop_bw})

    n = len(rows)
    log(f"\nreplayed {n} misparsed questions (oracle = the gold chain)\n")

    log("ARM A — forward query with the CORRECT chain (the parse taken out):")
    for k, v in armA.most_common():
        log(f"  {k:28s} {v:4d}  ({v / max(1, n):.0%})")
    log(f"  per hop: {dict(hopA)}")

    log("\nARM B — backward (object-side) retrieval, now run on the misparses:")
    for k, v in armB.most_common():
        log(f"  {k:28s} {v:4d}  ({v / max(1, n):.0%})")
    log(f"  per hop: {dict(hopB)}")

    either = sum(1 for r in rows if r["forward_oracle_ok"] or r["backward_ok"])
    both = sum(1 for r in rows if r["forward_oracle_ok"] and r["backward_ok"])
    log(f"\n  recoverable by EITHER direction: {either}/{n}  ({either / max(1, n):.0%})")
    log(f"  recoverable by BOTH:             {both}/{n}")

    log("\nARM C — would a cheaper chain splitter have got the hop count right?")
    for k in sorted(split_fix):
        log(f"  {k:34s} {split_fix[k]:4d}")

    log("\n" + "=" * 72)
    a_ok_n = armA["all_hops_served"]
    if n:
        if a_ok_n / n >= 0.5:
            log("VERDICT: the facts WERE retrievable once the chain was right.")
            log(f"  {a_ok_n}/{n} misparsed questions complete under an oracle parse.")
            log("  The parse is the binding constraint, and getting the frame right is")
            log("  the emitter's job. This class is a REAL emitter target.")
        elif a_ok_n / n <= 0.2:
            log("VERDICT: fixing the parse does NOT rescue these walks.")
            log(f"  only {a_ok_n}/{n} complete even with the gold chain handed to them.")
            log("  These are retrieval failures wearing a parser's label. They belong")
            log("  with the backward-retrieval fix; an emitter buys little here.")
        else:
            log(f"VERDICT: mixed — {a_ok_n}/{n} complete under an oracle parse.")
            log("  Split the class and treat the two halves separately before training.")
    log("=" * 72)

    out = {"decomp": str(DECOMP), "tau_ret": tau_ret, "top_k": TOP_K,
           "n_misparsed": len(misparsed), "n_replayed": n,
           "arm_a_forward_oracle": dict(armA), "arm_a_per_hop": dict(hopA),
           "arm_b_backward": dict(armB), "arm_b_per_hop": dict(hopB),
           "recoverable_either": either, "recoverable_both": both,
           "arm_c_split_probe": dict(split_fix),
           "git_rev": m._git_rev(), "wall_s": time.perf_counter() - t0, "details": rows}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.1f}s")
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
