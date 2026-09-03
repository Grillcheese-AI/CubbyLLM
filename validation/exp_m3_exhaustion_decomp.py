"""exp_m3_exhaustion_decomp — why do 246 of 800 harvest walks exhaust retrieval?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

The GoT challenge pre-checks (docs/research/2026-09-03-got-challenge-scored.md
§0) showed the chain path fails by finding nothing above `tau_ret`, not by
choosing the wrong candidate. The harvest logs candidates only for ACCEPTED
hops, so the failing hop has to be replayed. This script rebuilds the eval
store, the encoder and the retriever exactly as exp_m3_cot_pipeline.py does
(same seed), replays every `retrieval_exhausted` question hop by hop with an
unbounded acceptance scan, and classifies the failing hop:

  threshold_bound   a fact that SERVES the hop is retrieved, best score in
                    [floor, tau_ret)  -> the lower-tau_ret experiment applies
  far_below         a serving fact is retrieved but below `floor`
  not_retrieved     the gold chain's fact is in the store but no serving fact
                    appears in the top-K for the forward query
  absent            the gold chain's fact is not in the store (impossible by construction here:
                    the eval store IS the sample's own fact pool; kept for other stores)

and for every not-retrieved / far-below case runs Gemini's backward check:
does querying from the OBJECT side (the gold fact's object text) retrieve the
gold fact above tau_ret?

  python validation/exp_m3_exhaustion_decomp.py            # -> validation/logs/exp_m3_exhaustion_decomp.{json,log}
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import exp_m3_cot_pipeline as m  # noqa: E402
from cubbyllm.reasoning.pipeline import _accept  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question  # noqa: E402

HARVEST = ROOT / "validation" / "logs" / "cot_harvest_v3cf.jsonl"
OUT_JSON = ROOT / "validation" / "logs" / "exp_m3_exhaustion_decomp.json"
OUT_LOG = ROOT / "validation" / "logs" / "exp_m3_exhaustion_decomp.log"
TOP_K = 50
FLOOR = 0.50


def main() -> None:
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    recs = [json.loads(l) for l in open(HARVEST, encoding="utf-8")]
    taus = recs[0].get("taus") or {}
    tau_ret = float(taus.get("tau_ret", 0.5959))
    exhausted = {r["question"]: r for r in recs if r.get("reason") == "retrieval_exhausted"}
    log(f"harvest {HARVEST.name}: {len(recs)} records, {len(exhausted)} retrieval_exhausted, tau_ret {tau_ret:.4f}")

    sw = m._load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(m.V4_TABLE)
    questions, answers, hops, chains, store = m.load_sample(800, seed=0)
    retrieve = m.make_retriever(store, enc)
    store_norm = {normalize(f) for f in store}
    log(f"eval sample rebuilt: {len(questions)} questions, {len(store)} store facts (seed 0, as the harvest run)")

    def gold_facts(i: int) -> list[str]:
        return [store[j] for j in chains[i]]          # chains are indices into the sample's own fact pool

    cls: Counter = Counter()
    misparse_forms: Counter = Counter()
    misparse_phrases: Counter = Counter()
    per_hop_fail: Counter = Counter()
    scores_bound: list[float] = []
    backward_hits = backward_tries = 0
    details: list[dict] = []
    for i, q in enumerate(questions):
        if q not in exhausted:
            continue
        plan = parse_question(q)
        if plan is None:
            cls["unparseable_now"] += 1
            continue
        entity = None
        hop = 0
        outcome = None
        gold = gold_facts(i)
        gold_norm = [normalize(g) for g in gold]
        while hop < plan.n_hop:
            query = f"what is the {plan.tail}" if hop == 0 else f"{entity} {plan.relations[hop]}"
            best = None                                # (score, fact, triple) of the best SERVING candidate
            for score, fact in retrieve(query, TOP_K):
                t = parse_fact(fact)
                if t is not None and _accept(plan, hop, entity, t):
                    best = (float(score), fact, t)
                    break                              # results come sorted by score
            if best is None:
                # the gold fact for this hop: in the store? retrievable from the object side?
                g = gold_norm[hop] if hop < len(gold_norm) else None
                in_store = bool(g) and g in store_norm
                kind = "not_retrieved" if in_store else "absent"
                bw = None
                if in_store and plan.n_hop == len(chains[i]):   # the hop index only aligns with the gold chain when the parse is right
                    gt = parse_fact(gold[hop])
                    if gt is not None:
                        backward_tries += 1
                        bw_hits = [(float(s), f) for s, f in retrieve(gt.obj, TOP_K) if normalize(f) == g]
                        bw = bw_hits[0][0] if bw_hits else None
                        if bw is not None and bw >= tau_ret:
                            backward_hits += 1
                outcome = (kind, hop, None, bw)
                break
            score, fact, t = best
            if score < tau_ret:
                kind = "threshold_bound" if score >= FLOOR else "far_below"
                if kind == "threshold_bound":
                    scores_bound.append(score)
                bw = None
                if kind == "far_below":
                    backward_tries += 1
                    bw_hits = [(float(s), f) for s, f in retrieve(t.obj, TOP_K) if f == fact]
                    bw = bw_hits[0][0] if bw_hits else None
                    if bw is not None and bw >= tau_ret:
                        backward_hits += 1
                outcome = (kind, hop, score, bw)
                break
            entity = t.obj
            hop += 1
        if outcome is None:
            outcome = ("walk_completes_now", hop, None, None)   # every hop served above tau: not exhausted on replay
        kind, hop_i, score, bw = outcome
        if kind in ("not_retrieved", "absent") and plan.n_hop != len(chains[i]):
            kind = "misparsed_chain"                    # the grammar parsed a different hop count than the gold chain has
            misparse_forms[(plan.n_hop, len(chains[i]))] += 1
            misparse_phrases[" ".join(q.split()[:5]).lower()] += 1
        cls[kind] += 1
        per_hop_fail[hop_i] += 1
        details.append({"question": q, "gold_answer": answers[i], "n_hop": plan.n_hop, "kind": kind, "hop": hop_i,
                        "serving_score": score, "backward_score": bw})

    n = sum(cls.values())
    log(f"\nclassified {n} exhausted walks:")
    for k, v in cls.most_common():
        log(f"  {k:20s} {v:4d}  ({v / max(1, n):.0%})")
    log(f"failing hop index: {dict(sorted(per_hop_fail.items()))}")
    if misparse_forms:
        log(f"misparsed chains (parsed n_hop, gold n_hop): {dict(misparse_forms)}")
        log(f"misparsed openers: {misparse_phrases.most_common(8)}")
        ex_mis = [d for d in details if d["kind"] == "misparsed_chain"][:4]
        for d in ex_mis:
            log(f"  misparsed e.g. parsed {d['n_hop']} hops | {d['question'][:120]}")
    if scores_bound:
        sb = sorted(scores_bound)
        log(f"threshold_bound serving scores: min {sb[0]:.4f} p50 {sb[len(sb) // 2]:.4f} max {sb[-1]:.4f} "
            f"(tau_ret {tau_ret:.4f}; a tau of {sb[0]:.4f} would admit all of them)")
    log(f"backward (object-side) retrieval on the not_retrieved/far_below cases: {backward_hits}/{backward_tries} "
        f"reach tau_ret")
    log(f"wall {time.perf_counter() - t0:.1f}s")
    for d in details[:8]:
        log(f"  e.g. [{d['kind']}] hop {d['hop']} score {d['serving_score']} bw {d['backward_score']} | {d['question'][:90]}")

    out = {"harvest": str(HARVEST), "tau_ret": tau_ret, "floor": FLOOR, "top_k": TOP_K, "n_exhausted": len(exhausted),
           "classified": dict(cls), "failing_hop": {str(k): v for k, v in per_hop_fail.items()},
           "misparsed_forms": {f"{k[0]}->{k[1]}": v for k, v in misparse_forms.items()},
           "misparsed_openers": misparse_phrases.most_common(12),
           "threshold_bound_scores": sorted(scores_bound),
           "backward": {"hits": backward_hits, "tries": backward_tries},
           "git_rev": m._git_rev(), "wall_s": time.perf_counter() - t0, "details": details}
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
