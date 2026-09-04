"""exp_m4_triple_lookup — retrieval as SEARCH (cosine top-k) vs retrieval as LOOKUP (a triple index).

Wired: STANDALONE (validation script; never imported by cubbyllm/).

The chain walk (cubbyllm/reasoning/pipeline._walk) retrieves the top-k facts by
fastword cosine for a hop query, then filters them with an EXACT acceptance
test on the parsed triple (`_accept`: the tail string at hop 0, relation +
subject afterwards). The 246/800 `retrieval_exhausted` walks in the harvest
(exp_m3_exhaustion_decomp) are the cases where the serving fact never reached
the top-k above tau_ret — the search half failing at a job the acceptance
half already defines exactly. Every fact in the store is a template triple, so
the same acceptance test can be run as an INDEX LOOKUP: (relation, subject)
-> facts, no threshold, no k. This script measures both on the same walks:

  cosine@3>=tau   the live operating point (top_k=3, tau_ret 0.5959): a serving fact is in the top 3 above tau
  cosine@3        ... in the top 3 at any score
  cosine@50       ... in the top 50 at any score (the ceiling of "search harder")
  lookup          the (relation, subject) index holds a serving fact
  ambiguous       the index holds >1 serving fact (where a frontier walk would branch)

and walks each question end to end with the index (a frontier over all
candidates), scoring the final entity set against the gold answer.

  python validation/exp_m4_triple_lookup.py     # -> validation/logs/exp_m4_triple_lookup.{json,log}
"""
from __future__ import annotations

import collections
import json
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import exp_m3_cot_pipeline as m  # noqa: E402
from cubbyllm.reasoning.pipeline import _accept  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402

HARVEST = ROOT / "validation" / "logs" / "cot_harvest_v3cf.jsonl"
OUT_JSON = ROOT / "validation" / "logs" / "exp_m4_triple_lookup.json"
OUT_LOG = ROOT / "validation" / "logs" / "exp_m4_triple_lookup.log"
K_LIVE = 3          # ReasoningCortex.k_facts at serve
K_WIDE = 50


class TripleIndex:
    """(relation, subject) -> facts, over parseable template facts. The lookup half of FactStore-to-be."""

    def __init__(self, facts: list[str]) -> None:
        self.by_tail: dict[str, list[tuple[str, object]]] = collections.defaultdict(list)   # normalize("rel of subj")
        self.by_subj: dict[str, list[tuple[str, object]]] = collections.defaultdict(list)   # normalize(subj)
        self.n_parsed = 0
        for f in facts:
            t = parse_fact(f)
            if t is None:
                continue
            self.n_parsed += 1
            self.by_tail[normalize(f"{t.rel} of {t.subj}")].append((f, t))
            self.by_subj[normalize(t.subj)].append((f, t))

    def hop(self, plan, hop: int, entity: str | None) -> list[tuple[str, object]]:
        if hop == 0:
            return list(self.by_tail.get(normalize(plan.tail), []))
        expected = plan.relations[hop]
        return [(f, t) for f, t in self.by_subj.get(normalize(entity or ""), []) if relation_matches(expected, t.rel)]


def main() -> None:
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    log(f"exp_m4_triple_lookup  git {rev}  python {platform.python_version()}  {platform.platform()}")
    recs = {r["question"]: r for r in (json.loads(l) for l in open(HARVEST, encoding="utf-8"))}
    tau_ret = float(next(iter(recs.values())).get("taus", {}).get("tau_ret", 0.5959))
    sw = m._load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(m.V4_TABLE)
    questions, answers, hops, chains, store = m.load_sample(800, seed=0)
    retrieve = m.make_retriever(store, enc)
    index = TripleIndex(store)
    log(f"harvest {HARVEST.name}: {len(recs)} records, tau_ret {tau_ret:.4f}; store {len(store)} facts, "
        f"{index.n_parsed} parse as triples ({100 * index.n_parsed / len(store):.1f}%)")

    hop_stats = collections.Counter()
    hop_n = 0
    walk = collections.Counter()
    by_reason = collections.defaultdict(collections.Counter)
    examples: list[dict] = []
    for i, q in enumerate(questions):
        rec = recs.get(q, {})
        reason = rec.get("reason") or ("verified" if rec.get("verified") else "other")
        plan = parse_question(q)
        if plan is None:
            walk["unparseable"] += 1
            by_reason[reason]["unparseable"] += 1
            continue
        gold_chain = [store[j] for j in chains[i]]
        # --- the same hops the live walk would take, entity from the GOLD chain (so every hop is measured, not
        #     only the ones a failing walk reached) ---
        entity = None
        ok_chain = True
        for h in range(plan.n_hop):
            hop_n += 1
            query = f"what is the {plan.tail}" if h == 0 else f"{entity} {plan.relations[h]}"
            wide = retrieve(query, K_WIDE)
            serving = [(s, f) for s, f in wide if (t := parse_fact(f)) is not None and _accept(plan, h, entity, t)]
            top3 = wide[:K_LIVE]
            if any((t := parse_fact(f)) is not None and _accept(plan, h, entity, t) and s >= tau_ret for s, f in top3):
                hop_stats["cosine@3>=tau"] += 1
            if any((t := parse_fact(f)) is not None and _accept(plan, h, entity, t) for s, f in top3):
                hop_stats["cosine@3"] += 1
            if serving:
                hop_stats["cosine@50"] += 1
            cands = index.hop(plan, h, entity)
            if cands:
                hop_stats["lookup"] += 1
                if len(cands) > 1:
                    hop_stats["ambiguous"] += 1
            else:
                ok_chain = False
                if len(examples) < 12:
                    examples.append({"q": q, "hop": h, "query": query, "entity": entity, "gold_fact": gold_chain[h] if h < len(gold_chain) else None,
                                     "reason": reason})
            # advance along the gold chain, as the harvest's accepted walk would
            gt = parse_fact(gold_chain[h]) if h < len(gold_chain) else None
            if gt is None or not _accept(plan, h, entity, gt):
                hop_stats["gold_chain_not_accepted_by_plan"] += 1
                ok_chain = False
                break
            entity = gt.obj
        # --- end to end with the index alone: a frontier over every candidate ---
        frontier = {None}
        dead = False
        for h in range(plan.n_hop):
            nxt = set()
            for e in frontier:
                for f, t in index.hop(plan, h, e):
                    nxt.add(t.obj)
            if not nxt:
                dead = True
                break
            frontier = nxt
        gold = normalize(answers[i])
        if dead:
            out = "lookup_dead"
        elif gold in {normalize(e) for e in frontier}:
            out = "lookup_correct" if len(frontier) == 1 else "lookup_correct_in_set"
        else:
            out = "lookup_wrong"
        walk[out] += 1
        by_reason[reason][out] += 1

    log()
    log(f"per hop (n={hop_n} hops over {len(questions)} questions, gold-chain entities):")
    for k in ("cosine@3>=tau", "cosine@3", "cosine@50", "lookup", "ambiguous", "gold_chain_not_accepted_by_plan"):
        log(f"  {k:32s} {hop_stats[k]:5d}  {100 * hop_stats[k] / hop_n:5.1f}%")
    log()
    log("end to end, the index alone (frontier walk):")
    for k, v in walk.most_common():
        log(f"  {k:24s} {v:4d}  {100 * v / len(questions):5.1f}%")
    log()
    log("by the harvest's outcome (cosine walk + VM):")
    for reason, c in sorted(by_reason.items(), key=lambda kv: -sum(kv[1].values())):
        log(f"  {reason:22s} n={sum(c.values()):3d}  " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    log()
    for e in examples[:8]:
        log(f"  e.g. lookup dead [{e['reason']}] hop {e['hop']} query={e['query']!r} gold_fact={e['gold_fact']!r}")
    wall = time.perf_counter() - t0
    log(f"wall {wall:.1f}s")
    OUT_JSON.write_text(json.dumps({"git": rev, "tau_ret": tau_ret, "store": len(store), "parsed": index.n_parsed,
                                    "hops": hop_n, "hop_stats": dict(hop_stats), "walk": dict(walk),
                                    "by_reason": {k: dict(v) for k, v in by_reason.items()}, "examples": examples,
                                    "wall_s": round(wall, 1)}, indent=1), encoding="utf-8")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON.name, OUT_LOG.name)


if __name__ == "__main__":
    main()
