#!/usr/bin/env python3
"""exp_r1_gate_diagnostics — the five zero-GPU queries that precede the emitter.

Rung 1 of the evidence ladder says Know -> Build -> Test. These are the "Know"
queries: every one is a scan or a groupby over logs that already exist, so none
of them costs a GPU-hour and any of them can change what the emitter is for.

Inputs (all in validation/logs/):
  cot_harvest_v3cf.jsonl          the 800-question harvest, pre-lookup   (coverage 0.646)
  cot_harvest_lookup.jsonl        the same eval under lookup-first        (coverage 0.710)
  exp_m3_exhaustion_decomp.json   the 246 classified failures, with per-case detail

Usage:  python validation/exp_r1_gate_diagnostics.py [--logs validation/logs] [--json OUT]

Every number this prints is a measurement over those files. Nothing is projected.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

# the six literal misparsed openers named in exp_m3_exhaustion_decomp.log
OPENERS = ["what is the source that", "which is the administrative territorial",
           "what is the instance of", "where is the history of",
           "who is the editor of", "which is the office held"]


def norm(s: str) -> str:
    """Comparison form. Deliberately shallow: the corpus is chain-consistently
    typo'd and normalisation is a consumer-side concern (harvest schema, Rules)."""
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def load(logs: Path):
    recs = [json.loads(l) for l in (logs / "cot_harvest_v3cf.jsonl").open(encoding="utf-8")]
    decomp = json.loads((logs / "exp_m3_exhaustion_decomp.json").read_text(encoding="utf-8"))
    lookup = None
    p = logs / "cot_harvest_lookup.jsonl"
    if p.exists():
        lookup = [json.loads(l) for l in p.open(encoding="utf-8")]
    return recs, decomp, lookup


def hop_candidates(rec):
    """candidates_topk is [[ [score, text], ... ], ...] — one list per walked hop."""
    c = rec.get("candidates_topk") or []
    return [h for h in c if h]


# ---------------------------------------------------------------- Q3
def q3_crosstab(decomp, out):
    print("\n" + "=" * 78)
    print("Q3  cause x failing-hop cross-tab")
    print("    Both marginals are published; the joint was never computed, and the")
    print("    two readings are opposite. If misparses die at hop 0 the hop count was")
    print("    never operative and 'misparsed_chain' is a mislabel.")
    print("=" * 78)
    det = decomp["details"]
    tab = defaultdict(Counter)
    for d in det:
        tab[d["kind"]][d["hop"]] += 1
    hops = sorted({d["hop"] for d in det})
    print(f"\n{'cause':<22}" + "".join(f"{'hop '+str(h):>9}" for h in hops) + f"{'total':>8}{'@hop0':>8}")
    rows = {}
    for kind in sorted(tab, key=lambda k: -sum(tab[k].values())):
        row = [tab[kind][h] for h in hops]
        tot = sum(row)
        share0 = tab[kind][0] / tot if tot else 0
        rows[kind] = {"by_hop": {str(h): tab[kind][h] for h in hops},
                      "total": tot, "share_at_hop0": round(share0, 4)}
        print(f"{kind:<22}" + "".join(f"{v:>9}" for v in row) + f"{tot:>8}{share0:>7.0%}")
    mis = rows.get("misparsed_chain", {})
    s0 = mis.get("share_at_hop0", 0)
    print(f"\n  VERDICT: {s0:.0%} of misparsed_chain failures occur at hop 0.")
    if s0 >= 0.5:
        print("  -> The chain died before hop-count planning could matter. 'misparsed_chain'")
        print("     is substantially a FIRST-HOP RETRIEVAL failure wearing a parser label,")
        print("     and that mass belongs with the retrieval bucket, not the emitter's.")
    else:
        print("  -> Misparses mostly fail late: hop-count planning is a real defect and")
        print("     is a legitimate target for the emitter.")
    out["q3_crosstab"] = rows
    return rows


# ---------------------------------------------------------------- Q1
def q1_gold_rank(recs, decomp, out):
    print("\n" + "=" * 78)
    print("Q1  is the gold answer already inside candidates_topk, and at what rank?")
    print("    Splits 'retrieval never surfaced it' (needs a backward walk) from")
    print("    'retrieval surfaced it and the cut discarded it' (needs re-ranking).")
    print("=" * 78)
    by_q = {r["question"]: r for r in recs}
    det = decomp["details"]
    stats = Counter()
    ranks, scores_found = [], []
    per_kind = defaultdict(Counter)
    for d in det:
        r = by_q.get(d["question"])
        kind = d["kind"]
        if r is None:
            stats["no_harvest_record"] += 1; continue
        cands = hop_candidates(r)
        if not cands:
            stats["no_candidate_data"] += 1; per_kind[kind]["no_candidate_data"] += 1; continue
        gold = norm(d.get("gold_answer"))
        if not gold:
            stats["no_gold"] += 1; continue
        stats["scanned"] += 1
        hit_rank, hit_score = None, None
        for hop in cands:
            for i, (sc, txt) in enumerate(hop):
                if gold and gold in norm(txt):
                    if hit_rank is None or i < hit_rank:
                        hit_rank, hit_score = i, sc
        if hit_rank is None:
            stats["gold_absent"] += 1; per_kind[kind]["gold_absent"] += 1
        else:
            stats["gold_present"] += 1; per_kind[kind]["gold_present"] += 1
            ranks.append(hit_rank); scores_found.append(hit_score)
    print(f"\n  exhausted walks                     {len(det)}")
    print(f"  with candidate data logged          {stats['scanned']}")
    print(f"  candidates_topk EMPTY               {stats['no_candidate_data']}"
          f"   <- the diagnostic is unavailable for these")
    if stats["scanned"]:
        gp, ga = stats["gold_present"], stats["gold_absent"]
        print(f"\n  of the {stats['scanned']} scannable:")
        print(f"    gold text PRESENT in candidates   {gp:>4}  ({gp/stats['scanned']:.0%})  -> re-ranking / threshold problem")
        print(f"    gold text ABSENT                  {ga:>4}  ({ga/stats['scanned']:.0%})  -> genuine retrieval miss")
        if ranks:
            ranks_sorted = sorted(ranks)
            med = ranks_sorted[len(ranks_sorted) // 2]
            print(f"    rank of gold when present: min {min(ranks)}  median {med}  max {max(ranks)}")
            print(f"    score of gold when present: min {min(scores_found):.4f}  max {max(scores_found):.4f}")
        print("\n  by failure class (scannable only):")
        print(f"    {'kind':<22}{'present':>9}{'absent':>9}{'no data':>9}")
        for k in sorted(per_kind, key=lambda k: -sum(per_kind[k].values())):
            c = per_kind[k]
            print(f"    {k:<22}{c['gold_present']:>9}{c['gold_absent']:>9}{c['no_candidate_data']:>9}")
    out["q1_gold_rank"] = {"stats": dict(stats), "ranks": ranks,
                           "per_kind": {k: dict(v) for k, v in per_kind.items()}}
    return stats


# ---------------------------------------------------------------- Q2
def q2_margin(recs, decomp, out):
    print("\n" + "=" * 78)
    print("Q2  on the threshold_bound hops: gold score minus best DISTRACTOR score")
    print("    Lowering tau only helps if the correct fact outranks the distractors.")
    print("    A negative margin means the lower tau admits the distractor FIRST.")
    print("=" * 78)
    tau = decomp["tau_ret"]
    by_q = {r["question"]: r for r in recs}
    tb = [d for d in decomp["details"] if d["kind"] == "threshold_bound"]
    print(f"\n  tau_ret in force: {tau:.4f}   threshold_bound cases: {len(tb)}")
    rows, no_data = [], 0
    for d in tb:
        r = by_q.get(d["question"])
        cands = hop_candidates(r) if r else []
        if not cands:
            no_data += 1; continue
        gold = norm(d.get("gold_answer"))
        best_gold, best_distractor = None, None
        for hop in cands:
            for sc, txt in hop:
                if gold and gold in norm(txt):
                    if best_gold is None or sc > best_gold: best_gold = sc
                else:
                    if best_distractor is None or sc > best_distractor: best_distractor = sc
        if best_gold is None:
            rows.append({"q": d["question"][:60], "gold": None,
                         "distractor": best_distractor, "margin": None}); continue
        margin = best_gold - (best_distractor if best_distractor is not None else 0.0)
        rows.append({"q": d["question"][:60], "gold": best_gold,
                     "distractor": best_distractor, "margin": margin})
    have = [r for r in rows if r["margin"] is not None]
    print(f"  with candidate data: {len(rows)}   gold locatable: {len(have)}   no candidate data: {no_data}")
    if have:
        pos = [r for r in have if r["margin"] > 0]
        neg = [r for r in have if r["margin"] <= 0]
        print(f"\n    POSITIVE margin (gold outranks every distractor)  {len(pos):>4}  ({len(pos)/len(have):.0%})"
              f"  -> lowering tau is a real win")
        print(f"    NEGATIVE or zero margin                           {len(neg):>4}  ({len(neg)/len(have):.0%})"
              f"  -> lowering tau admits a DISTRACTOR first")
        ms = sorted(r["margin"] for r in have)
        print(f"    margin: min {ms[0]:+.4f}  median {ms[len(ms)//2]:+.4f}  max {ms[-1]:+.4f}")
        if neg:
            print("\n    examples where the 'free win' is a precision trap:")
            for r in sorted(neg, key=lambda r: r["margin"])[:4]:
                print(f"      margin {r['margin']:+.4f}  gold {r['gold']:.4f} vs distractor {r['distractor']:.4f}"
                      f" | {r['q']}")
    out["q2_margin"] = {"tau_ret": tau, "n": len(tb), "no_candidate_data": no_data,
                         "rows": [{k: v for k, v in r.items() if k != 'q'} for r in have]}
    return rows


# ---------------------------------------------------------------- Q4
def q4_openers(decomp, out):
    print("\n" + "=" * 78)
    print("Q4  would a six-string opener override fix the misparses?")
    print("    Three of four competition submissions proposed a serve-time hop-retry")
    print("    instead, which multiplies walk cost for a prefix match.")
    print("=" * 78)
    mis = [d for d in decomp["details"] if d["kind"] == "misparsed_chain"]
    hit = Counter(); matched = []
    for d in mis:
        q = norm(d["question"])
        for o in OPENERS:
            if q.startswith(norm(o)):
                hit[o] += 1; matched.append(d); break
    print(f"\n  misparsed_chain cases: {len(mis)}")
    for o in OPENERS:
        print(f"    {o:<42}{hit[o]:>4}")
    tot = sum(hit.values())
    print(f"    {'TOTAL matched by the six openers':<42}{tot:>4}  ({tot/len(mis):.0%} of misparses)")
    # does one opener imply one hop count?
    print("\n  is the mapping opener -> gold n_hop deterministic?")
    per = defaultdict(Counter)
    for d in matched:
        q = norm(d["question"])
        for o in OPENERS:
            if q.startswith(norm(o)): per[o][d["n_hop"]] += 1; break
    ok = 0
    for o, c in per.items():
        det = len(c) == 1
        ok += sum(c.values()) if det else 0
        print(f"    {o:<42}{dict(c)}   {'deterministic' if det else 'AMBIGUOUS'}")
    print(f"\n  VERDICT: {ok} of {len(mis)} misparses ({ok/len(mis):.0%}) are fixable by a")
    print("  deterministic prefix->hop-count table. The rest need the parser.")
    out["q4_openers"] = {"n_misparsed": len(mis), "matched": tot,
                          "deterministic": ok, "per_opener": {o: dict(c) for o, c in per.items()}}


# ---------------------------------------------------------------- Q5
def q5_lookup_overlap(recs, lookup, decomp, out):
    print("\n" + "=" * 78)
    print("Q5  re-run the decomposition under lookup-first")
    print("    The 246 were classified at coverage 0.646. Lookup-first is already")
    print("    banked at 0.710. Adding the free wins to 517 double-counts against")
    print("    the +51 the index already recovered.")
    print("=" * 78)
    if lookup is None:
        print("\n  cot_harvest_lookup.jsonl not present — cannot measure the overlap.")
        return
    pre = {r["question"]: r for r in recs}
    post = {r["question"]: r for r in lookup}
    both = set(pre) & set(post)
    v_pre = {q for q in both if pre[q].get("verified")}
    v_post = {q for q in both if post[q].get("verified")}
    print(f"\n  questions in both runs: {len(both)}")
    print(f"  verified pre-lookup   : {len(v_pre)}  ({len(v_pre)/len(both):.3f})")
    print(f"  verified under lookup : {len(v_post)}  ({len(v_post)/len(both):.3f})")
    print(f"  recovered by lookup   : {len(v_post - v_pre)}")
    print(f"  LOST under lookup     : {len(v_pre - v_post)}")
    ex = {d["question"]: d["kind"] for d in decomp["details"]}
    rec_by_kind = Counter(ex[q] for q in (v_post - v_pre) if q in ex)
    print("\n  which failure classes did lookup already fix?")
    for k, n in rec_by_kind.most_common():
        tot = sum(1 for d in decomp["details"] if d["kind"] == k)
        print(f"    {k:<22}{n:>4} of {tot:<4}  ({n/tot:.0%} already recovered)")
    still = [d for d in decomp["details"] if d["question"] in post and not post[d["question"]].get("verified")]
    print(f"\n  STILL failing under lookup-first: {len(still)} of {len(decomp['details'])}")
    print("  remaining, by original class:")
    for k, n in Counter(d["kind"] for d in still).most_common():
        tot = sum(1 for d in decomp["details"] if d["kind"] == k)
        print(f"    {k:<22}{n:>4} of {tot:<4}  ({n/tot:.0%} still open)")
    base = len(v_post)
    print(f"\n  => the free wins must be counted against {base}/{len(both)} = {base/len(both):.3f},")
    print(f"     and only against the {len(still)} failures that survive lookup-first.")
    out["q5_lookup_overlap"] = {
        "n": len(both), "verified_pre": len(v_pre), "verified_post": len(v_post),
        "recovered": len(v_post - v_pre), "lost": len(v_pre - v_post),
        "recovered_by_kind": dict(rec_by_kind),
        "still_failing": len(still),
        "still_by_kind": dict(Counter(d["kind"] for d in still))}


# ---------------------------------------------------------------- Q6
def q6_backward_on_remainder(decomp, lookup, out):
    """What backward retrieval is worth AFTER lookup-first has taken its share."""
    print("\n" + "=" * 78)
    print("Q6  backward (object-side) retrieval, on what actually survives lookup-first")
    print("=" * 78)
    if lookup is None:
        print("\n  no lookup harvest — skipped"); return
    tau = decomp["tau_ret"]
    post = {r["question"]: r for r in lookup}
    still = [x for x in decomp["details"]
             if x["question"] in post and not post[x["question"]].get("verified")]
    print(f"\n  tau_ret {tau:.4f} | surviving failures: {len(still)}\n")
    print(f"  {'kind':<22}{'n':>5}{'has bw':>8}{'bw>=tau':>9}{'share':>8}")
    tot = 0
    per = {}
    for k in ["misparsed_chain", "not_retrieved", "walk_completes_now", "far_below", "threshold_bound"]:
        rows = [x for x in still if x["kind"] == k]
        has = [x for x in rows if x.get("backward_score") is not None]
        ok = [x for x in has if x["backward_score"] >= tau]
        tot += len(ok); per[k] = {"n": len(rows), "probed": len(has), "reaches_tau": len(ok)}
        print(f"  {k:<22}{len(rows):>5}{len(has):>8}{len(ok):>9}{(len(ok)/len(rows) if rows else 0):>7.0%}")
    base = sum(1 for r in lookup if r.get("verified")); n = len(lookup)
    print(f"\n  backward clears tau on {tot} of {len(still)}")
    print(f"  ceiling if every one then verifies: ({base}+{tot})/{n} = {(base+tot)/n:.3f}  [UPPER BOUND]")
    unprobed = [k for k, v in per.items() if v["n"] and not v["probed"]]
    if unprobed:
        print(f"\n  NOT PROBED AT ALL: {', '.join(unprobed)}")
        print("  The original decomposition only ran the backward probe on")
        print("  not_retrieved/far_below. Until the misparses are probed too, whether")
        print("  they are a PARSE failure or a DIRECTION failure is unknown — and that")
        print("  is precisely what decides whether the emitter is needed for them.")
    out["q6_backward_remainder"] = {"still": len(still), "reaches_tau": tot,
                                    "ceiling": round((base + tot) / n, 4), "per_kind": per}


# ---------------------------------------------------------------- Q7
def q7_gold_in_store(recs, lookup, decomp, out):
    """Is the answer even in the store? Separates retrieval failure from absence."""
    print("\n" + "=" * 78)
    print("Q7  is the gold answer present anywhere in the observed store?")
    print("    Retrieval failures are fixable. Store absence is not, by any emitter.")
    print("=" * 78)
    if lookup is None:
        print("\n  no lookup harvest — skipped"); return
    store = set()
    for src in (recs, lookup):
        for r in src:
            for h in (r.get("trace") or []):
                if h.get("fact"): store.add(h["fact"])
            for hop in (r.get("candidates_topk") or []):
                for sc, txt in hop: store.add(txt)
    nstore = [norm(f) for f in store]
    post = {r["question"]: r for r in lookup}
    still = [x for x in decomp["details"]
             if x["question"] in post and not post[x["question"]].get("verified")]
    print(f"\n  observed store: {len(store)} distinct fact strings")
    print("  CAVEAT: only facts that surfaced in a trace or a candidate list are visible")
    print("  here, so 'in store' is a LOWER bound and 'absent' is an over-count.\n")
    print(f"  {'kind':<22}{'n':>5}{'gold in store':>15}{'absent':>9}")
    tot = 0; per = {}
    for k in ["misparsed_chain", "not_retrieved", "walk_completes_now", "far_below", "threshold_bound"]:
        rows = [x for x in still if x["kind"] == k]
        ins = sum(1 for x in rows if norm(x.get("gold_answer")) and
                  any(norm(x["gold_answer"]) in f for f in nstore))
        tot += ins; per[k] = {"n": len(rows), "gold_in_store": ins}
        print(f"  {k:<22}{len(rows):>5}{ins:>15}{len(rows)-ins:>9}")
    base = sum(1 for r in lookup if r.get("verified")); n = len(lookup)
    print(f"  {'TOTAL':<22}{len(still):>5}{tot:>15}{len(still)-tot:>9}")
    print(f"\n  {tot}/{len(still)} = {tot/len(still):.0%} of surviving failures have their answer in reach.")
    print(f"  ceiling if all of them were recovered: ({base}+{tot})/{n} = {(base+tot)/n:.3f}  [UPPER BOUND]")
    out["q7_gold_in_store"] = {"observed_store": len(store), "still": len(still),
                               "gold_in_store": tot, "ceiling": round((base + tot) / n, 4),
                               "per_kind": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="validation/logs")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    logs = Path(a.logs)
    if not (logs / "cot_harvest_v3cf.jsonl").exists():
        sys.exit(f"harvest not found under {logs}")
    recs, decomp, lookup = load(logs)
    print(f"exp_r1_gate_diagnostics | harvest {len(recs)} records | "
          f"decomp {decomp['n_exhausted']} exhausted | git_rev {decomp['git_rev'][:12]}")
    out = {"harvest_n": len(recs), "n_exhausted": decomp["n_exhausted"],
           "decomp_git_rev": decomp["git_rev"]}
    q3_crosstab(decomp, out)
    q1_gold_rank(recs, decomp, out)
    q2_margin(recs, decomp, out)
    q4_openers(decomp, out)
    q5_lookup_overlap(recs, lookup, decomp, out)
    q6_backward_on_remainder(decomp, lookup, out)
    q7_gold_in_store(recs, lookup, decomp, out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
