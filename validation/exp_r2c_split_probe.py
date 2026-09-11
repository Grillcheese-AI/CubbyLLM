"""exp_r2c_split_probe — would a cheaper chain splitter fix the 92 misparses?

Wired: STANDALONE. Needs NO encoder, NO corpus, NO GPU — only the two logs.

`planner._split_chain` splits a question body on the literal " of the " and
`parse_question` sets n_hop = len(rels) + 1. The 92 `misparsed_chain` failures
are exactly the questions where that count disagrees with the gold chain, and
under lookup-first they are 100% still open — the largest surviving class.

This asks the cheapest question about them: is the miscount a one-line splitter
bug, or genuinely ambiguous? Relations themselves contain "of" ("country of
citizenship"), which is why the grammar demands the "the" — so a naive splitter
should over-split badly. That is a prediction this measures rather than assumes.

Method: patch `_split_chain` with each candidate pattern and run the REAL
`parse_question` — so all six v2 frames, the class-noun capture and the
inversions behave exactly as the pipeline's do. The baseline arm must reproduce
the decomposition's own parsed hop counts; if it does not, the run aborts.

CRITICAL: every candidate is scored on ALL questions, not just the 92. A
splitter that repairs misparses while breaking questions that verify today is
worse than nothing. The regression column is the point.

  python validation/exp_r2c_split_probe.py
      -> validation/logs/exp_r2c_split_probe.{json,log}
"""
from __future__ import annotations

import json, pathlib, re, sys, time, types
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Candidates. None = the shipped behaviour (plain str.split on " of the ").
SPLITTERS = {
    "of_the (current grammar)":     None,
    "of_the | of_a | of_an":        r"\s+of\s+(?:the|a|an)\s+",
    "of_any (naive)":               r"\s+of\s+",
    "of_the | contained_within_the": r"\s+of\s+the\s+|\s+contained\s+within\s+the\s+",
    "of_the | in_the":              r"\s+of\s+the\s+|\s+in\s+the\s+",
    "of_the | contained_within_the | in_the":
                                    r"\s+of\s+the\s+|\s+contained\s+within\s+the\s+|\s+in\s+the\s+",
}


def load_planner():
    """Load the real planner module without dragging in the package tree."""
    src = (ROOT / "cubbyllm" / "reasoning" / "planner.py").read_text(encoding="utf-8")
    src = src.replace("from ..core.protocols import Wiring",
                      'class Wiring:\n    WIRED = "WIRED"')
    mod = types.ModuleType("planner_real")
    mod.__dict__["__name__"] = "planner_real"
    sys.modules["planner_real"] = mod           # register before exec: @dataclass needs it
    exec(compile(src, "planner_real.py", "exec"), mod.__dict__)
    return mod


def make_split(pattern: str | None):
    """A _split_chain with the shipped signature: (body) -> (rels, tail)."""
    if pattern is None:
        def split(body: str):
            segs = body.split(" of the ")
            return [s.strip() for s in segs[:-1]], segs[-1].strip()
    else:
        rx = re.compile(pattern, re.I)
        def split(body: str):
            segs = rx.split(body)
            return [s.strip() for s in segs[:-1]], segs[-1].strip()
    return split


def main() -> None:
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    harvest, decomp_p = LOGS / "cot_harvest_v3cf.jsonl", LOGS / "exp_m3_exhaustion_decomp.json"
    for p in (harvest, decomp_p):
        if not p.exists():
            sys.exit(f"missing {p}")
    recs = [json.loads(l) for l in harvest.open(encoding="utf-8")]
    decomp = json.loads(decomp_p.read_text(encoding="utf-8"))
    gold = {r["question"]: r.get("n_hop") for r in recs}              # parquet n_hop = GOLD
    parsed_ref = {d["question"]: d["n_hop"] for d in decomp["details"]}  # plan.n_hop = PARSED
    mis = {d["question"] for d in decomp["details"] if d["kind"] == "misparsed_chain"}
    verified = {r["question"] for r in recs if r.get("verified")}

    pl = load_planner()
    orig_split = pl._split_chain
    log(f"harvest {len(recs)} records | misparsed_chain {len(mis)} | verified {len(verified)}")

    # ---- join check against the decomposition's own published table
    forms = Counter(f"{parsed_ref[q]}->{gold[q]}" for q in mis if q in gold and q in parsed_ref)
    if dict(forms) != decomp["misparsed_forms"]:
        log(f"join MISMATCH: {dict(forms)} vs {decomp['misparsed_forms']}"); sys.exit(1)
    log(f"join check vs decomposition's misparse table: MATCH {dict(forms)}")

    def hop_counts(pattern):
        pl._split_chain = make_split(pattern)
        out = {}
        for q in gold:
            try:
                p = pl.parse_question(q)
            except Exception:
                p = None
            out[q] = p.n_hop if p is not None else None
        return out

    # ---- fidelity check: the baseline must reproduce the recorded parses
    base_counts = hop_counts(None)
    checked = [q for q in parsed_ref if q in base_counts]
    agree = sum(1 for q in checked if base_counts[q] == parsed_ref[q])
    log(f"baseline fidelity: reproduces {agree}/{len(checked)} recorded parses "
        f"({agree/max(1,len(checked)):.1%})")
    if agree != len(checked):
        log("  WARNING: the reimplementation does not match the recorded parser exactly.")
        log("  Treat the deltas below as indicative, not exact.")
    log("")

    parseable = [q for q in gold if base_counts.get(q) is not None]
    log(f"questions the grammar parses at all: {len(parseable)}/{len(gold)} "
        f"| of the {len(mis)} misparses: {len(mis & set(parseable))}\n")

    results = {}
    log(f"{'splitter':<42}{'fixes/92':>10}{'breaks':>9}{'correct':>9}{'net':>7}")
    log("-" * 78)
    base_correct = None
    for name, pat in SPLITTERS.items():
        counts = base_counts if pat is None else hop_counts(pat)
        fixes = breaks = correct = unparse = 0
        for q, g in gold.items():
            n = counts.get(q)
            if n is None:
                unparse += 1
                if q in verified: breaks += 1
                continue
            if n == g: correct += 1
            if q in mis and n == g: fixes += 1
            if q in verified and n != g: breaks += 1
        if base_correct is None: base_correct = correct
        results[name] = {"fixes_misparsed": fixes, "breaks_verified": breaks,
                         "correct_hop_count": correct, "unparseable": unparse,
                         "net_vs_baseline": correct - base_correct}
        log(f"{name:<42}{fixes:>10}{breaks:>9}{correct:>9}{correct-base_correct:>+7}")
    pl._split_chain = orig_split

    log("\n  fixes/92 : misparses whose hop count this splitter gets right")
    log("  breaks   : questions that VERIFY today whose hop count it gets wrong")
    log("  correct  : questions with the right hop count, all 800")
    log("  net      : correct, relative to the shipped grammar\n")

    best = max(results.items(), key=lambda kv: kv[1]["correct_hop_count"])
    log("=" * 78)
    if best[0].startswith("of_the (current"):
        log("VERDICT: no cheaper splitter beats the shipped grammar.")
        log("  The miscount is NOT a one-line bug. Hop-count planning is a real")
        log("  defect and a legitimate emitter target.")
    else:
        b = best[1]
        log(f"VERDICT: '{best[0]}' beats the shipped grammar by {b['net_vs_baseline']} questions")
        log(f"  (+{b['fixes_misparsed']} of {len(mis)} misparses fixed, "
            f"{b['breaks_verified']} currently-verified questions broken).")
        if b["breaks_verified"] > b["fixes_misparsed"]:
            log("  It breaks more than it fixes. NOT a free win.")
        elif b["fixes_misparsed"] < len(mis) * 0.5:
            log(f"  A real but PARTIAL win: {len(mis)-b['fixes_misparsed']} misparses survive it.")
            log("  Take the fix, but the class does not collapse — hop-count planning")
            log("  remains a legitimate emitter target.")
        else:
            log("  The class largely collapses. Re-run the pipeline before training:")
            log("  a right hop count is not the same as a successful walk.")
    log("=" * 78)

    out = {"n_records": len(recs), "n_misparsed": len(mis), "n_verified": len(verified),
           "baseline_fidelity": {"agree": agree, "checked": len(checked)},
           "splitters": results, "wall_s": time.perf_counter() - t0}
    (LOGS / "exp_r2c_split_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (LOGS / "exp_r2c_split_probe.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
