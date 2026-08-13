"""exp_m3_cot_pipeline — the CoT kill-criterion eval (spec section 7).

800 eval questions (seed 0, the exp_m3_injection sample), a DISJOINT
200-question calibration slice (seed 1) for tau_vm/tau_ret. Store = the
eval sample's unique facts. Arms: retrieval-only (top-1 fact's parsed
object), chase-only (n_hop-step chase, final fact's object), CoT pipeline
(walk + VM verify + readout). Normalized exact match vs the dataset answer,
by hop; claimed-answer precision; control pass rate.

Kill criterion (all four): CoT beats retrieval-only on 2-hop AND 3-hop;
claimed precision >= 0.90; zero verified results with a sub-tau hop
(asserted in the pipeline itself); control role below tau_vm in >= 95% of
programs. Standalone; never imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# cubelang mojibake-corrupts non-ASCII fillers on the way back through
# run-proto (see the "Facts that matter" note): a corrupted symbol can
# still print. Reconfigure so an unlucky console codepage (cp1252 on
# Windows) can't crash the run on a stray printed example.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import _load_semantic_words, youden_tau  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning import parse_fact, parse_question  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402

PQ_FILE = pathlib.Path(r"E:\valid_scaling_law_with_facts.pq")
V4_TABLE = pathlib.Path(r"D:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz")


# --------------------------------------------------------------------------
# 1. Sample loading (mirrors exp_m3_injection.load_rows's dedup trick).
# --------------------------------------------------------------------------
def load_sample(n: int, seed: int, exclude: set[str] | None = None):
    """-> (questions, answers, hops, chains, store).

    `chains[i]` is question i's own supporting-fact chain, in walk order,
    as indices into `store` (the whitespace-normalized, deduped fact pool
    built from exactly this sample -- same trick as exp_m3_injection).
    """
    import pyarrow.parquet as pq

    t = pq.read_table(PQ_FILE, columns=["question_prompt", "facts", "answer", "n_hop"])
    rows = [(q, f, a, int(h)) for q, f, a, h in zip(
        t.column("question_prompt").to_pylist(), t.column("facts").to_pylist(),
        t.column("answer").to_pylist(), t.column("n_hop").to_pylist())
        if q and f and a and len(q) > 15]
    if exclude:
        rows = [r for r in rows if r[0] not in exclude]

    rng = np.random.default_rng(seed)
    if len(rows) > n:
        pick = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[j] for j in sorted(pick)]

    fact_id: dict[str, int] = {}
    store: list[str] = []

    def fid(s: str) -> int:
        s = " ".join(s.split())
        if s not in fact_id:
            fact_id[s] = len(store)
            store.append(s)
        return fact_id[s]

    chains = [[fid(x) for x in f] for _, f, _, _ in rows]
    return ([r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
            chains, store)


# --------------------------------------------------------------------------
# 2. Retriever: unit-cosine top-k over a precomputed store.
# --------------------------------------------------------------------------
def _unit_vec(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def make_retriever(store_texts: list[str], enc):
    if store_texts:
        M = np.stack([enc.encode(t).reshape(-1) for t in store_texts]).astype(np.float32)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    else:
        M = np.zeros((0, 80 * 128), dtype=np.float32)

    def retrieve(query: str, k: int) -> list[tuple[float, str]]:
        if not len(store_texts):
            return []
        qv = _unit_vec(enc.encode(query).reshape(-1).astype(np.float32))
        scores = M @ qv
        k = min(k, len(store_texts))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(float(scores[i]), store_texts[i]) for i in idx]

    return retrieve


# --------------------------------------------------------------------------
# 4. Baselines.
# --------------------------------------------------------------------------
def retrieval_only(question: str, retrieve) -> str | None:
    """Top-1 fact, parsed object. None on empty retrieval or unparseable fact."""
    r = retrieve(question, 1)
    if not r:
        return None
    _, fact = r[0]
    t = parse_fact(fact)
    return t.obj if t is not None else None


def chase_only(question: str, n_hop: int, retrieve, k: int = 5) -> str | None:
    """n_hop rounds of top-1 with text-level query expansion (exp_m3_injection's
    chase protocol, reused inline): each round re-encodes question + facts
    retrieved so far; already-picked facts are skipped so expansion can't
    just re-pick the same top hit."""
    got: list[str] = []
    for step in range(max(n_hop, 1)):
        query = question if step == 0 else question + " " + " ".join(got)
        cands = retrieve(query, k)
        pick = next((f for _, f in cands if f not in got), None)
        if pick is None:
            break
        got.append(pick)
    if not got:
        return None
    t = parse_fact(got[-1])
    return t.obj if t is not None else None


def dataset_objects(chain_ids: list[int], store: list[str]) -> list[str | None]:
    """The dataset chain's own fact objects, in walk order -- the gold
    per-hop targets `hop_correct` is checked against."""
    out = []
    for fid_ in chain_ids:
        t = parse_fact(store[fid_])
        out.append(t.obj if t is not None else None)
    return out


def control_violation(sim: float | None, result: str | None, tau: float) -> bool:
    """Mirrors pipeline.answer's control-leak check exactly."""
    return (sim is not None and sim >= tau) or (result is not None and sim is None)


def hops_symbol_ok(trace, gold_objs: list[str | None]) -> bool:
    if len(trace) > len(gold_objs):
        return False
    return all(normalize(ht.symbol or "") == normalize(gold_objs[j] or "")
               for j, ht in enumerate(trace))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=800)
    ap.add_argument("--calibration", type=int, default=200)
    ap.add_argument("--table", type=pathlib.Path, default=V4_TABLE)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-repairs", type=int, default=3)
    ap.add_argument("--exe", type=str, default=None, help="cubelang exe override")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {args.table.name} | rows {PQ_FILE.name} | cubelang {cc.find_cubelang_exe(args.exe)}\n")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(args.table)

    # -- eval sample -------------------------------------------------------
    questions, answers, hops, chains, store = load_sample(args.questions, seed=0)
    hop_counts = {h: hops.count(h) for h in sorted(set(hops))}
    print(f"eval: {len(questions)} questions {hop_counts} | {len(store)} unique facts in store")
    retrieve = make_retriever(store, enc)

    # -- calibration sample (disjoint by question text) --------------------
    cal_q, cal_a, cal_h, cal_chains, cal_store = load_sample(
        args.calibration, seed=1, exclude=set(questions))
    cal_hop_counts = {h: cal_h.count(h) for h in sorted(set(cal_h))}
    print(f"calibration: {len(cal_q)} questions {cal_hop_counts} | "
          f"{len(cal_store)} unique facts in its own store (disjoint from eval)\n")
    cal_retrieve = make_retriever(cal_store, enc)

    def run_fn(source: str, fn: str, exe=args.exe) -> dict:
        return cc.run_program_proto(source, fn=fn, exe=exe)

    # =======================================================================
    # 3. Calibration: permissive taus, collect (similarity, hop_correct).
    # =======================================================================
    print("=== calibration (tau_vm=0.0, tau_ret=-1.0) ===")
    per_hop_obs: list[tuple[float, bool]] = []
    accepted_ret_scores: list[float] = []
    cal_unparseable_q = 0
    cal_mojibake_excluded = 0
    t0 = time.perf_counter()
    for i, (q, chain_ids, h) in enumerate(zip(cal_q, cal_chains, cal_h)):
        gold_objs = dataset_objects(chain_ids, cal_store)
        result = pipeline_answer(q, cal_retrieve, run_fn, tau_vm=0.0, tau_ret=-1.0,
                                 top_k=args.top_k, max_repairs=args.max_repairs)
        if result.reason == "unparseable":
            cal_unparseable_q += 1
            continue
        for hop_i, ht in enumerate(result.trace):
            accepted_ret_scores.append(ht.ret_score)
            if ht.similarity is None:
                continue
            if not ht.fact.isascii():
                # cubelang mojibake-corrupts non-ASCII fillers on the way back
                # through run-proto (ledger note): the VM's returned symbol
                # bytes get mangled independent of similarity/genuine binding
                # quality (verified empirically: sim=1.0 "correct" 1-hop
                # recoveries and mojibake-corrupted "incorrect" ones land at
                # the same similarity), so a string-equality hop_correct check
                # here measures an encoding bug, not reasoning quality. Counted
                # separately in the eval loop's non_ascii_questions instead of
                # poisoning the tau_vm separation.
                cal_mojibake_excluded += 1
                continue
            correct_obj = gold_objs[hop_i] if hop_i < len(gold_objs) else None
            hop_correct = correct_obj is not None and normalize(ht.symbol or "") == normalize(correct_obj)
            per_hop_obs.append((ht.similarity, hop_correct))
        if (i + 1) % 50 == 0:
            print(f"  calibration {i + 1}/{len(cal_q)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"calibration done in {time.perf_counter() - t0:.0f}s | "
          f"{cal_unparseable_q} unparseable calibration questions | "
          f"{cal_mojibake_excluded} non-ASCII hop obs excluded (mojibake) | "
          f"{len(per_hop_obs)} per-hop observations "
          f"({sum(c for _, c in per_hop_obs)} correct)")

    if per_hop_obs:
        sims_arr = np.array([s for s, _ in per_hop_obs])
        correct_arr = np.array([c for _, c in per_hop_obs])
        # youden_tau handles a one-sided (all-correct or all-wrong) split on
        # its own: with an empty "diff" side, fpr stays 0 throughout and the
        # argmax lands on the last (lowest-scored) "same" point, so tau ends
        # up at min(correct sims) rather than an artificially strict value.
        tau_vm, tau_vm_auc = youden_tau(sims_arr[correct_arr], sims_arr[~correct_arr])
    else:
        tau_vm, tau_vm_auc = 0.3, float("nan")
        print("  WARNING: zero calibration hop observations -- falling back to tau_vm=0.3")
    tau_ret = float(np.percentile(accepted_ret_scores, 5)) if accepted_ret_scores else 0.0
    print(f"tau_vm = {tau_vm:.4f} (Youden AUC {tau_vm_auc:.4f} on {len(per_hop_obs)} hop obs) | "
          f"tau_ret = {tau_ret:.4f} (5th pct of {len(accepted_ret_scores)} accepted-hop scores)\n")

    # =======================================================================
    # 5. Main eval loop: three arms per question.
    # =======================================================================
    print(f"=== eval ({len(questions)} questions, tau_vm={tau_vm:.4f} tau_ret={tau_ret:.4f}) ===")
    acc: dict[str, dict[int, list[int]]] = {
        "retrieval_only": defaultdict(list), "chase_only": defaultdict(list), "cot": defaultdict(list)}
    wall_ms: dict[str, list[float]] = {"retrieval_only": [], "chase_only": [], "cot": []}
    repairs_hist: Counter = Counter()
    verified_count = 0
    claimed_correct = 0
    control_pass = 0
    control_total = 0
    unparseable_questions = 0
    unparseable_facts = 0
    non_ascii_questions = 0
    sub_tau_violations = 0
    sensitivity_records: list[dict] = []
    example_printed = False

    t0 = time.perf_counter()
    for i, (q, a, h, chain_ids) in enumerate(zip(questions, answers, hops, chains)):
        gold = normalize(a)
        gold_objs = dataset_objects(chain_ids, store)

        # -- retrieval-only ---------------------------------------------
        s0 = time.perf_counter()
        r1 = retrieval_only(q, retrieve)
        wall_ms["retrieval_only"].append((time.perf_counter() - s0) * 1000)
        if r1 is None:
            unparseable_facts += 1
        acc["retrieval_only"][h].append(int(r1 is not None and normalize(r1) == gold))

        # -- chase-only ---------------------------------------------------
        s0 = time.perf_counter()
        r2 = chase_only(q, h, retrieve)
        wall_ms["chase_only"].append((time.perf_counter() - s0) * 1000)
        if r2 is None:
            unparseable_facts += 1
        acc["chase_only"][h].append(int(r2 is not None and normalize(r2) == gold))

        # -- CoT pipeline ---------------------------------------------------
        ctrl_calls: list[dict] = []

        def vm_run_fn(source: str, fn: str, _log=ctrl_calls) -> dict:
            out = run_fn(source, fn)
            if fn == "control":
                _log.append(out)
            return out

        s0 = time.perf_counter()
        result = pipeline_answer(q, retrieve, vm_run_fn, tau_vm=tau_vm, tau_ret=tau_ret,
                                 top_k=args.top_k, max_repairs=args.max_repairs)
        wall_ms["cot"].append((time.perf_counter() - s0) * 1000)
        repairs_hist[result.repairs_used] += 1
        if result.reason == "unparseable":
            unparseable_questions += 1

        cot_pred = result.answer if result.verified else None
        cot_correct = int(cot_pred is not None and normalize(cot_pred) == gold)
        acc["cot"][h].append(cot_correct)

        if result.verified:
            verified_count += 1
            claimed_correct += cot_correct
            if not all(ht.similarity is not None and ht.similarity >= tau_vm for ht in result.trace):
                sub_tau_violations += 1

        if ctrl_calls:
            last_ctrl = ctrl_calls[-1]
            control_total += 1
            if not control_violation(last_ctrl.get("similarity"), last_ctrl.get("result"), tau_vm):
                control_pass += 1

        if any(not ht.fact.isascii() for ht in result.trace):
            non_ascii_questions += 1

        # cheap ±20% tau_vm sensitivity: reuses this attempt's cached
        # similarities/control observation, no re-retrieval / no re-VM-call
        if (result.reason in (None, "vm_verify_failed") and result.trace
                and all(ht.similarity is not None for ht in result.trace)):
            symbol_ok = hops_symbol_ok(result.trace, gold_objs)
            last_ctrl = ctrl_calls[-1] if ctrl_calls else {}
            sensitivity_records.append({
                "hop": h,
                "sims": [ht.similarity for ht in result.trace],
                "ctrl_sim": last_ctrl.get("similarity"),
                "ctrl_result": last_ctrl.get("result"),
                "symbol_ok": symbol_ok,
                "predicted": normalize(result.trace[-1].symbol or ""),
                "gold": gold,
            })

        if not example_printed and result.verified:
            print(f"\n  example (q{i}): {q}")
            for ht in result.trace:
                print(f"    hop: fact={ht.fact!r} symbol={ht.symbol!r} sim={ht.similarity}")
            print(f"    -> answer={result.answer!r} verified={result.verified} "
                 f"repairs={result.repairs_used}\n")
            example_printed = True

        if (i + 1) % 50 == 0:
            print(f"  eval {i + 1}/{len(questions)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    if not example_printed:
        print("\n  (no verified example encountered to print)\n")
    print(f"eval done in {time.perf_counter() - t0:.0f}s\n")

    # =======================================================================
    # Aggregate stats.
    # =======================================================================
    def acc_by_hop(arm: str) -> dict[str, float]:
        return {str(h_): float(np.mean(v)) for h_, v in sorted(acc[arm].items())}

    def acc_overall(arm: str) -> float:
        vals = [x for v in acc[arm].values() for x in v]
        return float(np.mean(vals)) if vals else 0.0

    claimed_precision = claimed_correct / verified_count if verified_count else 0.0
    verified_coverage = verified_count / len(questions) if questions else 0.0
    control_pass_rate = control_pass / control_total if control_total else 0.0

    # ±20% tau_vm sensitivity (re-verification only, no re-retrieval)
    def sensitivity_at(scale: float) -> dict:
        tau = tau_vm * scale
        would_verify_all = 0
        would_correct = 0
        for rec in sensitivity_records:
            wv = (rec["symbol_ok"]
                  and all(s is not None and s >= tau for s in rec["sims"])
                  and not control_violation(rec["ctrl_sim"], rec["ctrl_result"], tau))
            if wv:
                would_verify_all += 1
                if rec["predicted"] == rec["gold"]:
                    would_correct += 1
        coverage = would_verify_all / len(questions) if questions else 0.0
        precision = would_correct / would_verify_all if would_verify_all else 0.0
        return {"tau_vm": tau, "verified_coverage": coverage,
               "claimed_precision": precision, "verified_count": would_verify_all}

    sensitivity = {"0.8x": sensitivity_at(0.8), "1.0x": sensitivity_at(1.0), "1.2x": sensitivity_at(1.2)}

    print("=== results ===")
    for arm in ("retrieval_only", "chase_only", "cot"):
        print(f"  {arm:16s} overall={acc_overall(arm):.3f} by-hop={acc_by_hop(arm)} "
              f"mean_wall_ms={float(np.mean(wall_ms[arm])):.1f}")
    print(f"\n  CoT verified-coverage: {verified_coverage:.3f} ({verified_count}/{len(questions)})")
    print(f"  CoT claimed-answer precision: {claimed_precision:.3f} ({claimed_correct}/{verified_count if verified_count else 0})")
    print(f"  control pass rate: {control_pass_rate:.3f} ({control_pass}/{control_total})")
    print(f"  unparseable questions: {unparseable_questions} | unparseable facts: {unparseable_facts}")
    print(f"  non-ascii-walked-fact questions: {non_ascii_questions}")
    print(f"  sub-tau verified violations: {sub_tau_violations}")
    print(f"  repairs histogram: {dict(sorted(repairs_hist.items()))}")
    print(f"\n  tau_vm sensitivity (re-verify cached decisions only):")
    for k_, v_ in sensitivity.items():
        print(f"    {k_} (tau={v_['tau_vm']:.4f}): coverage={v_['verified_coverage']:.3f} "
              f"precision={v_['claimed_precision']:.3f} (n={v_['verified_count']})")

    # =======================================================================
    # 6. Verdict.
    # =======================================================================
    cot_by_hop = acc_by_hop("cot")
    ret_by_hop = acc_by_hop("retrieval_only")
    chase_by_hop = acc_by_hop("chase_only")
    beats_2hop = "2" in cot_by_hop and "2" in ret_by_hop and cot_by_hop["2"] > ret_by_hop["2"]
    beats_3hop = "3" in cot_by_hop and "3" in ret_by_hop and cot_by_hop["3"] > ret_by_hop["3"]
    clause1 = beats_2hop and beats_3hop
    clause2 = claimed_precision >= 0.90
    clause3 = sub_tau_violations == 0
    clause4 = control_pass_rate >= 0.95
    overall = clause1 and clause2 and clause3 and clause4

    print("\n=== verdict ===")
    print(f"  [{'PASS' if clause1 else 'FAIL'}] (1) CoT beats retrieval-only on 2-hop AND 3-hop: "
          f"2-hop cot={cot_by_hop.get('2', float('nan')):.3f} vs ret={ret_by_hop.get('2', float('nan')):.3f} "
          f"({'beats' if beats_2hop else 'does not beat'}); "
          f"3-hop cot={cot_by_hop.get('3', float('nan')):.3f} vs ret={ret_by_hop.get('3', float('nan')):.3f} "
          f"({'beats' if beats_3hop else 'does not beat'})")
    print(f"  [{'PASS' if clause2 else 'FAIL'}] (2) claimed-answer precision >= 0.90: {claimed_precision:.4f}")
    print(f"  [{'PASS' if clause3 else 'FAIL'}] (3) zero verified results with a sub-tau hop: "
          f"{sub_tau_violations} violations")
    print(f"  [{'PASS' if clause4 else 'FAIL'}] (4) control role below tau_vm in >= 95% of programs: "
          f"{control_pass_rate:.4f}")
    print(f"\n  informative (not a kill-criterion clause): CoT vs chase-only -- "
          f"2-hop cot={cot_by_hop.get('2', float('nan')):.3f} vs chase={chase_by_hop.get('2', float('nan')):.3f}; "
          f"3-hop cot={cot_by_hop.get('3', float('nan')):.3f} vs chase={chase_by_hop.get('3', float('nan')):.3f}")
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")

    out = {
        "config": {k_: (str(v_) if isinstance(v_, pathlib.Path) else v_) for k_, v_ in vars(args).items()},
        "n_eval_questions": len(questions),
        "n_calibration_questions": len(cal_q),
        "eval_hop_counts": {str(k_): v_ for k_, v_ in hop_counts.items()},
        "calibration_hop_counts": {str(k_): v_ for k_, v_ in cal_hop_counts.items()},
        "n_store_facts": len(store),
        "calibration": {
            "cal_unparseable_questions": cal_unparseable_q,
            "cal_mojibake_excluded_hop_obs": cal_mojibake_excluded,
            "n_hop_observations": len(per_hop_obs),
            "tau_vm": tau_vm, "tau_vm_youden_auc": tau_vm_auc,
            "tau_ret": tau_ret,
        },
        "arms": {
            arm: {"overall": acc_overall(arm), "by_hop": acc_by_hop(arm),
                 "mean_wall_ms": float(np.mean(wall_ms[arm]))}
            for arm in ("retrieval_only", "chase_only", "cot")
        },
        "cot_verified_coverage": verified_coverage,
        "cot_verified_count": verified_count,
        "claimed_answer_precision": claimed_precision,
        "claimed_correct": claimed_correct,
        "control_pass_rate": control_pass_rate,
        "control_total": control_total,
        "unparseable_questions": unparseable_questions,
        "unparseable_facts": unparseable_facts,
        "non_ascii_questions": non_ascii_questions,
        "sub_tau_violations": sub_tau_violations,
        "repairs_histogram": {str(k_): v_ for k_, v_ in sorted(repairs_hist.items())},
        "tau_vm_sensitivity": sensitivity,
        "verdict": {
            "clause1_cot_beats_retrieval_2h_3h": clause1,
            "clause2_claimed_precision_ge_0.90": clause2,
            "clause3_zero_sub_tau_violations": clause3,
            "clause4_control_pass_rate_ge_0.95": clause4,
            "overall": overall,
        },
    }

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_cot_pipeline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_cot_pipeline.json'}")


if __name__ == "__main__":
    main()
