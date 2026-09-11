"""exp_m3_cot_pipeline — the CoT kill-criterion eval (spec section 7).

800 eval questions (seed 0, the exp_m3_injection sample), a DISJOINT
200-question calibration slice (seed 1) for tau_vm/tau_ret. Store = the
eval sample's unique facts (+N DBpedia distractors, retrieval-only, when
--distractors is set). Arms: retrieval-only (top-1 fact's parsed object),
chase-only (n_hop-step full-store masked-argmax chase, final fact's
object), CoT pipeline (walk + VM verify + readout). Normalized exact match
vs the dataset answer, by hop; claimed-answer precision; control pass rate.

Calibration v2 (eval-wave, 3-model literature panel, Q21) -- REVISED
same-day after a first full run exposed a design flaw in the original
FPR-target plan: the VM's `similarity` measures binding FIDELITY (was
something cleanly recovered), not semantic TRUTH (was the correct thing
recovered). A wrong-entity/inverted-direction planted fault binds a
definitely-wrong filler and recovers it FAITHFULLY -- often at ~1.0
similarity in a single-binding (1-hop, zero-crosstalk) frame -- because
content-correctness is caught by the exact STRING-MATCH check
`pipeline.answer()` already runs alongside tau_vm
(`normalize(symbol) != normalize(triples[i].obj)`), not by the similarity
score. Calibrating an FPR-target threshold over fault-class similarities
therefore pins to the similarity ceiling and starves multi-hop coverage as
a side effect -- measured, not hypothesized, in the first eval-wave run
(2-hop/3-hop CoT accuracy collapsed to 0.000 under a tau_vm=1.0000 deployed
from that scheme). Fixed: DEPLOYED tau_vm is now a set of FRAME-SIZE-
CONDITIONAL FLOORS -- for each n_hop seen in calibration, tau_vm[n] = the
bootstrap-95%-CI LOWER bound of the 1st-percentile CORRECT-recovery
similarity at that frame size (a scalar fallback = min over sizes covers
frame sizes unseen in calibration). Each floor is reported alongside its
margin over the control role's own similarity (~0.01-0.04) -- that margin
is what tau_vm actually defends: a real, confident recovery vs an absent/
garbled one, NOT correct-vs-wrong content. Planted-fault instances
(wrong-entity / wrong-relation / inverted-direction / wrong-hop-order)
still get built and run, but are now reported as a CATCH RATE (does the
REAL verify decision -- symbol mismatch OR sub-floor similarity -- reject
the corrupted recovery?), not an FPR-calibration signal; their similarity
distributions stay in the calibration card as informational binding-
fidelity scores only. Organic cross-chain confusables (held-out) and
random-distractor substitutions (a unit sanity test) are reported the same
way. The OLD Youden number (on the walk's own organic structural near-
misses) stays as `tau_vm_youden_reference`, comparison only. Every eval
question is also harvested to validation/logs/cot_harvest{tag}.jsonl per
docs/schemas/cot-harvest-schema.md.

Counterfactual neighborhood (2026-08-28; harvest-schema `counterfactuals[]`,
the Q23 "regretted discard" and N4 of docs/research/2026-08-28-oracle-
competition-scored.md): for EVERY verified eval chain, the four planted-fault
classes are re-planted on the pipeline's OWN accepted triples (same generator
as the calibration pass; swap objects sourced from the calibration store,
never from distractors), each corrupted hop is recovered through the VM at the
deployed floor, and the per-hop outcome -- caught by symbol mismatch / below
floor / no similarity, or ESCAPED -- is written into that question's harvest
record. Caught records are per-hop hard negatives for the verifier's
successor and a self-refilling per-fault-class calibration stratum; escaped
records are the would-be false accepts the calibration must defend against.
Reported as an aggregate catch map (class x hop) in the json, plus the first
escaped examples. Disable with --no-counterfactuals; VM-call cap
--max-cf-vm-calls (default MAX_CF_VM_CALLS).

Kill criterion (all four): CoT beats retrieval-only on 2-hop AND 3-hop;
claimed precision >= 0.90; zero verified results with a sub-tau hop
(asserted in the pipeline itself); control role below tau_vm in >= 95% of
programs (every control call across every attempt). Standalone; never
imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

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
from exp_m3_haystack import iter_distractors  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning import TripleIndex  # noqa: E402
from cubbyllm.reasoning import build_chain_program, parse_fact, parse_question  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations, VMRelations, write_vocab_jsonl  # noqa: E402
from cubbyllm.reasoning.planner import Triple, normalize  # noqa: E402

PQ_FILE = pathlib.Path(r"E:\valid_scaling_law_with_facts.pq")
V4_TABLE = pathlib.Path(r"I:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz")

FAULT_CLASSES = ["wrong_entity", "wrong_relation", "inverted_direction", "wrong_hop_order"]
MAX_FAULTS_PER_CLASS_PER_CHAIN = 2
# Soft cap on total planted-fault VM calls -- stops the fault-calibration
# pass once comfortably past what a default-sized (200-question) run needs
# (~550 calls observed), regardless of how many verified calibration
# chains are available.
MAX_FAULT_VM_CALLS = 900
# Soft cap on counterfactual-neighborhood VM calls over the whole eval loop
# (v2 shape: ~517 verified chains x ~5-7 attempted corrupted hops ~= 3k).
MAX_CF_VM_CALLS = 6000


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
# 3. Baselines.
# --------------------------------------------------------------------------
def retrieval_only(question: str, retrieve) -> str | None:
    """Top-1 fact, parsed object. None on empty retrieval or unparseable fact."""
    r = retrieve(question, 1)
    if not r:
        return None
    _, fact = r[0]
    t = parse_fact(fact)
    return t.obj if t is not None else None


def chase_only(question: str, n_hop: int, retrieve, store_size: int) -> str | None:
    """n_hop rounds of full-store masked argmax -- exp_m3_injection.chase's
    fidelity bar, not a small top-k window. Each round ranks the WHOLE
    store (requesting `store_size` candidates from the shared retriever's
    top-k closure forces a full sort of every row, since k >= N), and picks
    the single best row not already chosen. "First not-yet-picked entry in
    a full ranking" IS the masked argmax over score[picked]=-inf: same
    operation, reusing the shared scorer instead of a second dense-matmul
    implementation."""
    got: list[str] = []
    for step in range(max(n_hop, 1)):
        query = question if step == 0 else question + " " + " ".join(got)
        cands = retrieve(query, store_size)
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
    """NOTE (sensitivity-table gating, fixed post-review): this compares
    each walked hop's recovered SYMBOL against the DATASET's own gold
    chain object for that hop position (`gold_objs`, from
    `dataset_objects`) -- a cheap proxy for "would this hop still verify
    at a different tau_vm," used ONLY by the cached +/-tau_vm sensitivity
    re-verification below (no re-retrieval, no re-VM-call). It is NOT the
    same check the live pipeline runs: `cubbyllm.reasoning.pipeline.answer`
    verifies each hop against the WALKED triple's OWN retrieved object
    (`triples[i].obj`), which can legitimately differ from the dataset's
    gold chain if retrieval found a different, still relation-correct fact
    that reaches the same final answer through a different path. So a
    sensitivity-table row can diverge slightly from what actually re-
    running the pipeline at that tau would report -- it's a proxy for "how
    sensitive is the verified set to tau_vm," not a byte-identical replay
    of the pipeline's own accept gate. Computation unchanged; comment only.
    """
    if len(trace) > len(gold_objs):
        return False
    return all(normalize(ht.symbol or "") == normalize(gold_objs[j] or "")
               for j, ht in enumerate(trace))


# --------------------------------------------------------------------------
# 4. Shared plumbing for planted-fault calibration + harvest.
# --------------------------------------------------------------------------
def store_hash(texts: list[str]) -> str:
    """sha256 over sorted fact texts -- a store snapshot's invalidation key."""
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        rev = out.stdout.strip()
        return rev if rev else "unknown"
    except Exception:
        return "unknown"


def _display_rels(plan, triples: list[Triple]) -> list[str]:
    """Exactly mirrors pipeline.answer's own display-relation computation,
    so a rebuilt program's role names are byte-identical to what the
    pipeline actually used for the same (plan, triples)."""
    return [(t.rel if i == 0 else plan.relations[i]) or t.rel
            for i, t in enumerate(triples)]


def _dataset_plan_and_triples(q: str, chain_ids: list[int], store: list[str]):
    """-> (plan, triples, display_rels, true_objs) built from the
    DATASET's own gold chain (not a pipeline walk) -- used by the
    organic-confusable and random-distractor held-out tests, neither of
    which needs `pipeline_answer` to have run at all. None if the question
    or any of its own chain facts fails to parse, or the chain length
    doesn't match the parsed plan's hop count."""
    plan = parse_question(q)
    if plan is None:
        return None
    triples = []
    for fid_ in chain_ids:
        t = parse_fact(store[fid_])
        if t is None:
            return None
        triples.append(t)
    if not triples or len(triples) != plan.n_hop:
        return None
    return plan, triples, _display_rels(plan, triples), [t.obj for t in triples]


def _run_fault_program(triples: list[Triple], display_rels: list[str],
                       compare_objs: list[str | None], run_fn,
                       tau_vm_floor: float | None = None):
    """Build the (possibly corrupted) chain program and recover every hop
    role (never control -- that's a separate always-on sanity check
    elsewhere). For each attempted hop (one with a comparison target and
    ASCII-safe content), determine:
      - a "negative" (binding-fidelity) observation: the hop's own
        similarity, recorded ONLY when the recovered symbol does NOT
        normalize-match its comparison target -- kept for the calibration
        card's INFORMATIONAL fidelity stats, not for setting tau_vm (see
        the module docstring: fidelity != truth).
      - "caught": whether the REAL pipeline verify decision would reject
        this hop -- symbol mismatch (content wrong) OR no similarity
        returned OR similarity below `tau_vm_floor` (when given). This is
        the honest hardness measurement: the string-match check catches
        content corruption; the floor catches absent/garbled recovery.
    A hop with no comparison target (compare_objs entry None -- "not
    constructible" for that hop, e.g. wrong-relation's last hop) is
    skipped entirely. Mojibake-affected (non-ASCII) hops are excluded,
    same rationale as the calibration mojibake guard.
    -> (negatives: list[float], n_vm_calls: int, n_mojibake_excluded: int,
        n_caught: int)
    """
    source, fns = build_chain_program(triples, display_rels)
    negatives: list[float] = []
    n_calls = 0
    n_mojibake = 0
    n_caught = 0
    for k, fn in enumerate(fns[:-1]):
        compare_obj = compare_objs[k] if k < len(compare_objs) else None
        if compare_obj is None:
            continue
        t = triples[k]
        if not (t.obj.isascii() and t.subj.isascii() and t.rel.isascii()):
            n_mojibake += 1
            continue
        out = run_fn(source, fn)
        n_calls += 1
        sim = out.get("similarity")
        symbol = out.get("result")
        mismatch = normalize(symbol or "") != normalize(compare_obj)
        if mismatch and sim is not None:
            negatives.append(float(sim))
        below_floor = (sim is None) or (tau_vm_floor is not None and sim < tau_vm_floor)
        if mismatch or below_floor:
            n_caught += 1
    return negatives, n_calls, n_mojibake, n_caught


def _pick_other_object(store_texts: list[str], exclude_norm: set[str], rng) -> str | None:
    """A parsed fact's object from elsewhere in the (calibration-only)
    store, normalized-distinct from every true object already in this
    chain. Bounded retries; None if the store can't offer one."""
    if not store_texts:
        return None
    for _ in range(8):
        f = store_texts[int(rng.integers(0, len(store_texts)))]
        t = parse_fact(f)
        if t is not None and normalize(t.obj) not in exclude_norm:
            return t.obj
    return None


def _fault_instances(triples: list[Triple], cal_store_texts: list[str], rng
                     ) -> dict[str, list[tuple[list[Triple], list[str | None]]]]:
    """-> {class_name: [(corrupted_triples, compare_objs), ...]} for the
    four planted-fault classes, capped at MAX_FAULTS_PER_CLASS_PER_CHAIN
    instances/class (VM-cost cap). `cal_store_texts` MUST be the
    calibration slice's own store -- distractors are never calibration
    negatives, and this is the only place "another calibration fact's
    object" is sourced from, so that invariant holds structurally (the
    eval store, where distractors live, never reaches this function)."""
    n = len(triples)
    true_objs = [t.obj for t in triples]
    exclude_norm = {normalize(o) for o in true_objs}
    out: dict[str, list[tuple[list[Triple], list[str | None]]]] = {c: [] for c in FAULT_CLASSES}

    hop_order = list(range(n))
    rng.shuffle(hop_order)
    hop_pick = hop_order[:MAX_FAULTS_PER_CLASS_PER_CHAIN]

    # wrong-entity: swap ONE hop's object for another calibration fact's
    # object. compare_objs is None everywhere EXCEPT the corrupted hop --
    # the other hops in this chain's frame are innocent bystanders (still
    # correctly bound), not faults, so they must not count toward
    # "attempted"/"caught": including them would dilute the catch rate
    # with hops that never needed catching in the first place.
    for hop_i in hop_pick:
        swap_obj = _pick_other_object(cal_store_texts, exclude_norm, rng)
        if swap_obj is None:
            continue
        corrupted = list(triples)
        t = triples[hop_i]
        corrupted[hop_i] = Triple(obj=swap_obj, rel=t.rel, subj=t.subj)
        compare: list[str | None] = [None] * n
        compare[hop_i] = true_objs[hop_i]
        out["wrong_entity"].append((corrupted, compare))

    # inverted-direction: bind the SUBJECT as the filler instead of the
    # object -- same single-hop-fault scoping as wrong-entity above.
    for hop_i in hop_pick:
        t = triples[hop_i]
        if normalize(t.subj) == normalize(t.obj):
            continue                       # not a real corruption, skip
        corrupted = list(triples)
        corrupted[hop_i] = Triple(obj=t.subj, rel=t.rel, subj=t.subj)
        compare = [None] * n
        compare[hop_i] = true_objs[hop_i]
        out["inverted_direction"].append((corrupted, compare))

    # wrong-hop-order: swap two hops' objects between each other (needs
    # n>=2) -- only the two swapped positions are faults.
    if n >= 2:
        pair_pool = [(a, b) for a in range(n) for b in range(a + 1, n)]
        rng.shuffle(pair_pool)
        for a, b in pair_pool[:MAX_FAULTS_PER_CLASS_PER_CHAIN]:
            ta, tb = triples[a], triples[b]
            if normalize(ta.obj) == normalize(tb.obj):
                continue                   # no-op swap, skip
            corrupted = list(triples)
            corrupted[a] = Triple(obj=tb.obj, rel=ta.rel, subj=ta.subj)
            corrupted[b] = Triple(obj=ta.obj, rel=tb.rel, subj=tb.subj)
            compare = [None] * n
            compare[a] = true_objs[a]
            compare[b] = true_objs[b]
            out["wrong_hop_order"].append((corrupted, compare))

    # wrong-relation: NO binding corruption -- query role H_{k+1} (hop k's
    # own recover call) but compare its answer against a DIFFERENT hop's
    # true object, shifted by 1 (and by 2 when a 3rd hop exists to shift
    # into) -- exactly "ask for H1, compare against hop-2's object."
    for shift in range(1, MAX_FAULTS_PER_CLASS_PER_CHAIN + 1):
        if shift >= n:
            break
        shifted: list[str | None] = list(true_objs[shift:]) + [None] * shift
        # A chain whose hop-k and hop-(k+shift) objects coincide makes this a
        # no-op "fault" (the recovery is compared against itself) -- skip that
        # hop, exactly as wrong-hop-order skips a same-object swap. Found by
        # the counterfactual harvest 2026-08-28: 3/236 wrong_relation
        # "escapes" in the first v3cf run were all this degenerate case.
        shifted = [None if (o is not None and normalize(o) == normalize(true_objs[k])) else o
                   for k, o in enumerate(shifted)]
        if all(o is None for o in shifted):
            continue
        out["wrong_relation"].append((list(triples), shifted))

    return out


def _cf_outcome(symbol: str | None, sim: float | None, compare_obj: str,
                tau_vm_floor: float | None) -> tuple[bool, bool, bool, str | None]:
    """Classify one corrupted-hop recovery against the REAL verify decision
    (mirrors pipeline.answer: symbol mismatch OR sub-floor similarity OR no
    similarity rejects). -> (symbol_mismatch, below_floor, caught, caught_by)
    with caught_by in {"symbol_mismatch", "below_floor", None}: content
    corruption is the primary catch (it is what the string check defends);
    below_floor is credited only when the content was faithful."""
    mismatch = normalize(symbol or "") != normalize(compare_obj)
    below_floor = (sim is None) or (tau_vm_floor is not None and sim < tau_vm_floor)
    caught = mismatch or below_floor
    caught_by = "symbol_mismatch" if mismatch else ("below_floor" if below_floor else None)
    return mismatch, below_floor, caught, caught_by


def _counterfactual_neighborhood(triples: list[Triple], display_rels: list[str],
                                 cal_store_texts: list[str], rng, run_fn,
                                 tau_vm_floor: float | None, budget: list[int]
                                 ) -> tuple[list[dict], int, int, bool]:
    """The counterfactual neighborhood of ONE verified chain (harvest-schema
    `counterfactuals[]`): plant the four fault classes on the chain's own
    accepted triples via `_fault_instances` (swap objects from the
    calibration store only -- the same never-a-distractor invariant), rebuild
    the program per instance, recover each corrupted hop through the VM, and
    return one record per attempted hop:
      {"cls", "hop" (0-based), "n_hop", "planted": {obj, rel, subj} (the
       triple bound at that hop in the corrupted chain), "compare_obj" (what
       the recovery was checked against), "symbol", "similarity",
       "symbol_mismatch", "below_floor", "caught", "caught_by"}
    Innocent bystander hops (compare None) are never attempted; non-ASCII
    hops are excluded (mojibake guard, same as calibration). `budget` is the
    shared remaining-VM-calls counter ([n]); planting stops when it reaches
    zero and the result is flagged truncated.
    -> (records, n_vm_calls, n_mojibake_excluded, truncated)"""
    records: list[dict] = []
    n_calls = 0
    n_moji = 0
    truncated = False
    n = len(triples)
    instances = _fault_instances(triples, cal_store_texts, rng)
    for cls in FAULT_CLASSES:
        for corrupted, compare in instances[cls]:
            source, fns = build_chain_program(corrupted, display_rels)
            for k, fn in enumerate(fns[:-1]):
                compare_obj = compare[k] if k < len(compare) else None
                if compare_obj is None:
                    continue
                t = corrupted[k]
                if not (t.obj.isascii() and t.subj.isascii() and t.rel.isascii()):
                    n_moji += 1
                    continue
                if budget[0] <= 0:
                    truncated = True
                    return records, n_calls, n_moji, truncated
                out = run_fn(source, fn)
                budget[0] -= 1
                n_calls += 1
                sim = out.get("similarity")
                symbol = out.get("result")
                mismatch, below, caught, caught_by = _cf_outcome(symbol, sim, compare_obj, tau_vm_floor)
                records.append({
                    "cls": cls, "hop": k, "n_hop": n,
                    "planted": {"obj": t.obj, "rel": t.rel, "subj": t.subj},
                    "compare_obj": compare_obj,
                    "symbol": symbol,
                    "similarity": (float(sim) if sim is not None else None),
                    "symbol_mismatch": mismatch, "below_floor": below,
                    "caught": caught, "caught_by": caught_by,
                })
    return records, n_calls, n_moji, truncated


def bootstrap_quantile_ci(sims: np.ndarray, q: float, n_boot: int, rng
                          ) -> tuple[float, float, float]:
    """-> (point estimate, ci95 lo, ci95 hi) of the q-th percentile of
    `sims`, via case resampling (n_boot draws, each the same size as
    `sims`)."""
    n = len(sims)
    point = float(np.percentile(sims, q))
    if n < 2:
        return point, point, point
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = sims[rng.integers(0, n, size=n)]
        boots[b] = np.percentile(sample, q)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _quantile_stats(sims: list[float]) -> dict:
    arr = np.array(sims, dtype=np.float64)
    n = len(arr)
    return {"n": n,
           "mean": float(arr.mean()) if n else float("nan"),
           "q50": float(np.percentile(arr, 50)) if n else float("nan"),
           "q90": float(np.percentile(arr, 90)) if n else float("nan"),
           "q99": float(np.percentile(arr, 99)) if n else float("nan")}


def _harvest_record(q: str, plan, gold_norm: str, gold_raw, h: int, result,
                    ret_calls: list[dict], tau_vm: float, tau_ret: float,
                    store_hash_: str, table_path_: str, git_rev_: str,
                    timestamp_: str, counterfactuals: list[dict] | None = None) -> dict:
    """One record per docs/schemas/cot-harvest-schema.md (canonical v1 +
    the `counterfactuals[]` field, 2026-08-28: None when not computed for
    this record -- unverified, disabled, or VM budget exhausted -- else the
    list from `_counterfactual_neighborhood`, possibly empty).
    `tau_vm` is the FRAME-SIZE-CONDITIONAL floor actually used for this
    question (tau_vm_floors[h] or the fallback), not a single global
    scalar -- see the module docstring."""
    trace_out = []
    for ht in result.trace:
        hop_verified = (ht.similarity is not None and ht.similarity >= tau_vm
                        and ht.triple is not None
                        and normalize(ht.symbol or "") == normalize(ht.triple.obj))
        trace_out.append({
            "query": ht.query, "fact": ht.fact,
            "triple": ({"obj": ht.triple.obj, "rel": ht.triple.rel, "subj": ht.triple.subj}
                      if ht.triple is not None else None),
            "ret_score": ht.ret_score, "symbol": ht.symbol,
            "similarity": ht.similarity, "hop_verified": hop_verified,
            "source": getattr(ht, "source", "search"),
        })

    candidates_topk = []
    for ht in result.trace:
        cands = next((c["candidates"] for c in reversed(ret_calls)
                     if c["query"] == ht.query), [])
        candidates_topk.append([[float(s), f] for s, f in cands])

    hops_verified_before_failure = 0
    for row in trace_out:
        if not row["hop_verified"]:
            break
        hops_verified_before_failure += 1

    cot_pred = result.answer if result.verified else None
    correct = bool(cot_pred is not None and normalize(cot_pred) == gold_norm)

    return {
        "question": q,
        "parsed": plan is not None,
        "n_hop": h,
        "answer_class": plan.answer_class if plan is not None else None,
        "program_source": result.source,
        "verified": result.verified,
        "answer": result.answer,
        "gold_answer": gold_raw,
        "correct": correct,
        "trace": trace_out,
        "candidates_topk": candidates_topk,
        "reason": result.reason,
        "refused": result.refused,
        "repairs_used": result.repairs_used,
        "banned_facts": result.repairs,
        "hops_verified_before_failure": hops_verified_before_failure,
        "counterfactuals": counterfactuals,
        "taus": {"tau_vm": tau_vm, "tau_ret": tau_ret},
        "store_snapshot_hash": store_hash_,
        "table_path": table_path_,
        "git_rev": git_rev_,
        "timestamp": timestamp_,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=800)
    ap.add_argument("--calibration", type=int, default=200)
    ap.add_argument("--table", type=pathlib.Path, default=V4_TABLE)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-repairs", type=int, default=1,   # was 3; lossless at 1 on the 800-question set (rb1 run, 2026-09-03)
                    help="repair budget per question (3 -> 1 measured lossless, 2.6x faster; see logs/exp_m3_cot_pipeline_rb1.json)")
    ap.add_argument("--exe", type=str, default=None, help="cubelang exe override")
    ap.add_argument("--lookup", action="store_true",
                    help="lookup-first walk: a TripleIndex over the eval store, cosine only for the hops it misses (exp_m4, 2026-09-04)")
    ap.add_argument("--verify-plan", choices=["host", "vm"], default=None,
                    help="dispose of the plan before the walk (plan_verify, 2026-09-11): 'host' answers "
                         "'is this relation known' with StoreRelations(store); 'vm' with the VM's QUERY over "
                         "the same vocabulary loaded as knowledge (one subprocess per distinct relation, memoized)")
    ap.add_argument("--resident", action="store_true",
                    help="one resident `cubelang run-proto` process for EVERY VM call (chain programs, "
                         "control, counterfactuals, and --verify-plan vm) instead of a spawn per call "
                         "(CubelangSession, 2026-09-11). Same wire, same fresh-VM-per-request semantics.")
    ap.add_argument("--distractors", type=int, default=0,
                    help="N DBpedia distractor texts appended to the EVAL "
                         "retrieval store only (never calibration)")
    ap.add_argument("--tag", default="", help="suffix for the output json/log/harvest files")
    ap.add_argument("--no-counterfactuals", action="store_true",
                    help="skip the per-verified-chain counterfactual neighborhood harvest")
    ap.add_argument("--max-cf-vm-calls", type=int, default=MAX_CF_VM_CALLS,
                    help="VM-call cap for the counterfactual neighborhood over the whole eval loop")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {args.table.name} | rows {PQ_FILE.name} | cubelang {cc.find_cubelang_exe(args.exe)}\n")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(args.table)

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    json_path = logs / f"exp_m3_cot_pipeline{args.tag}.json"
    harvest_path = logs / f"cot_harvest{args.tag}.jsonl"

    # Fixed seeds (spec section 7): eval=0 is exp_m3_injection's sample seed;
    # calibration=1 is sampled disjoint from it. Recorded in the json config
    # below so a run is reproducible from the json alone.
    eval_seed = 0
    calibration_seed = 1

    # -- eval sample -------------------------------------------------------
    questions, answers, hops, chains, store = load_sample(args.questions, seed=eval_seed)
    hop_counts = {h: hops.count(h) for h in sorted(set(hops))}
    n_facts_only = len(store)
    print(f"eval: {len(questions)} questions {hop_counts} | {n_facts_only} unique facts in store")

    # -- distractors (retrieval store ONLY; never a calibration negative) --
    n_distractors_added = 0
    if args.distractors > 0:
        print(f"loading {args.distractors} DBpedia distractors into the EVAL "
              f"retrieval store only ...", flush=True)
        td0 = time.perf_counter()
        for chunk in iter_distractors(args.distractors):
            store.extend(chunk)
            n_distractors_added += len(chunk)
            print(f"  {n_distractors_added} distractors loaded "
                  f"({time.perf_counter() - td0:.0f}s)", flush=True)
        print(f"eval store: {n_facts_only} facts + {n_distractors_added} distractors "
              f"= {len(store)} total ({time.perf_counter() - td0:.0f}s)\n", flush=True)

    retrieve = make_retriever(store, enc)
    store_size = len(store)
    index = TripleIndex(store) if args.lookup else None
    session = cc.CubelangSession(exe=args.exe) if args.resident else None
    if session is not None:
        print(f"  resident VM: {session.exe} (pid {session._proc.pid})")
    known = None
    if args.verify_plan == "host":
        known = StoreRelations(store)
        print(f"  verify-plan: host  ({len(known)} relations)")
    elif args.verify_plan == "vm":
        vocab_path = ROOT / "validation" / "logs" / f"exp_m3_vocab{args.tag}.jsonl"
        n_rel = write_vocab_jsonl(store, vocab_path)
        known = VMRelations(knowledge=vocab_path, exe=args.exe, session=session)
        print(f"  verify-plan: vm    ({n_rel} relations as knowledge -> {vocab_path.name}"
              f"{', resident' if session is not None else ', one spawn per relation'})")
    if index is not None:
        print(f"lookup arm: triple index over the eval store, {len(index)}/{index.n_facts} facts parse as triples\n")

    # -- calibration sample (disjoint by question text; NEVER sees distractors) --
    cal_q, cal_a, cal_h, cal_chains, cal_store = load_sample(
        args.calibration, seed=calibration_seed, exclude=set(questions))
    cal_hop_counts = {h: cal_h.count(h) for h in sorted(set(cal_h))}
    print(f"calibration: {len(cal_q)} questions {cal_hop_counts} | "
          f"{len(cal_store)} unique facts in its own store (disjoint from eval)\n")
    cal_retrieve = make_retriever(cal_store, enc)

    def run_fn(source: str, fn: str, exe=args.exe) -> dict:
        if session is not None:
            return session.run(source, fn=fn)
        return cc.run_program_proto(source, fn=fn, exe=exe)

    table_path_str = str(args.table)
    git_rev = _git_rev()

    # =======================================================================
    # 5. Calibration pass 1: permissive taus, collect (similarity,
    #    hop_correct, frame_size) for the frame-size-conditional floors,
    #    control-role similarity by frame size (the margin floors defend),
    #    the OLD Youden reference number, tau_ret, and the set of CLEAN
    #    walks that feed planted-fault calibration.
    #
    #    Real finding: `result.verified` is near-unreachable at tau_vm=0.0.
    #    The control role's cosine cleanup returns a small POSITIVE
    #    similarity (measured ~0.01-0.04, not 0 and not None), so
    #    `ctrl_sim >= tau_vm` trips on almost every walk when tau_vm=0.0 --
    #    independent of whether the retrieved chain is correct. Pipeline.
    #    answer()'s repair heuristic then makes it worse: it blames the
    #    LOWEST-similarity HOP for a failure that was actually the control
    #    check, bans that hop's (perfectly good) fact, and retries -- which
    #    often can't find a distinct alternate for a tightly-constrained
    #    hop, burning the walk down to reason="retrieval_exhausted" even
    #    though the ORIGINAL attempt-0 walk was a complete, correct chain
    #    (still recoverable via `last_trace`, which `answer()` does
    #    return). So this loop follows the same precedent the v1 script
    #    already used for `per_hop_obs`: judge chain quality from
    #    `result.trace`'s own content (complete + every triple's object
    #    matches the dataset's gold chain), never from `result.verified`
    #    or `result.reason`.
    # =======================================================================
    print("=== calibration pass 1 (tau_vm=0.0, tau_ret=-1.0) ===")
    per_hop_obs: list[tuple[float, bool, int]] = []   # (similarity, hop_correct, frame_size)
    control_obs_by_size: dict[int, list[float]] = defaultdict(list)
    accepted_ret_scores: list[float] = []
    cal_unparseable_q = 0
    cal_mojibake_excluded = 0
    cal_results: list[tuple[int, object]] = []       # (idx into cal_q, CoTResult) for clean walks
    t0 = time.perf_counter()
    for i, (q, chain_ids, h) in enumerate(zip(cal_q, cal_chains, cal_h)):
        gold_objs = dataset_objects(chain_ids, cal_store)
        cal_ctrl_calls: list[dict] = []

        def cal_run_fn(source: str, fn: str, _log=cal_ctrl_calls) -> dict:
            out = run_fn(source, fn)
            if fn == "control":
                _log.append(out)
            return out

        result = pipeline_answer(q, cal_retrieve, cal_run_fn, tau_vm=0.0, tau_ret=-1.0,
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
                # quality, so a string-equality hop_correct check here would
                # measure an encoding bug, not reasoning quality.
                cal_mojibake_excluded += 1
                continue
            correct_obj = gold_objs[hop_i] if hop_i < len(gold_objs) else None
            hop_correct = correct_obj is not None and normalize(ht.symbol or "") == normalize(correct_obj)
            per_hop_obs.append((ht.similarity, hop_correct, h))
        for ctrl in cal_ctrl_calls:
            csim = ctrl.get("similarity")
            if csim is not None:
                control_obs_by_size[h].append(csim)
        chain_ok = (bool(result.trace) and len(result.trace) == len(gold_objs)
                   and all(ht.triple is not None
                           and normalize(ht.triple.obj) == normalize(gold_objs[j] or "")
                           for j, ht in enumerate(result.trace)))
        if chain_ok:
            cal_results.append((i, result))
        if (i + 1) % 25 == 0:
            print(f"  calibration {i + 1}/{len(cal_q)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"calibration pass 1 done in {time.perf_counter() - t0:.0f}s | "
          f"{cal_unparseable_q} unparseable calibration questions | "
          f"{cal_mojibake_excluded} non-ASCII hop obs excluded (mojibake) | "
          f"{len(per_hop_obs)} per-hop observations "
          f"({sum(1 for _, c, _ in per_hop_obs if c)} correct) | "
          f"{len(cal_results)} clean gold-matching walks (planted-fault source)", flush=True)

    # -- OLD Youden number: computed-and-reported for comparison ONLY, per
    # the eval-wave redesign -- it no longer sets the deployed tau_vm. ------
    n_incorrect_obs = sum(1 for _, c, _ in per_hop_obs if not c)
    tau_vm_calibration_degenerate = n_incorrect_obs == 0
    if per_hop_obs:
        sims_arr = np.array([s for s, _, _ in per_hop_obs])
        correct_arr = np.array([c for _, c, _ in per_hop_obs])
        tau_vm_youden_reference, tau_vm_youden_auc = youden_tau(sims_arr[correct_arr], sims_arr[~correct_arr])
    else:
        tau_vm_youden_reference, tau_vm_youden_auc = float("nan"), float("nan")
    tau_ret = float(np.percentile(accepted_ret_scores, 5)) if accepted_ret_scores else 0.0
    print(f"  [reference only] tau_vm_youden = {tau_vm_youden_reference:.4f} "
          f"(AUC {tau_vm_youden_auc:.4f} on {len(per_hop_obs)} hop obs, "
          f"{n_incorrect_obs} incorrect, degenerate={tau_vm_calibration_degenerate}) | "
          f"tau_ret = {tau_ret:.4f} (5th pct of {len(accepted_ret_scores)} accepted-hop scores)\n")

    # =======================================================================
    # 6. Frame-size-conditional tau_vm floors (the actual deployed
    #    calibration -- see module docstring for why this replaced the
    #    FPR-target planted-fault scheme).
    # =======================================================================
    per_hop_obs_by_size: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for sim, correct, size in per_hop_obs:
        per_hop_obs_by_size[size].append((sim, correct))

    floor_rng = np.random.default_rng(505)
    tau_vm_floors: dict[int, float] = {}
    floor_details: dict[int, dict] = {}
    for size in sorted(per_hop_obs_by_size):
        correct_sims = np.array([s for s, c in per_hop_obs_by_size[size] if c], dtype=np.float64)
        n = len(correct_sims)
        if n == 0:
            continue
        point, lo, hi = bootstrap_quantile_ci(correct_sims, 1.0, 1000, floor_rng)
        floor = lo                                    # DEPLOY THE LOWER BOUND (conservative)
        ctrl_sims = control_obs_by_size.get(size, [])
        if ctrl_sims:
            ctrl_mean = float(np.mean(ctrl_sims))
        else:
            all_ctrl = [s for v in control_obs_by_size.values() for s in v]
            ctrl_mean = float(np.mean(all_ctrl)) if all_ctrl else float("nan")
        control_margin = (floor - ctrl_mean) if not np.isnan(ctrl_mean) else float("nan")
        tau_vm_floors[size] = floor
        floor_details[size] = {
            "n_correct_obs": n, "q01_point_estimate": point,
            "q01_ci95_lo": lo, "q01_ci95_hi": hi, "deployed_floor": floor,
            "n_control_obs": len(ctrl_sims), "control_role_mean_similarity": ctrl_mean,
            "control_margin": control_margin,
        }

    fallback_tau_vm = float(min(tau_vm_floors.values())) if tau_vm_floors else 0.3
    if not tau_vm_floors:
        print("  WARNING: zero per-frame-size floors computable -- "
              "falling back to tau_vm=0.3 for every question", flush=True)

    def tau_vm_for_size(size: int) -> float:
        return tau_vm_floors.get(size, fallback_tau_vm)

    print("=== tau_vm floors (frame-size-conditional; bootstrap-95%-CI LOWER "
          "bound of q01 correct-recovery similarity) ===")
    for size in sorted(floor_details):
        d = floor_details[size]
        print(f"  n_hop={size}: floor={d['deployed_floor']:.4f} "
              f"(q01 point={d['q01_point_estimate']:.4f}, "
              f"CI=[{d['q01_ci95_lo']:.4f},{d['q01_ci95_hi']:.4f}], n={d['n_correct_obs']}) | "
              f"control_role_mean={d['control_role_mean_similarity']:.4f} (n={d['n_control_obs']}) | "
              f"margin={d['control_margin']:.4f}")
    print(f"  fallback tau_vm (frame sizes unseen in calibration) = {fallback_tau_vm:.4f}\n")

    # =======================================================================
    # 7. Planted-fault calibration (Q21): four corruption classes on
    #    verified calibration chains, now reported as a CATCH RATE against
    #    the REAL verify decision (symbol mismatch OR sub-floor
    #    similarity), not an FPR-target threshold -- see module docstring.
    # =======================================================================
    print("=== planted-fault calibration ===")
    fault_rng = np.random.default_rng(303)
    fault_obs: dict[str, list[float]] = {c: [] for c in FAULT_CLASSES}       # informational fidelity scores
    fault_attempted: dict[str, int] = {c: 0 for c in FAULT_CLASSES}
    fault_caught: dict[str, int] = {c: 0 for c in FAULT_CLASSES}
    fault_vm_calls = 0
    fault_mojibake_excluded = 0
    n_fault_chains = 0
    t1 = time.perf_counter()
    for ci, (qi, result) in enumerate(cal_results):
        if fault_vm_calls >= MAX_FAULT_VM_CALLS:
            print(f"  VM-call cap reached ({MAX_FAULT_VM_CALLS}) -- stopping early "
                  f"at chain {ci}/{len(cal_results)}", flush=True)
            break
        q = cal_q[qi]
        plan = parse_question(q)
        if plan is None or not result.trace:
            continue
        triples = [ht.triple for ht in result.trace if ht.triple is not None]
        if len(triples) != len(result.trace) or not triples:
            continue
        display_rels = _display_rels(plan, triples)
        floor = tau_vm_for_size(len(triples))
        instances = _fault_instances(triples, cal_store, fault_rng)
        n_fault_chains += 1
        for cls, insts in instances.items():
            for corrupted, compare_objs in insts:
                negs, n_calls, n_moji, n_caught = _run_fault_program(
                    corrupted, display_rels, compare_objs, run_fn, tau_vm_floor=floor)
                fault_obs[cls].extend(negs)
                fault_attempted[cls] += n_calls
                fault_caught[cls] += n_caught
                fault_vm_calls += n_calls
                fault_mojibake_excluded += n_moji
        if (ci + 1) % 25 == 0:
            print(f"  fault-calibration {ci + 1}/{len(cal_results)} chains "
                  f"({time.perf_counter() - t1:.0f}s, {fault_vm_calls} VM calls so far)", flush=True)
    print(f"planted-fault calibration done in {time.perf_counter() - t1:.0f}s | "
          f"{n_fault_chains} verified chains used | {fault_vm_calls} VM calls | "
          f"{fault_mojibake_excluded} mojibake-excluded", flush=True)

    class_stats = {c: _quantile_stats(fault_obs[c]) for c in FAULT_CLASSES}
    catch_rate = {c: (fault_caught[c] / fault_attempted[c] if fault_attempted[c] else float("nan"))
                  for c in FAULT_CLASSES}
    untested_against = [c for c in FAULT_CLASSES if fault_attempted[c] < 20]

    print("  catch rate (symbol mismatch OR sub-floor similarity -- the REAL verify decision):")
    for c in FAULT_CLASSES:
        print(f"    {c:20s} attempted={fault_attempted[c]:4d} caught={fault_caught[c]:4d} "
              f"catch_rate={catch_rate[c]:.4f} | fidelity(informational, NOT truth): "
              f"n={class_stats[c]['n']:4d} mean={class_stats[c]['mean']:.4f} "
              f"q50={class_stats[c]['q50']:.4f} q99={class_stats[c]['q99']:.4f}")
    if untested_against:
        print(f"  untested_against (n_attempted<20): {untested_against}")

    # =======================================================================
    # 8. Organic-confusable held-out test (never used for calibration):
    #    ~100 eval questions, substitute a fact from an OTHER eval chain
    #    sharing an entity token; report catch rate at the deployed floors.
    # =======================================================================
    print("\n=== organic-confusable held-out test (never calibration) ===", flush=True)
    STOP_LEN = 3
    token_index: dict[str, set[int]] = defaultdict(set)
    for qi, chain_ids in enumerate(chains):
        for fid_ in chain_ids:
            t = parse_fact(store[fid_])
            if t is None:
                continue
            for tok in (normalize(t.subj) + " " + normalize(t.obj)).split():
                if len(tok) >= STOP_LEN:
                    token_index[tok].add(qi)

    organic_rng = np.random.default_rng(101)
    organic_sims: list[float] = []
    organic_attempted = 0
    organic_caught = 0
    organic_n_chains = 0
    for qi in organic_rng.permutation(len(questions)):
        if organic_n_chains >= 100:
            break
        qi = int(qi)
        parsed = _dataset_plan_and_triples(questions[qi], chains[qi], store)
        if parsed is None:
            continue
        plan, triples, display_rels, true_objs = parsed
        own_tokens = set()
        for t in triples:
            own_tokens |= set(normalize(t.subj).split()) | set(normalize(t.obj).split())
        own_tokens = {tok for tok in own_tokens if len(tok) >= STOP_LEN}
        own_fact_ids = set(chains[qi])
        candidate_qs: set[int] = set()
        for tok in own_tokens:
            candidate_qs |= token_index.get(tok, set())
        candidate_qs.discard(qi)
        candidate_facts = [fid_ for j in candidate_qs for fid_ in chains[j]
                           if fid_ not in own_fact_ids]
        if not candidate_facts:
            continue
        fid_pick = int(candidate_facts[int(organic_rng.integers(0, len(candidate_facts)))])
        sub = parse_fact(store[fid_pick])
        if sub is None:
            continue
        hop_i = int(organic_rng.integers(0, len(triples)))
        corrupted = list(triples)
        t0_ = triples[hop_i]
        corrupted[hop_i] = Triple(obj=sub.obj, rel=t0_.rel, subj=t0_.subj)
        # Only the corrupted hop is a fault -- the chain's other (still
        # correctly-bound) hops must not count toward attempted/caught.
        compare_only_hop_i: list[str | None] = [None] * len(triples)
        compare_only_hop_i[hop_i] = true_objs[hop_i]
        negs, n_calls, _n_moji, n_caught = _run_fault_program(
            corrupted, display_rels, compare_only_hop_i, run_fn,
            tau_vm_floor=tau_vm_for_size(len(triples)))
        organic_sims.extend(negs)
        organic_attempted += n_calls
        organic_caught += n_caught
        organic_n_chains += 1
    organic_stats = _quantile_stats(organic_sims)
    organic_catch_rate = organic_caught / organic_attempted if organic_attempted else float("nan")
    print(f"  organic-confusable: {organic_n_chains} chains, {organic_attempted} attempted, "
          f"caught={organic_caught}, catch_rate={organic_catch_rate:.4f} | "
          f"fidelity(informational): mean_sim={organic_stats['mean']:.4f}")

    # =======================================================================
    # 9. Random-distractor sanity gate (unit test, NEVER calibration):
    #    ~50 distractor-substituted bindings; report catch rate (expect ~100%).
    # =======================================================================
    print("\n=== random-distractor sanity gate (unit test, not calibration) ===", flush=True)
    distractor_titles: list[str] = []
    try:
        for chunk in iter_distractors(50):
            distractor_titles.extend(t.split(".", 1)[0].strip() for t in chunk)
            break
    except Exception as e:
        print(f"  WARNING: could not load distractor titles for the sanity gate: {e}")

    random_rng = np.random.default_rng(202)
    random_sims: list[float] = []
    random_attempted = 0
    random_caught = 0
    random_n = 0
    if distractor_titles:
        di = 0
        for qi in random_rng.permutation(len(questions)):
            if random_n >= 50:
                break
            qi = int(qi)
            parsed = _dataset_plan_and_triples(questions[qi], chains[qi], store)
            if parsed is None:
                continue
            plan, triples, display_rels, true_objs = parsed
            hop_i = int(random_rng.integers(0, len(triples)))
            title = distractor_titles[di % len(distractor_titles)]
            di += 1
            t0_ = triples[hop_i]
            corrupted = list(triples)
            corrupted[hop_i] = Triple(obj=title, rel=t0_.rel, subj=t0_.subj)
            # Only the corrupted hop is a fault -- see the organic-confusable
            # test's identical scoping note above.
            compare_only_hop_i: list[str | None] = [None] * len(triples)
            compare_only_hop_i[hop_i] = true_objs[hop_i]
            negs, n_calls, _n_moji, n_caught = _run_fault_program(
                corrupted, display_rels, compare_only_hop_i, run_fn,
                tau_vm_floor=tau_vm_for_size(len(triples)))
            random_sims.extend(negs)
            random_attempted += n_calls
            random_caught += n_caught
            random_n += 1
    else:
        print("  no distractor titles available -- random-distractor gate skipped")
    random_stats = _quantile_stats(random_sims)
    random_catch_rate = random_caught / random_attempted if random_attempted else float("nan")
    print(f"  random-distractor: {random_n} bindings, {random_attempted} attempted, "
          f"caught={random_caught}, catch_rate={random_catch_rate:.4f} (expect ~1.0) | "
          f"fidelity(informational): mean_sim={random_stats['mean']:.4f}\n")

    # -- calibration card ----------------------------------------------------
    cal_store_snapshot_hash = store_hash(cal_store)
    hardness_entries = (
        [("random_distractor", {"n_attempted": random_attempted, "n_caught": random_caught,
                                "catch_rate": random_catch_rate, **random_stats})]
        + [("organic_confusable", {"n_attempted": organic_attempted, "n_caught": organic_caught,
                                   "catch_rate": organic_catch_rate, **organic_stats})]
        + [(c, {"n_attempted": fault_attempted[c], "n_caught": fault_caught[c],
               "catch_rate": catch_rate[c], **class_stats[c]}) for c in FAULT_CLASSES]
    )
    # Ascending catch_rate -- the LOWEST-catch (most dangerous / hardest for
    # the verify layer to catch) first; untested (n_attempted=0) classes
    # sort last (nan catch_rate treated as "no evidence of danger yet").
    hardness_ladder = [
        {"class": name, **stats} for name, stats in
        sorted(hardness_entries,
              key=lambda kv: kv[1]["catch_rate"] if kv[1]["n_attempted"] else 2.0)
    ]
    calibration_card = {
        "note": ("Similarity measures binding FIDELITY (was something cleanly "
                 "recovered), not semantic TRUTH (was the correct thing "
                 "recovered). A wrong-entity/inverted-direction planted fault "
                 "binds a definitely-wrong filler and recovers it faithfully "
                 "-- often at ~1.0 similarity in a single-binding (1-hop, "
                 "zero-crosstalk) frame -- so fault-class similarity "
                 "distributions below CANNOT be calibrated into a truth-"
                 "discriminating threshold; they are informational binding-"
                 "fidelity scores only. Content-correctness is enforced by "
                 "the exact string-match check pipeline.answer() already "
                 "runs alongside tau_vm ('normalize(symbol) != "
                 "normalize(triples[i].obj)'), not by similarity "
                 "thresholding. tau_vm is therefore calibrated as a FRAME-"
                 "SIZE-CONDITIONAL FLOOR over CORRECT recoveries' own "
                 "fidelity (defending against absent/garbled recoveries -- "
                 "the control-role failure mode); the semantic hardness "
                 "ladder for content-correctness belongs to the RETRIEVAL "
                 "side (relation+entity structural filtering), not this VM "
                 "verify layer. 'catch_rate' below is the honest measurement "
                 "of that string-match defense (expect ~1.0 for every class, "
                 "by construction: a corrupted binding almost always fails "
                 "the string match against the true object)."),
        "tau_vm_floors": {str(size): floor_details[size] for size in sorted(floor_details)},
        "fallback_tau_vm": fallback_tau_vm,
        "bootstrap_draws": 1000,
        "fault_classes": {
            c: {**class_stats[c], "n_attempted": fault_attempted[c],
               "n_caught": fault_caught[c], "catch_rate": catch_rate[c]}
            for c in FAULT_CLASSES
        },
        "hardness_ladder_by_catch_rate": hardness_ladder,
        "mix_counts": {c: class_stats[c]["n"] for c in FAULT_CLASSES},
        "n_fault_chains_processed": n_fault_chains,
        "n_fault_vm_calls": fault_vm_calls,
        "fault_mojibake_excluded": fault_mojibake_excluded,
        "untested_against": untested_against,
        "organic_confusable": {"n_chains": organic_n_chains, "n_attempted": organic_attempted,
                               "n_caught": organic_caught, "catch_rate": organic_catch_rate,
                               **organic_stats},
        "random_distractor_sanity": {"n_bindings": random_n, "n_attempted": random_attempted,
                                     "n_caught": random_caught, "catch_rate": random_catch_rate,
                                     **random_stats},
        "tau_vm_youden_reference": tau_vm_youden_reference,
        "tau_vm_youden_auc": tau_vm_youden_auc,
        "tau_vm_youden_reference_degenerate": tau_vm_calibration_degenerate,
        "store_snapshot_hash": cal_store_snapshot_hash,
    }
    print("=== calibration card ===")
    print(f"  hardness ladder (by catch rate, ascending = most dangerous first): " + " < ".join(
        f"{e['class']}(catch={e['catch_rate']:.3f})" if e["n_attempted"] else f"{e['class']}(untested)"
        for e in hardness_ladder))
    print(f"  mix counts: {calibration_card['mix_counts']} | "
          f"organic n={organic_n_chains} | random n={random_n}")
    print(f"  tau_vm_floors: { {k_: round(v_, 4) for k_, v_ in tau_vm_floors.items()} } | "
          f"fallback={fallback_tau_vm:.4f}\n")

    # =======================================================================
    # 10. Main eval loop: three arms per question + harvest.
    # =======================================================================
    print(f"=== eval ({len(questions)} questions, tau_vm_floors={ {k_: round(v_, 4) for k_, v_ in tau_vm_floors.items()} } "
          f"fallback={fallback_tau_vm:.4f} tau_ret={tau_ret:.4f}) ===")
    acc: dict[str, dict[int, list[int]]] = {
        "retrieval_only": defaultdict(list), "chase_only": defaultdict(list), "cot": defaultdict(list)}
    wall_ms: dict[str, list[float]] = {"retrieval_only": [], "chase_only": [], "cot": []}
    repairs_hist: Counter = Counter()
    verified_count = 0
    reason_hist: Counter = Counter()   # outcome per question, incl. plan-time refusals
    claimed_correct = 0
    control_pass = 0
    control_total = 0
    unparseable_questions = 0
    unparseable_facts = 0
    non_ascii_questions = 0
    sub_tau_violations = 0
    sensitivity_records: list[dict] = []
    example_printed = False
    eval_store_hash = store_hash(store)
    run_timestamp = datetime.now(timezone.utc).isoformat()

    # counterfactual-neighborhood state (schema `counterfactuals[]`)
    cf_enabled = not args.no_counterfactuals
    cf_budget = [int(args.max_cf_vm_calls)]
    cf_rng = np.random.default_rng(404)
    cf_chains = 0
    cf_vm_calls = 0
    cf_mojibake = 0
    cf_truncated = 0
    cf_skipped_budget = 0
    cf_wall_ms: list[float] = []
    cf_attempted: dict[str, int] = {c: 0 for c in FAULT_CLASSES}
    cf_caught: dict[str, int] = {c: 0 for c in FAULT_CLASSES}
    cf_caught_by: Counter = Counter()
    cf_hop_map: dict[str, dict[str, list[int]]] = {c: {} for c in FAULT_CLASSES}   # cls -> hop -> [attempted, caught]
    cf_frame_map: dict[str, list[int]] = {}                                         # n_hop -> [attempted, caught]
    cf_escaped_examples: list[dict] = []
    cf_escaped_sims: list[float] = []
    cf_caught_sims: list[float] = []

    t2 = time.perf_counter()
    with open(harvest_path, "w", encoding="utf-8") as harvest_f:
        for i, (q, a, h, chain_ids) in enumerate(zip(questions, answers, hops, chains)):
            gold = normalize(a)
            gold_objs = dataset_objects(chain_ids, store)
            plan = parse_question(q)
            tau_vm_q = tau_vm_for_size(h)

            # -- retrieval-only ---------------------------------------------
            s0 = time.perf_counter()
            r1 = retrieval_only(q, retrieve)
            wall_ms["retrieval_only"].append((time.perf_counter() - s0) * 1000)
            if r1 is None:
                unparseable_facts += 1
            acc["retrieval_only"][h].append(int(r1 is not None and normalize(r1) == gold))

            # -- chase-only (full-store masked argmax) -----------------------
            s0 = time.perf_counter()
            r2 = chase_only(q, h, retrieve, store_size)
            wall_ms["chase_only"].append((time.perf_counter() - s0) * 1000)
            if r2 is None:
                unparseable_facts += 1
            acc["chase_only"][h].append(int(r2 is not None and normalize(r2) == gold))

            # -- CoT pipeline ---------------------------------------------------
            ctrl_calls: list[dict] = []
            ret_calls: list[dict] = []

            def vm_run_fn(source: str, fn: str, _log=ctrl_calls) -> dict:
                out = run_fn(source, fn)
                if fn == "control":
                    _log.append(out)
                return out

            def retrieve_log(query: str, k: int, _log=ret_calls) -> list[tuple[float, str]]:
                out = retrieve(query, k)
                _log.append({"query": query, "candidates": out})
                return out

            s0 = time.perf_counter()
            result = pipeline_answer(q, retrieve_log, vm_run_fn, tau_vm=tau_vm_q, tau_ret=tau_ret,
                                     top_k=args.top_k, max_repairs=args.max_repairs,
                                     lookup=(index.hop if index is not None else None),
                                     known=known)
            wall_ms["cot"].append((time.perf_counter() - s0) * 1000)
            repairs_hist[result.repairs_used] += 1
            reason_hist["verified" if result.verified else (result.reason or "vm_verify_failed")] += 1
            if result.reason == "unparseable":
                unparseable_questions += 1

            cot_pred = result.answer if result.verified else None
            cot_correct = int(cot_pred is not None and normalize(cot_pred) == gold)
            acc["cot"][h].append(cot_correct)

            if result.verified:
                verified_count += 1
                claimed_correct += cot_correct
                if not all(ht.similarity is not None and ht.similarity >= tau_vm_q for ht in result.trace):
                    sub_tau_violations += 1

            # -- counterfactual neighborhood (every verified chain) ------------
            cf_records: list[dict] | None = None
            if cf_enabled and result.verified and plan is not None:
                cf_triples = [ht.triple for ht in result.trace]
                if cf_triples and all(t is not None for t in cf_triples):
                    if cf_budget[0] <= 0:
                        cf_skipped_budget += 1
                    else:
                        s0 = time.perf_counter()
                        cf_records, n_cf_calls, n_cf_moji, cf_trunc = _counterfactual_neighborhood(
                            cf_triples, _display_rels(plan, cf_triples), cal_store,
                            cf_rng, run_fn, tau_vm_q, cf_budget)
                        cf_wall_ms.append((time.perf_counter() - s0) * 1000)
                        cf_chains += 1
                        cf_vm_calls += n_cf_calls
                        cf_mojibake += n_cf_moji
                        cf_truncated += int(cf_trunc)
                        for r_ in cf_records:
                            cf_attempted[r_["cls"]] += 1
                            hm = cf_hop_map[r_["cls"]].setdefault(str(r_["hop"]), [0, 0])
                            fm = cf_frame_map.setdefault(str(r_["n_hop"]), [0, 0])
                            hm[0] += 1
                            fm[0] += 1
                            if r_["caught"]:
                                cf_caught[r_["cls"]] += 1
                                cf_caught_by[r_["caught_by"]] += 1
                                hm[1] += 1
                                fm[1] += 1
                                if r_["similarity"] is not None:
                                    cf_caught_sims.append(r_["similarity"])
                            else:
                                if r_["similarity"] is not None:
                                    cf_escaped_sims.append(r_["similarity"])
                                if len(cf_escaped_examples) < 20:
                                    cf_escaped_examples.append({"question": q, **r_})

            # Per-program control accounting: EVERY control-role call across
            # every attempt (not just the last) counts toward the denominator
            # -- a question that retried once can log up to 2 control calls.
            for ctrl in ctrl_calls:
                control_total += 1
                if not control_violation(ctrl.get("similarity"), ctrl.get("result"), tau_vm_q):
                    control_pass += 1

            if any(not ht.fact.isascii() for ht in result.trace):
                non_ascii_questions += 1

            # cheap +/-20% tau_vm sensitivity: reuses this attempt's cached
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
                    "tau_vm_used": tau_vm_q,
                })

            # -- harvest (every eval question, per docs/schemas/cot-harvest-schema.md) --
            rec = _harvest_record(q, plan, gold, a, h, result, ret_calls, tau_vm_q, tau_ret,
                                  eval_store_hash, table_path_str, git_rev, run_timestamp,
                                  counterfactuals=cf_records)
            harvest_f.write(json.dumps(rec) + "\n")

            if not example_printed and result.verified:
                print(f"\n  example (q{i}): {q}")
                for ht in result.trace:
                    print(f"    hop: fact={ht.fact!r} symbol={ht.symbol!r} sim={ht.similarity}")
                print(f"    -> answer={result.answer!r} verified={result.verified} "
                     f"repairs={result.repairs_used}\n")
                example_printed = True

            if (i + 1) % 25 == 0:
                print(f"  eval {i + 1}/{len(questions)} ({time.perf_counter() - t2:.0f}s)", flush=True)

    if not example_printed:
        print("\n  (no verified example encountered to print)\n")
    print(f"eval done in {time.perf_counter() - t2:.0f}s")
    print(f"wrote {harvest_path} ({len(questions)} records)\n")

    # =======================================================================
    # 10b. Counterfactual-neighborhood summary (class x hop catch map).
    # =======================================================================
    cf_total_attempted = sum(cf_attempted.values())
    cf_total_caught = sum(cf_caught.values())
    cf_catch_rate = {c: (cf_caught[c] / cf_attempted[c] if cf_attempted[c] else float("nan"))
                     for c in FAULT_CLASSES}
    print("=== counterfactual neighborhood (harvest `counterfactuals[]`) ===")
    if not cf_enabled:
        print("  disabled (--no-counterfactuals)")
    else:
        print(f"  chains={cf_chains} (of {verified_count} verified; {cf_skipped_budget} skipped: budget) | "
              f"VM calls={cf_vm_calls} (cap {args.max_cf_vm_calls}; truncated chains={cf_truncated}) | "
              f"mojibake-excluded hops={cf_mojibake} | mean {float(np.mean(cf_wall_ms)) if cf_wall_ms else 0.0:.0f} ms/chain")
        print(f"  hard negatives (caught)={cf_total_caught} | ESCAPED (would-be false accepts)="
              f"{cf_total_attempted - cf_total_caught} | overall catch="
              f"{cf_total_caught / cf_total_attempted if cf_total_attempted else float('nan'):.4f}")
        print(f"  caught_by: {dict(cf_caught_by)}")
        for c in FAULT_CLASSES:
            hops = " ".join(f"h{k_}={v_[1]}/{v_[0]}" for k_, v_ in sorted(cf_hop_map[c].items()))
            print(f"    {c:20s} attempted={cf_attempted[c]:4d} caught={cf_caught[c]:4d} "
                  f"catch={cf_catch_rate[c]:.4f} | by hop: {hops}")
        frames = " ".join(f"n{k_}={v_[1]}/{v_[0]}" for k_, v_ in sorted(cf_frame_map.items()))
        print(f"  by frame size: {frames}")
        if cf_escaped_examples:
            print(f"  first escaped examples ({len(cf_escaped_examples)} shown):")
            for ex in cf_escaped_examples[:5]:
                print(f"    [{ex['cls']} hop{ex['hop']}/{ex['n_hop']}] planted={ex['planted']['obj']!r} "
                      f"compare={ex['compare_obj']!r} symbol={ex['symbol']!r} sim={ex['similarity']}")
    print()

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

    # +/-20% tau_vm sensitivity (re-verification only, no re-retrieval);
    # scales EACH record's own frame-size floor, since there's no single
    # global tau_vm anymore.
    def sensitivity_at(scale: float) -> dict:
        would_verify_all = 0
        would_correct = 0
        for rec in sensitivity_records:
            tau = rec["tau_vm_used"] * scale
            wv = (rec["symbol_ok"]
                  and all(s is not None and s >= tau for s in rec["sims"])
                  and not control_violation(rec["ctrl_sim"], rec["ctrl_result"], tau))
            if wv:
                would_verify_all += 1
                if rec["predicted"] == rec["gold"]:
                    would_correct += 1
        coverage = would_verify_all / len(questions) if questions else 0.0
        precision = would_correct / would_verify_all if would_verify_all else 0.0
        return {"scale": scale, "verified_coverage": coverage,
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
    print("  outcomes: " + "  ".join(f"{k}={v}" for k, v in reason_hist.most_common()))
    if known is not None and hasattr(known, "n_calls"):
        print(f"  verify-plan VM QUERY calls: {known.n_calls}")
    if session is not None:
        print(f"  resident VM: {session.n_requests} requests on one process")
        session.close()
    print(f"  non-ascii-walked-fact questions: {non_ascii_questions}")
    print(f"  sub-tau verified violations: {sub_tau_violations}")
    print(f"  repairs histogram: {dict(sorted(repairs_hist.items()))}")
    print(f"\n  tau_vm sensitivity (re-verify cached decisions only, scaling each "
          f"record's own frame-size floor):")
    for k_, v_ in sensitivity.items():
        print(f"    {k_}: coverage={v_['verified_coverage']:.3f} "
              f"precision={v_['claimed_precision']:.3f} (n={v_['verified_count']})")

    # =======================================================================
    # 11. Verdict.
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

    config = {k_: (str(v_) if isinstance(v_, pathlib.Path) else v_) for k_, v_ in vars(args).items()}
    config["eval_seed"] = eval_seed
    config["calibration_seed"] = calibration_seed
    config["n_facts_only"] = n_facts_only
    config["n_distractors_added"] = n_distractors_added
    config["store_size"] = store_size

    out = {
        "config": config,
        "n_eval_questions": len(questions),
        "n_calibration_questions": len(cal_q),
        "eval_hop_counts": {str(k_): v_ for k_, v_ in hop_counts.items()},
        "calibration_hop_counts": {str(k_): v_ for k_, v_ in cal_hop_counts.items()},
        "n_store_facts": len(store),
        "calibration": {
            "cal_unparseable_questions": cal_unparseable_q,
            "cal_mojibake_excluded_hop_obs": cal_mojibake_excluded,
            "n_hop_observations": len(per_hop_obs),
            "n_incorrect_hop_observations": n_incorrect_obs,
            "n_verified_calibration_walks": len(cal_results),
            "tau_vm_floors": {str(k_): v_ for k_, v_ in tau_vm_floors.items()},
            "fallback_tau_vm": fallback_tau_vm,
            "tau_vm_youden_reference": tau_vm_youden_reference,
            "tau_vm_youden_auc": tau_vm_youden_auc,
            "tau_vm_calibration_degenerate": tau_vm_calibration_degenerate,
            "tau_ret": tau_ret,
        },
        "calibration_card": calibration_card,
        "arms": {
            arm: {"overall": acc_overall(arm), "by_hop": acc_by_hop(arm),
                 "mean_wall_ms": float(np.mean(wall_ms[arm]))}
            for arm in ("retrieval_only", "chase_only", "cot")
        },
        "cot_verified_coverage": verified_coverage,
        "cot_verified_count": verified_count,
        "outcomes": dict(reason_hist),
        "verify_plan": args.verify_plan,
        "resident": bool(args.resident),
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
        "harvest_file": str(harvest_path),
        "n_harvest_records": len(questions),
        "counterfactual_neighborhood": {
            "enabled": cf_enabled,
            "note": ("Per verified eval chain: the four planted-fault classes re-planted on the "
                     "pipeline's own accepted triples (swap objects from the calibration store, "
                     "never distractors), each corrupted hop recovered through the VM at the "
                     "deployed frame-size floor. caught = the real verify decision rejects it; "
                     "escaped = a would-be false accept. Per-record detail lives in the harvest "
                     "file's `counterfactuals[]`."),
            "n_chains": cf_chains,
            "n_verified_chains": verified_count,
            "n_skipped_budget": cf_skipped_budget,
            "n_vm_calls": cf_vm_calls,
            "vm_call_cap": int(args.max_cf_vm_calls),
            "n_truncated_chains": cf_truncated,
            "n_mojibake_excluded_hops": cf_mojibake,
            "mean_wall_ms_per_chain": (float(np.mean(cf_wall_ms)) if cf_wall_ms else 0.0),
            "n_attempted": cf_total_attempted,
            "n_caught": cf_total_caught,
            "n_escaped": cf_total_attempted - cf_total_caught,
            "overall_catch_rate": (cf_total_caught / cf_total_attempted if cf_total_attempted else float("nan")),
            "caught_by": dict(cf_caught_by),
            "by_class": {c: {"n_attempted": cf_attempted[c], "n_caught": cf_caught[c],
                             "catch_rate": cf_catch_rate[c],
                             "by_hop": {k_: {"n_attempted": v_[0], "n_caught": v_[1]}
                                        for k_, v_ in sorted(cf_hop_map[c].items())}}
                         for c in FAULT_CLASSES},
            "by_frame_size": {k_: {"n_attempted": v_[0], "n_caught": v_[1]}
                              for k_, v_ in sorted(cf_frame_map.items())},
            "caught_similarity": _quantile_stats(cf_caught_sims),
            "escaped_similarity": _quantile_stats(cf_escaped_sims),
            "escaped_examples": cf_escaped_examples,
        },
        "git_rev": git_rev,
        "verdict": {
            "clause1_cot_beats_retrieval_2h_3h": clause1,
            "clause2_claimed_precision_ge_0.90": clause2,
            "clause3_zero_sub_tau_violations": clause3,
            "clause4_control_pass_rate_ge_0.95": clause4,
            "overall": overall,
        },
    }

    json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
