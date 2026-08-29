"""exp_m3_cascade — teacher only in the ambiguous band (queue item 5).

The one metric where the live teacher still clearly leads the table is
open-set CONFIDENCE (route-vs-spawn separation: AUC 0.825 vs 0.666 in the
open-set screen). The cascade thesis: the table decides everything far from
the boundary; only challenges inside an ambiguity band of width delta
around the decision boundary escalate to the teacher. If most decisions are
easy, a small teacher-call fraction should recover most of the gap.

Protocol: exp_m3_domain_routing.main's exact sampling (rng(42) open-set
split, per-domain seeds), centroid scorer per domain (the validated rule).
Decision scalar d = min(s1 - tau_match, margin - tau_margin); route iff
d >= 0; |d| < delta -> the teacher's own d_ref (its own calibrated taus)
decides, and the teacher's domain pick replaces the table's. delta=0 is the
pure table; delta=inf is the pure teacher. Both tau_match values are
Youden-calibrated in-run on closed-vs-open best scores — in-sample, like
every recorded screen's tau (the screens are the calibration method).

Metrics per delta: teacher-call fraction; spawn detection (frac of
never-seen-domain challenges sent to spawn); routed precision and coverage
on closed-set challenges; estimated mean latency per challenge (table
encode always + teacher encode on escalations only, using this run's
measured per-passage times). Standalone; never imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import (  # noqa: E402
    DOMAINS_DIR, EXCLUDE, _load_semantic_words, sample_passages, youden_tau)

V4 = r"I:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz"
TAU_MARGIN = 0.02                    # the shipped bridge default


def sample_with_open(exemplars: int, challenges: int, files: int, open_n: int):
    domains = sorted(d.name for d in DOMAINS_DIR.iterdir()
                     if d.is_dir() and d.name not in EXCLUDE)
    rng = np.random.default_rng(42)
    open_set = sorted(rng.choice(domains, size=open_n, replace=False))
    ex_t, ex_d, ch_t, ch_d, op_t = [], [], [], [], []
    for d in domains:
        ps = sample_passages(DOMAINS_DIR / d, files, exemplars + challenges,
                             seed=int(rng.integers(1 << 31)))
        if len(ps) < 10:
            continue
        half = min(exemplars, len(ps) // 2)
        if d in open_set:
            op_t.extend(ps[half:half + challenges])
            continue
        ex_t.extend(ps[:half])
        ex_d.extend([d] * half)
        take = ps[half:half + challenges]
        ch_t.extend(take)
        ch_d.extend([d] * len(take))
    return ex_t, ex_d, ch_t, ch_d, op_t, open_set


def centroid_stats(CH: np.ndarray, EX: np.ndarray, ex_dom: list[str]):
    """-> (s1, margin, picked_domain) per challenge row, centroid scorer."""
    a = CH / (np.linalg.norm(CH, axis=1, keepdims=True) + 1e-12)
    b = EX / (np.linalg.norm(EX, axis=1, keepdims=True) + 1e-12)
    ed = np.asarray(ex_dom)
    doms = sorted(set(ex_dom))
    C = np.stack([b[ed == d].mean(axis=0) for d in doms])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = a @ C.T
    order = np.argsort(-S, axis=1)
    s1 = S[np.arange(len(S)), order[:, 0]]
    s2 = S[np.arange(len(S)), order[:, 1]]
    picked = np.asarray(doms)[order[:, 0]]
    return s1, s1 - s2, picked


def decision_scalar(s1, margin, tau_match):
    return np.minimum(s1 - tau_match, margin - TAU_MARGIN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=V4)
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 9.0])
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table: {pathlib.Path(args.table).name} | teacher: MiniLM-L6-v2\n")

    ex_t, ex_d, ch_t, ch_d, op_t, open_set = sample_with_open(40, 40, 8, 5)
    print(f"{len(ex_t)} exemplars / {len(ch_t)} closed challenges / "
          f"{len(op_t)} open-set (held out: {open_set})")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(args.table)
    all_ch = ch_t + op_t
    t0 = time.perf_counter()
    EX = np.stack([enc.encode(t).ravel() for t in ex_t]).astype(np.float32)
    CH = np.stack([enc.encode(t).ravel() for t in all_ch]).astype(np.float32)
    tbl_us = (time.perf_counter() - t0) / (len(ex_t) + len(all_ch)) * 1e6

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t0 = time.perf_counter()
    EXr = np.asarray(model.encode(ex_t, batch_size=64, show_progress_bar=False))
    CHr = np.asarray(model.encode(all_ch, batch_size=64, show_progress_bar=False))
    ref_us = (time.perf_counter() - t0) / (len(ex_t) + len(all_ch)) * 1e6
    print(f"encoded: table {tbl_us:.0f} us/passage, teacher {ref_us:.0f} us/passage (batched)\n")

    is_closed = np.array([1] * len(ch_t) + [0] * len(op_t), dtype=bool)
    truth = np.asarray(ch_d + ["<open>"] * len(op_t))

    s1_t, mg_t, pick_t = centroid_stats(CH, EX, ex_d)
    s1_r, mg_r, pick_r = centroid_stats(CHr, EXr, ex_d)
    tau_t, auc_t = youden_tau(s1_t[is_closed], s1_t[~is_closed])
    tau_r, auc_r = youden_tau(s1_r[is_closed], s1_r[~is_closed])
    print(f"closed-vs-open AUC on best-centroid score: table {auc_t:.4f} | teacher {auc_r:.4f}")
    print(f"calibrated tau_match: table {tau_t:.4f} | teacher {tau_r:.4f}\n")

    d_t = decision_scalar(s1_t, mg_t, tau_t)
    d_r = decision_scalar(s1_r, mg_r, tau_r)

    out = {"config": vars(args), "auc": {"table": auc_t, "teacher": auc_r},
           "tau_match": {"table": float(tau_t), "teacher": float(tau_r)},
           "encode_us": {"table": tbl_us, "teacher_batched": ref_us},
           "sweep": {}}
    print(f"{'delta':>6} {'tch%':>6} {'spawn-det':>9} {'routed%':>8} "
          f"{'routed-acc':>10} {'us/chal':>8}")
    for delta in args.deltas:
        amb = np.abs(d_t) < delta
        route = np.where(amb, d_r >= 0, d_t >= 0)
        pick = np.where(amb, pick_r, pick_t)
        spawn_det = float((~route[~is_closed]).mean())
        routed_cl = route[is_closed]
        acc = float((pick[is_closed][routed_cl] == truth[is_closed][routed_cl]).mean()) \
            if routed_cl.any() else 0.0
        frac = float(amb.mean())
        us = tbl_us + frac * ref_us
        out["sweep"][f"delta={delta}"] = {
            "teacher_frac": frac, "spawn_detect": spawn_det,
            "frac_closed_routed": float(routed_cl.mean()),
            "acc_routed": acc, "est_us_per_challenge": us}
        print(f"{delta:>6.2f} {frac:>6.1%} {spawn_det:>9.3f} "
              f"{routed_cl.mean():>8.3f} {acc:>10.3f} {us:>8.0f}")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_cascade.json").write_text(json.dumps(out, indent=1),
                                             encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_cascade.json'}")


if __name__ == "__main__":
    main()
