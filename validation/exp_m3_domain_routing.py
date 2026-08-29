"""exp_m3_domain_routing — the production table on real 42-domain text.

E:\\datasets\\domains holds 42 folder-labeled corpora (books/documents). This
screen asks the production questions the axiom screen could only approximate:

  1. ROUTING: seed N exemplar passages per domain, route held-out passages by
     max cosine to individual exemplars (CubbyBridge's shipped rule). Macro /
     micro vs matched nulls AND a pure-noise encoder null (the baseline is a
     property of the procedure -- measured, not borrowed).
  2. TAU: same/different max-cosine distributions on production-shaped data ->
     ROC-AUC + Youden tau. M2 proved no earlier tau transfers; this is the
     first calibratable one.
  3. OPEN-SET (spawn detection): challenges from domains NEVER seeded must
     score LOW against every exemplar -- the route-vs-spawn decision. Reported
     as separation between closed-set and open-set best-cosine distributions.

Encoder: the production FastWordEncoder table (I:\\CUBBY-TRAINED-MODELS\\
fastword_table_v1.npz), loaded module-by-file-path so no mowm package import
occurs (semantic_words is numpy-only by design). MiniLM runs as the quality
reference only. Standalone: never imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAINS_DIR = pathlib.Path(r"E:\datasets\domains")
TABLE = pathlib.Path(r"I:\CUBBY-TRAINED-MODELS\fastword_table_v1.npz")
SEMANTIC_WORDS = pathlib.Path(r"C:\Users\grill\Documents\GitHub\mowm\mowm\encoding\semantic_words.py")
EXCLUDE = {"_unsure", "my_identity", "gutemberg_books_unclassified"}  # unlabeled/personal

K, L = 80, 128


def _load_semantic_words():
    spec = importlib.util.spec_from_file_location("semantic_words", SEMANTIC_WORDS)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_passages(domain_dir: pathlib.Path, n_files: int, n_passages: int,
                    chunk_lo: int = 300, chunk_hi: int = 900,
                    seed: int = 0) -> list[str]:
    """Paragraph-merged passages from up to n_files texts of one domain."""
    rng = np.random.default_rng(seed)
    files = sorted(p for p in domain_dir.iterdir() if p.suffix == ".txt")
    if not files:
        return []
    rng.shuffle(files)
    out: list[str] = []
    for f in files[:n_files]:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")[:400_000]
        except OSError:
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if len(para) < 40:
                continue
            buf = f"{buf} {para}".strip() if buf else para
            if len(buf) >= chunk_lo:
                out.append(buf[:chunk_hi])
                buf = ""
        if len(out) >= n_passages * 3:
            break
    if len(out) > n_passages:
        idx = rng.choice(len(out), size=n_passages, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


def route_metrics(chal: np.ndarray, chal_dom: list[str],
                  ex: np.ndarray, ex_dom: list[str]) -> dict:
    """Max-cosine-to-individual-exemplar routing (the shipped bridge rule)."""
    a = chal / (np.linalg.norm(chal, axis=1, keepdims=True) + 1e-12)
    b = ex / (np.linalg.norm(ex, axis=1, keepdims=True) + 1e-12)
    sims = a @ b.T
    ed = np.asarray(ex_dom)
    cd = np.asarray(chal_dom)
    picked = ed[sims.argmax(axis=1)]
    hit = picked == cd
    per_domain = {d: float(hit[cd == d].mean()) for d in sorted(set(chal_dom))}
    same = np.array([sims[i][ed == cd[i]].max() if (ed == cd[i]).any() else np.nan
                     for i in range(len(cd))])
    diff = np.array([sims[i][ed != cd[i]].max() for i in range(len(cd))])
    return {"micro": float(hit.mean()),
            "macro": float(np.mean(list(per_domain.values()))),
            "per_domain": per_domain, "best": sims.max(axis=1),
            "same": same, "diff": diff}


def youden_tau(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate([same, diff])
    labels = np.concatenate([np.ones(len(same)), np.zeros(len(diff))])
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    tpr, fpr = tp / max(tp[-1], 1), fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return float(s[int((tpr - fpr).argmax())]), auc


def _domain_scores(chal: np.ndarray, ex: np.ndarray, ex_dom: list[str],
                   base: str, topk: int = 5) -> tuple[np.ndarray, list[str]]:
    """(n_chal, n_domains) score matrix under a base scoring rule.

    max      -- best single exemplar per domain (the shipped rule)
    topk     -- mean of the top-k exemplar cosines per domain (noise-robust)
    centroid -- cosine to the normalized per-domain mean (34 comparisons
                instead of ~1300: also the cheapest serving form)
    """
    a = chal / (np.linalg.norm(chal, axis=1, keepdims=True) + 1e-12)
    b = ex / (np.linalg.norm(ex, axis=1, keepdims=True) + 1e-12)
    ed = np.asarray(ex_dom)
    doms = sorted(set(ex_dom))
    S = np.zeros((len(chal), len(doms)))
    if base == "centroid":
        C = np.stack([b[ed == d].mean(axis=0) for d in doms])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = a @ C.T
    else:
        sims = a @ b.T
        for j, d in enumerate(doms):
            block = sims[:, ed == d]
            if base == "max":
                S[:, j] = block.max(axis=1)
            else:                                   # topk mean
                k_ = min(topk, block.shape[1])
                S[:, j] = np.sort(block, axis=1)[:, -k_:].mean(axis=1)
    return S, doms


def _znorm(S: np.ndarray, ex: np.ndarray, ex_dom: list[str], doms: list[str],
           topk: int = 5) -> np.ndarray:
    """Per-domain calibration: z-score each domain's column against the
    distribution of its OWN exemplars' leave-one-out top-k scores."""
    b = ex / (np.linalg.norm(ex, axis=1, keepdims=True) + 1e-12)
    ed = np.asarray(ex_dom)
    Z = np.zeros_like(S)
    for j, d in enumerate(doms):
        own = b[ed == d]
        sims = own @ own.T
        np.fill_diagonal(sims, -np.inf)
        k_ = min(topk, own.shape[0] - 1)
        self_scores = np.sort(sims, axis=1)[:, -k_:].mean(axis=1)
        mu, sd = float(self_scores.mean()), float(self_scores.std() + 1e-9)
        Z[:, j] = (S[:, j] - mu) / sd
    return Z


def run_open_screen(args) -> None:
    """Which novelty scorer separates known-domain from never-seen-domain
    challenges best -- i.e. which rule should drive route-vs-spawn?"""
    import platform

    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"mode: --open-screen | {args.splits} rotated open-set splits x "
          f"{args.open_set} held-out domains | table {TABLE.name}\n")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    domains_all = sorted(d.name for d in DOMAINS_DIR.iterdir()
                         if d.is_dir() and d.name not in EXCLUDE)

    scorers = ("max/top1", "max/margin", "topk/top1", "topk/margin",
               "centroid/top1", "centroid/margin", "topk/znorm")
    agg: dict[str, dict[str, list[float]]] = {s_: {"auc": [], "route": [],
                                                   "routed": [], "acc_routed": [],
                                                   "spawned": []} for s_ in scorers}
    ref_agg: dict[str, list[float]] = {"auc": [], "route": []}

    for split in range(args.splits):
        rng = np.random.default_rng(42 + split)
        open_set = sorted(rng.choice(domains_all, size=args.open_set, replace=False))
        ex_texts: list[str] = []; ex_dom: list[str] = []
        ch_texts: list[str] = []; ch_dom: list[str] = []
        open_texts: list[str] = []
        for d in domains_all:
            ps = sample_passages(DOMAINS_DIR / d, args.files,
                                 args.exemplars + args.challenges,
                                 seed=int(rng.integers(1 << 31)))
            if len(ps) < 10:
                continue
            half = min(args.exemplars, len(ps) // 2)
            if d in open_set:
                open_texts.extend(ps[half:half + args.challenges])
                continue
            ex_texts.extend(ps[:half]); ex_dom.extend([d] * half)
            take = ps[half:half + args.challenges]
            ch_texts.extend(take); ch_dom.extend([d] * len(take))

        EX = np.stack([enc.encode(t).ravel() for t in ex_texts])
        CH = np.stack([enc.encode(t).ravel() for t in ch_texts])
        OP = np.stack([enc.encode(t).ravel() for t in open_texts])
        cd = np.asarray(ch_dom)
        print(f"split {split}: open={open_set} | {len(ex_texts)} ex / "
              f"{len(ch_texts)} closed / {len(open_texts)} open")

        for s_ in scorers:
            base, nov = s_.split("/")
            S_ch, doms = _domain_scores(CH, EX, ex_dom, base)
            S_op, _ = _domain_scores(OP, EX, ex_dom, base)
            if nov == "znorm":
                S_ch = _znorm(S_ch, EX, ex_dom, doms)
                S_op = _znorm(S_op, EX, ex_dom, doms)
            top_ch = np.sort(S_ch, axis=1)[:, -2:]
            top_op = np.sort(S_op, axis=1)[:, -2:]
            if nov == "margin":
                nov_ch = top_ch[:, 1] - top_ch[:, 0]
                nov_op = top_op[:, 1] - top_op[:, 0]
            else:                                    # top1 or znorm
                nov_ch, nov_op = top_ch[:, 1], top_op[:, 1]
            picked = np.asarray(doms)[S_ch.argmax(axis=1)]
            hit = picked == cd
            per_dom = [float(hit[cd == d].mean()) for d in sorted(set(ch_dom))]
            tau, auc = youden_tau(nov_ch, nov_op)
            routed = nov_ch >= tau
            agg[s_]["auc"].append(auc)
            agg[s_]["route"].append(float(np.mean(per_dom)))
            agg[s_]["routed"].append(float(routed.mean()))
            agg[s_]["acc_routed"].append(float(hit[routed].mean()) if routed.any() else 0.0)
            agg[s_]["spawned"].append(float((nov_op < tau).mean()))

        if args.with_ref:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            EXr = np.asarray(model.encode(ex_texts, batch_size=64, show_progress_bar=False))
            CHr = np.asarray(model.encode(ch_texts, batch_size=64, show_progress_bar=False))
            OPr = np.asarray(model.encode(open_texts, batch_size=64, show_progress_bar=False))
            S_ch, doms = _domain_scores(CHr, EXr, ex_dom, "topk")
            S_op, _ = _domain_scores(OPr, EXr, ex_dom, "topk")
            _, auc = youden_tau(np.sort(S_ch, axis=1)[:, -1],
                                np.sort(S_op, axis=1)[:, -1])
            picked = np.asarray(doms)[S_ch.argmax(axis=1)]
            ref_agg["auc"].append(auc)
            ref_agg["route"].append(float((picked == cd).mean()))

    print(f"\n{'scorer':>16} {'spawn-AUC':>12} {'route-macro':>12} "
          f"{'routed@tau':>11} {'acc|routed':>11} {'open->spawn':>12}")
    out: dict = {"config": vars(args), "scorers": {}}
    for s_ in scorers:
        a_ = agg[s_]
        line = {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in a_.items()}
        out["scorers"][s_] = line
        print(f"{s_:>16} "
              f"{line['auc'][0]:7.4f}±{line['auc'][1]:.3f} "
              f"{line['route'][0]:7.4f}±{line['route'][1]:.3f} "
              f"{line['routed'][0]:10.0%} {line['acc_routed'][0]:10.0%} "
              f"{line['spawned'][0]:11.0%}")
    if args.with_ref and ref_agg["auc"]:
        print(f"{'MiniLM topk/top1':>16} {np.mean(ref_agg['auc']):7.4f}±{np.std(ref_agg['auc']):.3f} "
              f"{np.mean(ref_agg['route']):7.4f}±{np.std(ref_agg['route']):.3f}"
              f"{'':>10}{'':>11}{'':>12}   (reference ceiling)")
        out["ref"] = {k_: (float(np.mean(v)), float(np.std(v)))
                      for k_, v in ref_agg.items()}

    best = max(scorers, key=lambda s_: np.mean(agg[s_]["auc"]))
    print(f"\nbest spawn scorer: {best} (AUC {np.mean(agg[best]['auc']):.4f} "
          f"vs shipped max/top1 {np.mean(agg['max/top1']['auc']):.4f})")
    out["best"] = best

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_open_set.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print(f"wrote {logs / 'exp_m3_open_set.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exemplars", type=int, default=40, help="seeded passages/domain")
    ap.add_argument("--challenges", type=int, default=40, help="held-out passages/domain")
    ap.add_argument("--files", type=int, default=8, help="files sampled/domain")
    ap.add_argument("--open-set", type=int, default=5, help="domains held out entirely")
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--with-ref", action="store_true", help="also run MiniLM reference")
    ap.add_argument("--open-screen", action="store_true",
                    help="screen novelty scorers for route-vs-spawn (thread 1)")
    ap.add_argument("--splits", type=int, default=3,
                    help="rotated open-set splits for --open-screen")
    args = ap.parse_args()

    if args.open_screen:
        run_open_screen(args)
        return

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table: {TABLE.name} | domains root: {DOMAINS_DIR}")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    print(f"encoder: {enc.meta.get('n_words')} words, k={enc.k} l={enc.l}\n")

    domains = sorted(d.name for d in DOMAINS_DIR.iterdir()
                     if d.is_dir() and d.name not in EXCLUDE)
    rng = np.random.default_rng(42)
    open_set = sorted(rng.choice(domains, size=args.open_set, replace=False))
    closed = [d for d in domains if d not in open_set]
    print(f"{len(domains)} usable domains -> {len(closed)} seeded, "
          f"open-set held out entirely: {open_set}\n")

    t0 = time.perf_counter()
    ex_texts: list[str] = []; ex_dom: list[str] = []
    ch_texts: list[str] = []; ch_dom: list[str] = []
    open_texts: list[str] = []
    skipped: list[str] = []
    for d in domains:
        ps = sample_passages(DOMAINS_DIR / d, args.files,
                             args.exemplars + args.challenges,
                             seed=int(rng.integers(1 << 31)))
        if len(ps) < 10:
            skipped.append(f"{d}({len(ps)})")
            continue
        half = min(args.exemplars, len(ps) // 2)
        if d in open_set:
            open_texts.extend(ps[half:half + args.challenges])
            continue
        ex_texts.extend(ps[:half]); ex_dom.extend([d] * half)
        take = ps[half:half + args.challenges]
        ch_texts.extend(take); ch_dom.extend([d] * len(take))
    print(f"sampled {len(ex_texts)} exemplars / {len(ch_texts)} challenges / "
          f"{len(open_texts)} open-set in {time.perf_counter() - t0:.0f}s"
          + (f" | skipped thin domains: {skipped}" if skipped else ""))

    t0 = time.perf_counter()
    EX = np.stack([enc.encode(t).ravel() for t in ex_texts])
    CH = np.stack([enc.encode(t).ravel() for t in ch_texts])
    OP = (np.stack([enc.encode(t).ravel() for t in open_texts])
          if open_texts else np.zeros((0, K * L)))
    n_all = len(ex_texts) + len(ch_texts) + len(open_texts)
    us = (time.perf_counter() - t0) / max(n_all, 1) * 1e6
    print(f"encoded {n_all} passages at {us:.0f} us/passage\n")

    out: dict = {"config": vars(args), "open_set_domains": open_set,
                 "n_exemplars": len(ex_texts), "n_challenges": len(ch_texts),
                 "n_open": len(open_texts), "encode_us_per_passage": us,
                 "skipped": skipped}

    # -- routing + measured noise null ----------------------------------------
    r = route_metrics(CH, ch_dom, EX, ex_dom)
    nulls = {"macro": [], "micro": [], "auc": []}
    for s_ in range(args.noise_seeds):
        g = np.random.default_rng(1000 + s_)
        rn = route_metrics(g.standard_normal(CH.shape), ch_dom,
                           g.standard_normal(EX.shape), ex_dom)
        _, a_ = youden_tau(rn["same"][~np.isnan(rn["same"])], rn["diff"])
        nulls["macro"].append(rn["macro"]); nulls["micro"].append(rn["micro"])
        nulls["auc"].append(a_)
    nm, nmi, na = (float(np.mean(nulls[k_])) for k_ in ("macro", "micro", "auc"))
    tau, auc = youden_tau(r["same"][~np.isnan(r["same"])], r["diff"])
    print(f"routing  : macro {r['macro']:.4f} (noise null {nm:.4f}) | "
          f"micro {r['micro']:.4f} (null {nmi:.4f})")
    print(f"tau      : Youden tau = {tau:.4f} | ROC-AUC {auc:.4f} "
          f"(noise-null AUC {na:.4f})")
    worst = sorted(r["per_domain"].items(), key=lambda kv: kv[1])[:5]
    best = sorted(r["per_domain"].items(), key=lambda kv: -kv[1])[:5]
    print(f"best domains : {', '.join(f'{d} {v:.2f}' for d, v in best)}")
    print(f"worst domains: {', '.join(f'{d} {v:.2f}' for d, v in worst)}")

    # -- open-set / spawn detection -------------------------------------------
    if len(OP):
        b = EX / (np.linalg.norm(EX, axis=1, keepdims=True) + 1e-12)
        o = OP / (np.linalg.norm(OP, axis=1, keepdims=True) + 1e-12)
        open_best = (o @ b.T).max(axis=1)
        closed_best = r["best"]
        spawn_tau, spawn_auc = youden_tau(closed_best, open_best)
        frac_open_below = float((open_best < tau).mean())
        frac_closed_above = float((closed_best >= tau).mean())
        print(f"open-set : closed best-cos {closed_best.mean():.4f} +/- {closed_best.std():.4f} "
              f"vs open {open_best.mean():.4f} +/- {open_best.std():.4f}")
        print(f"           spawn-detection AUC {spawn_auc:.4f} | at routing tau "
              f"{tau:.3f}: {frac_closed_above:.0%} closed routed, "
              f"{frac_open_below:.0%} open sent to spawn")
        out["open_set"] = {"spawn_auc": spawn_auc, "spawn_tau": spawn_tau,
                           "closed_best_mean": float(closed_best.mean()),
                           "open_best_mean": float(open_best.mean()),
                           "frac_open_below_tau": frac_open_below,
                           "frac_closed_above_tau": frac_closed_above}

    out["routing"] = {"macro": r["macro"], "micro": r["micro"],
                      "per_domain": r["per_domain"], "tau": tau, "auc": auc,
                      "noise_null": {"macro": nm, "micro": nmi, "auc": na}}

    # -- optional MiniLM reference --------------------------------------------
    if args.with_ref:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        t0 = time.perf_counter()
        EXr = np.asarray(model.encode(ex_texts, batch_size=64, show_progress_bar=False))
        CHr = np.asarray(model.encode(ch_texts, batch_size=64, show_progress_bar=False))
        ref_us = (time.perf_counter() - t0) / (len(ex_texts) + len(ch_texts)) * 1e6
        rr = route_metrics(CHr, ch_dom, EXr, ex_dom)
        rtau, rauc = youden_tau(rr["same"][~np.isnan(rr["same"])], rr["diff"])
        print(f"\nMiniLM ref: macro {rr['macro']:.4f} | micro {rr['micro']:.4f} | "
              f"AUC {rauc:.4f} | {ref_us:.0f} us/passage batched")
        out["ref"] = {"macro": rr["macro"], "micro": rr["micro"], "auc": rauc,
                      "tau": rtau, "batched_us_per_passage": ref_us}
        print(f"table/ref: macro {r['macro'] / rr['macro']:.2f}x | "
              f"AUC {auc / rauc:.2f}x")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_domain_routing.json").write_text(json.dumps(out, indent=1),
                                                    encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_domain_routing.json'}")


if __name__ == "__main__":
    main()
