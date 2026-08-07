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

Encoder: the production FastWordEncoder table (D:\\CUBBY-TRAINED-MODELS\\
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
TABLE = pathlib.Path(r"D:\CUBBY-TRAINED-MODELS\fastword_table_v1.npz")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exemplars", type=int, default=40, help="seeded passages/domain")
    ap.add_argument("--challenges", type=int, default=40, help="held-out passages/domain")
    ap.add_argument("--files", type=int, default=8, help="files sampled/domain")
    ap.add_argument("--open-set", type=int, default=5, help="domains held out entirely")
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--with-ref", action="store_true", help="also run MiniLM reference")
    args = ap.parse_args()

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
