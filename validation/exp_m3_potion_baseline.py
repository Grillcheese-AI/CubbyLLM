"""exp_m3_potion_baseline — head-to-head vs released model2vec checkpoints.

The paper's closest prior art is model2vec/potion (static distillation of a
sentence encoder into a per-token table). This screen runs a released potion
checkpoint through the SAME protocols our numbers come from, same machine,
same metric code:

  (1) 42-domain production routing (exp_m3_domain_routing's exact sampling:
      identical rng(42) open-set split and per-domain passage seeds, so the
      texts are the ones the recorded table/MiniLM runs saw)
  (2) NYT year-world placement (exp_m3_nyt_years' protocol, 100/year)

Differences to keep in mind when reading: potion distills into a FREE dense
space (256-d) with no binding algebra and no compact integer form; the
FastWord table distills into (80,128) block-code space, keeping qFHRR
binding + the 80-byte store. Quality parity at these tasks would mean the
algebra came for free. Standalone; never imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import (  # noqa: E402
    DOMAINS_DIR, EXCLUDE, TABLE, _load_semantic_words, route_metrics,
    sample_passages, youden_tau)
from exp_m3_nyt_years import load_nyt  # noqa: E402
from exp_m3_temporal_causal import _unit, loo_group_scores, year_metrics  # noqa: E402


def sample_domain_split(exemplars: int, challenges: int, files: int,
                        open_set_n: int):
    """Exactly exp_m3_domain_routing.main's sampling: sorted domains,
    rng(42) open-set choice, per-domain seeds drawn in domain order."""
    domains = sorted(d.name for d in DOMAINS_DIR.iterdir()
                     if d.is_dir() and d.name not in EXCLUDE)
    rng = np.random.default_rng(42)
    open_set = sorted(rng.choice(domains, size=open_set_n, replace=False))
    ex_texts, ex_dom, ch_texts, ch_dom = [], [], [], []
    for d in domains:
        ps = sample_passages(DOMAINS_DIR / d, files, exemplars + challenges,
                             seed=int(rng.integers(1 << 31)))
        if len(ps) < 10:
            continue
        half = min(exemplars, len(ps) // 2)
        if d in open_set:
            continue
        ex_texts.extend(ps[:half])
        ex_dom.extend([d] * half)
        take = ps[half:half + challenges]
        ch_texts.extend(take)
        ch_dom.extend([d] * len(take))
    return ex_texts, ex_dom, ch_texts, ch_dom, open_set


def routing_arm(name: str, encode, ex_texts, ex_dom, ch_texts, ch_dom) -> dict:
    t0 = time.perf_counter()
    EX = encode(ex_texts)
    CH = encode(ch_texts)
    us = (time.perf_counter() - t0) / (len(ex_texts) + len(ch_texts)) * 1e6
    r = route_metrics(CH, ch_dom, EX, ex_dom)
    tau, auc = youden_tau(r["same"][~np.isnan(r["same"])], r["diff"])
    out = {"macro": r["macro"], "micro": r["micro"], "auc": auc,
           "encode_us_per_passage": us}
    print(f"  {name:16s} macro {r['macro']:.4f} micro {r['micro']:.4f} "
          f"auc {auc:.4f} @ {us:.0f} us/passage")
    return out


def year_arm(name: str, encode, texts, years) -> dict:
    V = encode(texts)
    V = _unit(np.asarray(V, dtype=np.float32))
    ycount = Counter(years.tolist())
    world_years = sorted(y for y, c in ycount.items() if c >= 3)
    ygroups = {y: np.where(years == y)[0] for y in world_years}
    S, ylabels = loo_group_scores(V, ygroups)
    own = np.isin(years, world_years)
    m = year_metrics(S, ylabels, years, own)
    yl = np.asarray(ylabels)
    err = np.abs(yl[np.argmax(S, axis=1)] - years)
    m["within10"] = float((err <= 10).mean())
    print(f"  {name:16s} exact {m['exact']:.4f} ±10y {m['within10']:.4f} "
          f"MAE med {m['mae_median']:.0f} mean {m['mae_mean']:.0f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--potion", default="minishlab/potion-base-8M")
    ap.add_argument("--per-year", type=int, default=100)
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")

    from model2vec import StaticModel
    import model2vec
    potion = StaticModel.from_pretrained(args.potion)
    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    print(f"potion: {args.potion} (model2vec {model2vec.__version__}) vs "
          f"table {TABLE.name}\n")

    def enc_potion(texts):
        return np.asarray(potion.encode(texts, show_progress_bar=False),
                          dtype=np.float32)

    def enc_table(texts):
        return np.stack([enc.encode(t).ravel() for t in texts]).astype(np.float32)

    # potion->block: distill the POTION checkpoint into our block space via
    # the identical recipe (same vocab, same QR seed, same 0.5 signal blend,
    # same IDF) — isolates the question "is the block-code target space the
    # bottleneck, or the teacher/recipe?" If quality survives, the algebra
    # came for free and the remaining gap is teacher strength.
    vocab = sorted(enc._vec.keys())
    t0 = time.perf_counter()
    p2b = sw.FastWordEncoder.build(
        vocab, lambda ws: potion.encode(ws, show_progress_bar=False),
        enc._idf, enc.default_idf, k=enc.k, l=enc.l,
        meta={"teacher": args.potion})
    print(f"potion->block table built: {len(vocab)} words in "
          f"{time.perf_counter() - t0:.0f}s (in-vocab anisotropy fit)\n")

    def enc_p2b(texts):
        return np.stack([p2b.encode(t).ravel() for t in texts]).astype(np.float32)

    out: dict = {"config": vars(args), "results": {}}

    print("=== 42-domain routing (exp_m3_domain_routing protocol; recorded "
          "table 0.5946 / MiniLM 0.7973)")
    ex_texts, ex_dom, ch_texts, ch_dom, open_set = sample_domain_split(
        40, 40, 8, 5)
    print(f"  {len(ex_texts)} exemplars / {len(ch_texts)} challenges "
          f"({len(set(ex_dom))} domains; open-set excluded: {open_set})")
    out["results"]["routing"] = {
        "potion": routing_arm("potion", enc_potion, ex_texts, ex_dom,
                              ch_texts, ch_dom),
        "potion->block": routing_arm("potion->block", enc_p2b, ex_texts,
                                     ex_dom, ch_texts, ch_dom),
        "table": routing_arm("table", enc_table, ex_texts, ex_dom,
                             ch_texts, ch_dom),
    }

    print("\n=== NYT year placement (exp_m3_nyt_years protocol; recorded "
          "table exact 0.064 / MAE med 19, MiniLM 0.046 / 21)")
    texts, years, _ = load_nyt(args.per_year)
    print(f"  {len(texts)} texts, {len(set(years.tolist()))} years")
    out["results"]["nyt_years"] = {
        "potion": year_arm("potion", enc_potion, texts, years),
        "potion->block": year_arm("potion->block", enc_p2b, texts, years),
        "table": year_arm("table", enc_table, texts, years),
    }

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_potion_baseline.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_potion_baseline.json'}")


if __name__ == "__main__":
    main()
