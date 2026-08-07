"""exp_m3_nyt_years — year-worlds on real NYT archive data (thread 3b).

D:\\grillcheese_training_data\\temporal\\nyt_data holds 294 NYT Archive API
month files (YEAR_MONTH.json, 1851-2024, 148 distinct years) — the dense,
REAL counterpart to train_augmented's 4,000 curated events, already proven
as real dated keys in H-G2. Per the user's frame: each year is a world;
months are too granular for worlds, so month files aggregate into year-worlds
(the month is kept as row metadata only).

This is deliberately hard mode: NYT abstracts include timeless content
(weddings, recipes) with no temporal signal at all — the honest ceiling on
"content alone places you in time" for real text, vs the curated screen's
event descriptions.

Evals (imports the machinery from exp_m3_temporal_causal):
  year routing -- LOO centroid routing over 148 year-worlds: exact, +/-1,
                  +/-5, +/-10, +/-25 years, MAE; measured noise null
  gradient     -- year-centroid cosine vs |delta years|, fine bins: is 1851
                  more like 1852 than 1951 in a real newspaper's language?
  delegation   -- inter-world query at the member level: consult worlds in
                  centroid order, answer when the best member clears tau,
                  else ask the next world. No duplicate-sibling ground truth
                  here (exact dups are removed), so the reported outcome is
                  the ANSWERING WORLD's year error vs consultation cost.

Exact-duplicate abstracts are removed before sampling; wire-service near-
duplicates may remain and are noted, not modeled. Standalone; never imported
by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import TABLE, _load_semantic_words  # noqa: E402
from exp_m3_temporal_causal import (  # noqa: E402
    _unit, delegation_eval, loo_group_scores, year_metrics)

NYT = pathlib.Path(r"D:\grillcheese_training_data\temporal\nyt_data")
K, L = 80, 128


def load_nyt(per_year: int, seed: int = 0):
    """-> (texts, years, months), deduped and sampled per YEAR."""
    by_year: dict[int, list[tuple[str, int]]] = defaultdict(list)
    seen: set[str] = set()
    for f in sorted(NYT.glob("*.json")):
        year, month = (int(x) for x in f.stem.split("_"))
        try:
            arts = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for a in arts:
            t = (a.get("abstract") or a.get("lead_paragraph")
                 or a.get("snippet") or "")
            t = " ".join(t.split())
            key = t.lower()
            if len(t) < 60 or key in seen:
                continue
            seen.add(key)
            by_year[year].append((t[:600], month))
    rng = np.random.default_rng(seed)
    texts, years, months = [], [], []
    for y in sorted(by_year):
        rows = by_year[y]
        if len(rows) > per_year:
            idx = rng.choice(len(rows), size=per_year, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        for t, m in rows:
            texts.append(t)
            years.append(y)
            months.append(m)
    return texts, np.array(years), months


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-year", type=int, default=100)
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--with-ref", action="store_true", help="MiniLM anchor")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {TABLE.name} | nyt_data ({args.per_year}/year)\n")

    texts, years, months = load_nyt(args.per_year)
    ycount = Counter(years.tolist())
    print(f"{len(texts)} texts across {len(ycount)} years "
          f"({min(ycount)}..{max(ycount)}, median {int(np.median(list(ycount.values())))}/year)")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)

    def table_encode(ts: list[str]) -> np.ndarray:
        return _unit(np.stack([enc.encode(t).reshape(-1) for t in ts]))

    t0 = time.perf_counter()
    V = table_encode(texts)
    print(f"encoded in {time.perf_counter() - t0:.0f}s "
          f"({(time.perf_counter() - t0) / len(texts) * 1e6:.0f} us/text)\n")

    world_years = sorted(y for y, c in ycount.items() if c >= 3)
    ygroups = {y: np.where(years == y)[0] for y in world_years}
    eg_of = np.arange(len(texts))          # no duplicate groups after dedup

    out: dict = {"config": vars(args), "n_texts": len(texts),
                 "n_year_worlds": len(world_years), "results": {}}

    def year_run(V, C=None) -> dict:
        S, ylabels = loo_group_scores(V, ygroups, C)
        own = np.isin(years, world_years)
        m = year_metrics(S, ylabels, years, own)
        yl = np.asarray(ylabels)
        routed = yl[np.argmax(S, axis=1)]
        err = np.abs(routed - years)
        m["within1"] = float((err <= 1).mean())
        m["within5"] = float((err <= 5).mean())
        m["within25"] = float((err <= 25).mean())
        m["comparisons"] = len(ylabels)
        return m

    res: dict = {"year_flat": year_run(V)}

    # gradient: fine-grained decades-scale bins
    yl = np.asarray(world_years)
    cents = _unit(np.stack([V[ygroups[y]].sum(axis=0) for y in world_years]))
    D_ = np.abs(yl[:, None] - yl[None, :])
    Cw = cents @ cents.T
    iu = np.triu_indices(len(yl), k=1)
    d, c = D_[iu], Cw[iu]
    bins = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200)]
    res["gradient"] = {f"{a}-{b}": [float(c[(d >= a) & (d < b)].mean()),
                                    int(((d >= a) & (d < b)).sum())]
                       for a, b in bins if ((d >= a) & (d < b)).any()}

    ch = np.arange(len(texts))
    res["delegation"] = delegation_eval(V, ch, ygroups, eg_of, years,
                                        taus=[0.0, 0.4, 0.5, 0.6, 1.01])
    out["results"]["table"] = res

    nulls = []
    for seed in range(args.noise_seeds):
        rng = np.random.default_rng(seed)
        Vn = _unit(rng.standard_normal((len(texts), K * L)).astype(np.float32))
        mn = year_run(Vn)
        nulls.append({"exact": mn["exact"], "within10": mn["within10"],
                      "mae_median": mn["mae_median"]})
    out["noise_null"] = {k: [float(np.mean([n[k] for n in nulls])),
                             float(np.std([n[k] for n in nulls]))]
                         for k in nulls[0]}

    if args.with_ref:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        t0 = time.perf_counter()
        Vr = _unit(np.asarray(model.encode(texts, batch_size=64,
                                           show_progress_bar=False)))
        ref_us = (time.perf_counter() - t0) / len(texts) * 1e6
        rr: dict = {"year_flat": year_run(Vr), "batched_us_per_text": ref_us}
        rr["delegation"] = delegation_eval(Vr, ch, ygroups, eg_of, years,
                                           taus=[0.0, 0.4, 0.5, 0.6, 1.01])
        out["results"]["ref"] = rr

    for tag, r in out["results"].items():
        y = r["year_flat"]
        print(f"=== {tag}")
        print(f"  year  exact {y['exact']:.3f} ±1y {y['within1']:.3f} "
              f"±5y {y['within5']:.3f} ±10y {y['within10']:.3f} "
              f"±25y {y['within25']:.3f} MAE med {y['mae_median']:.0f} "
              f"mean {y['mae_mean']:.0f}")
        if "gradient" in r:
            print("  gradient " + " ".join(f"{k}:{v[0]:.3f}"
                                           for k, v in r["gradient"].items()))
        d = r["delegation"]
        print(f"  delegation (n={d['n_challenges']}, flat={d['flat_member_comparisons']}):")
        for k, v in d.items():
            if k.startswith("tau="):
                print(f"    {k:9s} yr-exact {v['answer_year_exact']:.3f} "
                      f"±50y {v['answer_year_within50']:.3f} "
                      f"worlds {v['mean_worlds_consulted']:5.1f} "
                      f"member-cmps {v['mean_member_comparisons']:6.0f} "
                      f"delegated {v['frac_delegated']:.2f}")
    print(f"noise null: { {k: [round(v[0], 4), round(v[1], 4)] for k, v in out['noise_null'].items()} }")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_nyt_years.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_nyt_years.json'}")


if __name__ == "__main__":
    main()
