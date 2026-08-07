"""exp_m3_temporal_causal — temporal world hierarchy + causal events (thread 3).

The user's frame: the world was NOT the same from one era or one year to the
next — each era and each year is a world (months would be too granular), and
worlds need CAUSAL events connecting them (MoWM worlds are transition
machines; a cause->effect pair across years is transition data).

Spine: D:\\grillcheese_training_data\\temporal\\historical\\train_augmented.jsonl
— 4,000 curated events with text, year_start (negative = BC), a named period
(6 eras), region, and an `impact` field (the event's downstream consequence
text) — clean labels, no extraction pipeline in the loop. The HUGE_ENTITIES
book corpus (5,804 raw book txts) was probed and deferred: year/causal-marker
extraction works (~76-159 anchors per 400KB) but the corpus has misfiled
content (a 1943-46 novel under AncientClassical) and PDF-artifact files with
no extractable spacing; it is the scale-up path, not the labeled spine.

Evals (production FastWordEncoder table; MiniLM anchors era/year/causal):
  era      -- LOO centroid routing over the 6 period-worlds, vs measured
              noise null (3 seeds through the identical pipeline)
  year     -- flat routing over year-worlds (years with >= 3 events, LOO
              centroids): exact hit, +/-10, +/-50 years, MAE; then
              HIERARCHICAL era->year (route era first, then only that era's
              years) vs flat, with comparison counts
  masked   -- same year routing with all digit tokens stripped from the
              CHALLENGE text: can content alone place an event in time?
              (only 393/4000 texts mention their own year, so the gap is
              expected small — measured, not assumed)
  gradient -- year-world centroid cosine vs |delta years|, binned: does the
              representation know 1914 is more like 1915 than 1815?
  causal   -- event->impact retrieval among all 4,000 impact texts (and the
              reverse), hit@1/5/20 + MRR: can the table link an event to its
              own stated consequence among confusables?
  forward  -- route each impact text to its nearest year-world (own year
              excluded): does the consequence land LATER than the event more
              often than the matched null (the per-event fraction of
              candidate years that are later)? Causality points forward.

Deep-time events (year < -3500, e.g. -1,000,000 Paleolithic) stay in the era
eval but are excluded from year-level metrics (MAE in years is meaningless
at that scale); the exclusion count is reported. Standalone; never imported
by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import TABLE, _load_semantic_words  # noqa: E402

EVENTS = pathlib.Path(r"D:\grillcheese_training_data\temporal\historical\train_augmented.jsonl")
K, L = 80, 128
YEAR_FLOOR = -3500          # year-level metrics restricted to written history
MIN_PER_YEAR = 3            # a year-world needs >= 3 events (LOO leaves >= 2)


def load_events():
    rows = []
    with open(EVENTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("text") and len(r["text"]) > 40 and r.get("period")
                    and r.get("year_start") is not None and r.get("impact")
                    and len(r["impact"]) > 40):
                rows.append({"text": r["text"], "impact": r["impact"],
                             "year": int(r["year_start"]), "era": r["period"],
                             "title": r.get("title", "")})
    return rows


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def loo_group_scores(V: np.ndarray, groups: dict[object, np.ndarray],
                     C: np.ndarray | None = None,
                     event_groups: list[np.ndarray] | None = None,
                     ) -> tuple[np.ndarray, list]:
    """Cosine of each challenge row to each group's LOO centroid.

    groups: label -> row indices into V (the world-building vectors). C is
    the challenge-side matrix, row-aligned with V (defaults to V; the masked
    arm passes digit-stripped encodings). Row i scored against its own group
    uses the centroid with V[i] excluded — in EVERY arm, so a masked
    challenge never gets credit for its own unmasked vector sitting in the
    world. event_groups (a partition of rows into same-event duplicate sets)
    upgrades that to event-level exclusion — see below."""
    if C is None:
        C = V
    labels = sorted(groups, key=str)
    n = C.shape[0]
    S = np.zeros((n, len(labels)), dtype=np.float32)
    for g, lab in enumerate(labels):
        idx = groups[lab]
        gsum = V[idx].sum(axis=0)
        full = gsum / (np.linalg.norm(gsum) + 1e-12)
        S[:, g] = C @ full
        if len(idx) > 1:
            loo = _unit(gsum[None, :] - V[idx])
            S[idx, g] = np.einsum("ij,ij->i", C[idx], loo)
        else:
            S[idx, g] = -1.0        # a singleton group cannot vote for itself
    if event_groups is not None:
        # EVENT-level exclusion: train_augmented duplicates events across
        # rows (529 repeated titles covering 3,244/4,000 rows; 1914 holds
        # three copies of the Franz Ferdinand assassination). Row-level LOO
        # leaves a challenge's paraphrase siblings inside the centroid —
        # the same leakage shape as the retracted PQ result. For every
        # world a challenge's event-group intersects, score against the
        # centroid with ALL of that event's rows removed.
        world_of = {}
        for g, lab in enumerate(labels):
            for i in groups[lab]:
                world_of[int(i)] = g
        for eg in event_groups:
            if len(eg) == 0:
                continue
            by_world: dict[int, list[int]] = {}
            for i in eg:
                g = world_of.get(int(i))
                if g is not None:
                    by_world.setdefault(g, []).append(int(i))
            for g, members in by_world.items():
                idx = groups[labels[g]]
                rem = V[idx].sum(axis=0) - V[members].sum(axis=0)
                nrm = np.linalg.norm(rem)
                if nrm < 1e-9:      # the world was ONLY this event: no vote
                    S[eg, g] = -1.0
                else:
                    S[eg, g] = C[eg] @ (rem / nrm)
    return S, labels


def era_metrics(S: np.ndarray, labels: list, true_eras: list[str]) -> dict:
    pred = np.argmax(S, axis=1)
    lab_i = {e: i for i, e in enumerate(labels)}
    y = np.array([lab_i[e] for e in true_eras])
    per = {}
    for e, i in lab_i.items():
        m = y == i
        per[e] = float((pred[m] == i).mean())
    return {"macro": float(np.mean(list(per.values()))),
            "micro": float((pred == y).mean()), "per_era": per}


def year_metrics(S: np.ndarray, ylabels: list[int], true_years: np.ndarray,
                 own_world: np.ndarray) -> dict:
    """own_world: bool mask — challenge's own year has a world (exact hit
    is defined). MAE/±k computed over all challenge rows."""
    yl = np.asarray(ylabels)
    routed = yl[np.argmax(S, axis=1)]
    err = np.abs(routed - true_years)
    out = {"n": int(len(true_years)), "n_own_world": int(own_world.sum()),
           "exact": float((routed[own_world] == true_years[own_world]).mean())
           if own_world.any() else 0.0,
           "within10": float((err <= 10).mean()),
           "within50": float((err <= 50).mean()),
           "mae_mean": float(err.mean()), "mae_median": float(np.median(err))}
    return out


def delegation_eval(V: np.ndarray, ch: np.ndarray, ygroups: dict,
                    eg_of: np.ndarray, years: np.ndarray,
                    taus: list[float]) -> dict:
    """Inter-world query: a world missing the answer asks the next world.

    Worlds are ranked by centroid cosine. The protocol consults the top
    world's MEMBERS; if the best member score >= tau_answer it answers,
    otherwise it queries the next world; if every world is exhausted it
    answers with the global best member seen. tau=0 is 'trust your first
    world' (pure routing, cheap); tau>1 degenerates to flat search over all
    members (the ceiling). The answer is correct when the retrieved member
    is a duplicate sibling of the challenge event (the stored copy of the
    answer). Comparisons = n_world centroids + members of consulted worlds.

    World ranking uses per-challenge SELF-EXCLUDED centroids (the query is
    not in the store; its stored siblings legitimately are). The first
    version ranked against full centroids — on NYT, where year-centroids sit
    at 0.86-0.92 mutual cosine, the 1/100 self-nudge alone lifted routed
    exact-year from 0.064 to 0.372: the same leakage shape caught twice
    before this session, now at the row level.
    """
    Sw_all, labels = loo_group_scores(V, ygroups)
    yl = np.asarray(labels)
    G = len(yl)
    Sw = Sw_all[ch]                                  # (B, G) world ranking
    order = np.argsort(-Sw, axis=1)

    # best member per (challenge, world), self-row excluded
    Sm = V[ch] @ V.T
    Sm[np.arange(len(ch)), ch] = -np.inf
    best_s = np.full((len(ch), G), -np.inf, dtype=np.float32)
    best_m = np.zeros((len(ch), G), dtype=np.int64)
    for g, y in enumerate(yl):
        mem = ygroups[y]
        sub = Sm[:, mem]
        best_s[:, g] = sub.max(axis=1)
        best_m[:, g] = mem[np.argmax(sub, axis=1)]
    sizes = np.array([len(ygroups[y]) for y in yl])

    bs_ord = np.take_along_axis(best_s, order, axis=1)   # (B, G) in visit order
    out = {}
    for tau in taus:
        hit_mask = bs_ord >= tau
        first = np.where(hit_mask.any(axis=1), np.argmax(hit_mask, axis=1), G - 1)
        exhausted = ~hit_mask.any(axis=1)
        # exhausted challenges answer the globally best member seen
        pick_rank = np.where(exhausted, np.argmax(bs_ord, axis=1), first)
        gsel = order[np.arange(len(ch)), pick_rank]
        ans = best_m[np.arange(len(ch)), gsel]
        found = eg_of[ans] == eg_of[ch]
        yerr = np.abs(yl[gsel] - years[ch])
        consulted = np.where(exhausted, G, first + 1)
        cum = np.cumsum(sizes[order], axis=1)
        cmp_members = np.where(exhausted, cum[:, -1],
                               cum[np.arange(len(ch)),
                                   np.minimum(first, G - 1)])
        out[f"tau={tau}"] = {
            "found": float(found.mean()),
            "answer_year_exact": float((yerr == 0).mean()),
            "answer_year_within50": float((yerr <= 50).mean()),
            "mean_worlds_consulted": float(consulted.mean()),
            "mean_member_comparisons": float(cmp_members.mean()),
            "frac_delegated": float((consulted > 1).mean()),
        }
    out["flat_member_comparisons"] = int(sizes.sum())
    out["n_challenges"] = int(len(ch))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--with-ref", action="store_true", help="MiniLM anchors")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {TABLE.name} | events {EVENTS.name}\n")

    rows = load_events()
    eras = [r["era"] for r in rows]
    years = np.array([r["year"] for r in rows])
    from collections import Counter
    print(f"{len(rows)} events | eras {dict(Counter(eras))}")
    deep = years < YEAR_FLOOR
    print(f"deep-time events excluded from year-level metrics: {int(deep.sum())}")

    # same-event duplicate sets: train_augmented repeats events across rows
    # (augmented paraphrases). Key = (normalized title, year); every metric
    # below excludes/credits at this level, not the row level.
    eg_map: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        eg_map.setdefault((" ".join(r["title"].lower().split()), r["year"]),
                          []).append(i)
    event_groups = [np.array(v) for v in eg_map.values()]
    eg_of = np.zeros(len(rows), dtype=np.int64)
    for gi, v in enumerate(event_groups):
        eg_of[v] = gi
    n_dup_rows = sum(len(v) for v in event_groups if len(v) > 1)
    print(f"event groups: {len(event_groups)} "
          f"({n_dup_rows} rows are duplicates of another row's event)")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)

    def table_encode(texts: list[str]) -> np.ndarray:
        return _unit(np.stack([enc.encode(t).reshape(-1) for t in texts]))

    t0 = time.perf_counter()
    V = table_encode([r["text"] for r in rows])
    VI = table_encode([r["impact"] for r in rows])
    enc_us = (time.perf_counter() - t0) / (2 * len(rows)) * 1e6
    print(f"encoded {2 * len(rows)} texts ({enc_us:.0f} us/text)\n")

    out: dict = {"config": vars(args), "n_events": len(rows),
                 "encode_us_per_text": enc_us, "results": {}}

    def run_all(V, VI, tag: str, challenge_V=None) -> dict:
        """All evals for one encoder. challenge_V overrides the challenge-side
        vectors (the masked arm); worlds are always built from V."""
        C = challenge_V                     # None -> challenges are V itself
        res: dict = {}
        # --- era ---
        era_groups = {e: np.where(np.array(eras) == e)[0] for e in set(eras)}
        S, elabels = loo_group_scores(V, era_groups, C, event_groups)
        res["era"] = era_metrics(S, elabels, eras)
        era_pred = [elabels[i] for i in np.argmax(S, axis=1)]

        # --- year worlds (written history only) ---
        hist = ~deep
        ycount = Counter(years[hist].tolist())
        world_years = sorted(y for y, c in ycount.items() if c >= MIN_PER_YEAR)
        ygroups = {y: np.where((years == y) & hist)[0] for y in world_years}
        n_world_events = sum(len(v) for v in ygroups.values())
        res["n_year_worlds"] = len(world_years)
        res["n_events_in_year_worlds"] = n_world_events

        ch = np.where(hist)[0]                      # all written-history events
        own = np.isin(years[ch], world_years)
        Sy, ylabels = loo_group_scores(V, ygroups, C, event_groups)
        Sy = Sy[ch]
        res["year_flat"] = year_metrics(Sy, ylabels, years[ch], own)
        res["year_flat"]["comparisons"] = len(ylabels)

        # --- hierarchical era -> year ---
        yl = np.asarray(ylabels)
        year_era = {}
        for y in ylabels:
            e_cnt = Counter(np.array(eras)[ygroups[y]].tolist())
            year_era[y] = e_cnt.most_common(1)[0][0]
        Sh = Sy.copy()
        ncomp = []
        for r_i, i in enumerate(ch):
            allowed = np.array([year_era[y] == era_pred[i] for y in ylabels])
            ncomp.append(len(elabels) + int(allowed.sum()))
            if allowed.any():
                Sh[r_i, ~allowed] = -np.inf
        res["year_hier"] = year_metrics(Sh, ylabels, years[ch], own)
        res["year_hier"]["comparisons"] = float(np.mean(ncomp))

        # --- causal: event -> its impact, among all impacts. Row-strict
        # ranking is unfair under duplication (a sibling's near-identical
        # impact outranking your own is not an error), so the primary metric
        # credits ANY impact of the same event group (any-positive rank, the
        # haystack convention); row-strict is kept for reference.
        Sc = (V if C is None else C) @ VI.T
        diag = Sc[np.arange(len(rows)), np.arange(len(rows))]
        rk = (Sc > diag[:, None]).sum(axis=1) + 1
        pos_best = np.empty(len(rows), dtype=np.float32)
        for eg in event_groups:
            pos_best[eg] = Sc[np.ix_(eg, eg)].max(axis=1)
        rke = (Sc > pos_best[:, None]).sum(axis=1) + 1
        res["causal_event_to_impact"] = {
            "hit1": float((rke <= 1).mean()), "hit5": float((rke <= 5).mean()),
            "hit20": float((rke <= 20).mean()), "mrr": float((1.0 / rke).mean()),
            "strict_hit1": float((rk <= 1).mean()),
            "strict_mrr": float((1.0 / rk).mean()),
            "chance_hit1": 1.0 / len(rows)}
        rk2 = (Sc.T > diag[:, None]).sum(axis=1) + 1
        res["causal_impact_to_event"] = {
            "strict_hit1": float((rk2 <= 1).mean()),
            "strict_mrr": float((1.0 / rk2).mean())}

        # --- forward: impacts should land at later year-worlds than their event
        cents = _unit(np.stack([V[ygroups[y]].sum(axis=0) for y in ylabels]))
        Sf = VI @ cents.T
        fwd, null = [], []
        for i in range(len(rows)):
            if deep[i]:
                continue
            cand = yl != years[i]
            if not cand.any() or (yl[cand] > years[i]).all() \
                    or (yl[cand] < years[i]).all():
                continue                      # no both-sides candidates: skip
            j = int(np.argmax(np.where(cand, Sf[i], -np.inf)))
            fwd.append(1.0 if yl[j] > years[i] else 0.0)
            null.append(float((yl[cand] > years[i]).mean()))
        res["forward"] = {"n": len(fwd), "frac_later": float(np.mean(fwd)),
                          "matched_null": float(np.mean(null))}

        # --- inter-world query (delegation): a world missing the answer
        # asks the next world. Challenges: events with a duplicate sibling
        # stored in a year-world (the sibling IS the stored answer).
        if tag in ("table", "ref"):
            eg_size = np.array([len(event_groups[g]) for g in eg_of])
            chd = np.where(hist & (eg_size > 1) & np.isin(years, world_years))[0]
            res["delegation"] = delegation_eval(
                V, chd, ygroups, eg_of, years,
                taus=[0.0, 0.3, 0.4, 0.5, 0.6, 1.01])

        # --- temporal gradient (table arm only; descriptive) ---
        if tag == "table":
            D_ = np.abs(yl[:, None] - yl[None, :])
            Cw = cents @ cents.T
            iu = np.triu_indices(len(yl), k=1)
            d, c = D_[iu], Cw[iu]
            bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250),
                    (250, 500), (500, 100000)]
            res["gradient"] = {f"{a}-{b}": [float(c[(d >= a) & (d < b)].mean()),
                                            int(((d >= a) & (d < b)).sum())]
                               for a, b in bins if ((d >= a) & (d < b)).any()}
        return res

    from collections import Counter  # noqa: F811  (used inside run_all)

    res = run_all(V, VI, "table")
    out["results"]["table"] = res

    # masked arm: digits stripped from the challenge side only
    masked = [re.sub(r"\d+", " ", r["text"]) for r in rows]
    Vm = table_encode(masked)
    rm = run_all(V, VI, "masked", challenge_V=Vm)
    out["results"]["table_masked"] = {k: rm[k] for k in
                                      ("era", "year_flat", "year_hier")}

    # measured noise null through the identical pipeline
    nulls = []
    for seed in range(args.noise_seeds):
        rng = np.random.default_rng(seed)
        Vn = _unit(rng.standard_normal((len(rows), K * L)).astype(np.float32))
        VIn = _unit(rng.standard_normal((len(rows), K * L)).astype(np.float32))
        rn = run_all(Vn, VIn, "noise")
        nulls.append({"era_macro": rn["era"]["macro"],
                      "year_exact": rn["year_flat"]["exact"],
                      "year_mae_median": rn["year_flat"]["mae_median"],
                      "causal_hit1": rn["causal_event_to_impact"]["hit1"],
                      "forward_frac": rn["forward"]["frac_later"]})
    out["noise_null"] = {k: [float(np.mean([n[k] for n in nulls])),
                             float(np.std([n[k] for n in nulls]))]
                         for k in nulls[0]}

    if args.with_ref:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        t0 = time.perf_counter()
        Vr = _unit(np.asarray(model.encode([r["text"] for r in rows],
                                           batch_size=64, show_progress_bar=False)))
        VIr = _unit(np.asarray(model.encode([r["impact"] for r in rows],
                                            batch_size=64, show_progress_bar=False)))
        ref_us = (time.perf_counter() - t0) / (2 * len(rows)) * 1e6
        rr = run_all(Vr, VIr, "ref")
        rr["batched_us_per_text"] = ref_us
        out["results"]["ref"] = rr

    # ---- report ----
    for tag, r in out["results"].items():
        print(f"=== {tag}")
        if "era" in r:
            print(f"  era     macro {r['era']['macro']:.3f} micro {r['era']['micro']:.3f}")
        for k in ("year_flat", "year_hier"):
            if k in r:
                y = r[k]
                print(f"  {k:9s} exact {y['exact']:.3f} ±10y {y['within10']:.3f} "
                      f"±50y {y['within50']:.3f} MAE med {y['mae_median']:.0f} "
                      f"(cmp {y['comparisons']:.0f})")
        if "causal_event_to_impact" in r:
            c = r["causal_event_to_impact"]
            print(f"  causal  e->i hit@1 {c['hit1']:.3f} hit@5 {c['hit5']:.3f} "
                  f"MRR {c['mrr']:.3f} (chance {c['chance_hit1']:.5f})")
        if "forward" in r:
            f = r["forward"]
            print(f"  forward frac-later {f['frac_later']:.3f} vs null "
                  f"{f['matched_null']:.3f} (n={f['n']})")
        if "delegation" in r:
            d = r["delegation"]
            print(f"  delegation (n={d['n_challenges']}, flat={d['flat_member_comparisons']} member cmps):")
            for k, v in d.items():
                if k.startswith("tau="):
                    print(f"    {k:9s} found {v['found']:.3f} yr-exact "
                          f"{v['answer_year_exact']:.3f} worlds {v['mean_worlds_consulted']:5.1f} "
                          f"member-cmps {v['mean_member_comparisons']:6.0f} "
                          f"delegated {v['frac_delegated']:.2f}")
        if "gradient" in r:
            g = " ".join(f"{k}:{v[0]:.3f}" for k, v in r["gradient"].items())
            print(f"  gradient {g}")
    print(f"noise null: { {k: [round(v[0], 4), round(v[1], 4)] for k, v in out['noise_null'].items()} }")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_temporal_causal.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_temporal_causal.json'}")


if __name__ == "__main__":
    main()
