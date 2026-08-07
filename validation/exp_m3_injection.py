"""exp_m3_injection — fact-injection accessibility eval (thread 2).

E:\\valid_scaling_law_with_facts.pq is built for exactly this question: each
row has a question, its full supporting-fact chain (`facts`, 1-3 hops), one
designated `fact_to_inject`, and an answer that always lives in the LAST fact
(verified 10000/10000). Pre-injection the chain is incomplete by construction
(the injected fact is absent from memory), so full-chain retrievability is 0;
this eval measures the POST state: once the fact is in the store, does the
question actually reach it — and the rest of its chain — through the
production FastWordEncoder table?

Store: all unique supporting facts (~13.7k; natural confusables — questions
share fact templates) + N DBpedia distractors. Dense scoring only; the
compact-form cost curve is already measured in exp_m3_haystack.

Conditions per (N, encoder):
  inject   -- rank of the question's own fact_to_inject: hit@1/5/20, MRR,
              split by hop (1-hop facts share ~75% of their words with the
              question; 2/3-hop only ~55% — the measured surface-form gap)
  chain    -- single-shot chain coverage@20, full-chain@20 (the answerability
              proxy: every supporting fact in top-20), answer-fact hit@k
  chase    -- iterative retrieval for multi-hop: n_hop rounds of top-1, each
              round re-encoding the question text + facts retrieved so far
              (text-level query expansion; pure table lookups, no model).
              Baseline at the same budget: top-n_hop of the single query.

Chance for hit@k of one specific row is k/N_store (~0.0015 at N=0, k=20);
stated analytically — far below every measured figure. MiniLM anchors N=0.
Standalone; never imported by cubbyllm/.
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

from exp_m3_domain_routing import TABLE, _load_semantic_words  # noqa: E402
from exp_m3_haystack import PQ_FILE, iter_distractors  # noqa: E402

K, L = 80, 128


def load_rows(max_questions: int, seed: int = 0):
    """-> (questions, hops, targets, sup_idx, inj_idx, ans_idx) with facts
    deduped into `targets`; per question: supporting-fact rows, the injected
    fact's row, and the answer-bearing (last) fact's row."""
    import pyarrow.parquet as pq

    t = pq.read_table(PQ_FILE, columns=["question_prompt", "facts",
                                        "fact_to_inject", "n_hop"])
    rows = [(q, f, i, int(h)) for q, f, i, h in zip(
        t.column("question_prompt").to_pylist(), t.column("facts").to_pylist(),
        t.column("fact_to_inject").to_pylist(), t.column("n_hop").to_pylist())
        if q and f and i and len(q) > 15 and i in f]
    rng = np.random.default_rng(seed)
    if len(rows) > max_questions:
        pick = rng.choice(len(rows), size=max_questions, replace=False)
        rows = [rows[j] for j in sorted(pick)]

    fact_id: dict[str, int] = {}
    targets: list[str] = []

    def fid(s: str) -> int:
        s = " ".join(s.split())
        if s not in fact_id:
            fact_id[s] = len(targets)
            targets.append(s)
        return fact_id[s]

    sup_idx = [sorted({fid(x) for x in f}) for _, f, _, _ in rows]
    inj_idx = [fid(i) for _, _, i, _ in rows]
    ans_idx = [fid(f[-1]) for _, f, _, _ in rows]
    return ([r[0] for r in rows], [r[3] for r in rows],
            targets, sup_idx, inj_idx, ans_idx)


def dense_scores(Qf: np.ndarray, store: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    S = np.zeros((Qf.shape[0], store.shape[0]), dtype=np.float32)
    for a in range(0, store.shape[0], chunk):
        b = min(a + chunk, store.shape[0])
        S[:, a:b] = Qf @ store[a:b].astype(np.float32).T
    return S


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def by_hop(vals: list[float], hops: list[int]) -> dict:
    agg: dict[int, list[float]] = {}
    for v, h in zip(vals, hops):
        agg.setdefault(h, []).append(v)
    return {str(h): float(np.mean(v)) for h, v in sorted(agg.items())}


def rank_of(S: np.ndarray, idx: list[int]) -> np.ndarray:
    """1-based rank of store row idx[i] in question i's score row."""
    B = S.shape[0]
    tgt = S[np.arange(B), idx]
    return (S > tgt[:, None]).sum(axis=1) + 1


def single_shot(S: np.ndarray, hops, sup_idx, inj_idx, ans_idx) -> dict:
    B = S.shape[0]
    r_inj = rank_of(S, inj_idx)
    r_ans = rank_of(S, ans_idx)
    top20 = np.argsort(-S, axis=1)[:, :20]
    cov, full, budget_cov = [], [], []
    for i in range(B):
        t20 = set(top20[i].tolist())
        sup = sup_idx[i]
        cov.append(len(t20 & set(sup)) / len(sup))
        full.append(1.0 if set(sup) <= t20 else 0.0)
        tb = set(top20[i, :len(sup)].tolist())
        budget_cov.append(len(tb & set(sup)) / len(sup))
    return {
        "inject": {"hit1": float((r_inj <= 1).mean()),
                   "hit5": float((r_inj <= 5).mean()),
                   "hit20": float((r_inj <= 20).mean()),
                   "mrr": float((1.0 / r_inj).mean()),
                   "hit5_by_hop": by_hop((r_inj <= 5).astype(float).tolist(), hops)},
        "answer_fact": {"hit1": float((r_ans <= 1).mean()),
                        "hit5": float((r_ans <= 5).mean()),
                        "hit20": float((r_ans <= 20).mean())},
        "chain": {"coverage20": float(np.mean(cov)),
                  "full20": float(np.mean(full)),
                  "full20_by_hop": by_hop(full, hops),
                  "coverage_at_nhop_budget": float(np.mean(budget_cov))},
    }


def chase(questions, hops, sup_idx, store, encode_fn, S0: np.ndarray,
          store_texts: list[str]) -> dict:
    """n_hop rounds of top-1 retrieval with text-level query expansion.
    Round 1 reuses the single-shot scores; later rounds re-encode
    question + retrieved texts for still-active multi-hop questions.
    Expansion uses whatever row was actually retrieved -- including a
    distractor: a wrong pick poisons the expanded query, which is part of
    what the metric measures."""
    B = len(questions)
    max_hop = max(hops)
    got: list[list[int]] = [[] for _ in range(B)]
    S = S0.copy()
    for step in range(max_hop):
        active = [i for i in range(B) if hops[i] > step]
        if step > 0:
            texts = [questions[i] + " " + " ".join(store_texts[j] for j in got[i])
                     for i in active]
            Qs = _unit(np.stack([encode_fn(t) for t in texts]))
            S_act = dense_scores(Qs, store)
        else:
            S_act = S[active]
        for row, i in enumerate(active):
            s = S_act[row].copy()
            s[got[i]] = -np.inf
            got[i].append(int(np.argmax(s)))
    cov = [len(set(g) & set(s)) / len(s) for g, s in zip(got, sup_idx)]
    full = [1.0 if set(s) <= set(g) else 0.0 for g, s in zip(got, sup_idx)]
    return {"coverage": float(np.mean(cov)), "full_chain": float(np.mean(full)),
            "full_by_hop": by_hop(full, hops), "coverage_by_hop": by_hop(cov, hops)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=800)
    ap.add_argument("--sweep", type=int, nargs="+", default=[0, 100_000])
    ap.add_argument("--with-ref", action="store_true", help="MiniLM anchor at N=0")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {TABLE.name} | rows {PQ_FILE.name}\n")

    questions, hops, targets, sup_idx, inj_idx, ans_idx = load_rows(args.questions)
    hop_counts = {h: hops.count(h) for h in sorted(set(hops))}
    print(f"{len(questions)} questions {hop_counts} | {len(targets)} unique facts in store rows 0..{len(targets) - 1}")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)

    def table_encode(text: str) -> np.ndarray:
        return enc.encode(text).reshape(-1)

    t0 = time.perf_counter()
    Qf = _unit(np.stack([table_encode(t) for t in questions]))
    Tf = _unit(np.stack([table_encode(t) for t in targets])).astype(np.float16)
    print(f"encoded {len(questions)}+{len(targets)} texts in {time.perf_counter() - t0:.0f}s")

    n_max = max(args.sweep)
    dis = np.zeros((n_max, K * L), dtype=np.float16)
    dis_texts: list[str] = []
    done = 0
    t0 = time.perf_counter()
    for chunk_texts in iter_distractors(n_max):
        V = np.stack([table_encode(t) for t in chunk_texts])
        dis[done:done + len(chunk_texts)] = _unit(V).astype(np.float16)
        dis_texts.extend(chunk_texts)
        done += len(chunk_texts)
        print(f"  encoded {done / 1e3:6.0f}k distractors ({time.perf_counter() - t0:.0f}s)", flush=True)

    out: dict = {"config": vars(args), "n_questions": len(questions),
                 "n_targets": len(targets), "hop_counts": {str(k): v for k, v in hop_counts.items()},
                 "results": {}}
    for N in args.sweep:
        store = np.concatenate([Tf, dis[:N]]) if N else Tf
        chance20 = 20 / store.shape[0]
        t0 = time.perf_counter()
        S = dense_scores(Qf, store)
        res = single_shot(S, hops, sup_idx, inj_idx, ans_idx)
        res["chase"] = chase(questions, hops, sup_idx, store, table_encode, S,
                             targets + dis_texts[:N])
        res["chance_hit20"] = chance20
        res["score_s"] = time.perf_counter() - t0
        out["results"][f"N={N}/table"] = res
        i, c, ch = res["inject"], res["chain"], res["chase"]
        print(f"\nN={N} store={store.shape[0]} (chance hit@20 = {chance20:.5f})")
        print(f"  inject   hit@1 {i['hit1']:.3f} hit@5 {i['hit5']:.3f} hit@20 {i['hit20']:.3f} "
              f"MRR {i['mrr']:.3f} | hit5 by hop {i['hit5_by_hop']}")
        print(f"  answer   hit@1 {res['answer_fact']['hit1']:.3f} hit@5 {res['answer_fact']['hit5']:.3f} "
              f"hit@20 {res['answer_fact']['hit20']:.3f}")
        print(f"  chain    cov@20 {c['coverage20']:.3f} full@20 {c['full20']:.3f} "
              f"by hop {c['full20_by_hop']} | cov@n_hop-budget {c['coverage_at_nhop_budget']:.3f}")
        print(f"  chase    cov {ch['coverage']:.3f} full {ch['full_chain']:.3f} "
              f"by hop {ch['full_by_hop']}")

    if args.with_ref:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def ref_encode(text: str) -> np.ndarray:
            return np.asarray(model.encode([text], show_progress_bar=False))[0]

        Qr = _unit(np.asarray(model.encode(questions, batch_size=64, show_progress_bar=False)))
        Tr = _unit(np.asarray(model.encode(targets, batch_size=64, show_progress_bar=False)))
        S = (Qr @ Tr.T).astype(np.float32)
        res = single_shot(S, hops, sup_idx, inj_idx, ans_idx)
        res["chase"] = chase(questions, hops, sup_idx, Tr.astype(np.float16),
                             ref_encode, S, targets)
        out["results"]["N=0/MiniLM"] = res
        i, c, ch = res["inject"], res["chain"], res["chase"]
        print(f"\nN=0 MiniLM anchor")
        print(f"  inject   hit@1 {i['hit1']:.3f} hit@5 {i['hit5']:.3f} hit@20 {i['hit20']:.3f} "
              f"MRR {i['mrr']:.3f} | hit5 by hop {i['hit5_by_hop']}")
        print(f"  chain    cov@20 {c['coverage20']:.3f} full@20 {c['full20']:.3f} "
              f"by hop {c['full20_by_hop']}")
        print(f"  chase    cov {ch['coverage']:.3f} full {ch['full_chain']:.3f} "
              f"by hop {ch['full_by_hop']}")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_injection.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_injection.json'}")


if __name__ == "__main__":
    main()
