"""exp_m3_haystack — needle-in-haystack retrieval vs store size (thread 2).

Targets: the deduped supporting facts of E:\\valid_scaling_law_with_facts.pq
(multi-hop QA; ground truth is each question's own `facts` list). Distractors:
DBpedia abstracts from BEIR dbpedia-entity, N in {0, 100k, 1M}. Questions
retrieve over targets+distractors with the production FastWordEncoder table.

Two store forms, measured side by side where RAM allows:
  dense   -- float vectors (k*l = 10240 dims); infeasible at 1M (20+ GB)
  compact -- argmax one-hot as 80 uint8 block indices = 80 BYTES/doc
             (1M docs = 80 MB); scored ADC-style: the DENSE query gathered
             at each doc's hot indices. No fitted codebook -> no leakage
             concerns; the quality cost of compaction is measured at the N
             where both forms fit.

MiniLM anchors N=0 only: at 1M docs the teacher's ~9 ms/text makes store
encoding infeasible (~2.5 h vs ~10 min for the table) -- which is the
feasibility argument for the fast path, stated by measurement.

Metrics per (N, form): any-supporting-fact hit@1/5/20, MRR of the best-ranked
supporting fact, split by n_hop. Standalone; never imported by cubbyllm/.
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

PQ_FILE = pathlib.Path(r"E:\valid_scaling_law_with_facts.pq")
DBPEDIA = pathlib.Path(r"E:\datasets\full-dbpedia\corpus\corpus-00000-of-00001.parquet")
K, L = 80, 128


def load_multihop(max_questions: int, seed: int = 0):
    """-> (questions, n_hops, targets, sup_idx: list[list[int]] per question)."""
    import pyarrow.parquet as pq

    t = pq.read_table(PQ_FILE, columns=["question_prompt", "facts", "n_hop"])
    qs = t.column("question_prompt").to_pylist()
    fs = t.column("facts").to_pylist()
    hops = t.column("n_hop").to_pylist()
    rows = [(q, f, h) for q, f, h in zip(qs, fs, hops)
            if q and f and len(q) > 15 and all(x for x in f)]
    rng = np.random.default_rng(seed)
    if len(rows) > max_questions:
        idx = rng.choice(len(rows), size=max_questions, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    fact_id: dict[str, int] = {}
    targets: list[str] = []
    sup_idx: list[list[int]] = []
    for _, facts, _ in rows:
        ids = []
        for f in facts:
            f = " ".join(f.split())
            if f not in fact_id:
                fact_id[f] = len(targets)
                targets.append(f)
            ids.append(fact_id[f])
        sup_idx.append(sorted(set(ids)))
    return ([r[0] for r in rows], [int(r[2]) for r in rows], targets, sup_idx)


def iter_distractors(n: int, chunk: int = 50_000):
    """Yield up to n DBpedia passages ('title. text', truncated)."""
    import pyarrow.parquet as pq

    got = 0
    pf = pq.ParquetFile(DBPEDIA)
    for batch in pf.iter_batches(batch_size=chunk, columns=["title", "text"]):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        out = []
        for ti, tx in zip(titles, texts):
            if got >= n:
                break
            out.append(f"{ti or ''}. {tx or ''}"[:700])
            got += 1
        if out:
            yield out
        if got >= n:
            return


def compact_scores(Qd: np.ndarray, idx: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """ADC scoring: dense queries (B, K, L) gathered at each doc's hot indices
    (N, K) uint8 -> (B, N) scores. Ranking-equivalent to cosine against the
    one-hot store (doc norms are constant sqrt(K))."""
    B = Qd.shape[0]
    N = idx.shape[0]
    S = np.zeros((B, N), dtype=np.float32)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        blk = idx[a:b]                                   # (n, K)
        acc = np.zeros((B, b - a), dtype=np.float32)
        for k_ in range(K):
            acc += Qd[:, k_, :][:, blk[:, k_]]
        S[:, a:b] = acc
    return S


def dense_scores(Qf: np.ndarray, store: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Cosine of unit queries (B, D) against unit store rows (N, D), chunked."""
    S = np.zeros((Qf.shape[0], store.shape[0]), dtype=np.float32)
    for a in range(0, store.shape[0], chunk):
        b = min(a + chunk, store.shape[0])
        S[:, a:b] = Qf @ store[a:b].astype(np.float32).T
    return S


def metrics(S: np.ndarray, sup_idx: list[list[int]], hops: list[int]) -> dict:
    """Supporting facts occupy store rows [0, n_targets); distractors follow."""
    order = np.argsort(-S, axis=1)
    out = {"hit1": 0.0, "hit5": 0.0, "hit20": 0.0, "mrr": 0.0}
    by_hop: dict[int, list[float]] = {}
    n = len(sup_idx)
    for i in range(n):
        sup = set(sup_idx[i])
        ranks = [int(np.where(order[i] == s)[0][0]) + 1 for s in sup]
        best = min(ranks)
        out["hit1"] += best <= 1
        out["hit5"] += best <= 5
        out["hit20"] += best <= 20
        out["mrr"] += 1.0 / best
        by_hop.setdefault(hops[i], []).append(1.0 if best <= 5 else 0.0)
    for k_ in out:
        out[k_] = float(out[k_] / n)
    out["hit5_by_hop"] = {str(h): float(np.mean(v)) for h, v in sorted(by_hop.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=400)
    ap.add_argument("--sweep", type=int, nargs="+", default=[0, 100_000, 1_000_000])
    ap.add_argument("--dense-max", type=int, default=100_000,
                    help="largest N at which the dense store is also scored")
    ap.add_argument("--with-ref", action="store_true", help="MiniLM anchor at N=0")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table {TABLE.name} | targets {PQ_FILE.name} | distractors {DBPEDIA.parent.parent.name}\n")

    questions, hops, targets, sup_idx = load_multihop(args.questions)
    print(f"{len(questions)} questions ({dict((h, hops.count(h)) for h in sorted(set(hops)))} hops) "
          f"| {len(targets)} unique supporting facts (store rows 0..{len(targets) - 1})")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)

    t0 = time.perf_counter()
    Q3 = np.stack([enc.encode(t) for t in questions])            # (B, K, L)
    T3 = np.stack([enc.encode(t) for t in targets])
    print(f"encoded questions+targets in {time.perf_counter() - t0:.0f}s")
    Qf = Q3.reshape(len(questions), -1)
    Qf = Qf / (np.linalg.norm(Qf, axis=1, keepdims=True) + 1e-12)
    Qd = Qf.reshape(len(questions), K, L)                        # unit, dense

    n_max = max(args.sweep)
    tgt_idx = T3.argmax(axis=2).astype(np.uint8)                 # (T, K)
    tgt_dense = T3.reshape(len(targets), -1)
    tgt_dense = (tgt_dense / (np.linalg.norm(tgt_dense, axis=1, keepdims=True)
                              + 1e-12)).astype(np.float16)

    dis_idx = np.zeros((n_max, K), dtype=np.uint8)
    dis_dense = np.zeros((min(n_max, args.dense_max), K * L), dtype=np.float16)
    done = 0
    t0 = time.perf_counter()
    for chunk_texts in iter_distractors(n_max):
        V = np.stack([enc.encode(t) for t in chunk_texts])       # (c, K, L)
        c = len(chunk_texts)
        dis_idx[done:done + c] = V.argmax(axis=2).astype(np.uint8)
        take = max(0, min(args.dense_max - done, c))
        if take:
            F = V[:take].reshape(take, -1)
            dis_dense[done:done + take] = (
                F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
            ).astype(np.float16)
        done += c
        el = time.perf_counter() - t0
        print(f"  encoded {done / 1e3:6.0f}k distractors ({el:.0f}s, "
              f"{el / done * 1e6:.0f} us/doc)", flush=True)
    encode_s = time.perf_counter() - t0

    out: dict = {"config": vars(args), "n_questions": len(questions),
                 "n_targets": len(targets), "distractor_encode_s": encode_s,
                 "results": {}}
    print(f"\n{'N':>10} {'form':>8} {'hit@1':>7} {'hit@5':>7} {'hit@20':>7} "
          f"{'MRR':>7} {'score-ms/q':>11}")
    for N in args.sweep:
        # compact form (always)
        idx_store = np.concatenate([tgt_idx, dis_idx[:N]])
        t0 = time.perf_counter()
        S = compact_scores(Qd, idx_store)
        ms = (time.perf_counter() - t0) / len(questions) * 1e3
        m = metrics(S, sup_idx, hops)
        m["score_ms_per_q"] = ms
        out["results"][f"N={N}/compact"] = m
        print(f"{N:>10} {'compact':>8} {m['hit1']:7.3f} {m['hit5']:7.3f} "
              f"{m['hit20']:7.3f} {m['mrr']:7.3f} {ms:11.1f}")
        # dense form (where it fits)
        if N <= args.dense_max:
            store = np.concatenate([tgt_dense, dis_dense[:N]])
            t0 = time.perf_counter()
            S = dense_scores(Qf.astype(np.float32), store)
            ms = (time.perf_counter() - t0) / len(questions) * 1e3
            m = metrics(S, sup_idx, hops)
            m["score_ms_per_q"] = ms
            out["results"][f"N={N}/dense"] = m
            print(f"{N:>10} {'dense':>8} {m['hit1']:7.3f} {m['hit5']:7.3f} "
                  f"{m['hit20']:7.3f} {m['mrr']:7.3f} {ms:11.1f}")

    if args.with_ref:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        t0 = time.perf_counter()
        Qr = np.asarray(model.encode(questions, batch_size=64, show_progress_bar=False))
        Tr = np.asarray(model.encode(targets, batch_size=64, show_progress_bar=False))
        ref_us = (time.perf_counter() - t0) / (len(questions) + len(targets)) * 1e6
        Qr = Qr / (np.linalg.norm(Qr, axis=1, keepdims=True) + 1e-12)
        Tr = Tr / (np.linalg.norm(Tr, axis=1, keepdims=True) + 1e-12)
        m = metrics(Qr @ Tr.T, sup_idx, hops)
        out["results"]["N=0/MiniLM"] = m
        out["results"]["N=0/MiniLM"]["encode_us_per_text"] = ref_us
        print(f"{0:>10} {'MiniLM':>8} {m['hit1']:7.3f} {m['hit5']:7.3f} "
              f"{m['hit20']:7.3f} {m['mrr']:7.3f}   ({ref_us:.0f} us/text; "
              f"at 1M docs the teacher store-encode alone would be "
              f"~{ref_us * 1e6 / 3.6e9:.1f} h)")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_haystack.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_haystack.json'}")


if __name__ == "__main__":
    main()
