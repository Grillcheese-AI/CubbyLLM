"""exp_m3_beir_dbpedia — the table on a published benchmark (BEIR dbpedia-entity).

Full BEIR dbpedia-entity: 4.64M corpus passages (E:\\datasets\\full-dbpedia),
400 test queries with graded qrels (0/1/2; qrels_test.tsv downloaded from
BeIR/dbpedia-entity-qrels). This anchors the paper against PUBLISHED numbers
instead of only self-relative ones. Literature anchors (verify against the
BEIR paper / MTEB before publication, they are quoted from memory here):
BM25 nDCG@10 ~0.313; all-MiniLM-L6-v2 ~0.32.

Arms, all encoded on this machine:
  table two-stage -- the validated serve shape: 80-byte compact store
                     (argmax/qFHRR form, 371 MB for 4.64M docs) ADC-scored
                     for a top-100 shortlist, then dense re-rank of the
                     shortlist (re-encoded on the fly; the dense store
                     itself would be 95 GB — the point of the compact form).
  table compact   -- shortlist stage alone (no re-rank), the ablation
  potion-base-8M  -- released model2vec checkpoint, full dense store
                     (256-d f16 = 2.4 GB), its natural serving form

Metrics: nDCG@10 (graded gains 2^rel - 1, the BEIR convention) and
recall@100 (fraction of a query's relevant docs in the top 100).
Standalone; never imported by cubbyllm/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import TABLE, _load_semantic_words  # noqa: E402
from exp_m3_haystack import compact_scores  # noqa: E402

CORPUS = pathlib.Path(r"E:\datasets\full-dbpedia\corpus\corpus-00000-of-00001.parquet")
QUERIES = pathlib.Path(r"E:\datasets\full-dbpedia\queries\queries-00000-of-00001.parquet")
QRELS = pathlib.Path(r"E:\datasets\full-dbpedia\qrels_test.tsv")
K, L = 80, 128


def load_qrels():
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for ln in open(QRELS, encoding="utf-8"):
        parts = ln.rstrip("\n").split("\t")
        if parts[0] == "query-id":
            continue
        qrels[parts[0]][parts[1]] = int(parts[2])
    return dict(qrels)


def load_queries(qids: set[str]):
    import pyarrow.parquet as pq

    t = pq.read_table(QUERIES, columns=["_id", "text"])
    rows = [(i, x) for i, x in zip(t.column("_id").to_pylist(),
                                   t.column("text").to_pylist()) if i in qids]
    return [r[0] for r in rows], [r[1] for r in rows]


def iter_corpus(chunk: int = 50_000):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(CORPUS)
    for batch in pf.iter_batches(batch_size=chunk,
                                 columns=["_id", "title", "text"]):
        ids = batch.column("_id").to_pylist()
        texts = [f"{a or ''}. {b or ''}"[:700]
                 for a, b in zip(batch.column("title").to_pylist(),
                                 batch.column("text").to_pylist())]
        yield ids, texts


def ndcg10(ranked_rows: np.ndarray, rel_of_row: dict[int, int]) -> float:
    dcg = 0.0
    for i, r in enumerate(ranked_rows[:10]):
        rel = rel_of_row.get(int(r), 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(rel_of_row.values(), reverse=True)[:10]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2)
               for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def eval_rankings(top_rows: np.ndarray, qids: list[str],
                  qrels: dict, row_rel: list[dict[int, int]]) -> dict:
    """top_rows: (nq, >=100) ranked corpus row indices per query."""
    nd, rec = [], []
    for qi in range(len(qids)):
        rels = row_rel[qi]
        nd.append(ndcg10(top_rows[qi], rels))
        n_rel = sum(1 for v in rels.values() if v > 0)
        if n_rel:
            got = sum(1 for r in top_rows[qi][:100] if rels.get(int(r), 0) > 0)
            rec.append(got / n_rel)
    return {"ndcg10": float(np.mean(nd)), "recall100": float(np.mean(rec)),
            "n_queries": len(nd)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-docs", type=int, default=0,
                    help="cap corpus size for smoke runs (0 = all)")
    ap.add_argument("--shortlist", type=int, default=100)
    ap.add_argument("--potion", default="minishlab/potion-base-8M")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")

    qrels = load_qrels()
    qids, qtexts = load_queries(set(qrels))
    print(f"{len(qids)} test queries | judged docs "
          f"{sum(len(v) for v in qrels.values())}")

    judged_ids = {cid for v in qrels.values() for cid in v}
    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    from model2vec import StaticModel
    potion = StaticModel.from_pretrained(args.potion)

    # ---- pass 1: stream-encode the corpus into both stores -----------------
    n_cap = args.max_docs or 5_000_000
    tbl_idx_parts, pot_parts = [], []
    row_of_id: dict[str, int] = {}          # judged ids only
    n_docs = 0
    t0 = time.perf_counter()
    t_tbl = t_pot = 0.0
    for ids, texts in iter_corpus():
        take = min(len(texts), n_cap - n_docs)
        if take <= 0:
            break
        ids, texts = ids[:take], texts[:take]
        s = time.perf_counter()
        V = np.stack([enc.encode(t) for t in texts])          # (c, K, L)
        t_tbl += time.perf_counter() - s
        tbl_idx_parts.append(V.argmax(axis=2).astype(np.uint8))
        s = time.perf_counter()
        P = np.asarray(potion.encode(texts, show_progress_bar=False),
                       dtype=np.float32)
        t_pot += time.perf_counter() - s
        P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        pot_parts.append(P.astype(np.float16))
        for j, cid in enumerate(ids):
            if cid in judged_ids:
                row_of_id[cid] = n_docs + j
        n_docs += take
        el = time.perf_counter() - t0
        print(f"  {n_docs / 1e6:5.2f}M docs ({el / 60:.1f} min; table "
              f"{t_tbl / n_docs * 1e6:.0f} us/doc, potion "
              f"{t_pot / n_docs * 1e6:.0f} us/doc)", flush=True)
    tbl_idx = np.concatenate(tbl_idx_parts); del tbl_idx_parts
    pot_store = np.concatenate(pot_parts); del pot_parts
    print(f"stores: compact {tbl_idx.nbytes / 1e6:.0f} MB (80 B/doc) | "
          f"potion dense {pot_store.nbytes / 1e9:.2f} GB")

    # judged relevance by corpus ROW, per query
    row_rel = [{row_of_id[cid]: rel for cid, rel in qrels[q].items()
                if cid in row_of_id} for q in qids]
    covered = sum(len(r) for r in row_rel)
    print(f"judged docs found in corpus: {covered} / "
          f"{sum(len(qrels[q]) for q in qids)}")

    # ---- queries -----------------------------------------------------------
    Q3 = np.stack([enc.encode(t) for t in qtexts])
    Qf = Q3.reshape(len(qids), -1)
    Qf = Qf / (np.linalg.norm(Qf, axis=1, keepdims=True) + 1e-12)
    Qd = Qf.reshape(len(qids), K, L)
    QP = np.asarray(potion.encode(qtexts, show_progress_bar=False),
                    dtype=np.float32)
    QP /= (np.linalg.norm(QP, axis=1, keepdims=True) + 1e-12)

    out: dict = {"config": vars(args), "n_docs": n_docs,
                 "n_queries": len(qids),
                 "table_encode_us_per_doc": t_tbl / n_docs * 1e6,
                 "potion_encode_us_per_doc": t_pot / n_docs * 1e6,
                 "results": {}, "literature_anchors_UNVERIFIED": {
                     "BM25_ndcg10": 0.313, "all-MiniLM-L6-v2_ndcg10": 0.32}}

    # ---- table: compact shortlist (query-chunked: a full 400 x 4.64M score
    # matrix would be 7.4 GB; 50 queries at a time keeps it under 1 GB) -----
    t0 = time.perf_counter()
    short = np.zeros((len(qids), args.shortlist), dtype=np.int64)
    for a in range(0, len(qids), 50):
        b = min(a + 50, len(qids))
        S = compact_scores(Qd[a:b], tbl_idx)
        short[a:b] = np.argsort(-S, axis=1)[:, :args.shortlist]
        del S
    ms = (time.perf_counter() - t0) / len(qids) * 1e3
    out["results"]["table_compact"] = eval_rankings(short, qids, qrels, row_rel)
    out["results"]["table_compact"]["score_ms_per_q"] = ms
    print(f"table compact: {out['results']['table_compact']}")

    # ---- pass 2: re-encode shortlisted docs dense, re-rank -----------------
    need = sorted({int(r) for row in short for r in row})
    pos_of = {r: i for i, r in enumerate(need)}
    dense = np.zeros((len(need), K * L), dtype=np.float32)
    row = 0
    ptr = 0
    for ids, texts in iter_corpus():
        if ptr >= len(need):
            break
        lo, hi = row, row + len(texts)
        while ptr < len(need) and need[ptr] < hi:
            r = need[ptr]
            v = enc.encode(texts[r - lo]).ravel()
            dense[pos_of[r]] = v / (np.linalg.norm(v) + 1e-12)
            ptr += 1
        row = hi
        if row >= n_docs:
            break
    rr = np.zeros_like(short)
    for qi in range(len(qids)):
        cand = short[qi]
        sc = dense[[pos_of[int(r)] for r in cand]] @ Qf[qi]
        rr[qi] = cand[np.argsort(-sc)]
    out["results"]["table_two_stage"] = eval_rankings(rr, qids, qrels, row_rel)
    print(f"table two-stage: {out['results']['table_two_stage']}")

    # ---- potion dense -------------------------------------------------------
    t0 = time.perf_counter()
    topP = np.zeros((len(qids), args.shortlist), dtype=np.int64)
    chunkN = 500_000
    best = np.full((len(qids), args.shortlist), -np.inf, dtype=np.float32)
    for a in range(0, n_docs, chunkN):
        b = min(a + chunkN, n_docs)
        Sc = QP @ pot_store[a:b].astype(np.float32).T
        loc = np.argsort(-Sc, axis=1)[:, :args.shortlist]
        cand = np.concatenate([topP, loc + a], axis=1)
        vals = np.concatenate([best, np.take_along_axis(Sc, loc, axis=1)], axis=1)
        keep = np.argsort(-vals, axis=1)[:, :args.shortlist]
        topP = np.take_along_axis(cand, keep, axis=1)
        best = np.take_along_axis(vals, keep, axis=1)
    ms = (time.perf_counter() - t0) / len(qids) * 1e3
    out["results"]["potion_dense"] = eval_rankings(topP, qids, qrels, row_rel)
    out["results"]["potion_dense"]["score_ms_per_q"] = ms
    print(f"potion dense: {out['results']['potion_dense']}")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_beir_dbpedia.json").write_text(json.dumps(out, indent=1),
                                                  encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_beir_dbpedia.json'}")


if __name__ == "__main__":
    main()
