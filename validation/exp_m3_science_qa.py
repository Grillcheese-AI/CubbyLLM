"""exp_m3_science_qa — real science questions against the fast word table.

H:\\datasets_facts\\KonstantyM___science_qa_prep holds 4.28M rows of
``input = "context: tag/<topic>/ question: <q>"`` / ``label = expert answer``.
Natural questions with self-labeled ground truth twice over: the topic tag
(routing) and the paired answer (retrieval). Two evals from one encoding pass,
using the open-set screen's winning rule (centroid routing + margin gate):

  1. Q->A retrieval: does a question retrieve ITS OWN answer among all stored
     answers? (hit@1/@5, MRR; chance = 1/n_store) -- the cross-modal
     question->explanation match, the production analog of name->formula.
  2. Tag routing: route each question to a topic by cosine to per-tag answer
     CENTROIDS; macro/micro vs a measured noise null, plus the margin gate's
     same/diff separation.

Standalone; never imported by cubbyllm/. MiniLM runs only as the reference.
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

from exp_m3_domain_routing import TABLE, _load_semantic_words, youden_tau  # noqa: E402

DATA = pathlib.Path(r"H:\datasets_facts\KonstantyM___science_qa_prep\default\0.0.0"
                    r"\9dea632d62d523d0a3fd2bf027950821f95090c2")
ROW_RE = re.compile(r"^context:\s*tag/([^/\s]+)/\s*question:\s*(.+)$", re.S)


def load_pairs(n_tags: int, per_tag: int, shards: int = 1,
               seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    """-> {tag: [(question, answer), ...]} for the top-n_tags by count."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    buckets: dict[str, list[tuple[str, str]]] = {}
    files = sorted(DATA.glob("science_qa_prep-train-*.arrow"))[:shards]
    for f in files:
        with open(f, "rb") as fh:
            try:
                t = ipc.open_stream(fh).read_all()
            except pa.lib.ArrowInvalid:
                fh.seek(0)
                t = ipc.open_file(fh).read_all()
        for inp, lab in zip(t.column("input").to_pylist(),
                            t.column("label").to_pylist()):
            m = ROW_RE.match(inp or "")
            if not m or not lab or len(lab) < 80:
                continue
            tag, q = m.group(1), " ".join(m.group(2).split())
            if len(q) < 20:
                continue
            buckets.setdefault(tag, []).append((q, " ".join(lab.split())[:900]))
    # top tags with enough rows, deterministic subsample per tag
    rng = np.random.default_rng(seed)
    good = {k_: v for k_, v in buckets.items() if len(v) >= per_tag}
    top = sorted(good, key=lambda k_: -len(good[k_]))[:n_tags]
    out = {}
    for k_ in top:
        rows = good[k_]
        idx = rng.choice(len(rows), size=per_tag, replace=False)
        out[k_] = [rows[i] for i in sorted(idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", type=int, default=40)
    ap.add_argument("--per-tag", type=int, default=20)
    ap.add_argument("--shards", type=int, default=15)
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--with-ref", action="store_true")
    args = ap.parse_args()

    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"table: {TABLE.name} | data: {DATA.parent.parent.parent.name} "
          f"({args.shards} shard(s))\n")

    t0 = time.perf_counter()
    pairs = load_pairs(args.tags, args.per_tag, args.shards)
    tags = sorted(pairs)
    Q_texts, A_texts, tag_of = [], [], []
    for tg in tags:
        for q, a in pairs[tg]:
            Q_texts.append(q); A_texts.append(a); tag_of.append(tg)
    n = len(Q_texts)
    print(f"loaded {n} QA pairs across {len(tags)} tags in "
          f"{time.perf_counter() - t0:.0f}s: {tags[:8]}...\n")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    t0 = time.perf_counter()
    Q = np.stack([enc.encode(t).ravel() for t in Q_texts])
    A = np.stack([enc.encode(t).ravel() for t in A_texts])
    us = (time.perf_counter() - t0) / (2 * n) * 1e6
    print(f"encoded {2 * n} texts at {us:.0f} us/text")

    def _evals(Qm: np.ndarray, Am: np.ndarray, label: str) -> dict:
        q = Qm / (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-12)
        a = Am / (np.linalg.norm(Am, axis=1, keepdims=True) + 1e-12)
        sims = q @ a.T
        rank_of_self = (sims >= sims[np.arange(n), np.arange(n)][:, None]).sum(axis=1)
        hit1 = float((sims.argmax(axis=1) == np.arange(n)).mean())
        hit5 = float((rank_of_self <= 5).mean())
        mrr = float((1.0 / rank_of_self).mean())
        td = np.asarray(tag_of)
        C = np.stack([a[td == tg].mean(axis=0) for tg in tags])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = q @ C.T
        picked = np.asarray(tags)[S.argmax(axis=1)]
        hit = picked == td
        per_tag = [float(hit[td == tg].mean()) for tg in tags]
        top2 = np.sort(S, axis=1)[:, -2:]
        same = np.array([S[i, tags.index(td[i])] for i in range(n)])
        diff_mask = ~np.eye(len(tags), dtype=bool)
        diff = np.array([S[i][np.arange(len(tags)) != tags.index(td[i])].max()
                         for i in range(n)])
        tau, auc = youden_tau(same, diff)
        print(f"{label:>12}: Q->A hit@1 {hit1:.3f} hit@5 {hit5:.3f} MRR {mrr:.3f} "
              f"(chance hit@1 {1 / n:.4f})")
        print(f"{'':>12}  tag-route macro {float(np.mean(per_tag)):.4f} "
              f"micro {float(hit.mean()):.4f} | margin mean "
              f"{float((top2[:, 1] - top2[:, 0]).mean()):.4f} | AUC {auc:.4f} tau {tau:.4f}")
        return {"hit1": hit1, "hit5": hit5, "mrr": mrr,
                "route_macro": float(np.mean(per_tag)), "route_micro": float(hit.mean()),
                "auc": auc, "tau": tau,
                "per_tag": dict(zip(tags, per_tag))}

    out: dict = {"config": vars(args), "n_pairs": n, "tags": tags,
                 "encode_us_per_text": us}
    out["table"] = _evals(Q, A, "table")

    # measured noise null (the baseline is a property of the procedure)
    nulls = {"hit1": [], "route_macro": []}
    for s_ in range(args.noise_seeds):
        g = np.random.default_rng(1000 + s_)
        rn = _evals(g.standard_normal(Q.shape), g.standard_normal(A.shape),
                    f"noise[{s_}]")
        nulls["hit1"].append(rn["hit1"]); nulls["route_macro"].append(rn["route_macro"])
    out["noise_null"] = {k_: (float(np.mean(v)), float(np.std(v)))
                        for k_, v in nulls.items()}

    if args.with_ref:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        t0 = time.perf_counter()
        Qr = np.asarray(model.encode(Q_texts, batch_size=64, show_progress_bar=False))
        Ar = np.asarray(model.encode(A_texts, batch_size=64, show_progress_bar=False))
        ref_us = (time.perf_counter() - t0) / (2 * n) * 1e6
        out["ref"] = _evals(Qr, Ar, "MiniLM ref")
        out["ref"]["batched_us_per_text"] = ref_us
        print(f"{'':>12}  (MiniLM {ref_us:.0f} us/text batched | table/ref: "
              f"hit@1 {out['table']['hit1'] / max(out['ref']['hit1'], 1e-9):.2f}x, "
              f"route {out['table']['route_macro'] / max(out['ref']['route_macro'], 1e-9):.2f}x)")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_science_qa.json").write_text(json.dumps(out, indent=1),
                                                encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_science_qa.json'}")


if __name__ == "__main__":
    main()
