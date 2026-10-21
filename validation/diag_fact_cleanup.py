"""diag_fact_cleanup -- is the answer in the read at all? (order 0001, A0; no training).

The far facts read 0.021 exact on the 450M (run 3) while the villages read 0.953. Before asking
why the trunk does not decode the read, ask whether there is anything to decode: at every answer
piece, the scene the trunk was given is unbound at the address the asking token carries -- the
link's ``lookup`` before the gate and ``o``, so no trained weight but the embedding table is
involved -- and the result is cleaned up against the table the scene was built from: argmax over
the whole vocabulary, the qFHRR paper's decoding (``docs/papers/2026-08-07-qfhrr-semantic-word-
tables.md`` §3, §5). The scene is ``sum e_i (*) S(p_i)`` over the writing tokens, with ``e`` what
``HybridEmbedding.embed`` returns: ``train_base.build_model`` builds it without a tail generator,
so ``embed(ids, ctx)`` is the core table's row and the table below is exactly what was stored.

Per shard, per piece (first / later), one read per asking token: the rank of the token stored at
the asked address among the vocabulary by cosine and by dot product (top-1, top-5), cos(read,
stored) against the best token not stored there, and the *absent* reads -- an asking token whose
address holds nothing (the last answer piece asks for the piece after it; a village question
answered "nothing") -- whose best cosine is the floor a present read must clear. Also how often
the piece the question wants IS the stored token: a fact's answer is its stored pieces, a
village answers `` pond`` (with its space) for a stored ``pond``, so the trunk there maps one
closed set to another rather than copying. Stored top-1 most of the time: the information is in
the read and the failure is the trunk's decoding (a copy path should work before training).
Not: crosstalk or the learned, correlated embeddings are the problem (identity codes the lead).

    CB_CKPT=<spatial_link_3_final.pt> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json>
    [CB_SHARDS=facts_far1k_lead,spatial_here] [CB_DOCS=400] [CB_OUT=<json>] python validation/diag_fact_cleanup.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CB_SHARDS", "facts_far1k_lead,spatial_here")
os.environ.setdefault("CB_DOCS", "400")
import exp_hypertoken_recall as hr  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE, SpatialLink  # noqa: E402

WRITE, ASK = 1, 2


def reads(link: SpatialLink, E: torch.Tensor, tk, x, places, roles):
    """Every answer piece's read, once per asking token: ``(read, stored, first, adjacent,
    answer)`` -- ``stored`` the token ids written at the asked address (the target set: a village
    name's pieces share one cell), ``adjacent`` whether the asker is the token right before the
    piece, ``answer`` whether the piece the question wants is one of them (a village answers
    `` pond`` for a stored ``pond``) -- and the absent reads, whose address holds nothing (the last
    piece asks for the one after it; a village question answered "nothing")."""
    present, absent = [], []
    for r in range(len(x)):
        P = torch.from_numpy(places[r]).float()
        placed = places[r, :, 0] != NO_PLACE
        w = placed & ((roles[r] & WRITE) != 0)
        a = placed & ((roles[r] & ASK) != 0)
        stored: dict = {}
        for i in np.flatnonzero(w):
            stored.setdefault(tuple(places[r, i]), []).append(int(x[r, i]))
        ids, wt = torch.from_numpy(x[r]), torch.from_numpy(w)
        scene = link.scene(link.bind(E[ids].unsqueeze(0), torch.where(wt.unsqueeze(-1), P, 0 * P).unsqueeze(0)),
                           wt.unsqueeze(0))[0]                                 # (S, d): V_t, causal
        seen = set()
        for j, k in hr.answers(tk, x[r]):
            for t in range(j, k + 2):                                          # k + 1: the piece after the last
                h = t - 1
                while h > 0 and not a[h]:
                    h -= 1
                if not a[h] or h in seen:
                    continue
                seen.add(h)
                rd = link.lookup(scene[h], link.encode(P[h]))
                there = stored.get(tuple(places[r, h]))
                if there is None:
                    absent.append((rd, [], False, h == t - 1, False))
                elif t <= k:
                    written = sorted(set(x[r, :h][w[:h]].tolist()))             # the document's writers so far
                    present.append((rd, there, t == j, h == t - 1, int(x[r, t]) in there, written, (r, j)))
    return present, absent


@torch.no_grad()
def cleanup(rows, E: torch.Tensor) -> dict:
    """Rank of the best stored token among the table's rows, by cosine and by dot; the cosines
    of that token, of the best row not stored there, and of the best row (for absent reads)."""
    En = F.normalize(E, dim=-1)
    out = {k: [] for k in ("rank_cos", "rank_dot", "cos_target", "cos_wrong", "cos_best", "first", "adjacent", "answer",
                           "doc_top1", "q")}
    for b in range(0, len(rows), 64):
        chunk = rows[b:b + 64]
        R = torch.stack([c[0] for c in chunk])
        cos = F.normalize(R, dim=-1) @ En.T                                     # (n, V)
        dot = R @ E.T
        out["cos_best"] += cos.max(1).values.tolist()
        for i, c in enumerate(chunk):
            if not c[1]:
                continue
            t = torch.tensor(c[1])
            ct, dt = cos[i, t].max(), dot[i, t].max()
            out["rank_cos"].append(int((cos[i] > ct).sum()))
            out["rank_dot"].append(int((dot[i] > dt).sum()))
            out["cos_target"].append(float(ct))
            out["cos_wrong"].append(float(cos[i].scatter(0, t, -2.0).max()))
            out["first"].append(c[2]); out["adjacent"].append(c[3]); out["answer"].append(c[4])
            cand = torch.tensor(c[5])                                           # a pointer: only what was written here
            out["doc_top1"].append(int(cand[cos[i, cand].argmax()]) in c[1])
            out["q"].append(c[6])
    return out


def summary(c: dict, sel) -> dict:
    sel = np.asarray(sel, dtype=bool)
    if not sel.any():
        return {"n": 0}
    rc, rd = np.asarray(c["rank_cos"])[sel], np.asarray(c["rank_dot"])[sel]
    ct, cw = np.asarray(c["cos_target"])[sel], np.asarray(c["cos_wrong"])[sel]
    dt = np.asarray(c["doc_top1"])[sel]
    return {"n": int(sel.sum()), "top1_cos": float((rc == 0).mean()), "top5_cos": float((rc < 5).mean()),
            "top1_among_written": float(dt.mean()),
            "top1_dot": float((rd == 0).mean()), "top5_dot": float((rd < 5).mean()),
            "median_rank_cos": float(np.median(rc)), "cos_target": float(ct.mean()), "cos_best_wrong": float(cw.mean()),
            "target_beats_wrong": float((ct > cw).mean())}


def main() -> None:
    t0 = time.time()
    from tokenizers import Tokenizer
    ck = torch.load(hr.CKPT, map_location="cpu", weights_only=False, mmap=True)
    meta = dict(ck["meta"])
    E = ck["params"][0].float().clone()                                         # model.parameters()[0]: embedding.core
    del ck
    if tuple(E.shape) != (meta["vocab"], meta["D"]):
        raise SystemExit(f"params[0] is {tuple(E.shape)}, not the ({meta['vocab']}, {meta['D']}) embedding")
    sp = meta["spatial"]
    link = SpatialLink(meta["D"], scale=float(sp.get("scale", 2.0)), seed=int(sp.get("seed", 0)))
    tk = Tokenizer.from_file(hr.SPM)
    norms = E.norm(dim=-1)
    report = {"ckpt": hr.CKPT, "meta": meta, "docs": hr.DOCS, "table_norm": {"mean": float(norms.mean()),
              "p5": float(norms.quantile(0.05)), "p95": float(norms.quantile(0.95))}, "shards": {},
              "env": f"{time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} torch {torch.__version__} "
                     f"numpy {np.__version__} {platform.processor() or platform.machine()} (CPU)"}
    print(f"cleanup of the read -- {os.path.basename(hr.CKPT)}: D={meta['D']} vocab {meta['vocab']}, "
          f"|e| mean {report['table_norm']['mean']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    for name in hr.SHARDS:
        x, pr = hr.heldout_docs(name, hr.DOCS)
        places, roles = pr
        writers = ((places[..., 0] != NO_PLACE) & ((roles & WRITE) != 0)).sum(1)
        present, absent = reads(link, E, tk, x, places, roles)
        c, ca = cleanup(present, E), cleanup(absent, E)
        first = np.asarray(c["first"])
        res = {"documents": len(x), "writers_per_doc": float(writers.mean()),
               "asker_adjacent": float(np.mean(c["adjacent"])) if present else 0.0,
               "answer_is_stored": float(np.mean(c["answer"])) if present else 0.0,
               "all": summary(c, np.ones(len(first), bool)), "first": summary(c, first), "later": summary(c, ~first),
               "absent": {"n": len(absent), "cos_best": float(np.mean(ca["cos_best"])) if absent else None}}
        qs: dict = {}                                                           # a whole answer: every piece top-1
        for q, top, doc in zip(c["q"], np.asarray(c["rank_cos"]) == 0, c["doc_top1"]):
            a_, b_ = qs.get(q, (True, True))
            qs[q] = (a_ and bool(top), b_ and bool(doc))
        res["exact_by_cleanup"] = float(np.mean([v[0] for v in qs.values()])) if qs else 0.0
        res["exact_among_written"] = float(np.mean([v[1] for v in qs.values()])) if qs else 0.0
        if absent and present:
            res["margin_present_over_absent"] = res["all"]["cos_target"] / max(res["absent"]["cos_best"], 1e-9)
        report["shards"][name] = res
        for part in ("all", "first", "later"):
            v = res[part]
            if v["n"]:
                print(f"  {name:17s} {part:5s} {v['n']:5d} reads  top-1 {v['top1_cos']:.3f} top-5 {v['top5_cos']:.3f} (cos)"
                      f"  top-1 {v['top1_dot']:.3f} top-5 {v['top5_dot']:.3f} (dot)  median rank {v['median_rank_cos']:.0f}"
                      f"  cos target {v['cos_target']:.3f} vs best wrong {v['cos_best_wrong']:.3f}"
                      f"  | among the document's writers top-1 {v['top1_among_written']:.3f}", flush=True)
        print(f"  {name:17s} absent {res['absent']['n']} reads, best cos {res['absent']['cos_best']}; "
              f"{res['writers_per_doc']:.1f} writers a document; asker right before the piece {res['asker_adjacent']:.3f}; "
              f"the answer piece is the stored token {res['answer_is_stored']:.3f}; whole answers by cleanup "
              f"{res['exact_by_cleanup']:.3f} (vocabulary) {res['exact_among_written']:.3f} (the document's writers)"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    if hr.OUT:
        os.makedirs(os.path.dirname(hr.OUT), exist_ok=True)
        json.dump(report, open(hr.OUT, "w", encoding="utf-8"), indent=1)
        print(f"  wrote {hr.OUT}")


if __name__ == "__main__":
    main()
