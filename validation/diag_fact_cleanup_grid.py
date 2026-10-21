"""diag_fact_cleanup_grid -- A0 extended (order 0001, Addendum 2): every read-side fix, offline.

``diag_fact_cleanup.py`` showed the far facts' answer is in run 3's read. This rebuilds the scene
from the checkpoint's stored vectors and reads it back under each read-side change, before
anything trains -- per-piece top-1 and whole-answer exact for every cell of

  codes       the table as the scene stores it | random unit identity codes (B4's best case)
  sub-scenes  K = 1, 2, 4: a WRITE goes into, and an ASK reads from, sub-scene hash(place) mod K
              only (B8; the full place is hashed, so an object's pieces spread)
  candidates  the whole vocabulary | the tokens written in this document so far (the ring, B1)

and, per cell with the written candidates, the median gap between the target's cosine and the best
wrong writer's -- what the ring's temperature starts from (``tau = log(2 / gap)``, Addendum 3) -- and
the read-aware gate's threshold: the midpoint between the present reads' cosine to their target and
the absent reads' best (the asker after a name's last piece), per candidate set (``copy_beta``).
``CB_CELLS=stored_K1,...`` runs only those cells.

    CB_CKPT=<spatial_link_3_final.pt> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json>
    [CB_SHARDS=facts_far1k_lead] [CB_DOCS=400] [CB_OUT=<json>] python validation/diag_fact_cleanup_grid.py
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
os.environ.setdefault("CB_SHARDS", "facts_far1k_lead")
os.environ.setdefault("CB_DOCS", "400")
import exp_hypertoken_recall as hr  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE, SpatialLink  # noqa: E402

WRITE, ASK, KS = 1, 2, (1, 2, 4)


def sub(p: np.ndarray, K: int) -> np.ndarray:
    """The sub-scene of each place (..., 3): a hash of the whole place, mod K."""
    p = p.astype(np.int64)
    return ((p[..., 0] * 73856093) ^ (p[..., 1] * 19349663) ^ (p[..., 2] * 83492791)) % K


def doc_reads(tk, x, P, R):
    """One document's reads: ``(asker h, stored ids at its address, question key, first piece)``,
    and the absent ones (the last piece asking for the next: ``stored`` None)."""
    placed = P[:, 0] != NO_PLACE
    w, a = placed & ((R & WRITE) != 0), placed & ((R & ASK) != 0)
    stored: dict = {}
    for i in np.flatnonzero(w):
        stored.setdefault(tuple(P[i]), []).append(int(x[i]))
    out, seen = [], set()
    for j, k in hr.answers(tk, x):
        for t in range(j, k + 1):
            h = t - 1
            while h > 0 and not a[h]:
                h -= 1
            there = stored.get(tuple(P[h])) if a[h] else None
            if there is None or h in seen:
                continue
            seen.add(h)
            out.append((h, there, j, t == j))
        if a[k] and tuple(P[k]) not in stored and k not in seen:
            seen.add(k)
            out.append((k, None, j, False))
    return w, out


@torch.no_grad()
def cell(link, C, Cn, tk, docs, K: int) -> dict:
    """top-1 over the vocabulary and over the written tokens, exact for both, the writer gap."""
    hv, hw, gaps, qv, qw, pres, absn = [], [], [], {}, {}, {"vocab": [], "written": []}, {"vocab": [], "written": []}
    for r, (x, P, R) in enumerate(docs):
        w, rd = doc_reads(tk, x, P, R)
        if not rd:
            continue
        wi = np.flatnonzero(w)
        bound = link.bind(C[torch.from_numpy(x[wi])], torch.from_numpy(P[wi]).float())     # (n_w, d)
        wsub = torch.from_numpy(sub(P[wi], K))
        reads = []
        for h, _there, _j, _f in rd:
            m = torch.from_numpy(wi < h) & (wsub == int(sub(P[h], K)))
            reads.append(link.lookup(bound[m].sum(0), link.encode(torch.from_numpy(P[h]).float())))
        cos = F.normalize(torch.stack(reads), dim=-1) @ Cn.T                                 # (m, V)
        for i, (h, there, j, f) in enumerate(rd):
            written = torch.tensor(sorted(set(x[wi[wi < h]].tolist())))
            if there is None:                                                               # an absent read
                absn["vocab"].append(float(cos[i].max())); absn["written"].append(float(cos[i, written].max()))
                continue
            t = torch.tensor(there)
            pres["vocab"].append(float(cos[i, t].max())); pres["written"].append(float(cos[i, t].max()))
            ok_v = int(cos[i].argmax()) in there
            ok_w = int(written[cos[i, written].argmax()]) in there
            others = written[~torch.isin(written, t)]
            if len(others):
                gaps.append(float(cos[i, t].max() - cos[i, others].max()))
            hv.append(ok_v); hw.append(ok_w)
            qv[(r, j)] = qv.get((r, j), True) and ok_v
            qw[(r, j)] = qw.get((r, j), True) and ok_w
    return {"reads": len(hv), "top1_vocab": float(np.mean(hv)), "top1_written": float(np.mean(hw)),
            "exact_vocab": float(np.mean(list(qv.values()))), "exact_written": float(np.mean(list(qw.values()))),
            "writer_gap_median": float(np.median(gaps)), "tau_for_2_nats": float(np.log(2.0 / np.median(gaps))),
            **{f"beta_{c}": (float(np.median(pres[c])) + float(np.median(absn[c]))) / 2 for c in pres if absn[c]},
            **{f"absent_best_{c}": float(np.median(absn[c])) for c in absn if absn[c]},
            "present_target_cos": float(np.median(pres["written"])), "absent_reads": len(absn["written"])}


def main() -> None:
    t0 = time.time()
    from tokenizers import Tokenizer
    ck = torch.load(hr.CKPT, map_location="cpu", weights_only=False, mmap=True)
    meta = dict(ck["meta"])
    E = ck["params"][0].float().clone()
    del ck
    sp = meta["spatial"]
    link = SpatialLink(meta["D"], scale=float(sp.get("scale", 2.0)), seed=int(sp.get("seed", 0)))
    tk = Tokenizer.from_file(hr.SPM)
    ident = F.normalize(torch.randn(E.shape, generator=torch.Generator().manual_seed(0)), dim=-1)
    codes = {"stored": (E, F.normalize(E, dim=-1)), "random": (ident, ident)}
    report = {"ckpt": hr.CKPT, "docs": hr.DOCS, "shards": {},
              "env": f"{time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} torch {torch.__version__} "
                     f"numpy {np.__version__} {platform.processor() or platform.machine()} (CPU)"}
    for name in hr.SHARDS:
        x, (places, roles) = hr.heldout_docs(name, hr.DOCS)
        docs = [(x[r], places[r], roles[r]) for r in range(len(x))]
        res = report["shards"][name] = {}
        for code, (C, Cn) in codes.items():
            for K in KS:
                if os.environ.get("CB_CELLS") and f"{code}_K{K}" not in os.environ["CB_CELLS"].split(","):
                    continue
                v = res[f"{code}_K{K}"] = cell(link, C, Cn, tk, docs, K)
                print(f"  {name} {code:6s} K={K}  {v['reads']} reads  per piece: vocabulary {v['top1_vocab']:.3f}  written "
                      f"{v['top1_written']:.3f}   whole answers: vocabulary {v['exact_vocab']:.3f}  written {v['exact_written']:.3f}"
                      f"   writer gap {v['writer_gap_median']:.3f} (tau {v['tau_for_2_nats']:.2f})   present target cos"
                      f" {v['present_target_cos']:.3f}, absent best {v.get('absent_best_written', float('nan')):.3f} written /"
                      f" {v.get('absent_best_vocab', float('nan')):.3f} vocabulary ({v['absent_reads']}) -> beta"
                      f" {v.get('beta_written', float('nan')):.3f} ring / {v.get('beta_vocab', float('nan')):.3f} table"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
    if hr.OUT:
        json.dump(report, open(hr.OUT, "w", encoding="utf-8"), indent=1)
        print(f"  wrote {hr.OUT}")


if __name__ == "__main__":
    main()
