"""recall_split -- the held-out questions of a loci-token shard scored as ``exp_hypertoken_recall``
scores them (teacher-forced: the first answer token, the whole answer), and split the way the toy's
evaluation splits them (``exp_k10_far_facts``; order 0005), so a gap between the probe's exact and
the read's ceiling (0.965 whole answers among the writers on run 3) can be located:

  per piece   the first piece, the later pieces, and the end (the newline after the last piece:
              an answer that runs on past its last piece is wrong however its pieces read);
  the gate    the copy's mean lam at the asking tokens whose address holds a written piece (a
              WRITE at that place, at or before the asker, in the same document), against those
              whose address is empty (the asker after a name's last piece, a village's "nothing").
              Only the asking tokens the copy acted on count: one with nothing written yet keeps the
              trunk's logits and has no lam.

``train_base.fact_probe`` calls ``score`` on each ``CB_PROBE_SHARDS`` shard at every eval; ``line``
is its ``[probe]`` line. Its first token and exact are ``exp_hypertoken_recall.score``'s, the same
forwards (``tests/model/test_train_only.py`` holds them equal).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_hypertoken_recall as hr  # noqa: E402   (its import defaults CB_NO_FLEX=1: train_base.fact_probe undoes it)
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402


def _forward(model, ids, P, R):
    """``exp_hypertoken_recall.score``'s forward, as it makes it: places when there are some and
    the model has the link, the roles when the shard has them."""
    if P is None or model.spatial is None:
        return model.forward(ids[:, :-1])
    placed = P[..., 0] != NO_PLACE
    positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
    kw = {} if R is None else dict(writes=(placed & ((R & 1) != 0))[:, :-1], asks=(placed & ((R & 2) != 0))[:, :-1])
    return model.forward(ids[:, :-1], positions=positions[:, :-1], placed=placed[:, :-1], **kw)


def _written(places: np.ndarray, roles) -> list:
    """Per document: each written place -> the first position that writes it (without roles a
    placed token writes, as the model takes it)."""
    out = []
    for r in range(len(places)):
        w = places[r, :, 0] != NO_PLACE
        if roles is not None:
            w &= (roles[r] & 1) != 0
        first: dict = {}
        for i in np.flatnonzero(w):
            first.setdefault(tuple(places[r, i]), int(i))
        out.append(first)
    return out


@torch.no_grad()
def score(model, tk, x: np.ndarray, places, dev) -> dict:
    """``x`` (n, doc) ids; ``places`` (n, doc, 3), ``(places, roles)``, or None
    (``exp_hypertoken_recall.heldout_docs``)."""
    roles = None
    if isinstance(places, tuple):
        places, roles = places
    if hr.STRIP and places is not None:
        places = hr.strip_questions(tk, x, places)
    cp = model.spatial.copy_path if model.spatial is not None else None
    calls, lam = [], {"present": [], "empty": []}
    hook = cp.register_forward_hook(lambda _m, inp, out: calls.append((inp[4], out[2][:, 0]))) if cp is not None else None
    n = dict.fromkeys(("questions", "first", "exact", "later", "n_later", "end", "n_end"), 0)
    written = _written(places, roles) if places is not None and cp is not None else None
    try:
        for b in range(0, len(x), hr.BATCH):
            ids = torch.from_numpy(x[b:b + hr.BATCH]).to(dev)
            P = None if places is None else torch.from_numpy(places[b:b + hr.BATCH]).to(dev)
            R = None if roles is None else torch.from_numpy(roles[b:b + hr.BATCH]).to(dev)
            calls.clear()
            pred = _forward(model, ids, P, R).argmax(-1).cpu().numpy()
            S = ids.shape[1] - 1
            for ix, lv in calls:                                 # the asking positions the copy acted on (flat)
                for j, v in zip(ix.tolist(), lv.float().tolist()):
                    r, t = divmod(j, S)
                    at = written[b + r].get(tuple(places[b + r, t]))
                    lam["present" if at is not None and at <= t else "empty"].append(v)
            for r in range(ids.shape[0]):
                row = x[b + r]
                for j, k in hr.answers(tk, row):
                    ok = [pred[r, t - 1] == row[t] for t in range(j, k + 1)]
                    n["questions"] += 1
                    n["first"] += int(ok[0])
                    n["exact"] += int(all(ok))
                    n["later"] += sum(map(int, ok[1:]))
                    n["n_later"] += len(ok) - 1
                    if k + 1 < len(row):                         # the newline after the last piece
                        n["end"] += int(pred[r, k] == row[k + 1])
                        n["n_end"] += 1
    finally:
        if hook is not None:
            hook.remove()
    q = max(n["questions"], 1)
    out = {"questions": n["questions"], "first_token": n["first"] / q, "exact": n["exact"] / q,
           "pieces": n["questions"] + n["n_later"], "later": n["later"] / max(n["n_later"], 1), "n_later": n["n_later"],
           "end": n["end"] / max(n["n_end"], 1), "n_end": n["n_end"]}
    if cp is not None:
        out.update(asks=len(lam["present"]) + len(lam["empty"]),
                   **{f"lam_{c}": (float(np.mean(v)) if v else None) for c, v in lam.items()},
                   **{f"n_{c}": len(v) for c, v in lam.items()})
    return out


def line(pr: dict) -> str:
    """The ``[probe]`` line of a ``train_base.fact_probe`` result."""
    def lam(v):
        if not v.get("asks"):
            return ""
        f = {c: "-" if v[f"lam_{c}"] is None else f"{v[f'lam_{c}']:.3f}" for c in ("present", "empty")}
        return f"  lam here {f['present']} ({v['n_present']}) empty {f['empty']} ({v['n_empty']})"
    return "  ".join(f"{k}: first {v['first_token']:.3f} exact {v['exact']:.3f} later {v['later']:.3f} end {v['end']:.3f}"
                     + lam(v) for k, v in pr.items() if not k.startswith("_")) + (
        f"  lam {pr['_lam']:.3f}" if pr.get("_lam") is not None else "")
