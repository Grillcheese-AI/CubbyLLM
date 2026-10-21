"""diag_fact_read -- why a fact link answers its facts' questions or not, piece by piece.

``exp_hypertoken_recall`` scores the first answer token and the whole answer. On the far facts
(run 2 of the spatial notebook, 2026-10-20) the link read 0.005 exact, first token 0.22: this
splits that number. In ``make_fact_shard``'s documents the address rides on ``here``, three
tokens before the answer (``here \\n Answer :``), so the first piece needs the read carried
forward by the trunk, while every later piece is asked for by the piece before it (its own
token carries the next piece's address). Three reads of the same held-out documents:

  trained : places and roles as the shard has them;
  moved   : the address ALSO on the token right before the answer (zero-shot: never trained so),
            which then asks for the first piece itself;
  none    : no places at all -- the trunk's prior on the same answers;
  unbound : places and roles as trained, but only the writing tokens bound at the input
            (``SpatialLink(bind="writes")``, zero-shot on a link trained with every placed
            token bound): the answer pieces enter as themselves.

Per read: the first piece, the later pieces (token accuracy, teacher-forced) and the whole answer.
``later`` trained >> none says the read works and only the hand-off to the first piece fails;
``first`` moved >> trained says the fix is in the shard (the token before the answer asks).

    CB_CKPT=<link final> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json> [CB_SHARDS=facts_far1k]
    [CB_DOCS=24] [CB_BATCH=4] [CB_OUT=<json>] python validation/diag_fact_read.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CB_SHARDS", "facts_far1k")
os.environ.setdefault("CB_DOCS", "24")
os.environ.setdefault("CB_BATCH", "4")
import exp_hypertoken_recall as hr  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402

ASK = 2


def moved(x: np.ndarray, places: np.ndarray, roles: np.ndarray, spans: list):
    """The address of each question's ``here`` copied onto the token right before its answer."""
    P, R = places.copy(), roles.copy()
    for r, row_spans in enumerate(spans):
        for j, _k in row_spans:
            h = j - 1
            while h > 0 and not (P[r, h, 0] != NO_PLACE and (R[r, h] & ASK)):
                h -= 1
            if h > 0 and h != j - 1:
                P[r, j - 1], R[r, j - 1] = P[r, h], ASK
    return P, R


@torch.no_grad()
def tally(model, x, places, roles, spans, mode: str) -> dict:
    first = later = n_later = exact = n = 0
    for b in range(0, len(x), hr.BATCH):
        ids = torch.from_numpy(x[b:b + hr.BATCH])
        if mode == "none":
            logits = model.forward(ids[:, :-1])
        else:
            P = torch.from_numpy(places[b:b + hr.BATCH])
            Rr = torch.from_numpy(roles[b:b + hr.BATCH])
            placed = P[..., 0] != NO_PLACE
            positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
            logits = model.forward(ids[:, :-1], positions=positions[:, :-1], placed=placed[:, :-1],
                                   writes=(placed & ((Rr & 1) != 0))[:, :-1], asks=(placed & ((Rr & 2) != 0))[:, :-1])
        pred = logits.argmax(-1).numpy()
        for r in range(ids.shape[0]):
            row = x[b + r]
            for j, k in spans[b + r]:
                ok = [pred[r, t - 1] == row[t] for t in range(j, k + 1)]
                first += int(ok[0]); exact += int(all(ok)); n += 1
                later += sum(ok[1:]); n_later += len(ok) - 1
    return {"questions": n, "first": first / max(n, 1), "later": later / max(n_later, 1),
            "later_tokens": n_later, "exact": exact / max(n, 1)}


def main() -> None:
    t0 = time.time()
    from train_base import build_model
    from tokenizers import Tokenizer
    ck = torch.load(hr.CKPT, map_location="cpu", weights_only=False)
    model = build_model(dict(ck["meta"]), torch.device("cpu"))
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.float())
    del ck
    tk = Tokenizer.from_file(hr.SPM)
    report = {"ckpt": hr.CKPT, "docs": hr.DOCS, "shards": {}}
    for name in hr.SHARDS:
        x, pr = hr.heldout_docs(name, hr.DOCS)
        places, roles = pr
        spans = [hr.answers(tk, row) for row in x]
        mp, mr = moved(x, places, roles, spans)
        res = {}
        modes = (("trained", (places, roles)), ("moved", (mp, mr)), ("none", (places, roles)), ("unbound", (places, roles)))
        for mode, (P, R) in modes:
            if mode in os.environ.get("CB_SKIP", "").split(","):
                continue
            model.spatial.bind_mode = "writes" if mode == "unbound" else model.spatial.bind_mode
            res[mode] = tally(model, x, P, R, spans, mode)
            v = res[mode]
            print(f"  {name} {mode:8s} {v['questions']} questions  first {v['first']:.3f}  later {v['later']:.3f} "
                  f"({v['later_tokens']} tokens)  exact {v['exact']:.3f}   [{time.time() - t0:.0f}s]", flush=True)
        report["shards"][name] = res
    if hr.OUT:
        json.dump(report, open(hr.OUT, "w", encoding="utf-8"), indent=1)
        print(f"  wrote {hr.OUT}")


if __name__ == "__main__":
    main()
