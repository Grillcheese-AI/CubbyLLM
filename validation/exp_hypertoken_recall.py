"""exp_hypertoken_recall -- the questions of a loci-token shard answered by a checkpoint: the
spatial link's and the fact link's gate on a trained trunk (H-S12 gate 1, H-K10 gate 4, H-E20
gate 2 on ``facts_text_far``).

Reads the held-out tail of a shard (``_holdout.json``, documents of ``_align.json`` tokens), runs
the trunk over each document -- with the ``.pos.i16`` places when the shard has them and the
checkpoint has the link, plain otherwise -- and scores every ``Answer:`` teacher-forced: the
first answer token and the whole answer (argmax at each piece). The same probe on the text
control says what attention alone does on the same questions; on ``facts_text_far`` (the
questions ~1,500 tokens past the facts, beyond the 512 window) it is the episodic memory's test.
``CB_STRIP=1`` removes the places from everything after the first ``Question:`` -- the facts stay
bound, the questions carry no address -- so the trunk's own query (the learned route, taught by
``CB_ROUTE_LOSS``) has to find it: the route's number, apart from the read's.

    CB_CKPT=<ckpt.pt> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json> [CB_SHARDS=facts,facts_text]
    [CB_DOCS=100] [CB_BATCH=8] [CB_STRIP=0] [CB_OUT=<json>] python validation/exp_hypertoken_recall.py
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402

CKPT, CORPUS, SPM = os.environ.get("CB_CKPT", ""), os.environ.get("CB_CORPUS", ""), os.environ.get("CUBBY_SPM", "")
SHARDS = os.environ.get("CB_SHARDS", "facts,facts_text").split(",")
DOCS, BATCH, OUT = int(os.environ.get("CB_DOCS", "100")), int(os.environ.get("CB_BATCH", "8")), os.environ.get("CB_OUT", "")
STRIP = os.environ.get("CB_STRIP", "0") == "1"
os.environ.setdefault("CB_NO_FLEX", "1")


def heldout_docs(name: str, n: int):
    """The last ``n`` held-out documents of the shard: ids (n, doc) and places (n, doc, 3) or
    None -- with the roles (n, doc) attached as ``places.roles`` when the shard has them."""
    doc = json.load(open(os.path.join(CORPUS, "_align.json")))[name]
    hold = json.load(open(os.path.join(CORPUS, "_holdout.json")))[name]
    ids = np.memmap(os.path.join(CORPUS, f"{name}.u32"), dtype=np.uint32, mode="r")
    n = min(n, hold // doc)
    start = len(ids) - n * doc
    x = np.array(ids[start:], dtype=np.int64).reshape(n, doc)
    pp = os.path.join(CORPUS, f"{name}.pos.i16")
    places = None
    if os.path.exists(pp):
        pos = np.memmap(pp, dtype=np.int16, mode="r").reshape(-1, 3)
        places = np.array(pos[start:], dtype=np.int64).reshape(n, doc, 3)
        rp = os.path.join(CORPUS, f"{name}.role.u8")
        if os.path.exists(rp):                                   # WRITE / ASK: a question reads, it does not write
            roles = np.array(np.memmap(rp, dtype=np.uint8, mode="r")[start:], dtype=np.int64).reshape(n, doc)
            places = (places, roles)
    return x, places


def answers(tk, row: np.ndarray):
    """``(first, last)`` token spans of every answer in a document: after ``Answer:``, past the
    whitespace, up to the newline."""
    tag, nl = tk.encode("Answer:", add_special_tokens=False).ids, tk.encode("\n", add_special_tokens=False).ids[-1]
    out, i, n = [], 0, len(row)
    while i < n - len(tag):
        if list(row[i:i + len(tag)]) == tag:
            j = i + len(tag)
            while j < n and tk.decode([int(row[j])]).strip() == "" and row[j] != nl:
                j += 1
            k = j
            while k < n and row[k] != nl:
                k += 1
            if k > j:
                out.append((j, k - 1))
            i = k
        else:
            i += 1
    return out


def strip_questions(tk, x: np.ndarray, places: np.ndarray) -> np.ndarray:
    """The places of everything from the first ``Question:`` on set to NO_PLACE: the questions
    carry no address, the trunk's own query must find it."""
    tag = tk.encode("Question:", add_special_tokens=False).ids
    out = places.copy()
    for r in range(len(x)):
        row = list(x[r])
        for i in range(len(row) - len(tag)):
            if row[i:i + len(tag)] == tag:
                out[r, i:] = NO_PLACE
                break
    return out


@torch.no_grad()
def score(model, tk, x: np.ndarray, places, spatial: bool, dev) -> dict:
    first = exact = n = 0
    roles = None
    if isinstance(places, tuple):
        places, roles = places
    if STRIP and places is not None:
        places = strip_questions(tk, x, places)
    for b in range(0, len(x), BATCH):
        ids = torch.from_numpy(x[b:b + BATCH]).to(dev)
        if spatial and places is not None:
            P = torch.from_numpy(places[b:b + BATCH]).to(dev)
            placed = P[..., 0] != NO_PLACE
            positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
            kw = {}
            if roles is not None:                                # each row is one document: no restarts needed
                R = torch.from_numpy(roles[b:b + BATCH]).to(dev)
                kw = dict(writes=(placed & ((R & 1) != 0))[:, :-1], asks=(placed & ((R & 2) != 0))[:, :-1])
            logits = model.forward(ids[:, :-1], positions=positions[:, :-1], placed=placed[:, :-1], **kw)
        else:
            logits = model.forward(ids[:, :-1])
        pred = logits.argmax(-1).cpu().numpy()
        for r in range(ids.shape[0]):
            row = x[b + r]
            for j, k in answers(tk, row):
                ok = [pred[r, t - 1] == row[t] for t in range(j, k + 1)]
                first += int(ok[0])
                exact += int(all(ok))
                n += 1
    return {"questions": n, "first_token": first / max(n, 1), "exact": exact / max(n, 1)}


def main() -> None:
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_base import build_model
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = dict(ck["meta"])
    model = build_model(meta, dev)
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    del ck
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(SPM)
    spatial = model.spatial is not None
    print(f"loci-token recall -- {os.path.basename(CKPT)}: d={meta['D']} L={meta['L']} window={meta['window']} "
          f"mem_every={meta.get('mem_every', 0)} spatial={meta.get('spatial')} | {dev}; {DOCS} held-out documents per shard")
    report = {"ckpt": CKPT, "meta": meta, "docs": DOCS, "shards": {}}
    for name in SHARDS:
        x, places = heldout_docs(name, DOCS)
        res = score(model, tk, x, places, spatial, dev)
        res["placed"] = bool(spatial and places is not None)
        res["roles"] = isinstance(places, tuple)
        res["questions_stripped"] = bool(STRIP and res["placed"])
        report["shards"][name] = res
        print(f"  {name:18s} {res['questions']:5d} questions   first token {res['first_token']:.3f}   exact {res['exact']:.3f}"
              f"   ({'places read, questions stripped: the route alone' if res['questions_stripped'] else 'places read' if res['placed'] else 'text only'})"
              f"   [{time.time() - t0:.0f}s]", flush=True)
    report["wall_s"] = round(time.time() - t0, 1)
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
