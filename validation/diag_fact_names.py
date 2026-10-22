"""diag_fact_names -- a fact shard rebuilt with readable names (order 0011, ``make_fact_shard --names
v2``) against the shard it replaces: the same facts at the same addresses and the same questions in
every document, and what the words became.

From the shards' bytes alone: per document, the set of addresses written (a fact's ``(x, y)``, the
subject's cell plus the relation's step) and asked (each question's), the questions' count, the
pieces an object's name takes; decoded, every question's doubled words (``of of``) and every answer.
The filler between the facts and the questions is real text drawn at the old offsets' rng, so it may
differ where a name's length moved the gap: not compared.

    CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json> [CB_OLD=facts_far1k_lead] [CB_NEW=facts_far1k_v2]
    [CB_OUT=<json>] python validation/diag_fact_names.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from make_fact_shard import doubled  # noqa: E402

CORPUS, SPM = os.environ.get("CB_CORPUS", ""), os.environ.get("CUBBY_SPM", "")
OLD, NEW = os.environ.get("CB_OLD", "facts_far1k_lead"), os.environ.get("CB_NEW", "facts_far1k_v2")
OUT = os.environ.get("CB_OUT", "") or os.path.join(HERE, "logs", f"diag_fact_names_{NEW}_{_dt.date.today().isoformat()}.json")
WRITE, ASK = 1, 2


def shard(name: str):
    doc = json.load(open(os.path.join(CORPUS, "_align.json")))[name]
    ids = np.memmap(os.path.join(CORPUS, f"{name}.u32"), dtype=np.uint32, mode="r").reshape(-1, doc)
    pos = np.memmap(os.path.join(CORPUS, f"{name}.pos.i16"), dtype=np.int16, mode="r").reshape(len(ids), doc, 3)
    roles = np.memmap(os.path.join(CORPUS, f"{name}.role.u8"), dtype=np.uint8, mode="r").reshape(len(ids), doc)
    return ids, pos, roles


def cells(pos: np.ndarray, roles: np.ndarray, role: int) -> set:
    return {(int(x), int(y)) for x, y in pos[roles == role][:, :2]}


def qa(tk, row: np.ndarray) -> list:
    """``(question, answer)`` for every question of a decoded document."""
    text = tk.decode([int(t) for t in row], skip_special_tokens=True)
    return re.findall(r"Question: (.*?) here\nAnswer: (.*)", text)


def main() -> None:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(SPM)
    (oi, op, orl), (ni, npo, nrl) = shard(OLD), shard(NEW)
    assert oi.shape == ni.shape, (oi.shape, ni.shape)
    n = len(oi)
    same_w = same_a = same_q = 0
    pieces_old, pieces_new = int((orl == WRITE).sum()), int((nrl == WRITE).sum())
    dbl = {"old": 0, "new": 0}
    of_of = {"old": 0, "new": 0}
    n_q, ex_new, ex_old = 0, [], []
    for d in range(n):
        o_r, n_r = np.asarray(orl[d]), np.asarray(nrl[d])
        same_w += cells(np.asarray(op[d]), o_r, WRITE) == cells(np.asarray(npo[d]), n_r, WRITE)
        same_a += cells(np.asarray(op[d]), o_r, ASK) == cells(np.asarray(npo[d]), n_r, ASK)
        qo, qn = qa(tk, oi[d]), qa(tk, ni[d])
        same_q += len(qo) == len(qn)
        n_q += len(qn)
        for key, qs in (("old", qo), ("new", qn)):
            for q, _a in qs:
                dbl[key] += bool(doubled(q))
                of_of[key] += " of of " in f" {q.casefold()} "
        if d >= n - 3:                                           # the last held-out documents, as the probe reads them
            ex_old += qo[:3]
            ex_new += qn[:3]
    rep = {"old": OLD, "new": NEW, "documents": n, "questions_new": n_q,
           "same_written_addresses": same_w, "same_asked_addresses": same_a, "same_question_count": same_q,
           "written_pieces": {"old": pieces_old, "new": pieces_new},
           "questions_with_any_doubled_word": dbl, "questions_with_of_of": of_of,
           "examples_old": [list(x) for x in ex_old], "examples_new": [list(x) for x in ex_new]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{n:,} documents: the same written addresses in {same_w:,}, asked in {same_a:,}, questions in {same_q:,}; "
          f"written pieces {pieces_old:,} -> {pieces_new:,}; questions with 'of of' {of_of['old']:,} -> {of_of['new']:,}, "
          f"with any doubled word {dbl['old']:,} -> {dbl['new']:,}")
    for (q0, a0), (q1, a1) in zip(ex_old, ex_new):
        print(f"  old: {q0} -> {a0}\n  new: {q1} -> {a1}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
