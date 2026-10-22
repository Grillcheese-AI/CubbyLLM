"""k10_free_docs -- the free-text toy's documents (order 0006): ``k10_far_docs``'s facts, asked in
free text.

A document: ``n_facts`` facts ``s r o_0 .. o_k-1 EOS`` (each piece a WRITE), filler, then one
question -- ``Q`` and one of 8 templates (subject and relation in either order, among function words
taken from the filler language, as real function words are) -- then ``HERE`` and the answer,
``o_0 .. o_k-1 EOS``, or ``NONE EOS`` when the document does not hold the fact. ``HERE`` asks for
piece 0 and every answer token for the next (``make_fact_shard --lead``'s layout). Each writer and
asker carries its address two ways: the text key's ``(s, r, piece)`` (``model/spatial_key.py``) and
today's hashed place ``cell(s) + step(r) + 6 i z``; each asker's gold pointer offsets (back to its
question's subject and relation tokens) come apart, for the pointer loss.

Templates 0-5 train, 6-7 are held out. 20% of the entities never appear in a training document
(``FreeLayout.unseen``); a held-out document may ask about one, its fact written in that document.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_k10_hypertoken as kh  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from k10_far_docs import ASK, WRITE, Z, Layout  # noqa: E402

FUNC = {w: kh.FILL0 + i for i, w in enumerate(("what", "is", "the", "of", "'s", "tell", "me", "about"))}
TEMPLATES = [["S", "R"], ["R", "S"],                                   # 0-1: today's, and reversed
             ["what", "is", "the", "R", "of", "S"], ["what", "is", "S", "'s", "R"],
             ["tell", "me", "the", "R", "of", "S"], ["tell", "me", "S", "'s", "R"],
             ["about", "S", "what", "is", "the", "R"], ["the", "R", "of", "S", "is", "what"]]   # 6-7: held out
TRAIN_T, HELD_T = range(6), range(6, 8)


class FreeLayout(Layout):
    """``Layout`` with the last id kept for NONE and the entities split: 20% never in training."""

    def __init__(self, vocab: int, entities: int = 0, seed: int = 4242) -> None:
        super().__init__(vocab, entities)
        self.n_ent -= 1                                              # ent(n_ent) would be vocab - 1: NONE
        self.none = vocab - 1
        perm = np.random.default_rng(seed).permutation(self.n_ent)
        self.unseen, self.seen = np.sort(perm[: self.n_ent // 5]), np.sort(perm[self.n_ent // 5:])


def document(rng, lang, lay: FreeLayout, n_facts: int, filler: int, seq: int, pieces: int, template: int,
             unseen: bool = False, absent: bool = False):
    """ids (seq+1,), text keys and places (seq+1, 3) int16 each, roles (seq+1,), gold (seq+1, 2) int16
    (an asker's offsets back to its subject / relation, -1 elsewhere), and the question's span
    ``(HERE index, answer tokens, {template, unseen, absent})``."""
    subj = rng.choice(lay.seen, n_facts, replace=False)
    if unseen:
        subj[0] = rng.choice(lay.unseen)                             # the asked fact's subject: never in training
    facts, ids, key, plc, rol = [], [], [], [], []
    for i in range(n_facts):
        s, r = lay.ent(subj[i]), int(rng.integers(kh.N_REL))
        o = [lay.ent(e) for e in rng.choice(lay.seen, int(rng.integers(1, pieces + 1)))]
        addr = kh.cell(s) + kh.step(r)
        facts.append((s, r, o))
        ids += [s, kh.REL0 + r] + o + [kh.EOS]
        key += [None, None] + [(s, kh.REL0 + r, j) for j in range(len(o))] + [None]
        plc += [None, None] + [addr + Z * j for j in range(len(o))] + [None]
        rol += [0, 0] + [WRITE] * len(o) + [0]
    fill = lang.tokens(filler)
    ids, key, plc, rol = ids + fill, key + [None] * len(fill), plc + [None] * len(fill), rol + [0] * len(fill)
    s, r, o = facts[0]
    if absent:                                                       # a fact the document does not hold
        if rng.random() < 0.5:
            r = (r + 1 + int(rng.integers(kh.N_REL - 1))) % kh.N_REL     # its subject, another relation
        else:
            pool = np.setdiff1d(lay.unseen if unseen else lay.seen, subj)
            s = lay.ent(rng.choice(pool))                                # a subject it never mentions
        o = [lay.none]
    words = [{"S": s, "R": kh.REL0 + r}.get(w, FUNC.get(w)) for w in TEMPLATES[template]]
    q0 = len(ids) + 1                                                # the template's first token
    here = q0 + len(words)
    ps, pr = q0 + TEMPLATES[template].index("S"), q0 + TEMPLATES[template].index("R")
    ids += [kh.Q] + words + [kh.HERE] + o + [kh.EOS]
    addr = kh.cell(s) + kh.step(r)
    key += [None] * (len(words) + 1) + [(s, kh.REL0 + r, j) for j in range(len(o) + 1)] + [None]
    plc += [None] * (len(words) + 1) + [addr + Z * j for j in range(len(o) + 1)] + [None]
    rol += [0] * (len(words) + 1) + [ASK] * (len(o) + 1) + [0]
    gold = [None] * (here) + [(here + j - ps, here + j - pr) for j in range(len(o) + 1)] + [None]
    rest = lang.tokens(seq + 1 - len(ids))
    n = seq + 1
    ids, rol = (ids + rest)[:n], (rol + [0] * len(rest))[:n]
    out = []
    for col, width, empty in ((key, 3, NO_PLACE), (plc, 3, NO_PLACE), (gold, 2, -1)):
        A = np.full((n, width), empty, dtype=np.int16)
        for i, v in enumerate((col + [None] * len(rest))[:n]):
            if v is not None:
                A[i] = v
        out.append(A)
    info = {"template": template, "unseen": bool(unseen), "absent": bool(absent)}
    return np.array(ids, dtype=np.int64), out[0], out[1], np.array(rol, dtype=np.uint8), out[2], (here, o, info)


def batch(rng, lang, lay, n: int, a, cells=None):
    """``n`` documents: training ones (a train template, a seen subject, absent with ``a.absent``), or the
    held-out ``cells`` -- one ``(template, unseen, absent)`` per document."""
    docs = []
    for i in range(n):
        t, u, ab = cells[i] if cells is not None else (int(rng.choice(TRAIN_T)), False, bool(rng.random() < a.absent))
        docs.append(document(rng, lang, lay, a.facts_per_doc, a.gap, a.seq, a.pieces, t, u, ab))
    stack = [torch.from_numpy(np.stack([d[k] for d in docs])).long() for k in range(5)]
    return (*stack, [d[5] for d in docs])


def held_cells(rng, n: int, absent: float) -> list:
    """Half the held-out questions in a held-out template, half about an unseen subject, independently."""
    return [(int(rng.choice(HELD_T if rng.random() < 0.5 else TRAIN_T)), bool(rng.random() < 0.5),
             bool(rng.random() < absent)) for _ in range(n)]
