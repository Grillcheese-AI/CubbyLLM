"""k10_far_docs -- the far-facts toy's documents (order 0001): ``exp_k10_hypertoken``'s, generalized
to a large vocabulary, multi-piece objects, roles and fewer questions, and at its settings its
documents exactly (``selfcheck``, which ``exp_k10_far_facts`` runs at every start).

A document: ``n_facts`` facts ``s r o_0 .. o_k-1 EOS`` with piece i of the object at the fact's
address + 6 i z (a WRITE with roles), ``filler`` tokens of the toy's language, then ``n_ask`` of
the facts asked ``Q s r HERE o_0 .. o_k-1 EOS``: HERE asks for piece 0 and, with roles, piece j asks
for j + 1 (``make_fact_shard --lead``'s layout), the last for one that is not there (it predicts
EOS); without roles HERE is placed (and writes, as in the toy) and the answer is not.
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

WRITE, ASK = 1, 2
Z = np.array([0, 0, kh.STEP], dtype=np.int64)          # piece i of a name sits 6 i cells up z


class Layout:
    """Entity index -> token id: the toy's 509 (3..511), then 1,024..vocab-1."""

    def __init__(self, vocab: int, entities: int = 0) -> None:
        self.vocab = vocab
        self.n_ent = entities or kh.N_ENT + vocab - 1024

    def ent(self, i) -> int:
        i = int(i)
        return kh.ENT0 + i if i < kh.N_ENT else 1024 + i - kh.N_ENT


def document(rng, lang, lay: Layout, n_facts: int, filler: int, seq: int, n_ask: int, pieces: int, roles: bool):
    """ids (seq+1,), places (seq+1, 3) int16, roles (seq+1,) uint8, spans [(HERE index, [pieces])]."""
    ents = rng.choice(lay.n_ent, 2 * n_facts, replace=False)
    ids, pos, rol, facts = [], [], [], []
    for i in range(n_facts):
        s, o = lay.ent(ents[i]), [lay.ent(ents[n_facts + i])]
        r = int(rng.integers(kh.N_REL))
        if pieces > 1:
            o += [lay.ent(e) for e in rng.choice(lay.n_ent, int(rng.integers(0, pieces)))]
        addr = kh.cell(s) + kh.step(r)
        facts.append((s, r, o, addr))
        ids += [s, kh.REL0 + r] + o + [kh.EOS]
        pos += [None, None] + [addr + Z * j for j in range(len(o))] + [None]
        rol += [0, 0] + [WRITE] * len(o) + [0]
    fill = lang.tokens(filler)
    ids, pos, rol = ids + fill, pos + [None] * len(fill), rol + [0] * len(fill)
    spans = []
    for s, r, o, addr in [facts[i] for i in rng.permutation(n_facts)][:n_ask]:
        q = [kh.Q, s, kh.REL0 + r, kh.HERE] + o + [kh.EOS]
        if len(ids) + len(q) > seq + 1:
            break
        spans.append((len(ids) + 3, o))
        ids += q
        if roles:                                       # HERE asks piece 0, piece j asks j + 1
            pos += [None] * 3 + [addr + Z * j for j in range(len(o) + 1)] + [None]
            rol += [0] * 3 + [ASK] * (len(o) + 1) + [0]
        else:                                           # the toy: HERE placed (and writing), the answer not
            pos += [None, None, None, addr] + [None] * (len(o) + 1)
            rol += [0] * (len(o) + 5)
    rest = lang.tokens(seq + 1 - len(ids))
    ids, pos, rol = (ids + rest)[:seq + 1], (pos + [None] * len(rest))[:seq + 1], (rol + [0] * len(rest))[:seq + 1]
    P = np.full((seq + 1, 3), NO_PLACE, dtype=np.int16)
    for i, p in enumerate(pos):
        if p is not None:
            P[i] = p
    return np.array(ids, dtype=np.int64), P, np.array(rol, dtype=np.uint8), spans


def batch(rng, lang, lay, n, a):
    docs = [document(rng, lang, lay, a.facts_per_doc, a.gap, a.seq, a.questions, a.pieces, a.roles) for _ in range(n)]
    return (torch.from_numpy(np.stack([d[0] for d in docs])), torch.from_numpy(np.stack([d[1] for d in docs])),
            torch.from_numpy(np.stack([d[2] for d in docs])) if a.roles else None, [d[3] for d in docs])


def selfcheck() -> None:
    """At the toy's settings the documents are the toy's, token for token, place for place."""
    lay = Layout(1024)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    l1, l2 = kh.Language(np.random.default_rng(6)), kh.Language(np.random.default_rng(6))
    for _ in range(4):
        ids, P, ask_at, answers = kh.document(r1, l1, 8, 120, 256)
        ids2, P2, _, spans = document(r2, l2, lay, 8, 120, 256, 8, 1, False)
        assert np.array_equal(ids, ids2) and np.array_equal(P, P2), "the generalized document is not the toy's"
        assert ask_at == [h for h, _ in spans] and answers == [o[0] for _, o in spans]
