"""Order 0050, stage 1: the shortlist index over a head's rows (``cubbyllm/model/vocab/shortlist.py``). Pinned: K equal
to the vocabulary is the full head; the index builds deterministically from a seed; the flag off leaves the head's
output bit-identical; the shortlist scores its rows exactly."""
from __future__ import annotations

import numpy as np
import pytest

from cubbyllm.model.vocab.shortlist import ShortlistIndex, kmeans, successors

torch = pytest.importorskip("torch")


def _head(v=500, d=16, seed=0):
    from cubbyllm.model.vocab import TopKRetrievalHead
    g = torch.Generator().manual_seed(seed)
    return TopKRetrievalHead(torch.randn(v, d, generator=g), learnable=True)


def test_the_index_builds_the_same_from_the_same_seed():
    rows = np.random.default_rng(1).standard_normal((800, 8)).astype(np.float32)
    c1, a1 = kmeans(rows, 32, seed=7)
    c2, a2 = kmeans(rows, 32, seed=7)
    assert np.array_equal(a1, a2) and np.array_equal(c1, c2)
    assert np.bincount(a1, minlength=32).min() >= 1                      # no empty cluster
    assert not np.array_equal(kmeans(rows, 32, seed=8)[1], a1)


def test_k_equal_to_the_vocabulary_is_the_full_head(monkeypatch):
    head = _head()
    idx = ShortlistIndex.build(head._codes().detach().numpy(), 20, seed=0)
    q = torch.randn(3, 2, 16)
    full = head.logits(q, head.vocab)
    head.attach_shortlist(idx, k=head.vocab)
    monkeypatch.setenv("CB_HEAD_SHORTLIST", "1")
    assert torch.allclose(head.logits(q, head.vocab), full, atol=1e-5)


def test_the_flag_off_leaves_the_head_bit_identical(monkeypatch):
    head = _head()
    q = torch.randn(2, 3, 16)
    before = head.logits(q, head.vocab)
    head.attach_shortlist(ShortlistIndex.build(head._codes().detach().numpy(), 20, seed=0), k=64)
    monkeypatch.delenv("CB_HEAD_SHORTLIST", raising=False)
    assert torch.equal(head.logits(q, head.vocab), before)


def test_a_shortlist_scores_its_rows_exactly_and_masks_the_rest(monkeypatch):
    head = _head()
    idx = ShortlistIndex.build(head._codes().detach().numpy(), 25, seed=0)
    head.attach_shortlist(idx, k=100)
    monkeypatch.setenv("CB_HEAD_SHORTLIST", "1")
    q = torch.randn(1, 1, 16)
    out = head.logits(q, head.vocab)[0, 0]
    full = head.logits.__func__(type("H", (), {"shortlist": None, "learnable": True, "temperature": 1.0,
                                               "_codes": lambda s: head._codes(), "vocab": head.vocab})(), q, head.vocab)[0, 0]
    kept = torch.isfinite(out)
    ids = idx.shortlist(q.reshape(1, -1).numpy(), 100)[0]
    assert kept.sum().item() == len(ids) >= 100
    assert torch.allclose(out[kept], full[kept], atol=1e-5)


def test_whole_clusters_are_taken_in_score_order_and_successors_join_arm_c():
    rows = np.random.default_rng(2).standard_normal((300, 8)).astype(np.float32)
    idx = ShortlistIndex.build(rows, 10, seed=0)
    h = rows[:1]
    ids = idx.shortlist(h, 50)[0]
    best = np.argsort(-(h @ idx.centroids.T)[0], kind="stable")
    assert set(idx.assign[ids]) == set(best[:len(set(idx.assign[ids]))])   # a prefix of the cluster ranking, whole
    toks = np.array([1, 2, 1, 2, 1, 3, 4, 1, 2])
    succ = successors(toks, 300, top=2)
    assert list(succ[1]) == [2, 3] and list(succ[4]) == [1, -1] and succ[0, 0] == -1
    idx.router, idx.succ = (idx.centroids, np.zeros(10, dtype=np.float32)), succ
    with_succ = idx.shortlist(h, 50, "router+succ", prev=np.array([1]))[0]
    assert {2, 3} <= set(with_succ) and set(ids) <= set(with_succ)


def test_the_screen_arm_lists_a_context_clusters_tokens_by_their_mass():
    from cubbyllm.model.vocab.shortlist import ScreenIndex
    rng = np.random.default_rng(3)
    hid = np.concatenate([rng.normal(5, 0.1, (40, 4)), rng.normal(-5, 0.1, (40, 4))]).astype(np.float32)
    top = np.concatenate([np.tile([7, 8], (40, 1)), np.tile([9, 7], (40, 1))])
    w = np.concatenate([np.tile([0.9, 0.1], (40, 1)), np.tile([0.6, 0.3], (40, 1))])
    sc = ScreenIndex.build(hid, top, w, vocab=20, n=2, seed=0)
    near_a = sc.shortlist(hid[:1], 1)[0]
    assert list(near_a) == [7]                                   # the cluster's heaviest token
    assert set(sc.shortlist(hid[:1], 3)[0]) == {7, 8, 9}         # k distinct: the next cluster fills the rest
    assert ScreenIndex.build(hid, top, w, 20, 2, seed=0).ids.tolist() == sc.ids.tolist()
