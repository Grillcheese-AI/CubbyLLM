"""Order 0068, route 5 wired: the Hamming pass over a head's rows (``cubbyllm/model/vocab/hamming.py``) and the "+ham"
arms of the shortlist. Pinned: the pass's score is the row's norm times (the signs that agree minus those that do
not); its top N is the brute-force top N, ties to the lower id; a "+ham" arm is its base arm's list joined with that
top N; the union's rows are scored exactly; the flag off leaves the head bit-identical; N equal to the vocabulary is
the full head; a "+ham" arm with no Hamming index refuses."""
from __future__ import annotations

import numpy as np
import pytest

from cubbyllm.model.vocab.hamming import HammingIndex, pack, popcount
from cubbyllm.model.vocab.shortlist import ARMS, HAM, ScreenIndex, ShortlistIndex, successors


def _rows(v=300, d=64, seed=0):
    return np.random.default_rng(seed).standard_normal((v, d)).astype(np.float32)


def _index(rows, seed=5):
    """A small index with clusters, a screen over made-up states and successors."""
    rng = np.random.default_rng(seed)
    v, d = rows.shape
    idx = ShortlistIndex.build(rows, 12, seed=0)
    hid = rng.standard_normal((400, d)).astype(np.float32)
    full = hid @ rows.T
    top = np.argsort(-full, axis=1)[:, :4]
    wt = np.take_along_axis(full, top, 1)
    idx.screen = ScreenIndex.build(hid, top, np.exp(wt - wt.max(1, keepdims=True)), vocab=v, n=8, seed=0)
    idx.succ = successors(rng.integers(0, v, 3000), v, top=4)
    return idx


def _head(rows):
    torch = pytest.importorskip("torch")
    from cubbyllm.model.vocab import TopKRetrievalHead
    return torch, TopKRetrievalHead(torch.from_numpy(rows.copy()), learnable=True)


def test_the_score_is_the_norm_times_the_signs_that_agree_minus_those_that_do_not():
    rows, h = _rows(300, 70), _rows(5, 70, seed=1)                       # d = 70: the last word is padded
    idx = HammingIndex.build(rows)
    assert idx.bits.shape == (300, 3) and idx.bits.dtype == np.uint32
    want = (np.sign(h) @ np.sign(rows).T) * np.linalg.norm(rows, axis=1)
    assert np.allclose(idx.scores(h), want, rtol=1e-5, atol=1e-4)


def test_a_zero_counts_as_negative_on_both_sides():
    rows = np.array([[1.0, 0.0, -2.0, 3.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert pack(rows).tolist() == [[0b1001], [0]]
    h = np.array([[0.0, -1.0, -1.0, 5.0]], dtype=np.float32)             # bits 1000: one bit from row 0
    assert np.allclose(HammingIndex.build(rows).scores(h), [[np.linalg.norm(rows[0]) * (4 - 2 * 1), 0.0]])


def test_the_top_n_is_the_brute_force_top_n_and_ties_go_to_the_lower_id():
    rows = _rows(200, 64, seed=2)
    rows[150:160] = rows[40]                                             # ten copies of one row: ties
    idx, h = HammingIndex.build(rows), _rows(6, 64, seed=3)
    s = idx.scores(h)
    for n in (1, 7, 45, 199, 200, 500):
        for i, ids in enumerate(idx.top(h, n)):
            assert np.array_equal(ids, np.sort(np.argsort(-s[i], kind="stable")[:n]))
    assert idx.top(rows[40:41], 5)[0].tolist() == [40, 150, 151, 152, 153]   # eleven rows tie for first
    assert idx.top(rows[17:18], 1)[0].tolist() == [17]                   # a state along a row finds that row
    assert all(len(x) == 0 for x in idx.top(h, 0))


def test_the_byte_table_counts_what_numpy_2_counts(monkeypatch):
    x = np.random.default_rng(4).integers(0, 2 ** 32, (50, 3), dtype=np.uint32)
    want = np.array([[bin(int(v)).count("1") for v in r] for r in x])
    assert np.array_equal(popcount(x), want)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert np.array_equal(popcount(x), want)


def test_a_ham_arm_is_its_base_arms_list_joined_with_the_top_n():
    rows, h, prev = _rows(), _rows(7, 64, seed=6), np.arange(7)
    idx = _index(rows).attach_hamming(rows, n=20)
    for base in ("clusters", "screen", "screen+succ"):
        plain, both = idx.shortlist(h, 30, base, prev), idx.shortlist(h, 30, base + HAM, prev)
        for a, t, u in zip(plain, idx.ham.top(h, 20), both):
            assert np.array_equal(u, np.union1d(a, t)) and len(t) == 20 and len(u) <= len(a) + 20
    assert all(a + HAM not in ARMS for a in ARMS)                        # 0050's five arms stay as they were


def test_a_ham_arm_with_no_hamming_index_refuses():
    rows = _rows()
    with pytest.raises(ValueError, match="attach_hamming"):
        _index(rows).shortlist(rows[:1], 30, "screen+succ" + HAM, np.array([0]))


def test_the_union_is_scored_exactly_and_the_flag_off_is_bit_identical(monkeypatch):
    rows = _rows()
    torch, head = _head(rows)
    q = torch.from_numpy(_rows(6, 64, seed=7)).reshape(2, 3, 64)
    prev = torch.arange(6).reshape(2, 3)
    monkeypatch.delenv("CB_HEAD_SHORTLIST", raising=False)
    full = head.logits(q, head.vocab)
    idx = _index(rows)
    head.attach_shortlist(idx, k=30, arm="screen+succ" + HAM, ham_n=20)
    assert idx.ham_n == 20 and np.array_equal(idx.ham.bits, pack(rows))  # built from the head's own rows
    assert torch.equal(head.logits(q, head.vocab, prev), full)           # off: nothing changes
    monkeypatch.setenv("CB_HEAD_SHORTLIST", "1")
    out, full = head.logits(q, head.vocab, prev).reshape(6, -1), full.reshape(6, -1)
    lists = idx.shortlist(q.reshape(6, -1).numpy(), 30, "screen+succ" + HAM, prev.reshape(-1).numpy())
    for i, ids in enumerate(lists):
        kept = torch.isfinite(out[i])
        assert kept.nonzero().reshape(-1).tolist() == ids.tolist()
        assert torch.allclose(out[i][kept], full[i][kept], atol=1e-5)


def test_n_equal_to_the_vocabulary_is_the_full_head(monkeypatch):
    rows = _rows()
    torch, head = _head(rows)
    q = torch.from_numpy(_rows(4, 64, seed=8))
    full = head.logits(q, head.vocab)
    head.attach_shortlist(_index(rows), k=10, arm="screen" + HAM, ham_n=300)
    monkeypatch.setenv("CB_HEAD_SHORTLIST", "1")
    assert torch.allclose(head.logits(q, head.vocab), full, atol=1e-5)
