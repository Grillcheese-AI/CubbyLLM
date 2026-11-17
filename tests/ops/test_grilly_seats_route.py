"""Order 0068, the wiring: the trunk attaches route 5 only with ``CB_HEAD_SHORTLIST=1`` and an index (off by default,
and off the trunk is the one before); ``grilly_seats.attach_route`` refuses an index over another head. With grilly2's
shortlist decode on the path, a route over every row decodes a tiny Cubby model exactly as its full head does, through
the seam the seats call."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from cubbyllm.emit.seats_450m import ROUTE_FILE, route_index
from cubbyllm.model.vocab.shortlist import ScreenIndex, ShortlistIndex, successors
from cubbyllm.ops import grilly_seats as gs


def _index(rows: np.ndarray, path: str, clusters: int = 6) -> None:
    v = len(rows)
    idx = ShortlistIndex.build(rows, clusters, seed=0)
    lists = [np.flatnonzero(idx.assign == c) for c in range(clusters)]
    idx.screen = ScreenIndex(idx.centroids, np.concatenate(lists).astype(np.int64),
                             np.concatenate([[0], np.cumsum([len(x) for x in lists])]), v)
    idx.succ = successors(np.random.default_rng(1).integers(0, v, 4000), v, top=4)
    idx.save(path)


def test_off_by_default_even_with_an_index(tmp_path, monkeypatch):
    monkeypatch.delenv("CB_HEAD_SHORTLIST", raising=False)
    (tmp_path / ROUTE_FILE).write_bytes(b"")
    assert route_index(str(tmp_path)) is None


def test_on_it_takes_the_named_index_then_the_one_beside_the_export(tmp_path, monkeypatch):
    monkeypatch.setenv("CB_HEAD_SHORTLIST", "1")
    monkeypatch.delenv("CB_HEAD_ROUTE_INDEX", raising=False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert route_index(str(tmp_path)) is None                       # on, and no index: the full head, warned
    assert any("the full head runs" in str(w.message) for w in seen)
    (tmp_path / ROUTE_FILE).write_bytes(b"")
    assert route_index(str(tmp_path)) == os.path.join(str(tmp_path), ROUTE_FILE)
    named = tmp_path / "elsewhere.npz"
    named.write_bytes(b"")
    monkeypatch.setenv("CB_HEAD_ROUTE_INDEX", str(named))
    assert route_index(str(tmp_path)) == str(named)


class _Old:                                     # a grilly2 model without the shortlist decode
    pass


def test_a_grilly2_without_the_decode_gets_nothing(tmp_path):
    assert gs.attach_route(_Old(), str(tmp_path / "unread.npz")) is None


def _tiny():
    pytest.importorskip("grilly.infer.shortlist")
    import grilly
    from grilly.infer.models import CubbyConfig, CubbyForCausalLM
    if not hasattr(CubbyForCausalLM, "hidden"):
        pytest.skip("this grilly2 has no shortlist decode (branch cowork/shortlist-generate, under review)")
    rng = np.random.default_rng(11)
    m = CubbyForCausalLM(CubbyConfig(vocab_size=300, hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
                                     window=8, attn_every=3, ctx_dim=8, n_slots=4, router_hidden=16, gen_hidden=16,
                                     gen_basis=4, gen_rank=2, gen_layers=3, vocab_real=296))
    with grilly.no_grad():
        for name, p in m.named_parameters():
            shape = tuple(p.shape)
            w = (1.0 + 0.1 * rng.standard_normal(shape) if name.endswith("norm.weight")
                 else rng.standard_normal(shape) / np.sqrt(shape[-1] if len(shape) > 1 else 1))
            p.copy_(grilly.from_numpy(w.astype(np.float32)))
    return grilly, m


def test_a_route_over_every_row_decodes_as_the_full_head(tmp_path):
    g, m = _tiny()
    rows = m.lm_head.weight.detach().numpy()
    _index(rows, str(tmp_path / ROUTE_FILE))
    prompt = [[5, 9, 14, 2, 77, 120]]
    full = gs.generate(m, prompt, 20, eos=-1, g=g)
    head = gs.attach_route(m, str(tmp_path / ROUTE_FILE), near=300, context=300)
    assert head is not None and m.shortlist is head
    assert gs.generate(m, prompt, 20, eos=-1, g=g) == full
    assert head.picks == 20
    gs.generate(m, prompt + prompt, 20, eos=-1, g=g)                   # a batch: the full head
    gs.generate(m, prompt, 20, eos=-1, temperature=0.7, seed=3, g=g)   # sampling: the full head
    assert head.picks == 20


def test_an_index_over_another_head_is_refused(tmp_path):
    g, m = _tiny()
    rows = m.lm_head.weight.detach().numpy()
    _index(rows[:200], str(tmp_path / "other.npz"))
    with pytest.raises(ValueError, match="head rows"):
        gs.attach_route(m, str(tmp_path / "other.npz"))
    assert m.shortlist is None
