"""The spatial shard (validation/make_spatial_shard.py) and the corpus that reads it.

Pinned: every answer in a document is the Scene's exact reading; exactly the name tokens of
the listing and the ``here`` of a question carry a place, everything else NO_PLACE; a
document is exactly ``--doc`` tokens and decodes back to its text; the text control places
nothing; ``train_base.Corpus`` reads the side file, starts every window on a document, and
hands ``forward_batch`` a batch the spatial link trains on (a loss with gradient at the gate).
"""
from __future__ import annotations

import os
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT / "validation"), str(ROOT / "tests" / "model")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.model.spatial import NO_PLACE, SpatialLink  # noqa: E402
from make_spatial_shard import questions, render, village  # noqa: E402

TOK = ROOT / "data" / "grillcheese_bbpe128k.json"
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="the bbpe tokenizer is not in this checkout")


@pytest.fixture(scope="module")
def tk():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK))


def test_the_answers_are_the_scenes_exact_reading():
    rng = random.Random(1)
    sc = village(rng, 16)
    assert len(sc.things) == 16 and all(len(sc.where(n)) == 3 for n in sc.things)
    for q, a, cell in questions(sc, rng, 60):
        if cell is not None:
            assert q == "What is at here?" and a == (sc.at(cell) or "nothing")
        elif q.startswith("What is the position of "):
            assert a == " ".join(map(str, sc.where(q[len("What is the position of "):-1])))
        elif " neighbour of " in q:
            d, name = q[len("What is the "):-1].split(" neighbour of ")
            assert a == sc.adjacent(name).get(d, "nothing")
        elif q.startswith("What is the nearest "):
            kind, name = q[len("What is the nearest "):-1].split(" of ")
            assert a == (sc.nearest(name, kind) or "nothing")
        elif q.startswith("What is the region of "):
            rs = sc.regions_of(q[len("What is the region of "):-1])
            assert a == (rs[0] if rs else "nothing")
        else:
            n, d, name = q[len("What is "):-1].split(" ", 2)
            assert a == (sc.at(sc.offset(name[3:], d, int(n))) or "nothing")


def test_a_placed_question_carries_the_cell_it_is_about(tk):
    """``--ask placed``: the host computes the address (neighbour, offset, at) and a ``here``
    token carries it; the answer is the scene's reading at that cell."""
    rng = random.Random(5)
    sc = village(rng, 12)
    qa = questions(sc, rng, 40, "placed")
    assert qa and all(cell is not None for _q, _a, cell in qa)
    for q, a, cell in qa:
        assert a == (sc.at(cell) or "nothing")
        if " neighbour of " in q:
            d, name = q[len("What is the "):-1].split(" neighbour of ")
            assert cell == sc.offset(name, d, 1)
        elif not q.startswith("What is at"):
            n, d, name = q[len("What is "):-1].split(" ", 2)
            assert cell == sc.offset(name[3:], d, int(n))
    eos = tk.token_to_id("</s>")
    ids, places = render(tk, sc, qa[:5], "places", eos, 400)
    text = tk.decode(ids.tolist(), skip_special_tokens=True)
    assert "? here\n" in text or "at here?" in text
    here = [j for j in range(len(ids)) if places[j, 0] != NO_PLACE and tk.decode([int(ids[j])]).strip() == "here"]
    assert len(here) == 5 and [tuple(int(v) for v in places[j]) for j in here] == [c for _q, _a, c in qa[:5]]
    ids_t, places_t = render(tk, sc, qa[:5], "text", eos, 400)     # the control: the cell as words, nothing placed
    assert not (places_t[:, 0] != NO_PLACE).any()
    assert " ".join(map(str, qa[0][2])) in tk.decode(ids_t.tolist(), skip_special_tokens=True)


def test_a_document_places_exactly_the_things_and_the_asked_cell(tk):
    rng = random.Random(2)
    sc = village(rng, 8)
    qa = questions(sc, rng, 6)
    qa.append(("What is at here?", sc.at((3, 3, 0)) or "nothing", (3, 3, 0)))
    eos = tk.token_to_id("</s>")
    ids, places = render(tk, sc, qa, "places", eos, 320)
    assert ids.shape == (320,) and places.shape == (320, 3) and ids[-1] == eos
    text = tk.decode(ids.tolist(), skip_special_tokens=True)
    assert text.startswith("Village:\n") and "Answer:" in text
    placed = places[:, 0] != NO_PLACE
    # the listing: each thing's name tokens carry its cell, in curve order; the newline after it does not
    listed = []
    i = 0
    for cell, name in sc.serialize(*_box(sc)):
        n = len(tk.encode(name, add_special_tokens=False).ids)
        at = next(j for j in range(i, len(ids)) if placed[j])
        assert np.all(places[at:at + n] == cell) and tk.decode(ids[at:at + n].tolist()) == name
        assert not placed[at + n]
        listed.append(name)
        i = at + n
    assert listed == [n for _c, n in sc.serialize(*_box(sc))]
    # after the listing only a question's `here` carries a place, and the last one carries (3, 3, 0)
    here = [j for j in range(i, len(ids)) if placed[j]]
    assert here and all(tk.decode([int(ids[j])]).strip() == "here" for j in here)
    assert np.all(places[here[-1]] == (3, 3, 0))
    assert not placed[ids == eos].any()
    # the control: the same village as words, nothing placed
    ids_t, places_t = render(tk, sc, qa, "text", eos, 320)
    assert not (places_t[:, 0] != NO_PLACE).any() and " at " in tk.decode(ids_t.tolist(), skip_special_tokens=True)
    assert not np.array_equal(ids_t, ids)
    with pytest.raises(ValueError):
        render(tk, village(rng, 32), [], "places", eos, 16)


def _box(sc):
    cs = np.array(list(sc.cells))
    return cs.min(axis=0), cs.max(axis=0)


def test_the_corpus_reads_the_shard_by_documents_and_the_link_trains_on_it(tmp_path, tk, monkeypatch):
    import make_spatial_shard as mss
    monkeypatch.setattr(sys, "argv", ["make_spatial_shard", "--cache", str(tmp_path), "--tokenizer", str(TOK),
                                      "--villages", "12", "--doc", "192", "--things", "4", "8",
                                      "--questions", "2", "4", "--holdout-frac", "0.25", "--seed", "3"])
    mss.main()
    assert (tmp_path / "spatial.u32").exists() and (tmp_path / "spatial.pos.i16").exists()
    import json
    assert json.load(open(tmp_path / "_align.json"))["spatial"] == 192
    assert json.load(open(tmp_path / "_holdout.json"))["spatial"] == 3 * 192
    from train_base import Corpus, Tokens, forward_batch
    import shutil
    shutil.copy(tmp_path / "spatial.u32", tmp_path / "spatial (1).u32")      # a sync tool's conflict copy: skipped
    corpus = Corpus(str(tmp_path), seq=128, val_tokens=0, seed=0)
    assert corpus.names == ["spatial"]
    x, y = corpus.batch(1, 0, 4, "stable")
    assert isinstance(x, Tokens) and x.ids.shape == (4, 128) and x.pos.shape == (4, 128, 3) and y.shape == (4, 128)
    first = tk.decode(x.ids[0, :3].tolist())
    assert first.startswith("Village")                                     # every window starts on a document
    assert (x.pos[..., 0] != NO_PLACE).any()
    vx, vy = corpus.val_batch("spatial", 2)
    assert isinstance(vx, Tokens) and tk.decode(vx.ids[1, :3].tolist()).startswith("Village")
    # the link trains on it: a loss, and gradient at the gate once o is non-zero
    from test_grow import META, _build
    meta = dict(META, vocab=tk.get_vocab_size())
    m = _build(meta)
    m.spatial = SpatialLink(meta["D"], scale=1.0, seed=0, at=2)
    with torch.no_grad():
        m.spatial.o.weight.normal_(0, 0.1)
    logits = forward_batch(m, Tokens(x.ids[:2, :32], x.pos[:2, :32]))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:2, :32].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss) and m.spatial.gate.weight.grad.abs().sum() > 0
    # the route loss (CB_ROUTE_LOSS): the learned query at a placed token is taught its address
    import train_base
    plain_fn, _ = train_base.make_fns(m, meta["vocab"], torch.device("cpu"))
    monkeypatch.setattr(train_base, "ROUTE_LOSS", 1.0)
    route_fn, _ = train_base.make_fns(m, meta["vocab"], torch.device("cpu"))
    xb, yb = Tokens(x.ids[:2, :32], x.pos[:2, :32]), y[:2, :32]
    m.spatial.q.weight.grad = None
    plain, routed = plain_fn(xb, yb), route_fn(xb, yb)
    assert routed > plain and torch.isfinite(routed)                     # 1 - cos(unitary(q), S(p)) > 0 at the placed tokens
    routed.backward()
    assert m.spatial.q.weight.grad.abs().sum() > 0                         # the query head gets the route's gradient
    assert torch.allclose(plain_fn(Tokens(x.ids[:2, :32], torch.full_like(x.pos[:2, :32], NO_PLACE)), yb),
                          route_fn(Tokens(x.ids[:2, :32], torch.full_like(x.pos[:2, :32], NO_PLACE)), yb))   # nothing placed: no term
    monkeypatch.setattr(train_base, "ROUTE_LOSS", 0.0)
    # a window with nothing placed takes the plain path, bit for bit
    none = torch.full_like(x.pos[:2, :32], NO_PLACE)
    with torch.no_grad():
        assert torch.equal(forward_batch(m, Tokens(x.ids[:2, :32], none)), m.forward(x.ids[:2, :32]))
    # the text control is a plain shard: no side file, plain tensors
    monkeypatch.setattr(sys, "argv", ["make_spatial_shard", "--cache", str(tmp_path / "text"), "--tokenizer", str(TOK),
                                      "--villages", "12", "--doc", "192", "--things", "4", "8", "--mode", "text",
                                      "--questions", "2", "4", "--holdout-frac", "0.25", "--seed", "3"])
    os.makedirs(tmp_path / "text")
    mss.main()
    assert (tmp_path / "text" / "spatial_text.u32").exists() and not (tmp_path / "text" / "spatial_text.pos.i16").exists()
    tx, _ = Corpus(str(tmp_path / "text"), seq=128, val_tokens=0, seed=0).batch(1, 0, 2, "stable")
    assert torch.is_tensor(tx) and tk.decode(tx[0, :3].tolist()).startswith("Village")
