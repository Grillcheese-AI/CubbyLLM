"""Growing a CubbyModel keeps its function (cubbyllm/model/grow.py).

The claim the growth plan rests on (docs/research/2026-09-25-grow-450m.md,
step 0): width ×2, depth ×2 with zeroed exits, FFN ×2, and depth + FFN
together, each give the same logits as the model they were grown from, before
any training. Every parameter is randomised first (the zero-initialised
adapter basis included), so an exact match is not an accident of zeros. The
plain copy (G_stack's) is the control: it must NOT match, or the test could
not see a wrong grow.
"""
import pytest
import torch

from cubbyllm.core.config import CubbyConfig
from cubbyllm.core.context import FrozenSlotRouter
from cubbyllm.core.generation import BasisHyperGenerator, SnapshotHardener
from cubbyllm.model.assembly import CubbyModel
from cubbyllm.model.backbone import HybridBackbone
from cubbyllm.model.binding import BindingHead
from cubbyllm.model.grow import grow_into, grown_meta, layer_map
from cubbyllm.model.memory import MemoryLayer
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

META = {"D": 16, "L": 5, "heads": 2, "window": 4, "attn_every": 3, "ctx": 6,
        "slots": 3, "gen_basis": 3, "gen_rank": 2, "vocab": 40}
S = 11                                          # crosses the attention window


def _build(meta, seed=0):
    torch.manual_seed(seed)
    d, n = meta["D"], meta["L"]
    gen = BasisHyperGenerator(ctx_dim=meta["ctx"], d_model=d, n_layers=n,
                              n_basis=meta["gen_basis"], rank=meta["gen_rank"])
    return CubbyModel(
        config=CubbyConfig(d_model=d, n_layers=n, ctx_dim=meta["ctx"], vocab_core=meta["vocab"]),
        context_source=FrozenSlotRouter(input_dim=d, n_slots=meta["slots"],
                                        ctx_dim=meta["ctx"]).freeze(),
        backbone=HybridBackbone(d, n, attn_every=meta["attn_every"], window=meta["window"],
                                heads=meta["heads"], ffn_mult=meta.get("ffn_mult", 2)),
        memory=MemoryLayer(gen, SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(meta["vocab"], d),
        head=TopKRetrievalHead(torch.randn(meta["vocab"], d) * 0.1, learnable=True),
        retrieval_k=meta["vocab"],
    )


def _source():
    m = _build(META)
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.context_source.parameters()):
            p.copy_(torch.randn_like(p) * 0.3)
    return m


def _grow(**kw):
    src = _source()
    dst = _build(grown_meta(META, kw.get("width", 1), kw.get("depth", 1), kw.get("ffn", 1)), seed=1)
    rep = grow_into(src, dst, **kw)
    x = torch.randint(0, META["vocab"], (2, S), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        return src.forward(x), dst.forward(x), rep, src, dst, x


@pytest.mark.parametrize("kw", [
    {"width": 2}, {"depth": 2}, {"ffn": 2}, {"depth": 2, "ffn": 2},
    {"width": 2, "depth": 2, "ffn": 2}, {"depth": 3},
], ids=["width2", "depth2", "ffn2", "depth2_ffn2", "all", "depth3"])
def test_the_grown_model_computes_the_same_logits(kw):
    a, b, rep, *_ = _grow(**kw)
    assert rep["params_dst"] > rep["params_src"]
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_the_plain_copy_is_not_exact():
    a, b, *_ = _grow(depth=2, exit="copy")
    assert (a - b).abs().max() > 1e-2


def test_decode_matches_the_grown_forward():
    """The step path (what serving runs) agrees with the forward on the grown model."""
    _, b, _, _, dst, x = _grow(width=2, depth=2, ffn=2)
    state = None
    with torch.no_grad():
        for t in range(S):
            logits, state = dst.step(x[:, t], state)
            assert torch.allclose(logits, b[:, t], atol=1e-4, rtol=1e-4)


def test_every_layer_keeps_its_kind():
    for L in (5, 6, 7, 32):
        for depth in (2, 3):
            m = layer_map(L, 3, depth)
            assert len(m) == L * depth
            assert sorted(i for i, c in m if not c) == list(range(L))
            assert all((j % 3 == 0) == (i % 3 == 0) for j, (i, _) in enumerate(m))


def test_meta_keeps_old_checkpoints_equal():
    assert "ffn_mult" not in grown_meta(META, width=2)
    assert grown_meta(META, ffn=2)["ffn_mult"] == 4
    g = grown_meta(META, width=2, depth=2)
    assert (g["D"], g["heads"], g["L"]) == (32, 4, 10)
