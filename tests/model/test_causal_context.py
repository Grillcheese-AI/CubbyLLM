"""The context path is causal, per sample, and identical in training and decode.

Until 2026-09-23 ``CubbyModel.infer_context`` mean-pooled the whole sequence
(position t's context saw the tokens after t) and ``BasisHyperGenerator.apply``
averaged its adapter over the batch (sample i was transformed by the batch's
mean adapter). Decode could reproduce neither, so a trained model ran under a
different context from the one it was trained with. These tests pin the fix:

* a later token cannot change an earlier position's logits (exactly);
* another sample in the batch cannot change this sample's logits (exactly);
* stepping the model one token at a time reproduces the training forward.
"""
import torch

from cubbyllm.core.config import CubbyConfig
from cubbyllm.core.context import FrozenSlotRouter
from cubbyllm.core.generation import BasisHyperGenerator, SnapshotHardener
from cubbyllm.model.assembly import CubbyModel
from cubbyllm.model.backbone import HybridBackbone
from cubbyllm.model.binding import BindingHead
from cubbyllm.model.memory import MemoryLayer
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

D, L, V, CTX, S = 32, 4, 50, 8, 12


def _model():
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D, n_layers=L, ctx_dim=CTX, vocab_core=V)
    gen = BasisHyperGenerator(ctx_dim=CTX, d_model=D, n_layers=L, n_basis=4, rank=2)
    with torch.no_grad():                      # zero-init B would make the path a no-op
        gen.B_basis.normal_(0, 0.2)
    return CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=D, n_slots=4, ctx_dim=CTX).freeze(),
        # window 4 < S, so the check also crosses the attention window
        backbone=HybridBackbone(D, L, attn_every=3, window=4, heads=2),
        memory=MemoryLayer(gen, SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(V, D),
        head=TopKRetrievalHead(torch.randn(V, D) * 0.1, learnable=True),
        retrieval_k=V,
    )


def test_context_is_per_position():
    model = _model()
    ctx = model.infer_context(torch.randint(0, V, (2, S)))
    assert ctx.vector.shape == (2, S, CTX)


def test_a_later_token_cannot_change_an_earlier_position():
    model = _model()
    a = torch.randint(0, V, (2, S))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % V              # only the last token differs
    with torch.no_grad():
        la, lb = model.forward(a), model.forward(b)
    assert torch.equal(la[:, :-1], lb[:, :-1]), (
        "future token leaked into the past; max |diff| = "
        f"{(la[:, :-1] - lb[:, :-1]).abs().max():.2e}")
    assert not torch.equal(la[:, -1], lb[:, -1])


def test_another_sample_cannot_change_this_one():
    model = _model()
    a = torch.randint(0, V, (3, S))
    b = a.clone()
    b[2] = torch.randint(0, V, (S,))           # only sample 2 differs
    with torch.no_grad():
        la, lb = model.forward(a), model.forward(b)
    assert torch.equal(la[:2], lb[:2]), (
        "samples share an adapter; max |diff| = "
        f"{(la[:2] - lb[:2]).abs().max():.2e}")


def test_stepwise_decode_reproduces_the_training_forward():
    model = _model()
    tokens = torch.randint(0, V, (2, S))
    with torch.no_grad():
        full = model.forward(tokens)
        state, seq = None, []
        for t in range(S):
            logits, state = model.step(tokens[:, t], state)
            seq.append(logits)
        stepped = torch.stack(seq, dim=1)
    assert torch.allclose(full, stepped, atol=1e-4), (
        f"decode diverges from training; max |diff| = {(full - stepped).abs().max():.2e}")
