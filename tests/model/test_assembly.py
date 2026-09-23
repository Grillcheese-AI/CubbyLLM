import cubbyllm.model.assembly as mod
from cubbyllm.core.config import CubbyConfig
from cubbyllm.core.context import FrozenSlotRouter
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener
from cubbyllm.core.protocols import Wiring
from cubbyllm.model.assembly import CubbyModel
from cubbyllm.model.backbone import MinGRUBackbone
from cubbyllm.model.binding import BindingHead
from cubbyllm.model.memory import MemoryLayer
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

D, V, CTX = 16, 40, 8


def _model():
    import torch

    cfg = CubbyConfig(d_model=D, n_layers=2, ctx_dim=CTX, vocab_core=V)
    router = FrozenSlotRouter(input_dim=D, n_slots=4, ctx_dim=CTX).freeze()
    gen = HyperGenerator(ctx_dim=CTX, n_out=D * D)
    memory = MemoryLayer(generator=gen, hardener=SnapshotHardener(), d_model=D)
    codebook = torch.randn(V, D)
    return CubbyModel(
        config=cfg,
        context_source=router,
        backbone=MinGRUBackbone(D, n_layers=2),
        memory=memory,
        binding=BindingHead(),
        embedding=HybridEmbedding(vocab_core=V, d_model=D),
        head=TopKRetrievalHead(codebook),
        retrieval_k=8,
    )


def test_end_to_end_forward_produces_token_logits():
    import torch

    torch.manual_seed(0)
    model = _model()
    tokens = torch.randint(0, V, (2, 7))          # (B=2, S=7)
    logits = model.forward(tokens)
    assert logits.shape == (2, 7, V)
    # exactly retrieval_k candidates survive per position (local softmax set)
    finite = torch.isfinite(logits[0, 0])
    assert int(finite.sum()) == 8


def test_context_is_threaded_and_frozen():
    model = _model()
    assert model.context_source.is_frozen is True
    import torch

    ctx = model.infer_context(torch.randint(0, V, (2, 5)))
    assert ctx.vector.shape == (2, 5, CTX)        # one context per position (causal)


def test_forward_docstring_threads_context():
    doc = CubbyModel.forward.__doc__ or ""
    assert "context-ignoring path cannot be written" in doc


def test_wiring_is_wired():
    assert mod.__wiring__ is Wiring.WIRED
