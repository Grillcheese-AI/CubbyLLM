import cubbyllm.training.data as data_mod
import cubbyllm.training.loop as loop_mod
import cubbyllm.training.probing as probe_mod
from cubbyllm.core.config import CubbyConfig
from cubbyllm.core.context import Context, FrozenSlotRouter
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener
from cubbyllm.core.probing import WrongContextProbe
from cubbyllm.core.protocols import Wiring
from cubbyllm.model.assembly import CubbyModel
from cubbyllm.model.backbone import MinGRUBackbone
from cubbyllm.model.binding import BindingHead
from cubbyllm.model.memory import MemoryLayer
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead
from cubbyllm.training import DataPipeline, InMemoryDataPipeline, P5WrongContextProbe, TrainLoop

D, V, CTX = 16, 32, 8


def _build():
    import numpy as np
    import torch

    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D, n_layers=1, ctx_dim=CTX, vocab_core=V)
    router = FrozenSlotRouter(input_dim=D, n_slots=4, ctx_dim=CTX).freeze()
    gen = HyperGenerator(ctx_dim=CTX, n_out=D * D)
    memory = MemoryLayer(generator=gen, hardener=SnapshotHardener(), d_model=D)
    model = CubbyModel(
        config=cfg, context_source=router,
        backbone=MinGRUBackbone(D, n_layers=1),
        memory=memory, binding=BindingHead(),
        embedding=HybridEmbedding(vocab_core=V, d_model=D),
        head=TopKRetrievalHead(torch.randn(V, D)), retrieval_k=8,
    )
    data = InMemoryDataPipeline(
        tokens=np.random.default_rng(0).integers(0, V, size=5000),
        manifest="test-corpus/v0 random",
    )
    return model, data, memory


def test_data_pipeline_manifest_is_stable_and_content_sensitive():
    import numpy as np

    toks = np.arange(2000) % 20
    a = InMemoryDataPipeline(toks, manifest="m1")
    b = InMemoryDataPipeline(toks, manifest="m1")
    c = InMemoryDataPipeline(toks, manifest="m2")   # different manifest
    assert isinstance(a, DataPipeline)
    assert a.manifest_hash() == b.manifest_hash()
    assert a.manifest_hash() != c.manifest_hash()


def test_train_step_runs_and_updates_params():
    import torch

    model, data, _ = _build()
    loop = TrainLoop(model, data, SnapshotHardener(), lr=1e-2, batch_size=4, seq_len=8)
    before = [p.clone() for p in model.parameters()]
    loss = loop.step()
    assert isinstance(loss, float) and loss > 0
    after = list(model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_p5_probe_detects_load_bearing_context():
    import torch

    _, _, memory = _build()
    probe = P5WrongContextProbe()
    assert isinstance(probe, WrongContextProbe)
    x = torch.randn(6, D)
    right = Context(vector=torch.randn(1, CTX), source_id="right")
    wrong = [Context(vector=torch.randn(1, CTX) + 5.0, source_id=f"w{i}") for i in range(3)]
    out = probe.probe(memory, x, right, wrong)
    assert set(out) == {"right", "wrong_mean", "bypass"}
    # a fresh (non-collapsed) generator: wrong contexts should move the read
    assert out["wrong_mean"] > 0.0


def test_wiring_declared():
    assert data_mod.__wiring__ is Wiring.STANDALONE
    assert loop_mod.__wiring__ is Wiring.STANDALONE
    assert probe_mod.__wiring__ is Wiring.STANDALONE
