import cubbyllm.model.memory.base as mod
from cubbyllm.core.context import Context
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener
from cubbyllm.core.protocols import Generable, Wiring
from cubbyllm.model.memory import MemoryLayer

D = 16


def _memory():
    gen = HyperGenerator(ctx_dim=8, n_out=D * D)
    return MemoryLayer(generator=gen, hardener=SnapshotHardener(), d_model=D)


def test_memory_is_generable():
    assert isinstance(_memory(), Generable)


def test_forward_generated_is_context_conditioned_residual():
    import torch

    mem = _memory()
    x = torch.randn(5, D)
    ctx = Context(vector=torch.randn(1, 8), source_id="t0")
    out = mem.forward_generated(x, ctx)
    assert out.shape == x.shape

    # different contexts -> different reads (theta actually depends on c)
    ctx2 = Context(vector=torch.randn(1, 8) + 3.0, source_id="t1")
    out2 = mem.forward_generated(x, ctx2)
    assert not torch.allclose(out, out2, atol=1e-4)


def test_wiring_is_wired():
    assert mod.__wiring__ is Wiring.WIRED
