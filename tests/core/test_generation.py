import cubbyllm.core.generation as mod
from cubbyllm.core.context import Context
from cubbyllm.core.generation import (
    GeneratedParams,
    Hardener,
    HyperGenerator,
    ParameterGenerator,
    SnapshotHardener,
)
from cubbyllm.core.protocols import Wiring


def test_generated_params_carrier():
    gp = GeneratedParams(weights=[1.0, 2.0], meta={"target": "head"})
    assert gp.meta["target"] == "head"


def test_hyper_generator_maps_context_to_weights():
    import torch

    gen = HyperGenerator(ctx_dim=8, n_out=16)
    assert isinstance(gen, ParameterGenerator)
    ctx = Context(vector=torch.randn(1, 8), source_id="t0")
    gp = gen.generate(ctx)
    assert gp.weights.shape[-1] == 16


def test_snapshot_hardener_penalizes_drift():
    import torch

    gen = HyperGenerator(ctx_dim=8, n_out=16)
    hardener = SnapshotHardener(beta=10.0)
    assert isinstance(hardener, Hardener)
    ctxs = [Context(vector=torch.randn(1, 8), source_id=f"t{i}") for i in range(3)]

    # no snapshot yet -> zero penalty
    assert float(hardener.penalty(gen)) == 0.0

    hardener.snapshot(gen, ctxs)
    # immediately after snapshot, drift is ~zero
    assert float(hardener.penalty(gen)) < 1e-6

    # perturb the generator; penalty must become positive
    with torch.no_grad():
        for p in gen.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    assert float(hardener.penalty(gen)) > 0.0


def test_wiring_declared():
    assert mod.__wiring__ in (Wiring.WIRED, Wiring.STANDALONE)
