"""BasisHyperGenerator — the scalable theta=f(c) form (ported from cubemind MindForge).

The sensitivity tests below are not decoration. cubemind's own torch port of this
design (`model/cubby/heads.py::MindForgeLoRAHead`) is self-conditioned in every
call site — `forward(x, context=None)` then sets `context = x`, so it computes
theta=f(h) on the model's own hidden state and no external context ever reaches
it. Nothing in that repo fails when that happens. These tests fail.
"""
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.core.context import Context  # noqa: E402
from cubbyllm.core.generation import BasisHyperGenerator, HyperGenerator  # noqa: E402

CTX_D, D, NL = 8, 32, 4


def _gen(**kw):
    torch.manual_seed(0)
    return BasisHyperGenerator(ctx_dim=CTX_D, d_model=D, n_layers=NL,
                               n_basis=4, rank=2, hidden=16, **kw)


def _ctx(seed, b=2):
    g = torch.Generator().manual_seed(seed)
    return Context(vector=torch.randn(b, CTX_D, generator=g), source_id="test")


def test_generated_block_has_the_protocol_shape():
    gp = _gen().generate(_ctx(0))
    assert gp.weights.shape == (2, D * D)
    assert gp.meta["source_id"] == "test" and gp.meta["layer_id"] == 0


def test_output_actually_depends_on_the_context():
    """Guard against a generator that ignores c — the degenerate case where
    theta=f(c) silently becomes theta=f(anything-else)."""
    g = _gen()
    a = g.generate(_ctx(1)).weights
    b = g.generate(_ctx(2)).weights
    # B_basis is zero-init, so at step 0 every output is identically 0 and this
    # test would pass vacuously. Move it first, exactly as training would.
    with torch.no_grad():
        g.B_basis.normal_(0, 0.02)
    a, b = g.generate(_ctx(1)).weights, g.generate(_ctx(2)).weights
    assert not torch.allclose(a, b), "generator output is invariant to the context"


def test_output_actually_depends_on_the_layer():
    g = _gen()
    with torch.no_grad():
        g.B_basis.normal_(0, 0.02)
    c = _ctx(3)
    a, b = g.generate(c, layer_id=0).weights, g.generate(c, layer_id=1).weights
    assert not torch.allclose(a, b), "generator output is invariant to layer_id"


def test_factored_apply_matches_the_materialised_matrix():
    """`apply` must equal x @ W^T for the W `generate` would have built —
    otherwise the cheap path and the drop-in path have diverged."""
    g = _gen()
    with torch.no_grad():
        g.B_basis.normal_(0, 0.02)
    ctx = _ctx(4, b=1)
    x = torch.randn(3, 5, D)
    W = g.generate(ctx).weights.reshape(-1, D, D).mean(dim=0)
    assert torch.allclose(g.apply(x, ctx), x @ W.t(), atol=1e-5)


def test_zero_init_does_not_block_training():
    """B zero-init is standard LoRA (identity adapter at step 0). It trains fine
    PROVIDED the delta is consumed residually and the loss has a nonzero gradient
    where delta=0 — which is how MemoryLayer uses it (``x + tanh(delta)``).

    The caveat is real and was found by this test failing: with a loss that is
    quadratic *to zero* (``delta.pow(2).mean()``), delta=0 is already the minimum,
    so nothing anywhere receives gradient. That is a property of the objective,
    not of the init. cubemind's head sidesteps it by adding ``base(x)``; a comment
    there claiming zero-init 'kills gradients' conflates the two.
    """
    g = _gen()
    opt = torch.optim.Adam(list(g.parameters()), lr=1e-2)
    ctx, x = _ctx(5, b=1), torch.randn(4, D)
    target = torch.randn(4, D)

    def loss():                                    # residual read, real target
        return (x + torch.tanh(g.apply(x, ctx)) - target).pow(2).mean()

    loss().backward()
    assert g.B_basis.grad.abs().sum() > 0, "B_basis got no gradient at init"
    assert g.A_basis.grad.abs().sum() == 0, "A_basis should be blocked only at step 0"
    for _ in range(3):
        opt.zero_grad(); loss().backward(); opt.step()
    assert g.A_basis.grad.abs().sum() > 0, "A_basis never unblocked after B moved"


def test_a_loss_quadratic_to_zero_stalls_the_whole_generator():
    """The negative case the test above discovered — pinned so it stays known.
    If the generated delta is the ENTIRE output and the objective pulls it to
    zero, zero-init B sits at the minimum and NOTHING trains. Any future consumer
    of this generator must add the delta to something."""
    g = _gen()
    ctx, x = _ctx(6, b=1), torch.randn(4, D)
    g.apply(x, ctx).pow(2).mean().backward()
    assert g.B_basis.grad.abs().sum() == 0
    assert g.A_basis.grad.abs().sum() == 0


def test_is_dramatically_smaller_than_the_flat_generator():
    """The reason this class exists. At the real shape (d=2048) the flat form is
    272.6M params — 13.6% of a 2B model — to emit one d*d block."""
    d = 256
    flat = sum(p.numel() for p in HyperGenerator(ctx_dim=CTX_D, n_out=d * d).parameters())
    basis = sum(p.numel() for p in BasisHyperGenerator(
        ctx_dim=CTX_D, d_model=d, n_layers=8, n_basis=16, rank=8, hidden=64).parameters())
    assert basis * 10 < flat, f"expected a large saving; got flat={flat:,} basis={basis:,}"
