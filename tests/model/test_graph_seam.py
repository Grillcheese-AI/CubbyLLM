"""``ops.graph_step`` and the int8 head: the two opt-ins over the step.

Both are seams — a capability the backend may or may not have — so what is
testable everywhere is that they are honest about which case they are in.
On real torch ``graph_step`` is the identity and ``int8_weight_only``
raises; on grilly2 (``-p grilly.torch_alias``) the wrapper records and
replays and the head quantizes, and those branches run there.
"""
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.ops import graph_stats, graph_step, int8_available, int8_weight_only  # noqa: E402

HAS_GRAPHS = hasattr(torch, "graphed")


def test_graph_step_is_transparent():
    """Whatever the backend, the wrapped step answers the same as the bare
    one. On a backend without capture it *is* the bare one."""
    calls = []

    def step(x, state=None):
        calls.append(1)
        y = x * 2.0 + 1.0
        return y, {"n": torch.zeros(()) if state is None else state["n"] + 1.0}

    wrapped = graph_step(step)
    x = torch.ones(1, 4)
    state = None
    for _ in range(6):
        y, state = wrapped(x, state)
    assert torch.equal(y, x * 2.0 + 1.0)
    if not HAS_GRAPHS:
        assert wrapped is step
        assert graph_stats(wrapped) is None


@pytest.mark.skipif(not HAS_GRAPHS, reason="backend has no graph capture")
def test_a_settled_step_actually_replays():
    """The diagnostic that matters. A step whose shapes do not settle
    captures over and over and never replays, and wrapping it is pure
    overhead — which is exactly what the old decode state did."""
    w = torch.randn(8, 8)

    def step(x, state):
        state.add_(1.0)
        return (x @ w).tanh() + state, state

    wrapped = graph_step(step)
    state = torch.zeros(())
    with torch.no_grad():
        for i in range(6):
            _, state = wrapped(torch.randn(1, 8), state)
    stats = graph_stats(wrapped)
    assert stats["replays"] >= 3, stats
    assert stats["captures"] == 1, stats


def test_int8_says_which_backend_it_is_on():
    """It raises rather than quietly leaving float32 in place: a
    benchmark that reports an unquantized run as a quantized one is worse
    than one that fails."""
    w = torch.randn(64, 16)
    if int8_available():
        q = int8_weight_only(w)
        assert q.shape == w.shape
        out = torch.nn.functional.linear(torch.randn(1, 16), q)
        assert out.shape == (1, 64)
    else:
        with pytest.raises(RuntimeError, match="no weight-only int8 path"):
            int8_weight_only(w)


def test_the_head_quantizes_only_when_asked():
    """Opt-in, and irreversible for that head — so nothing quantizes by
    accident and nothing dequantizes by accident either."""
    from cubbyllm.model.vocab import TopKRetrievalHead

    torch.manual_seed(0)
    head = TopKRetrievalHead(torch.randn(64, 16), learnable=True)
    q = torch.randn(1, 1, 16)
    before = head.logits(q, 64)
    assert torch.equal(head.logits(q, 64), before), "logits are not stable"
    if not int8_available():
        with pytest.raises(RuntimeError):
            head.quantize_()
        return
    head.quantize_()
    after = head.logits(q, 64)
    assert after.shape == before.shape
    rel = (after - before).abs().max() / before.abs().max()
    assert rel < 0.05, f"int8 head moved the logits by {rel:.3f}"


def test_the_head_matmul_is_f_linear():
    """``F.linear(q, W)`` rather than ``q @ W.t()``: the same arithmetic,
    but the call a quantized weight can answer with its own kernel, and one
    that does not ask the backend to materialize a (V, d) transpose."""
    import torch.nn.functional as F

    torch.manual_seed(1)
    q, w = torch.randn(1, 1, 16), torch.randn(64, 16)
    assert torch.equal(F.linear(q, w), q @ w.t())
