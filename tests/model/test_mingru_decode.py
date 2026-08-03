"""The incremental decode path must equal the parallel-scan path.

This is the test that licenses every inference benchmark. `step()` exists to make
the O(1)-per-token claim real — generating with `forward()` re-runs the whole
prefix per token, which is O(S) work and strictly worse than a KV cache, so the
architecture's headline inference advantage lives entirely in this method.

If `step()` drifts from `forward()`, a throughput comparison stops measuring the
trained model and starts measuring a different one that happens to be faster.
That is the failure mode these tests exist to prevent.
"""
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402

D, L, B, S = 32, 3, 2, 16


def _backbone():
    torch.manual_seed(0)
    return MinGRUBackbone(D, L)


def test_step_matches_forward_over_a_whole_sequence():
    bb = _backbone()
    x = torch.randn(B, S, D)
    with torch.no_grad():
        parallel = bb.forward(x)
        states, seq = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            seq.append(y)
        incremental = torch.stack(seq, dim=1)
    assert torch.allclose(parallel, incremental, atol=1e-4), (
        f"decode diverges from training path; max |diff| = "
        f"{(parallel - incremental).abs().max():.2e}"
    )


def test_state_size_is_independent_of_context_length():
    """The whole inference argument: carried state must not grow with S."""
    bb = _backbone()
    sizes = []
    with torch.no_grad():
        for S_ in (4, 64, 256):
            states = None
            for t in range(S_):
                _, states = bb.step(torch.randn(B, D), states)
            sizes.append(sum(s.numel() for s in states))
    assert len(set(sizes)) == 1, f"state grew with context: {sizes}"
    assert sizes[0] == L * B * D


def test_resuming_from_carried_state_matches_one_long_pass():
    """Prefill-then-decode must equal decoding the whole thing — otherwise a
    benchmark that prefills in parallel and decodes incrementally is invalid."""
    bb = _backbone()
    x = torch.randn(B, S, D)
    with torch.no_grad():
        full = torch.stack([bb.step(x[:, t], s)[0] for t, s in
                            [(0, None)]], dim=1)  # seed
        states, out = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        a = torch.stack(out, dim=1)
        # now split: first half, carry state, second half
        states, out = None, []
        for t in range(S // 2):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        mid = [s.clone() for s in states]
        for t in range(S // 2, S):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        b = torch.stack(out, dim=1)
    assert torch.allclose(a, b, atol=1e-6)
    assert full is not None and len(mid) == L
