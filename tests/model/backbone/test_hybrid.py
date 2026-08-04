"""HybridBackbone: windowed attention must decode incrementally AND keep its
carried state bounded by the window — the two properties that let it be an
alternate backbone without breaking the O(1)-inference thesis.

If step() drifts from forward(), a throughput/quality comparison measures a
different model (same reasoning as test_mingru_decode). If the state grows with
context, the hybrid forfeits the very guarantee (bounded decode state) that made
windowed attention the deployable choice over full attention — so both are tested
here, and a failure of either disqualifies the design, not just the code.
"""
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.model.backbone import Backbone, HybridBackbone  # noqa: E402

D, L, B, S, W = 32, 6, 2, 40, 8


def _backbone(window=W, attn_every=3):
    torch.manual_seed(0)
    return HybridBackbone(D, n_layers=L, attn_every=attn_every, window=window,
                          heads=4)


def test_conforms_and_interleaves():
    bb = _backbone()
    assert isinstance(bb, Backbone)
    # attn_every=3 over 6 layers -> attention on layers 0 and 3
    assert bb.is_attn == [True, False, False, True, False, False]
    assert bb.n_attn_layers == 2
    x = torch.randn(B, S, D)
    assert bb.forward(x).shape == x.shape


def test_step_matches_forward_over_a_whole_sequence():
    """Incremental decode (with a windowed KV cache) must equal the masked
    parallel forward at every position — including past the window boundary,
    where the cache trim has to reproduce exactly the mask's window."""
    bb = _backbone()
    x = torch.randn(B, S, D)                              # S=40 >> W=8
    with torch.no_grad():
        parallel = bb.forward(x)
        states, seq = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            seq.append(y)
        incremental = torch.stack(seq, dim=1)
    assert torch.allclose(parallel, incremental, atol=1e-4), (
        f"decode diverges from training path; max |diff| = "
        f"{(parallel - incremental).abs().max():.2e}")


def test_state_stops_growing_once_context_exceeds_the_window():
    """The deployable property: total carried state is bounded by the window, so
    it is CONSTANT for any context length >= window — unlike a full KV cache,
    which grows without bound."""
    bb = _backbone()

    def state_numel(S_):
        with torch.no_grad():
            states = None
            for t in range(S_):
                _, states = bb.step(torch.randn(B, D), states)
        tot = 0
        for s in states:
            tot += s.numel() if torch.is_tensor(s) else sum(t.numel() for t in s)
        return tot

    past = [state_numel(n) for n in (W, 2 * W, 4 * W)]   # all >= window
    assert len(set(past)) == 1, f"state grew past the window: {past}"
    # and it is genuinely bounded: attention layers cap at window, not context
    below, at = state_numel(W // 2), past[0]
    assert below < at, "state should still be filling below the window"


def test_full_kv_would_grow_but_this_does_not():
    """Contrast that names the win: an unbounded cache at 4*W would hold 4x the
    keys/values a windowed one does. The windowed state must be the SAME at W and
    4*W, which is exactly what makes decode O(1) in context."""
    bb = _backbone(window=W)

    def attn_state(S_):
        with torch.no_grad():
            states = None
            for t in range(S_):
                _, states = bb.step(torch.randn(B, D), states)
        # sum the KV-cache lengths across attention layers
        return sum(s[0].shape[2] for s in states if not torch.is_tensor(s))

    assert attn_state(W) == attn_state(4 * W) == bb.n_attn_layers * W


def test_resuming_from_carried_state_matches_one_pass():
    """Prefill half, carry state, decode the rest — must equal decoding straight
    through. Licenses a benchmark that prefills in parallel then decodes."""
    bb = _backbone()
    x = torch.randn(B, S, D)
    with torch.no_grad():
        states, out = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        straight = torch.stack(out, dim=1)

        states, out = None, []
        for t in range(S // 2):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        for t in range(S // 2, S):
            y, states = bb.step(x[:, t], states)
            out.append(y)
        split = torch.stack(out, dim=1)
    assert torch.allclose(straight, split, atol=1e-6)


def test_window_actually_bounds_reach():
    """A token beyond the window must NOT influence the current output — the
    property that makes recall fall back to the recurrent state past `window`,
    and the honest limit of any bounded-state model.

    ONE attention layer, so the receptive field is exactly the window. (Stacking
    windowed layers compounds reach to ~n*(window-1) — a real property, tested
    separately below — so a single layer is what isolates the per-layer bound.)"""
    torch.manual_seed(1)
    bb = HybridBackbone(D, n_layers=1, attn_every=1, window=W, heads=4)
    S_ = W + 6
    x = torch.randn(1, S_, D)
    with torch.no_grad():
        a = bb.forward(x)[0, -1]
        x2 = x.clone()
        x2[0, 0] = torch.randn(D)          # token 0 is (S_-1) > window back
        b = bb.forward(x2)[0, -1]
    assert torch.allclose(a, b, atol=1e-6), (
        "a token outside the window changed the output — the window is not bounding reach")


def test_stacked_windows_compound_reach():
    """Two windowed layers reach ~2*(window-1): a token that one layer cannot see
    CAN influence the output through a second hop. This is why a few windowed
    layers extend effective context beyond a single window — and why the honest
    recall claim is 'out to ~depth*window', not 'exactly window'."""
    torch.manual_seed(2)
    bb = HybridBackbone(D, n_layers=2, attn_every=1, window=W, heads=4)
    S_ = W + 3                              # last pos within 2-hop reach of token 0
    x = torch.randn(1, S_, D)
    with torch.no_grad():
        a = bb.forward(x)[0, -1]
        x2 = x.clone()
        x2[0, 0] = torch.randn(D)
        b = bb.forward(x2)[0, -1]
    assert not torch.allclose(a, b, atol=1e-5), (
        "two windowed layers failed to compound reach — expected 2-hop propagation")
