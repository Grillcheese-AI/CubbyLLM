import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.recall import MemoryRead

D, DK, B, S = 32, 16, 2, 20


def test_read_is_per_position_not_a_dc_offset():
    """Two query positions with different content must get different reads —
    the exact failure MEMORY_PROBE.md diagnosed (one vector broadcast to all)."""
    torch.manual_seed(0)
    m = MemoryRead(D, d_key=DK, topk=4)
    h = torch.randn(B, S, D)
    r = m.forward(h, window=4)
    assert r.shape == (B, S, D)
    # positions far apart, both with beyond-window memory, differ
    assert not torch.allclose(r[:, 10], r[:, 18], atol=1e-5)


def test_read_only_sees_beyond_window_past():
    """Perturbing a token INSIDE the window of position i must NOT change i's read;
    perturbing one BEYOND the window must. That is the whole point of the mask."""
    torch.manual_seed(1)
    m = MemoryRead(D, d_key=DK, topk=8)
    W = 4
    h = torch.randn(1, S, D)
    with torch.no_grad():
        r0 = m.forward(h, window=W)[0, S - 1]
        h_in = h.clone(); h_in[0, S - 2] = torch.randn(D)        # inside window
        r_in = m.forward(h_in, window=W)[0, S - 1]
        h_far = h.clone(); h_far[0, 0] = torch.randn(D)          # beyond window
        r_far = m.forward(h_far, window=W)[0, S - 1]
    assert torch.allclose(r0, r_in, atol=1e-6), "read used a within-window token"
    assert not torch.allclose(r0, r_far, atol=1e-6), "read ignored the beyond-window past"


def test_early_positions_with_no_memory_read_zero():
    """Positions i <= window have no beyond-window past; their read must be a
    clean zero, not a NaN."""
    m = MemoryRead(D, d_key=DK, topk=4)
    h = torch.randn(1, S, D)
    r = m.forward(h, window=6)
    assert torch.isfinite(r).all()
    assert torch.allclose(r[0, 0], torch.zeros(D), atol=1e-6)
