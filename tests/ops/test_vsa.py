import numpy as np
import pytest

import cubbyllm.ops.vsa as mod
from cubbyllm.core.protocols import Wiring
from cubbyllm.ops import BlockCodeVSA

K, L = 8, 16


def _vsa():
    return BlockCodeVSA(k=K, l=L)


def test_backend_reports_a_known_tier():
    vsa = _vsa()
    assert vsa.backend() in {"grilly_bridge", "grilly_python", "numpy"}
    assert vsa.k == K and vsa.l == L


def test_unbind_recovers_bound_factor_exactly():
    """bind(a, b) then unbind(., b) recovers a (exact for discrete one-hot)."""
    vsa = _vsa()
    cb = vsa.codebook(4)
    a, b = cb[0], cb[1]
    comp = vsa.bind(a, b)
    rec = vsa.unbind(comp, b)
    assert vsa.similarity(rec, a) > 0.99
    # and the composite is dissimilar to each factor (binding is quasi-orthogonal)
    assert vsa.similarity(comp, a) < 0.5


def test_zero_is_binding_identity():
    vsa = _vsa()
    a = vsa.codebook(2)[0]
    assert vsa.similarity(vsa.bind(a, vsa.zero()), a) > 0.99


def test_bundle_is_similar_to_members():
    vsa = _vsa()
    cb = vsa.codebook(3)
    bundled = vsa.bundle([cb[0], cb[1], cb[2]])
    sims = [vsa.similarity(bundled, cb[i]) for i in range(3)]
    assert all(s > 0 for s in sims)
    # a non-member random code is less similar than the members on average
    other = vsa.codebook(6)[5]
    assert np.mean(sims) > vsa.similarity(bundled, other)


def test_similarity_batch_matches_scalar():
    vsa = _vsa()
    cb = vsa.codebook(5)
    q = cb[2]
    batch = vsa.similarity_batch(q, cb)
    assert int(np.argmax(batch)) == 2
    assert abs(float(batch[2]) - vsa.similarity(q, cb[2])) < 1e-4


def test_numpy_fallback_matches_grilly():
    """Force the numpy path and confirm it agrees with grilly on a roundtrip."""
    vsa = _vsa()
    if vsa._ops is None:
        pytest.skip("grilly not available; only numpy path exists")
    cb = vsa.codebook(4)
    comp = vsa.bind(cb[0], cb[1])
    # numpy reimplementation of unbind
    fc = np.fft.fft(comp, axis=-1)
    fk = np.fft.fft(cb[1], axis=-1)
    rec_np = np.real(np.fft.ifft(fc * np.conj(fk), axis=-1)).astype(np.float32)
    rec_grilly = vsa.unbind(comp, cb[1])
    assert np.allclose(rec_np, rec_grilly, atol=1e-4)


def _continuous(seed):
    """A per-block probability distribution — the shape bundling actually produces."""
    rng = np.random.default_rng(seed)
    v = rng.random((K, L)).astype(np.float32)
    return v / v.sum(axis=-1, keepdims=True)


def test_bind_of_continuous_codes_is_not_the_identity():
    """REGRESSION GUARD — do not delete without reading this.

    grilly ships TWO functions named blockcode_bind with DIFFERENT semantics:
      * grilly.experimental.vsa.block_ops.BlockCodeOps.bind — true per-block
        circular convolution (what this package must use), and
      * grilly.backend._bridge.blockcode_bind (GPU shader + numpy fallback) —
        a one-hot ARGMAX SHIFT.

    The second only works on discrete one-hot input. On a continuous block code
    every entry is < 0.5 (each block sums to 1 across l slots), so its
    `if (a[i] > 0.5) hot_a = i` scan finds nothing, both indices stay 0, and it
    returns the IDENTITY ELEMENT — silently, without raising. `BlockCodeVSA`
    is safe today only because it never dispatches to `_bridge`. If anyone
    "fixes" the unused-bridge-tier warning by wiring dispatch through, this test
    is what catches it. Verified against grilly's source 2026-08-02.
    """
    vsa = _vsa()
    out = vsa.bind(_continuous(0), _continuous(1))
    assert not np.allclose(out, vsa.zero()), (
        "bind() collapsed to the identity element on continuous input — the "
        "backend is using one-hot argmax-shift semantics, not circular convolution"
    )


def test_unbind_recovers_the_right_filler_from_a_bundle():
    """The actual use case, previously untested: every other test here binds a
    single pair of one-hot codes, which is the one regime where the broken and
    correct implementations agree. Superposition is where they diverge."""
    vsa = _vsa()
    n = 3
    cb = vsa.codebook(2 * n, orthogonal=False)   # independent atoms, per H-B5
    roles, fillers = cb[:n], cb[n:]
    composite = vsa.bundle([vsa.bind(roles[i], fillers[i]) for i in range(n)])
    for i in range(n):
        rec = vsa.unbind(composite, roles[i])
        sims = [vsa.similarity(rec, fillers[j]) for j in range(n)]
        assert int(np.argmax(sims)) == i, f"unbind({i}) retrieved filler {np.argmax(sims)}"


def test_numpy_fallback_matches_grilly_on_continuous_input():
    """test_numpy_fallback_matches_grilly only compares one-hot codes, so it
    cannot detect a backend whose semantics diverge on continuous input — which
    is precisely how the two grilly paths differ."""
    vsa = _vsa()
    if vsa._ops is None:
        pytest.skip("grilly not available; only numpy path exists")
    a, b = _continuous(2), _continuous(3)
    fa, fb = np.fft.fft(a, axis=-1), np.fft.fft(b, axis=-1)
    ref = np.real(np.fft.ifft(fa * fb, axis=-1)).astype(np.float32)
    assert np.allclose(ref, vsa.bind(a, b), atol=1e-4)


def test_wiring_is_wired():
    assert mod.__wiring__ is Wiring.WIRED
