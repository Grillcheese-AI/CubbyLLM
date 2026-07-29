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


def test_wiring_is_wired():
    assert mod.__wiring__ is Wiring.WIRED
