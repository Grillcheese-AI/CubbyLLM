"""The place encoding's theorems, each checked to float precision (cubbyllm/spatial/ssp.py).

What a shape test would miss: a phase table that is not conjugate-symmetric (S(p) complex, the
identity not a delta), a bind that is correlation rather than convolution (S(p) (*) S(d) =
S(p - d)), a box whose geometric series drops the theta = 0 rows, a kernel that depends on p
and q separately (no translation invariance), a decode that is only approximate on the lattice.
"""
from __future__ import annotations

import numpy as np
import pytest

from cubbyllm.spatial.ssp import SSP

RNG = np.random.default_rng(0)
TOL = 1e-12


@pytest.fixture(scope="module")
def ssp():
    return SSP(d=256, dims=3, scale=2.0, seed=3)


def test_a_place_is_a_real_unit_vector_and_the_phase_table_is_fixed_by_its_seed(ssp):
    p = RNG.uniform(-50, 50, size=(5, 3))
    v = ssp.encode(p)
    assert v.shape == (5, 256) and v.dtype == np.float64
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=TOL)
    assert np.array_equal(SSP(256, 3, 2.0, 3).phases, ssp.phases)
    assert not np.array_equal(SSP(256, 3, 2.0, 4).phases, ssp.phases)
    assert ssp.phases[0].tolist() == [0, 0, 0] and ssp.phases[-1].tolist() == [0, 0, 0]


def test_moving_is_binding(ssp):
    """S(p + d) = S(p) (*) S(d), for any p and d, fractional included."""
    p, d = RNG.uniform(-30, 30, size=3), RNG.uniform(-30, 30, size=3)
    assert np.abs(ssp.encode(p + d) - ssp.bind(ssp.encode(p), ssp.encode(d))).max() < TOL
    assert np.abs(ssp.encode(p - d) - ssp.unbind(ssp.encode(p), ssp.encode(d))).max() < TOL


def test_the_origin_is_the_identity(ssp):
    """S(0) is the delta vector: binding anything with it changes nothing."""
    e = ssp.identity()
    assert e[0] == pytest.approx(1.0) and np.abs(e[1:]).max() < TOL
    x = RNG.standard_normal(256)
    assert np.abs(ssp.bind(x, e) - x).max() < TOL
    assert np.abs(ssp.unbind(ssp.bind(x, ssp.encode([3.0, -1.0, 2.0])), ssp.encode([3.0, -1.0, 2.0])) - x).max() < TOL


def test_similarity_is_a_kernel_of_the_displacement_falling_with_distance(ssp):
    p, q = RNG.uniform(-30, 30, size=3), RNG.uniform(-30, 30, size=3)
    assert ssp.similarity(ssp.encode(p), ssp.encode(q)) == pytest.approx(ssp.kernel(q - p), abs=TOL)
    shift = RNG.uniform(-100, 100, size=3)
    assert ssp.similarity(ssp.encode(p + shift), ssp.encode(q + shift)) == pytest.approx(
        ssp.similarity(ssp.encode(p), ssp.encode(q)), abs=TOL)
    along = np.array([[r, 0.0, 0.0] for r in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)])
    k = ssp.kernel(along)
    assert k[0] == pytest.approx(1.0, abs=TOL)
    assert k[1] > k[2] > 0.5 > abs(k[3]) and np.all(np.abs(k[3:]) < 0.2)       # scale 2: a cell apart ~0.6, two apart ~0
    assert ssp.kernel([2.0, 0.0, 0.0]) == pytest.approx(ssp.kernel([-2.0, 0.0, 0.0]), abs=TOL)  # even


def test_a_box_is_the_closed_form_of_its_cells(ssp):
    lo, hi = np.array([-2, 0, 1]), np.array([4, 2, 3])
    cells = [[x, y, z] for x in range(-2, 5) for y in range(0, 3) for z in range(1, 4)]
    assert np.abs(ssp.box(lo, hi) - ssp.region(cells)).max() < 1e-10
    single = ssp.box([5, 5, 5], [5, 5, 5])
    assert np.abs(single - ssp.encode([5.0, 5.0, 5.0])).max() < TOL
    with pytest.raises(ValueError):
        ssp.box([1, 0, 0], [0, 0, 0])


def test_a_place_reads_back_exactly_on_the_lattice(ssp):
    cells = np.array([[x, y, z] for x in range(-6, 7) for y in range(-6, 7) for z in range(0, 4)], float)
    for _ in range(20):
        p = cells[RNG.integers(len(cells))]
        got, s = ssp.decode(ssp.encode(p), cells)
        assert np.array_equal(got, p) and s == pytest.approx(1.0, abs=TOL)
    # at scale 1 the lattice kernel is nearly a delta; at scale 2 neighbours are similar but never tie
    near = SSP(256, 3, 1.0, 3)
    assert abs(near.kernel([1.0, 0.0, 0.0])) < 0.2


def test_the_constructor_refuses_odd_or_tiny_widths():
    with pytest.raises(ValueError):
        SSP(d=255)
    with pytest.raises(ValueError):
        SSP(d=2)
    with pytest.raises(ValueError):
        SSP(d=64, dims=3).encode([1.0, 2.0])
