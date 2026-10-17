"""The curve's theorems (cubbyllm/spatial/hilbert.py): a bijection on the cube, consecutive
cells share a face, and a run of it fills a cube where a run of row-major order is a slab.
"""
from __future__ import annotations

import numpy as np
import pytest

from cubbyllm.spatial import hilbert as H


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_the_curve_is_a_bijection_whose_consecutive_cells_share_a_face(bits):
    n = 1 << (3 * bits)
    idx = np.arange(n)
    pts = H.point(idx, bits, 3)
    assert pts.shape == (n, 3) and pts.min() == 0 and pts.max() == (1 << bits) - 1
    assert len(np.unique(pts, axis=0)) == n                              # every cell once
    assert np.array_equal(H.index(pts, bits), idx)                       # and back
    steps = np.abs(np.diff(pts, axis=0)).sum(axis=1)
    assert np.all(steps == 1)                                            # one face at a time
    assert pts[0].tolist() == [0, 0, 0]


def test_two_dimensions_work_too():
    pts = H.point(np.arange(16), 2, 2)
    assert np.all(np.abs(np.diff(pts, axis=0)).sum(axis=1) == 1) and len(np.unique(pts, axis=0)) == 16


def test_a_run_of_the_curve_is_a_cube_where_a_run_of_row_major_order_is_a_slab():
    """The locality the window needs: 8**j consecutive cells of the curve fill an aligned cube
    of side 2**j (the recursive construction's theorem), so the 512 tokens a window sees at
    once are an 8 x 8 x 8 block of the world; the same 512 cells in row-major order are a
    2 x 16 x 16 slab, and 8 of them a 1 x 1 x 8 line."""
    bits, side = 4, 16
    pts = H.point(np.arange(side ** 3), bits, 3)
    row = np.array([[x, y, z] for x in range(side) for y in range(side) for z in range(side)])
    for j in (1, 2, 3):
        run = 8 ** j
        for order, extent in ((pts, 2 ** j - 1), (row, None)):
            blocks = order.reshape(-1, run, 3)
            spans = blocks.max(axis=1) - blocks.min(axis=1)              # (n_blocks, 3)
            if extent is not None:
                assert np.all(spans == extent), (j, np.unique(spans, axis=0))
                assert np.all(blocks.min(axis=1) % (2 ** j) == 0)        # aligned
            else:
                assert np.all(spans[:, 2] == side - 1) if run >= side else np.all(spans[:, 2] == run - 1)
    print("512 consecutive cells: curve 8x8x8, row-major 2x16x16")


def test_order_and_curve_shift_a_box_to_the_origin_and_keep_only_its_cells():
    cells = np.array([[-3, 2, 0], [-2, 2, 0], [5, 9, 1], [-3, 3, 0], [-3, 2, 1]])
    ordered = H.order(cells)
    assert sorted(map(tuple, ordered)) == sorted(map(tuple, cells))
    assert ordered[0].tolist() == [-3, 2, 0]
    box = H.curve([-2, -2, 0], [5, 3, 2])
    assert box.shape == (8 * 6 * 3, 3) and len(np.unique(box, axis=0)) == len(box)
    assert np.all(box.min(axis=0) == [-2, -2, 0]) and np.all(box.max(axis=0) == [5, 3, 2])
    assert np.all(np.abs(np.diff(box, axis=0)).sum(axis=1) >= 1)
    assert H.order(np.zeros((0, 3), dtype=np.int64)).shape == (0, 3)
    with pytest.raises(ValueError):
        H.index([[1, 2, 9]], bits=3)
