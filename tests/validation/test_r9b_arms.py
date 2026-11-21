"""Order 0046 R9b: each faster arm computes BlockCodes' own arithmetic (R9b's "equal results"), pinned on seeded samples."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "validation"))
pytest.importorskip("numba")
from retrieval_bench import r9_vectors as R  # noqa: E402
from retrieval_bench import r9b_arms as A  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(R.CUBEMIND), reason="the cubemind checkout is not beside this repo")


@pytest.fixture(params=[(80, 128), (16, 32)])
def setup(request):
    k, l = request.param
    bc = R.block_codes(k, l)
    return bc, bc.codebook_discrete(20, seed=11), k, l


def test_scipy_and_spectral_bind_unbind_equal_blockcodes(setup):
    bc, cb, k, l = setup
    frame = bc.bundle([bc.bind(cb[i], cb[i + 1]) for i in range(0, 8, 2)])
    for a, b in ((cb[0], cb[1]), (frame, cb[3])):
        ref = bc.bind(a, b)
        assert np.allclose(A.scipy_bind(a, b), ref, atol=1e-5)
        assert np.allclose(A.from_spec(A.spec_bind(A.to_spec(a), A.to_spec(b)), l), ref, atol=1e-5)
        refu = bc.unbind(a, b)
        assert np.allclose(A.scipy_unbind(a, b), refu, atol=1e-5)
        assert np.allclose(A.from_spec(A.spec_unbind(A.to_spec(a), A.to_spec(b)), l), refu, atol=1e-5)


def test_bundle_and_similarity_equal_blockcodes_dense_and_spectral(setup):
    bc, cb, k, l = setup
    vs = cb[:6]
    ref = bc.bundle(list(vs))
    assert np.allclose(A.bundle_dense(vs), ref, atol=1e-6)
    assert np.allclose(A.from_spec(A.spec_bundle(A.to_spec(vs)), l), ref, atol=1e-5)
    for a, b in ((cb[0], cb[0]), (cb[0], cb[1]), (ref, cb[2])):
        s = bc.similarity(a, b)
        assert abs(A.similarity_dense(a, b) - s) < 1e-6
        assert abs(A.spec_similarity(A.to_spec(a), A.to_spec(b), l) - s) < 1e-5
    batch = A.spec_similarity_batch(A.to_spec(ref), A.to_spec(cb), l)
    assert np.allclose(batch, bc.similarity_batch(ref, cb), atol=1e-5)


def test_the_index_form_is_exact_for_one_hot(setup):
    bc, cb, k, l = setup
    ix = cb.argmax(axis=-1).astype(np.uint8 if l <= 256 else np.uint16)
    one_hot = lambda v: np.eye(l, dtype=np.float32)[v]
    assert np.array_equal(one_hot(A.idx_bind(ix[0], ix[1], l)), bc.bind(cb[0], cb[1]).round())
    assert np.array_equal(one_hot(A.idx_unbind(ix[2], ix[1], l)), bc.unbind(cb[2], cb[1]).round())
    assert A.idx_similarity(ix[0], ix[1]) == pytest.approx(bc.similarity(cb[0], cb[1]))
    frame = bc.bundle([bc.bind(cb[i], cb[i + 1]) for i in range(0, 8, 2)])
    assert A.frame_against_key(frame, ix[3]) == pytest.approx(bc.similarity(frame, cb[3]), abs=1e-6)
    assert np.allclose(A.frame_shift(frame, ix[3]), bc.bind(frame, cb[3]), atol=1e-5)
    assert np.allclose(A.idx_scores(ix[4], ix), bc.similarity_batch(cb[4], cb), atol=1e-6)


def test_the_vote_bundle_keeps_a_majority_and_ties_to_the_lowest():
    keys = np.array([[3, 5], [3, 1], [7, 1]], dtype=np.uint8)
    assert A.idx_bundle(keys, 128).tolist() == [3, 1]
    assert A.idx_bundle(np.array([[9, 4], [2, 6]], dtype=np.uint8), 128).tolist() == [2, 4]


def test_map_packed_scores_equal_the_cosine():
    rng = np.random.default_rng(0)
    v = rng.choice(np.array([-1, 1], dtype=np.int8), size=(50, 640))
    got = A.map_scores_packed(A.map_pack(v[7]), A.map_pack(v), 640)
    assert np.allclose(got, (v.astype(np.int32) @ v[7].astype(np.int32)) / 640)
    assert A.top(got, 3)[0] == 7
