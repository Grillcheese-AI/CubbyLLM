"""Order 0046 R9: the bench's shortcuts are cubemind's own BlockCodes functions, pinned on samples."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "validation"))
from retrieval_bench import r9_vectors as R  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(R.CUBEMIND), reason="the cubemind checkout is not beside this repo")


def test_the_drawn_indices_are_codebook_discretes():
    bc = R.block_codes(16, 32)
    cb = bc.codebook_discrete(20, seed=7)
    assert np.array_equal(R.discrete_indices(20, 16, 32, 7), cb.argmax(axis=-1))


def test_recall_similarity_is_similarity_batch():
    bc = R.block_codes(16, 32)
    cb = bc.codebook_discrete(30, seed=3)
    idx = cb.argmax(axis=-1)
    q = cb[5].copy()
    q[:4] = 0
    q[:4, 0] = 1                                         # four blocks re-drawn to index 0
    qi = q.argmax(axis=-1)
    assert np.allclose((idx == qi).mean(axis=1), bc.similarity_batch(q, cb))


def test_roles_cleanup_is_similarity_batch():
    bc = R.block_codes(16, 32)
    fillers = bc.codebook_discrete(50, seed=1)
    roles = bc.codebook_discrete(3, seed=2)
    frame = bc.bundle([bc.bind(roles[i], fillers[i]) for i in range(3)])
    u = bc.unbind(frame, roles[0])
    assert np.allclose(u.reshape(1, -1) @ fillers.reshape(50, -1).T / 16, bc.similarity_batch(u, fillers), atol=1e-5)


def test_block_noise_keeps_the_expected_similarity():
    row = R.block_recall(80, 500, 0.5, 0)
    assert abs(row["query_key_sim"] - 0.5) < 0.02 and row["recall_at_1"] > 0.99
