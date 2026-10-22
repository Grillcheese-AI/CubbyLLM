"""standin/capsules.py -- facts as capsules: provenance and use, a code to find them by resemblance.

Pinned: a capsule remembers its world, source and use; a misspelt name finds the right fact first and an
unrelated query sits near 0.5; the code is the same in every process (pinned digest); a saved store loads
back identical and refuses codes it cannot recompute; grilly2's GPU path and numpy agree to the integer;
two worlds never share capsules; priority rises with verified use and only verified capsules are offered
to the sleep cycle.
Run: python -m pytest standin/tests/test_capsules.py -q
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from capsules import CapsuleStore, _grilly_packed, encode, hamming_numpy  # noqa: E402

FACTS = ["Jim Haslam is the father of Bill Haslam", "Queretaro City is in Mexico",
         "The Bank of England was founded in 1694", "Groove Billed Ani is a member of Congo Free State",
         "Georg Baselitz is a German painter"]


def test_a_capsule_remembers_where_and_how_it_was_used():
    s = CapsuleStore(name="wiki", gpu=False)
    assert s.add(FACTS[0], source="wikidata", when=100.0)
    assert not s.add("  Jim Haslam is the father   of Bill Haslam ")        # the same fact, once
    m = s.meta[FACTS[0]]
    assert (m["world"], m["source"], m["added"], m["uses"], m["verified"]) == ("wiki", "wikidata", 100.0, 0, 0)
    s.touch(FACTS[0], verified=True, when=200.0)
    s.touch(FACTS[0], when=300.0)
    assert (m["uses"], m["verified"], m["last"]) == (2, 1, 300.0)
    assert s.codebook().shape == (1, 32)


def test_a_misspelt_name_finds_the_right_fact_first():
    s = CapsuleStore(FACTS, gpu=False)
    (top, fact), *_ = s.similar("who is Bill Haslem's father", k=3)
    assert fact == FACTS[0] and top > 0.6
    unrelated = s.similar("photosynthesis in green algae", k=5)
    assert all(0.4 < sim < 0.6 for sim, _ in unrelated)                    # unrelated codes sit near 0.5


def test_the_code_is_the_same_in_every_process():
    """Pinned: a stored code must resolve tomorrow. If this fails, the key or the features changed and
    every saved store needs its codes rebuilt."""
    code = encode("Jim Haslam is the father of Bill Haslam")
    assert code.dtype == np.int32 and code.shape == (32,)
    assert hashlib.sha1(code.tobytes()).hexdigest() == PINNED


def test_a_saved_store_loads_back_identical(tmp_path):
    s = CapsuleStore(FACTS, name="wiki", gpu=False)
    s.provenance[FACTS[1]] = "wikidata"
    s.times[FACTS[2]] = {"point": "1694-07-27"}
    s.touch(FACTS[3], verified=True, when=5.0)
    s.save(tmp_path / "wiki")
    t = CapsuleStore.load(tmp_path / "wiki", gpu=False)
    assert t.texts == s.texts and t.meta == s.meta
    assert t.provenance[FACTS[1]] == "wikidata" and t.times[FACTS[2]] == {"point": "1694-07-27"}
    assert np.array_equal(t.codebook(), s.codebook())
    assert t.similar("Bank of England", 2) == s.similar("Bank of England", 2)
    np.save(tmp_path / "wiki" / "codes.npy", s.codebook() ^ 1)            # a code from another key
    with pytest.raises(ValueError):
        CapsuleStore.load(tmp_path / "wiki", gpu=False)


@pytest.mark.skipif(_grilly_packed() is None, reason="needs grilly2 (grilly.vsa.packed)")
def test_grilly2_and_numpy_agree_to_the_integer():
    from grilly.vsa import packed
    import grilly
    rng = np.random.default_rng(0)
    book = rng.integers(-2**31, 2**31, size=(3000, 32), dtype=np.int64).astype(np.int32)
    q = book[17] ^ rng.integers(0, 2, size=32, dtype=np.int64).astype(np.int32)
    gpu = packed.hamming(grilly.from_numpy(q), grilly.from_numpy(book)).numpy()
    assert np.array_equal(gpu, hamming_numpy(q, book))
    g, c = CapsuleStore(FACTS, gpu=True), CapsuleStore(FACTS, gpu=False)
    for query in ("Bill Haslem", "Mexico", "painter from Germany"):
        assert [round(x, 9) for x, _ in g.similar(query, 5)] == [round(x, 9) for x, _ in c.similar(query, 5)]
        assert g.similar(query, 1) == c.similar(query, 1)


def test_worlds_do_not_share_capsules():
    host, plugin = CapsuleStore(FACTS[:2], name="host", gpu=False), CapsuleStore(["Cubby Man eats pellets"], name="game", gpu=False)
    assert all(t in FACTS for _, t in host.similar("pellets", 5))
    assert [t for _, t in plugin.similar("Bill Haslam", 5)] == ["Cubby Man eats pellets"]
    assert {m["world"] for m in host.meta.values()} == {"host"}


def test_priority_rises_with_verified_use_and_only_verified_capsules_are_offered():
    s = CapsuleStore(FACTS, gpu=False)
    now = s.meta[FACTS[0]]["added"]
    s.touch(FACTS[1], verified=True, when=now)
    s.touch(FACTS[2], when=now)
    s.touch(FACTS[2], when=now)
    assert s.priority(FACTS[1], now) == 4.0 and s.priority(FACTS[2], now) == 3.0 and s.priority(FACTS[0], now) == 1.0
    assert s.priority(FACTS[1], now + 30 * 86400) == pytest.approx(2.0)   # halves in 30 days unused
    assert s.candidates(5, now) == [FACTS[1]]


PINNED = "c18b90db003ab6d37a5623ae3681d3e72a4101f4"
