"""The fast word-table encoder's port (order 0033): ``cubbyllm.worlds.fast_words`` gives mowm's output, bit for bit.

Pinned against ``mowm/encoding/semantic_words.py`` loaded from the mowm checkout beside this one (skipped when it is
not there): the words a text splits into, each word's signal code, the projection, a built table, the encoding of a
fixed list of texts, and each module loading the other's saved table. The production table is pinned by
``validation/pin_fast_words.py`` (1.1 GB: not a unit test's).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.local_paths import expand  # noqa: E402
from cubbyllm.worlds import fast_words as port  # noqa: E402

MOWM = pathlib.Path(expand("$CUBBY_GITHUB/mowm/mowm/encoding/semantic_words.py"))
WORDS = ["capital", "france", "paris", "river", "flows", "through", "cairo", "currency", "japan", "yen", "who", "wrote",
         "pride", "prejudice", "globe", "theatre", "boston", "southwark", "population", "toronto"]
TEXTS = ["What is the capital of France?", "Which river flows through Cairo?", "the currency of Japan is the yen",
         "Where is the Globe Theatre in Boston?", "zyxwv unseen words only", "", "a", "Population: Toronto, 2019 (est.)"]


@pytest.fixture(scope="module")
def mowm():
    if not MOWM.exists():
        pytest.skip("no mowm checkout beside this one")
    spec = importlib.util.spec_from_file_location("mowm_semantic_words", MOWM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _embed(words):                                       # a fixed teacher: a seeded vector per word
    out = []
    for w in words:
        rng = np.random.default_rng(sum(ord(c) * 31 ** i for i, c in enumerate(w)) % (2**32))
        out.append(rng.standard_normal(48))
    return np.asarray(out, dtype=np.float32)


def _pair(mod, k=16, l=32):                              # noqa: E741
    idf = {w: 1.0 + i / 10 for i, w in enumerate(WORDS)}
    return mod.FastWordEncoder.build(WORDS, _embed, idf, 2.5, k=k, l=l, proj_seed=3, meta={"test": True})


def test_the_pieces_are_mowms(mowm):
    for t in TEXTS:
        assert port.split_words(t) == mowm.split_words(t)
    for w in WORDS + ["zyxwv", "é"]:
        assert np.array_equal(port.signal_code(w, 80, 128), mowm.signal_code(w, 80, 128))
    assert np.array_equal(port.orthonormal_projection(48, 16, 32, seed=3), mowm.orthonormal_projection(48, 16, 32, seed=3))
    E = _embed(WORDS)
    for a, b in zip(port.fit_anisotropy(E), mowm.fit_anisotropy(E)):
        assert np.array_equal(a, b)


def test_a_built_table_encodes_as_mowms_does(mowm):
    a, b = _pair(port), _pair(mowm)
    assert a.meta == b.meta and (a.k, a.l, a.default_idf) == (b.k, b.l, b.default_idf)
    for t in TEXTS:
        x, y = a.encode(t), b.encode(t)
        assert x.dtype == y.dtype == np.float32 and x.shape == (16, 32) and np.array_equal(x, y), t


def test_each_loads_the_others_saved_table(mowm, tmp_path):
    _pair(mowm).save_npz(tmp_path / "m.npz", dtype=np.float16)
    _pair(port).save_npz(tmp_path / "p.npz", dtype=np.float16)
    a, b = port.FastWordEncoder.from_npz(tmp_path / "m.npz"), mowm.FastWordEncoder.from_npz(tmp_path / "p.npz")
    for t in TEXTS:
        assert np.array_equal(a.encode(t), b.encode(t)), t


def test_an_unseen_word_is_its_signal_code_at_the_default_idf():
    enc = _pair(port)
    assert np.allclose(enc.encode("zyxwv").ravel(), 2.5 * port.signal_code("zyxwv", 16, 32))
    assert not enc.encode("").any()
