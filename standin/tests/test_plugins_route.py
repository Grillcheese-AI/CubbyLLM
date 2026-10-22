"""Order 0028 review, addition 4: the logistic router.
- the Newton fit reaches sklearn's balanced, L2 logistic optimum (sklearn only as the reference);
- with a stand-in encoder, a plugin's heads wake it on its kind of question and not on its not-lines, and nothing is
  fitted until the first question;
- unset, ``CUBBY_PLUGINS_ROUTER`` leaves the nearest-line router exactly as before;
- the question encoder (when the cache has it): unit vectors, the same text the same vector, a paraphrase closer than
  a near miss.
Run: python -m pytest standin/tests/test_plugins_route.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import plugins_route as route  # noqa: E402
from plugins_index import discover  # noqa: E402


class Bag:
    """A stand-in encoder: hashed bag of words, unit length."""

    def __init__(self, dim=64):
        self.dim, self.calls = dim, 0

    def encode(self, text):
        self.calls += 1
        v = np.zeros(self.dim, np.float32)
        for w in text.lower().replace("?", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) + 1e-9)


def two(tmp_path):
    for name, ex, nots in (("weather", ["storm warning in town", "any storm alert today", "is a storm warning out"],
                            ["what causes a storm", "storm history book"]),
                           ("units", ["convert miles to km", "how many km in miles", "convert pounds to kg"],
                            ["how far is the moon", "who invented miles"])):
        d = tmp_path / name
        d.mkdir()
        (d / "PLUGIN.md").write_text(f"---\nname: {name}\ndescription: {name} things\nexamples: {ex}\nnot: {nots}\n"
                                     "kinds: [tool]\n---\n", encoding="utf-8")
    return discover([tmp_path])


def test_newton_fit_matches_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12))
    y = (X[:, 0] + 0.5 * rng.normal(size=80) > 0.8).astype(float)           # unbalanced on purpose
    w = route.logistic(X, y, C=4.0)
    ref = sk.LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000, tol=1e-10).fit(X, y)
    assert np.allclose(w[:-1], ref.coef_[0], atol=1e-4) and abs(w[-1] - ref.intercept_[0]) < 1e-4


def test_heads_route_and_fit_lazily(tmp_path):
    idx, enc = two(tmp_path), Bag()
    r = route.LogisticRouter(idx, enc, tau=0.5)
    assert enc.calls == 0                                               # nothing fitted at startup
    assert [n for n, _ in r.wake("is there a storm warning tonight")] == ["weather"]
    assert [n for n, _ in r.wake("convert 5 miles to km")] == ["units"]
    s = r.scores("what causes a storm")
    assert s["weather"] < 0.5                                           # its own not-line, a negative in training
    assert r.wake("is there a storm warning tonight", disabled={"weather"}) == []


def test_unset_flag_keeps_the_nearest_line_router(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBBY_PLUGINS_ROUTER", raising=False)
    idx = two(tmp_path)
    r = route.make_router(idx)
    assert type(r) is route.Router and r.tau == route.TAU_OVERLAP
    ref = route.Router(idx, tau=route.TAU_OVERLAP)
    for q in ("storm warning in town", "convert miles to km", "hello"):
        assert r.wake(q) == ref.wake(q)


def test_question_encoder():
    from question_encoder import QuestionEncoder
    qe = QuestionEncoder.find()
    if qe is None:
        pytest.skip("paraphrase-MiniLM-L6-v2 is not in the Hugging Face cache")
    a, b, c = (qe.encode(t) for t in ("is there a storm warning in Halifax?", "any weather alerts for Halifax today?",
                                      "how far is Halifax from Toronto?"))
    assert abs(float(np.linalg.norm(a)) - 1) < 1e-5 and np.array_equal(qe._encode("is there a storm warning in Halifax?"), a)
    assert float(a @ b) > float(a @ c) + 0.1


def test_the_flag_fits_in_the_background(tmp_path, monkeypatch):
    import question_encoder
    monkeypatch.setenv("CUBBY_PLUGINS_ROUTER", "minilm")
    monkeypatch.setattr(question_encoder.QuestionEncoder, "find", classmethod(lambda cls, repo=None: Bag()))
    r = route.make_router(two(tmp_path))
    assert isinstance(r, route.LogisticRouter) and r.tau == route.TAU_LOGISTIC
    assert [n for n, _ in r.wake("is there a storm warning tonight")] == ["weather"]   # waits for the fit if needed
