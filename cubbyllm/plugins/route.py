"""standin/plugins_route.py -- which plugin a question would wake (order 0028).

Two routers, one interface (``scores(question)``, ``wake(question, disabled)``):
- ``Router`` (the default): each plugin's ``description`` and ``examples`` are one line each in a ``worlds.FactStore``,
  the store worlds are routed with -- the brain's encoder when one is given (unit cosine), its IDF-weighted word
  overlap otherwise. A question's score for a plugin is its best line's; a plugin whose score is at least ``tau``
  wakes. ``tau`` started at ``Worlds.TAU`` (0.35) and is tuned on the plugins' ``examples`` / ``not`` lines, written
  first (the routing test: ``validation/plugins_routing.py``): 0.4 with the serve stack's encoder, 0.475 with word
  overlap alone.
- ``LogisticRouter`` (``CUBBY_PLUGINS_ROUTER=minilm``, review addition 4): a question encoder
  (``question_encoder.QuestionEncoder``) and, per plugin, an L2 logistic head fitted on its description and examples
  against every other line -- its own ``not`` lines and the other plugins' lines. A question's score is the head's
  probability; ``tau`` 0.675 is the lowest where no plugin's near misses wake it more than 10% of the time, read
  leave one out (every line scored by heads fitted without it): 103/120 new phrasings wake their plugin, none wakes
  another (``validation/logs/plugins_routing_minilm_logistic_2026-10-22.json``). The heads are fitted in a background
  thread from mount (``make_router``; 1.4 s for the four proof plugins' 244 lines), never at startup; a question
  arriving first waits for them. About 5 ms a question after that.
A disabled plugin never wakes; a pinned one is always loaded and needs no score.

Wired: WIRED (``plugins_load.Loader.wake``, through ``make_router``).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

__wiring__ = "WIRED"

from ..worlds.base import FactStore, Worlds  # noqa: E402

TAU_START = Worlds.TAU               # where the tuning started (0.35)
# tuned on the four proof plugins' examples / not lines (validation/logs/plugins_routing_enc_2026-10-22.json and
# plugins_routing_2026-10-22.json): the lowest tau where no plugin's not-lines wake it more than once in ten
TAU_ENCODER, TAU_OVERLAP = 0.4, 0.475
TAU_PLUGIN = TAU_ENCODER
TAU_LOGISTIC, C_LOGISTIC = 0.675, 4.0  # C: the heads' inverse L2 strength (sklearn's convention), set before tuning


def _key(line: str) -> str:
    return " ".join(line.split())


class Router:
    def __init__(self, index, enc=None, tau: float = TAU_PLUGIN, leave_out: str | None = None) -> None:
        """``leave_out``: a line not to index (the routing test's leave-one-out: an example scored against the others)."""
        self.tau = float(tau)
        self.owner: dict[str, str] = {}
        lines = []
        for p in index.plugins.values():
            for line in [p.description] + list(p.examples):
                key = _key(line)
                if key and key != leave_out and key not in self.owner:
                    self.owner[key] = p.name
                    lines.append(key)
        self.store = FactStore(lines, enc=enc, name="plugins")

    def scores(self, question: str, k: int = 32) -> dict[str, float]:
        """Each plugin's best line score for ``question`` (plugins with no matching line are left out)."""
        out: dict[str, float] = {}
        for s, line in self.store(question, k):
            name = self.owner.get(_key(line))
            if name is not None and s > out.get(name, 0.0):
                out[name] = float(s)
        return out

    def wake(self, question: str, disabled=()) -> list[tuple[str, float]]:
        """The plugins this question wakes, best first."""
        got = [(n, s) for n, s in self.scores(question).items() if s >= self.tau and n not in disabled]
        return sorted(got, key=lambda x: (-x[1], x[0]))


def logistic(X: np.ndarray, y: np.ndarray, C: float = C_LOGISTIC, iters: int = 50) -> np.ndarray:
    """Weights (last = intercept) minimising 0.5|w|^2 + C * sum(s_i * logloss_i), class-balanced s_i, the intercept
    unpenalised -- sklearn's ``LogisticRegression(class_weight="balanced")`` objective, by Newton steps."""
    Xa = np.hstack([X, np.ones((len(X), 1))]).astype(np.float64)
    y = y.astype(np.float64)
    n1 = max(y.sum(), 1.0)
    n0 = max(len(y) - y.sum(), 1.0)
    s = np.where(y > 0, len(y) / (2 * n1), len(y) / (2 * n0))
    r = np.ones(Xa.shape[1])
    r[-1] = 0.0
    w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xa @ w, -30, 30)))
        g = r * w + C * Xa.T @ (s * (p - y))
        Hm = np.diag(r + 1e-9) + C * (Xa * (s * p * (1 - p))[:, None]).T @ Xa
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class LogisticRouter:
    def __init__(self, index, enc, tau: float = TAU_LOGISTIC, leave_out: str | None = None, C: float = C_LOGISTIC) -> None:
        """``leave_out``: a line not to fit on (the routing test's leave-one-out, examples and not-lines alike)."""
        self.tau, self.enc, self.C = float(tau), enc, float(C)
        self.lines: list[tuple[str, str, bool]] = []          # (line, plugin, an example of it?)
        for p in index.plugins.values():
            for line, pos in [(p.description, True)] + [(x, True) for x in p.examples] + [(x, False) for x in p.not_]:
                key = _key(line)
                if key and key != leave_out:
                    self.lines.append((key, p.name, pos))
        self.names = sorted(index.plugins)
        self._W = None
        self._lock = threading.Lock()

    def _fit(self) -> None:
        with self._lock:
            if self._W is not None:
                return
            X = np.stack([self.enc.encode(t).reshape(-1) for t, _n, _p in self.lines])
            self._W = {n: logistic(X, np.array([o == n and pos for _t, o, pos in self.lines]), self.C)
                       for n in self.names if any(o == n and pos for _t, o, pos in self.lines)}

    def prefit(self) -> threading.Thread:
        """Fit the heads in a daemon thread (the server's mount); ``scores`` waits on the same lock."""
        t = threading.Thread(target=self._fit, name="plugins-route-fit", daemon=True)
        t.start()
        return t

    def scores(self, question: str) -> dict[str, float]:
        if self._W is None:
            self._fit()
        x = np.append(self.enc.encode(question).reshape(-1), 1.0)
        return {n: float(1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))) for n, w in self._W.items()}

    wake = Router.wake


def make_router(index, enc=None, tau: float | None = None, kind: str | None = None):
    """``kind`` ("minilm" / "nearest"; None reads ``CUBBY_PLUGINS_ROUTER``): "minilm", the question encoder and
    logistic heads (the server's encoder unused; the serve stack's default since 0028's review 2); otherwise the
    nearest-line ``Router`` exactly as before. "minilm" without the encoder in the cache falls back, and says so."""
    if (kind or os.environ.get("CUBBY_PLUGINS_ROUTER", "")).lower() == "minilm":
        from .question_encoder import QuestionEncoder
        qe = QuestionEncoder.find()
        if qe is not None:
            r = LogisticRouter(index, qe, tau=TAU_LOGISTIC if tau is None else tau)
            r.prefit()
            return r
        print("[plugins] CUBBY_PLUGINS_ROUTER=minilm but the encoder is not in the cache: nearest-line routing")
    return Router(index, enc=enc, tau=tau if tau is not None else (TAU_ENCODER if enc is not None else TAU_OVERLAP))
