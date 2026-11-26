"""Order 0083: the trunk counts its decode steps per role and head -- route 5 when grilly2 picks tokens with it (a route
attached, greedy, one row, no processor: grilly2's own rule, ``GenerationMixin._shortlist_for``), the full head and why
otherwise -- and the count changes nothing the decode returns. The serve switches parse."""
from __future__ import annotations

import collections
import threading

from cubbyllm.emit import seats_450m as S
from cubbyllm.ops import grilly_seats as gs


def trunk(monkeypatch):
    t = object.__new__(S.Cubby450mTrunk)                 # no export: the decode is faked below
    t.steps, t.route, t._lock, t._model, t._eos, t._grilly = collections.Counter(), None, threading.Lock(), object(), 0, None
    t.use = lambda role: None
    monkeypatch.setattr(gs, "generate", lambda model, rows, n, eos, **kw: [list(r) + [7] * n for r in rows])
    return t


def test_steps_are_counted_by_head_and_why(monkeypatch):
    t = trunk(monkeypatch)
    assert t.generate("programs", [[1, 2]], 3) == [[7, 7, 7]]
    assert t.steps == {"programs|full head: off": 3}
    t.route = object()
    t.generate("programs", [[1, 2]], 4)
    t.generate("programs", [[1], [2]], 2)
    t.generate("talk", [[1]], 5, temperature=0.7)
    t.generate("programs", [[1]], 1, logits_processor=object())
    assert t.steps == {"programs|full head: off": 3, "programs|route 5": 4, "programs|full head: batch": 4,
                       "talk|full head: sampled": 5, "programs|full head: processor": 1}


def test_the_serve_switches_parse():
    from cubby_serve.cli import parser
    a = parser().parse_args(["--cubby450m", "x", "--memory", "sleep_dir", "--head-route", "route.npz"])
    assert a.memory == "sleep_dir" and a.head_route == "route.npz"
    b = parser().parse_args(["--cubby450m", "x"])
    assert b.memory is None and b.head_route is None
