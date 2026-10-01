"""Pins for validation/exp_he19_gate_a.py (H-E19 gate A's readout): a program that never closes, a step body
repeated 3+ times (a loop), and the cuts after each complete step that the oracle-stop read runs in the VM.
No cubelang needed. Run: python -m pytest standin/tests/test_exp_he19_gate_a.py -q -p no:hypothesispytest"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation"), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_he19_gate_a as ga  # noqa: E402

HEAD = "program X implements ISolver {\n    public function solve(input: Input): Output {\n"
STEP = "        create s{k} : quantity;   # step {k}\n        assign s{k} = $N1;\n        add s{k}, $N2;\n"
TAIL = "        sum s{k};\n        query s{k};\n        return s{k};\n    }}\n}}\n"


def prog(n, closed=True):
    body = "".join(STEP.format(k=k) for k in range(n))
    return HEAD + body + (TAIL.format(k=n - 1) if closed else "")


def test_closed_and_loop():
    assert ga.closed(prog(2)) and not ga.closed(prog(5, closed=False))
    assert not ga.looped(prog(2)) and ga.looped(prog(3))          # the same body, index stripped, 3 times


def test_cuts_close_on_each_complete_step():
    cuts = ga.prefixes(prog(3))
    assert [k for k, _ in cuts] == [1, 2, 3]
    assert all(ga.closed(p) for _, p in cuts)
    assert "return s0;" in cuts[0][1] and "create s1" not in cuts[0][1]
    # cut at the token cap: the last block may be partial, so it is dropped
    assert [k for k, _ in ga.prefixes(prog(3, closed=False))] == [1, 2]


def test_stops_reads_a_perfect_stop(monkeypatch):
    import exp_he18_ab as ab
    monkeypatch.setattr(ab, "vm_run", lambda p: (True, str(p.count("create "))))   # value = steps kept
    monkeypatch.setattr(ab, "value_correct", lambda r, g: str(r) == str(g))
    outs = [{"generated": prog(4), "reference": prog(2), "gold": "2"},     # right at the reference length
            {"generated": prog(4), "reference": prog(2), "gold": "3"},     # right only past it
            {"generated": prog(4), "reference": prog(2), "gold": "9"}]     # never
    st = ga.stops(outs)
    assert abs(st["at_ref_len"] - 1 / 3) < 1e-9 and abs(st["any_cut"] - 2 / 3) < 1e-9
    assert abs(st["first_step_right"] - 1 / 3) < 1e-9
