"""Order 0067, the second reader's step 2: the program generator. Drawn specs never state a held-out test kind; the
compiled CubeLang uses only the forms the VM build runs; the VM's verdict is the Python evaluator's on every reply
(edges included: an item past the end fails, "all" of nothing holds); a view too long for the VM is refused, never
judged."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import view_compile as VC  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VR  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402


def specs(seed, n):
    rng, out = random.Random(seed), []
    while len(out) < n:
        reps = VR.pool(rng, 20)
        s = VP.sample(rng, [RV.views(t) for t in reps])
        if s:
            out.append((s, reps))
    return out


def test_held_out_forms_are_named_and_never_drawn():
    assert VP.held_out({"q": "none", "view": "char", "where": [{"col": "char", "op": "==", "v": ","}]})
    assert VP.held_out({"q": "count", "view": "word", "where": [{"col": "lower", "op": "==", "v": "x"}], "op": "==", "v": 0})
    assert VP.held_out({"q": "n", "view": "item", "op": "==", "v": 3})
    assert VP.held_out({"q": "at", "view": "paragraph", "pos": 2, "col": "first_word", "op": "==", "val": "so"})
    assert VP.held_out({"q": "at", "view": "word", "pos": 1, "col": "lower", "op": "==", "val": "so"})
    assert VP.held_out({"q": "count", "view": "word", "where": [{"col": "lower", "op": "==", "v": "x"}], "op": ">=", "v": 2}) is None
    for s, _ in specs(3, 60):
        assert VP.held_out(s) is None


def test_the_compiled_program_uses_only_forms_the_vm_runs():
    for s, _ in specs(4, 60):
        b = VC.body(s)
        assert not re.search(r"&&|\|\||!\(|else\s+if|\belse\b", b), b
        for cond in re.findall(r"if \(([^)]*)\)", b):
            assert "[" not in cond and "(" not in cond, cond
        assert b.strip().endswith(";") and "return c" in b


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    return cc.CubelangSession()


def verdict(vm, spec, text):
    return str(vm.run(VC.program(spec, text), fn="solve")["result"]).strip() in ("1", "1.0")


def test_the_vm_agrees_with_the_evaluator(vm):
    n = 0
    for s, reps in specs(5, 25):
        for t in reps[:6]:
            if all(len(RV.views(t)[v]) <= RV.MAX_ROWS for v in VC.views_of(s)):
                assert verdict(vm, s, t) == VP.evaluate(s, RV.views(t)), (s, t)
                n += 1
    assert n >= 60


def test_edges_past_the_end_and_all_of_nothing(vm):
    last_q = {"q": "at", "view": "sentence", "pos": 3, "col": "end_mark", "op": "==", "val": "?"}
    all_items = {"q": "all", "view": "item", "where": [{"col": "marker", "op": "==", "v": "-"}]}
    for text in ("", "One. Two?", "One. Two. Three?", "- a\n* b", "plain"):
        for s in (last_q, all_items, {"not": last_q}, {"unless": [last_q, all_items]}):
            assert verdict(vm, s, text) == VP.evaluate(s, RV.views(text)), (s, text)
    assert VP.evaluate(all_items, RV.views("plain")) is True
    assert VP.evaluate(last_q, RV.views("One. Two?")) is False


def test_a_view_too_long_for_the_vm_is_refused():
    long_text = " ".join(["word"] * (RV.MAX_ROWS + 1))
    with pytest.raises(RV.TooLong):
        RV.literals(long_text, ["word"])
    RV.literals(" ".join(["word"] * RV.MAX_ROWS), ["word"])
