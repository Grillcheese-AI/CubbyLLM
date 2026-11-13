"""Order 0072, pinned: the static guard refuses a program that compares an unbound name, reads a list outside a length
check (or with an index that can go below 0 unchecked) or binds more than 250 items (a shared 2-byte VM name key no
longer matters: cubelang P6-9); every program the templates write passes it; and the fake host is never reached for an
ask-first verb without a yes given in this turn."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT / "validation" / "builders")]
import verb_cases as VC  # noqa: E402
import verb_cases_act as VA  # noqa: E402
import verb_contracts as EV  # noqa: E402
import verb_fake_host as FH  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import verb_guard as G  # noqa: E402
from cubbyllm.speech import verb_programs as VP  # noqa: E402

HEAD = "program P implements ISolve { public function solve(mention: str): number { "


def test_an_unbound_name_is_refused():
    assert any("ghost" in w for w in G.check(HEAD + 'create ok: number; assign ok = 0; if (ghost == "x") { assign ok = 1; } return ok; } }'))


def test_an_index_outside_a_length_check_is_refused():
    assert G.check(HEAD + 'let in_val = ["a"]; create n_in: number; assign n_in = 1; let x = in_val[3]; return 0; } }')
    assert G.check(HEAD + 'let in_val = ["a"]; create n_in: number; assign n_in = 1; create i: number; assign i = 0; '
                          'while (i < 5) { let x = in_val[i]; add i, 1; } return 0; } }')
    assert G.check(HEAD + 'let in_val = ["a"]; create n_in: number; assign n_in = 1; create i: number; assign i = 0; '
                          'sub i, 1; if (i < n_in) { let x = in_val[i]; } return 0; } }')
    assert not G.check(HEAD + 'let in_val = ["a"]; create n_in: number; assign n_in = 1; create i: number; assign i = 0; '
                              'while (i < n_in) { let x = in_val[i]; add i, 1; } return 0; } }')


def test_long_lists_are_refused_and_2_byte_keys_no_longer_matter():
    assert G.check(HEAD + "let in_val = [" + ", ".join(str(i) for i in range(251)) + "]; return 0; } }")
    assert not G.check(HEAD + 'let a = "thing"; create ok: number; assign ok = 0; if (a == "government") { assign ok = 1; } return ok; } }')


def test_every_template_passes_the_guard():
    rng = random.Random(0)
    for (verb, kind), variant in sorted(VC.VARIANT_OF.items()):
        for brk in (None,):
            data = VC.build(verb, variant, rng, "en", brk)
            assert not G.check(VP.program(kind, variant, data))


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    return cc.CubelangSession()


def test_the_host_is_never_reached_for_ask_first_without_a_yes_this_turn(vm):
    host, n = FH.FakeHost(), 0
    for case in VA.gate_set():
        if case["class"] != "ask first" or case["yes_this_turn"]:
            continue
        before = len(host.calls)
        got = FH.run_act(case, vm, host)
        assert len(host.calls) == before and got in ("ask", "refuse"), case
        n += 1
    assert n >= 20


def test_vm_and_evaluator_agree_on_a_sample(vm):
    rng = random.Random(3)
    for (verb, kind) in sorted(VC.VARIANT_OF)[::3]:
        for case in VC.cases(verb, kind, rng, 4):
            src = VP.program(kind, case["variant"], case["data"])
            if G.check(src):
                continue
            assert int(float(vm.run(src, fn="solve")["result"])) == EV.evaluate(kind, case["variant"], case["data"]), case
