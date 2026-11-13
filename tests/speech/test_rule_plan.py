"""Order 0067, step 3: a rule is a plan over the reply world, run by the VM. For each of exp_if_follow's ten kinds, a
plan of the shape the program seat writes, run by the VM, gives the measure's own verdict on every reply of
test_reply_world; a plan that names no measurement is "asked", never "checked"."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.speech import rule_plan as rp

_spec = importlib.util.spec_from_file_location("_rwt", pathlib.Path(__file__).with_name("test_reply_world.py"))
RWT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RWT)


_spec2 = importlib.util.spec_from_file_location("_ift", pathlib.Path(__file__).parents[2] / "validation" / "exp_if_turn.py")
IFT = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(IFT)
plan_of = IFT.plan_of                                    # the mapping line 4's run measures with (one copy)


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    s = cc.CubelangSession()
    return lambda src, fn: s.run(src, fn=fn)


def test_every_kind_run_by_the_vm_agrees_with_the_measure(vm):
    for rule in RWT.RULES:
        plan = plan_of(rule)
        assert rp.check(plan) is None, rule
        for reply in RWT.REPLIES:
            got = rp.run(plan, reply, vm)
            assert got["how"] == "checked" and got["held"] == RWT.IF.check(rule, reply), (rule, reply, got["program"])


def test_not_and_inequality(vm):
    plan = {"not": {"or": [{"m": "case", "op": "==", "v": "upper"}, {"m": "digit count", "op": "!=", "v": 0}]}}
    assert rp.run(plan, "quiet words", vm)["held"] is True
    assert rp.run(plan, "LOUD", vm)["held"] is False and rp.run(plan, "year 2024", vm)["held"] is False


def test_a_plan_that_names_no_measurement_is_asked_never_checked():
    no_run = lambda src, fn: pytest.fail("the VM is never asked")   # noqa: E731
    for plan in ({"m": "tone", "op": "==", "v": "warm"}, {"m": "word count", "op": "about", "v": 3},
                 {"m": "word", "op": "<=", "v": 2}, {"and": []}, {"text": "be nice"}):
        got = rp.run(plan, "anything", no_run)
        assert got["how"] == "asked" and got["held"] is None and got["why"], plan


def test_the_program_binds_the_reply_statements_it_reads():
    src, sts = rp.program({"m": "word count", "op": "<=", "v": 3}, "one two")
    assert sts == ["2 is the word count of the reply"] and "assign m_word_count = 2;" in src


def test_a_relation_with_an_argument_runs_in_the_vm(vm):
    plan = {"and": [{"m": "count of the word", "arg": "river", "op": ">=", "v": 2},
                    {"m": "starts with", "arg": '"', "op": "==", "v": "yes"},
                    {"m": "count of the word", "arg": "sea", "op": "<", "v": 1}]}
    assert rp.check(plan) is None
    assert rp.run(plan, '"The river meets the river."', vm)["held"] is True
    assert rp.run(plan, '"The river meets the sea."', vm)["held"] is False
    assert rp.run(plan, 'The river, the river.', vm)["held"] is False
    src, sts = rp.program(plan, '"river river"')
    assert 'assign m_count_of_the_word_0 = 2;' in src and '2 is the count of the word "river" of the reply' in sts


def test_an_argument_is_required_where_it_is_taken_and_refused_elsewhere():
    assert "takes an argument" in rp.check({"m": "count of the word", "op": ">=", "v": 1})
    assert "takes an argument" in rp.check({"m": "starts with", "arg": "", "op": "==", "v": "yes"})
    assert "takes no argument" in rp.check({"m": "word count", "arg": "x", "op": "<=", "v": 3})


def test_a_json_key_is_compared_as_written(vm):
    """Found by line 4's run (2026-11-12): keys were lowercased, so "U1" passed a rule asking for "u1" while
    exp_if_follow's check (and JSON) said no. Words and openings are folded; a key is not."""
    plan = {"and": [{"m": "json", "op": "==", "v": "yes"}, {"m": "json key", "op": "has", "v": "u1"}]}
    assert rp.run(plan, '{"U1": "I"}', vm)["held"] is False and rp.run(plan, '{"u1": "I"}', vm)["held"] is True
    assert rp.run({"m": "word", "op": "has", "v": "Paris"}, "we saw paris", vm)["held"] is True
