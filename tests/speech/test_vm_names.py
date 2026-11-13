"""The VM once keyed names and string literals by 2 bytes of BLAKE3, so two strings sharing a key read as one
("thing" == "government"). cubelang P6-9 (`676df1d`, the build CUBELANG_EXE names since 2026-11-13) keys by 8 bytes and
raises on a collision: the VM now tells such strings apart, and a rule that once had to be "asked" is checked and right.
`vm_names` stays only to find 2-byte pairs (old logs); nothing runs it in front of the VM any more."""
from __future__ import annotations

import builtins

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.bridges import vm_names as N
from cubbyllm.speech import rule_plan as rp


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    s = cc.CubelangSession()
    return lambda src, fn: s.run(src, fn=fn)


def test_the_old_2_byte_pair_is_still_found():
    pytest.importorskip("blake3")
    assert N.key("thing") == N.key("government")
    assert N.collisions('let x = "thing"; if (x == "government") { }') == [["government", "thing"]]


def test_the_vm_tells_the_pair_apart(vm):
    src = ('program P implements ISolve { public function solve(mention: str): number { create r: number; assign r = 0; '
           'let x = "thing"; if (x == "government") { assign r = 1; } return r; } }')
    assert str(vm(src, "solve")["result"]).strip() in ("0", "0.0")


def test_a_rule_once_asked_is_now_checked_and_right(vm):
    got = rp.run({"m": "word", "op": "lacks", "v": "government"}, "a thing here", vm)
    assert got["how"] == "checked" and got["held"] is True
    got = rp.run({"m": "word", "op": "lacks", "v": "government"}, "the government spoke", vm)
    assert got["how"] == "checked" and got["held"] is False


def test_names_are_literals_and_identifiers_without_keywords():
    got = N.names('create k1: number; let x = "a \\"q\\" b"; if (x == "one") { assign k1 = 1; } return k1;')
    assert got == {"k1", "x", 'a "q" b', "one"}


def test_without_blake3_the_finder_fails_closed(monkeypatch):
    real = builtins.__import__

    def no_blake3(name, *a, **k):
        if name == "blake3":
            raise ImportError("no blake3")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_blake3)
    assert N.collisions('let x = "stone";')
