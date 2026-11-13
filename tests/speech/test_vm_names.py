"""The VM keys names and string literals by 16 bits of BLAKE3 (cubelang `name_hash`): two strings sharing a key read as
one. The host finds such pairs before a program runs, and a rule whose program carries one is "asked", never checked."""
from __future__ import annotations

import builtins

import pytest

from cubbyllm.bridges import vm_names as N
from cubbyllm.speech import rule_plan as rp


def test_the_known_pair_shares_a_key():
    pytest.importorskip("blake3")
    assert N.key("thing") == N.key("government")
    assert N.collisions('let x = "thing"; if (x == "government") { }') == [["government", "thing"]]
    assert N.collisions('let x = "stone"; if (x == "government") { }') == []


def test_names_are_literals_and_identifiers_without_keywords():
    got = N.names('create k1: number; let x = "a \\"q\\" b"; if (x == "one") { assign k1 = 1; } return k1;')
    assert got == {"k1", "x", 'a "q" b', "one"}


def test_a_rule_whose_program_shares_a_key_is_asked_never_run():
    pytest.importorskip("blake3")
    never = lambda src, fn: pytest.fail("the VM is never asked")       # noqa: E731
    got = rp.run({"m": "word", "op": "lacks", "v": "government"}, "a thing here", never)
    assert got["how"] == "asked" and got["held"] is None and "name key" in got["why"]


def test_without_blake3_nothing_is_checked(monkeypatch):
    real = builtins.__import__

    def no_blake3(name, *a, **k):
        if name == "blake3":
            raise ImportError("no blake3")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_blake3)
    assert N.collisions('let x = "stone";')
