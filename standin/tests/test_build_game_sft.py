"""Pins for the v4 game-SFT builder: every family's template program
executes on the real VM and returns its gold; the prompts are in the game's
words; the split/repeat schema matches v3's. Live parts skip without the exe.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_game_sft as g  # noqa: E402


def test_generators_are_deterministic_and_in_the_games_words():
    a = g.build(seed=7, n={"decision": 5, "compare": 5, "where": 6, "count": 4, "compose": 4})
    b = g.build(seed=7, n={"decision": 5, "compare": 5, "where": 6, "count": 4, "compose": 4})
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    subs = {r["subtype"].split(":n_hop")[0] for r in a}
    assert subs >= {"game:decision", "game:compare", "game:where", "game:count", "game:compose"}
    assert all(r["split"] in ("train", "val") and r["repeat"] in (1, 2) for r in a)
    assert any("ghost" in r["prompt"] for r in a if r["subtype"] == "game:decision")
    where = [r for r in a if r["subtype"].startswith("game:where")]
    assert all("level-" in r["prompt"] and "Facts:" in r["prompt"] for r in where)
    assert all(r["gold"] in r["prompt"] for r in where), "the answer cell is among the facts"


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_live_every_family_executes_on_the_vm_and_matches_gold():
    _exe_or_skip()
    recs = g.build(seed=3, n={"decision": 4, "compare": 4, "where": 5, "count": 3, "compose": 3})
    stats, _ = g.verify_all(recs, verbose_every=10 ** 9)
    bad = [(r["subtype"], r["gold"], r["vm_result"], r["vm_error"]) for r in recs
           if not r["vm_ok"] or r["gold_match"] is False]
    assert bad == [], f"template programs must be VM-true: {bad}"
    kinds = {r["subtype"].split(":n_hop")[0] for r in recs if not r["subtype"].startswith("game:notebook")}
    assert len(kinds) == 5, kinds                        # + whatever his notebook contributed
    two_hop = [r for r in recs if "n_hop=2" in r["subtype"]]
    if two_hop:
        assert all(r["gold_match"] for r in two_hop), "hop_2 must recover the second hop"
