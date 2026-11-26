"""Order 0083's bench: the report reads each arm's rows and states the gates as pre-registered (the second asking
against the first on its certified rows; the first asking's median against the base arm's; 0 wrong; the kill on a
remembered plan's wrong answer; the programs identical with the route and without). Synthetic rows, no card."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]

import exp_0083_repeat as X  # noqa: E402


def row(pass_, i, verified, answer, s, how=None, program=None, match=None):
    return {"pass": pass_, "i": i, "q": "q", "gold": "g", "wording": None, "verified": verified, "answer": answer,
            "reason": None if verified else "no_plan", "match": match or ("correct" if answer else None), "how": how,
            "episode": None, "restated": False, "plan": None, "seed": None, "s": s,
            "memory": {"order": ["hippocampus", "emitter"], "s": 0.01, "tried": []} if how or verified is False else None,
            "program": program, "calls": [{"role": "programs", "ms": 100.0, "steps": 5, "heads": {"programs|route 5": 5}}]}


def write(tmp_path, monkeypatch, arms):
    monkeypatch.setattr(X, "rows_file", lambda arm: tmp_path / f"_exp_0083_{arm}.rows.jsonl")
    for arm, rows in arms.items():
        lines = [{"head": {"arm": arm}}] + rows
        X.rows_file(arm).write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def test_the_report_states_the_gates(tmp_path, monkeypatch):
    a = [row("gen3-1", 0, True, "paris", 2.5, program="P0"), row("gen3-1", 1, False, None, 2.4, program="P1"),
         row("gen3-2", 0, True, "paris", 2.5, program="P0"), row("gen3-2", 1, False, None, 2.4, program="P1")]
    m = [row("gen3-1", 0, True, "paris", 2.52), row("gen3-1", 1, False, None, 2.43),
         row("gen3-2", 0, True, "paris", 0.08, how="a remembered plan"), row("gen3-2", 1, False, None, 2.4)]
    h = [row("gen3-1", 0, True, "paris", 2.4, program="P0"), row("gen3-1", 1, False, None, 2.3, program="P1x")]
    write(tmp_path, monkeypatch, {"A": a, "M": m, "H": h})
    X.report(str(tmp_path / "rep.json"))
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    gate = rep["gates"]["memory M vs A"]
    assert gate == {"second asking gives the first's certified answer on all": True, "second asking median <= 0.2 s": True,
                    "first asking median <= base + 0.05 s": True, "0 wrong on every pass": True,
                    "kill: a wrong answer from a remembered plan": False, "WRONG not on the base arm's row": []}
    assert rep["gates"]["route H vs A: answers differ on"] == []
    assert rep["arms"]["M"]["repeat"] == {"certified_first": 1, "same_answer": 1, "remembered": 1,
                                          "s_median_on_certified": 0.08, "lost": [],
                                          "seat_path": {"certified_first": 1, "same_answer": 1, "s_median": 0.08},
                                          "host_path": {"certified_first": 0, "same_answer": 0, "s_median": None}}
    assert rep["gates"]["route programs identical"]["identical"] == 1 and not rep["gates"]["route programs identical"][">= 99%"]
    assert rep["arms"]["A"]["passes"]["gen3-1"]["steps"] == {"programs|route 5": 10}


def test_a_remembered_wrong_answer_fires_the_kill(tmp_path, monkeypatch):
    a = [row("gen3-1", 0, True, "paris", 2.5), row("gen3-2", 0, True, "paris", 2.5)]
    m = [row("gen3-1", 0, True, "paris", 2.5), row("gen3-2", 0, True, "lyon", 0.05, how="a remembered plan", match="WRONG")]
    write(tmp_path, monkeypatch, {"A": a, "M": m})
    X.report(str(tmp_path / "rep.json"))
    gate = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))["gates"]["memory M vs A"]
    assert gate["kill: a wrong answer from a remembered plan"] and not gate["0 wrong on every pass"]
    assert not gate["second asking gives the first's certified answer on all"]


def test_the_gen3_rows_are_r17s():
    import random

    from cubbyllm.local_paths import data_dir
    path = data_dir("gen3_free_text.jsonl")
    if not pathlib.Path(path).exists():
        import pytest
        pytest.skip("the gen-3 set is not on this machine")
    held = [json.loads(l) for l in open(path, encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    assert [r["q"] for r in X.gen3_rows()] == [r["prompt"] for r in random.Random(7).sample(held, 600)]
