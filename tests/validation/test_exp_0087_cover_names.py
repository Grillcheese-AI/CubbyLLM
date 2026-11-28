"""Order 0087's read: the report states the pre-registered gates from both arms' rows (Ghana on the other-words pass;
the two self questions; gen-3's WRONG added and correct rows down; the live eight unchanged; the kill). Synthetic rows,
no card."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]

import exp_0087_cover_names as X  # noqa: E402


def row(pass_, i, answer, match=None, reason=None, seed="s", names=None):
    return {"pass": pass_, "i": i, "q": "q", "gold": "g", "wording": None, "verified": bool(answer), "answer": answer,
            "reason": None if answer else reason, "names": names, "match": match, "restated": False, "plan": ["r"],
            "seed": seed, "s": 2.0, "names_s": 0.001 if names is not None else 0.0, "program": None, "seat_ms": 1.0}


def arm(gen3, other, live8, self_):
    return ([row("gen3", i, *g) for i, g in enumerate(gen3)] + [row("other", i, *o) for i, o in enumerate(other)]
            + [row("live8", i, *l) for i, l in enumerate(live8)] + [row("self", i, *s) for i, s in enumerate(self_)])


def write(tmp_path, monkeypatch, off, on):
    monkeypatch.setattr(X, "rows_file", lambda a: tmp_path / f"_exp_0087_{a}.rows.jsonl")
    for name, rows in (("off", off), ("on", on)):
        X.rows_file(name).write_text("".join(json.dumps(r) + "\n" for r in [{"head": {"arm": name}}] + rows),
                                     encoding="utf-8")


def report(tmp_path):
    X.report(str(tmp_path / "rep.json"))
    return json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))


LIVE = [("Margaret Trudeau", "correct")]


def test_the_gates_pass_when_the_rule_refuses_only_the_wrong_rows(tmp_path, monkeypatch):
    off = arm([("paris", "correct"), ("ghana", "WRONG"), ("rome", "correct")],
              [("Ghana", "WRONG", None, "citizenship"), ("x", "correct")], LIVE,
              [("you -- actor", None), ("this loop", None)])
    on = arm([("paris", "correct"), (None, None, "seed_is_relation"), ("rome", "correct")],
             [(None, None, "seed_is_relation", "citizenship"), ("x", "correct")], LIVE,
             [(None, None, "seed_is_pronoun"), (None, None, "seed_is_pronoun")])
    write(tmp_path, monkeypatch, off, on)
    rep = report(tmp_path)
    assert rep["gates"] == {"other: Ghana on 0 rows": True, "other: a relation-word seed spoken": [],
                            "self: both refused with seed_is_pronoun": True, "gen3: no WRONG added": True,
                            "gen3: correct at most 6 down": True, "live8 unchanged": True,
                            "kill: a WRONG added or more than 12 down": False}
    assert rep["gen3_wrong_off_and_on"] == [{"i": 1, "on": "seed_is_relation"}]
    assert rep["passes"]["gen3"]["on"]["by_the_rule"] == {"seed_is_relation": 1, "name_left_over": 0, "seed_is_pronoun": 0}
    assert rep["other_ghana"] == {"off": 1, "on": 0}


def test_a_wrong_answer_the_flag_adds_fires_the_kill(tmp_path, monkeypatch):
    off = arm([("paris", "correct"), (None, None, "no_plan")], [], LIVE, [])
    on = arm([("lyon", "WRONG"), (None, None, "no_plan")], [], [("Margaret", "near")], [])
    write(tmp_path, monkeypatch, off, on)
    rep = report(tmp_path)
    assert rep["gen3_wrong_added"] == [0] and rep["gates"]["kill: a WRONG added or more than 12 down"]
    assert not rep["gates"]["live8 unchanged"] and rep["gen3_correct_down"] == 1
    assert rep["gen3_right_answers_refused"][0]["i"] == 0
