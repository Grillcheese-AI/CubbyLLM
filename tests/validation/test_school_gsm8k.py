"""Order 0078, S2 (validation/school_gsm8k.py): the school's pool keeps GSM8K train outside dev; a wrong step is named
on the gold cut to its first k+1 steps when the gold's steps are the plan's, else on the plan's values written as
steps; `school.mistake_rows` uses the namer it is handed and is unchanged without one. The VM is stubbed (the cube's
own reading is pinned in cubelang's tests and read against school.py in exp_gsm8k_cube)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import school_gsm8k as S  # noqa: E402

from cubbyllm.reasoning import school  # noqa: E402

GOLD = "Half: <<48/2=24>>24. Total: <<48+24=72>>72. #### 72"


def _namer(tmp_path, stub):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "gsm8k-train-0", "question": "48 clips, half as many in May.", "answer": GOLD}) + "\n" +
        json.dumps({"id": "gsm8k-train-1", "question": "q", "answer": "no steps #### 3"}) + "\n", encoding="utf-8")
    (tmp_path / "dev_ids.json").write_text(json.dumps(["gsm8k-train-1"]), encoding="utf-8")
    n = S.CubeNamer("vm", "gsm8k.cube", str(tmp_path))
    calls = []

    def call(fn, *args):
        calls.append((fn, *args))
        return stub(fn, *args)
    n.call = call
    return n, calls


def test_the_pool_is_gsm8k_train_outside_dev(tmp_path):
    n, _ = _namer(tmp_path, lambda *a: None)
    assert n.keep({"gsm8k_id": "gsm8k-train-0", "split": "train"})
    assert not n.keep({"gsm8k_id": "gsm8k-train-1", "split": "train"})          # dev
    assert not n.keep({"gsm8k_id": "gsm8k-train-0", "split": "val"})
    assert not n.keep({"split": "train"})                                       # TinyGSM: no gold text


def test_a_wrong_step_is_named_on_the_golds_own_steps_when_they_are_the_plans(tmp_path):
    replay = {"Half: <<48/2=24>>": 24.0, "Half: <<48/2=24>>24. Total: <<48+24=72>>": 72.0}

    def stub(fn, *args):
        return replay.get(args[0]) if fn == "solve" else 1
    n, calls = _namer(tmp_path, stub)
    row = {"gsm8k_id": "gsm8k-train-0", "stats": {"k": 1}, "plan_values": [24.0, 72.0]}
    kind = n.for_row(row)(24.0, 72.0, [48.0], [72.0], [24.0])
    assert kind == "stopped early"
    named = [c for c in calls if c[0] == "classify_error"][-1]
    assert named[1] == "Half: <<48/2=24>>24. Total: <<48+24=72>>" and named[3] == "24"
    assert n.source["the gold's own steps"] == 1


def test_otherwise_the_plans_values_are_written_as_steps(tmp_path):
    n, calls = _namer(tmp_path, lambda fn, *a: 5 if fn == "classify_error" else 0.0)
    row = {"gsm8k_id": "gsm8k-train-0", "stats": {"k": 0}, "plan_values": [24.0, 72.0, 9.5]}   # 3 steps, gold has 2
    assert n.for_row(row)(30.0, 24.0) == "other"
    assert [c for c in calls if c[0] == "classify_error"][-1][1] == "<<24=24>>"
    assert n.source["the plan's values as steps"] == 1


def test_mistake_rows_uses_the_classifier_it_is_handed():
    attempts = [{"program": "p1", "outcome": school.WRONG, "values": [24.0], "stated": [48.0]},
                {"program": "p2", "outcome": school.CORRECT, "values": [72.0], "stated": [48.0]}]
    result = {"weights": [-0.5, 0.5], "level": 0, "dopamine": 0.5, "expected": 0.5}
    default = school.mistake_rows("q", attempts, result)
    handed = school.mistake_rows("q", attempts, result, classify=lambda *a: "reversed")
    assert [r["error"] for r in default if r["kind"] == "mistake"] == [school.classify_error(24.0, 72.0, [48.0], [72.0], [24.0])]
    assert [r["error"] for r in handed if r["kind"] == "mistake"] == ["reversed"]
