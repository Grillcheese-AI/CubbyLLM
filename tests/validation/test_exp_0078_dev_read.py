"""Order 0078's dev read (validation/exp_0078_dev_read.py, exp_0078_dev_score.py): a sampled try is H-E19's loop step
for step (a try fed the gold rows reaches what `StepLoop.solve` reaches; the same refusals); chains sharing a prompt
are sampled in one batch; the score picks m on the tries' VM values with the speak rule, and drops the ids it is
handed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "tests" / "reasoning"))
import exp_0078_dev_read as R  # noqa: E402
import exp_0078_dev_score as S  # noqa: E402
from test_step_loop import QUESTION, REC, py_vm  # noqa: E402

from cubbyllm.reasoning import step_loop as sl  # noqa: E402

ROWS = sl.recut(REC)
WRONG = "        create s1 : quantity;   # step 1: $S1 + $N3\n        assign s1 = $S1;\n        add s1, $N3;\n"


class Sampler:
    """emit_samples: the gold row for its prompt, except every second sample at step 1 takes the wrong step."""

    def __init__(self):
        self.calls = []

    def emit(self, prompt, max_new_tokens=64, **kw):
        return next((r["program"] for r in ROWS if r["prompt"] == prompt), "off the rows")   # malformed: refused

    def emit_samples(self, prompt, n, **kw):
        self.calls.append((prompt, n, kw.get("seed")))
        gold = self.emit(prompt)
        if prompt == ROWS[1]["prompt"]:
            return [gold if i % 2 == 0 else WRONG for i in range(n)]
        return [gold] * n


def test_a_try_is_the_loops_step_for_step_and_shared_prompts_batch():
    e = Sampler()
    loop = sl.StepLoop(e, vm=py_vm)
    assert loop.solve(QUESTION)["answer"] == "1109"
    ts = R.tries(e, loop, QUESTION, qi=3, k=4)
    assert [t["answer"] for t in ts] == ["1109", None, "1109", None]   # the wrong step's chain goes off the rows
    assert ts[0]["steps"] == ["154", "924", "1109"] and ts[0]["refused"] is None
    assert e.calls[0][1] == 4 and e.calls[1][1] == 4                  # steps 0 and 1: one prompt, one batch of 4
    assert len({c[2] for c in e.calls}) == len(e.calls)                # every batch its own seed


def test_the_cap_refuses_as_the_loop_does():
    class Never:
        def emit_samples(self, prompt, n, **kw):
            return ["        create s0 : quantity;   # step 0: $N1 / $N2\n        assign s0 = $N1;\n        div s0, $N2;\n"] * n
    loop = sl.StepLoop(Never(), vm=py_vm, max_steps=3)
    t = R.tries(Never(), loop, QUESTION, qi=0, k=2)[0]
    assert t["answer"] is None and "no stop" in t["refused"] and len(t["steps"]) == 4


def test_score_picks_m_and_drops_ids(tmp_path):
    def row(i, gold, answers, greedy):
        return {"id": f"q{i}", "gold": str(gold), "greedy": {"answer": greedy}, "seconds": {"greedy": 1, "tries": 2},
                "tries": [{"answer": a} for a in answers]}
    rows = [row(i, 10, ["10"] * 9 + ["3"] * 7, "10") for i in range(50)]
    rows += [row(50 + i, 5, ["7"] * 6 + ["5"] * 2 + [None] * 8, None) for i in range(50)]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    s = S.score(str(p))
    assert s["speak rule"]["by k"][16]["m"] == 7                        # m < 7 speaks the 50 wrong 7s
    assert s["speak rule"]["by k"][16]["at m"] == {"right": 50, "wrong": 0, "refused": 50, "wrong share": 0.0}
    assert s["greedy"] == {"right": 50, "wrong": 0, "refused": 50}
    assert s["any try right (the oracle)"] == 100
    d = tmp_path / "drop.json"
    d.write_text(json.dumps({"drop": ["q0", "q1"]}), encoding="utf-8")
    assert S.score(str(p), str(d))["questions"] == 98
