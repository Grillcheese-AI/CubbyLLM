"""Order 0045 -- ``validation/grow_mem_readout.py`` gives H-E20's gates 0-3 from fixture metrics exactly as the notebook's
readout cell did (its formulas, restated here line for line), plus gate 2 from the probes' reports."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import grow_mem_readout as gr  # noqa: E402

SEQ = 2048
KEYS = ("fineweb_edu", "books", "nemotron_qa", "nemotron_code", "instruct", "agent", "program")


def metrics(first: dict, last: dict, steps: int = 1000, tps: int = 131072, hours: float = 10.0) -> list[dict]:
    rows = []
    for s in range(100, steps + 1, 100):
        rows.append({"step": s, "loss": 2.5, "tokens": s * tps, "hours": hours * s / steps, "phase": "stable"})
        if s in (100, steps):
            v = dict(first if s == 100 else last)
            rows.append({"step": s, "val": v})
    return rows


def val(mix, copy=1.2, **over):
    v = {k: 2.4 for k in KEYS}
    v.update(over)
    v["_mix"], v["_copy"] = mix, copy
    return v


def notebook_checks(grown, control, seq):
    """The notebook cell's own formulas (notebooks/_build_grow_mem.py, 2026-10-17), verbatim in spirit."""
    vals = lambda rows: [r for r in rows if "val" in r]  # noqa: E731
    step_size = lambda rows: round([r for r in rows if "tokens" in r and r.get("step")][-1]["tokens"] / [r for r in rows if "tokens" in r and r.get("step")][-1]["step"])  # noqa: E731
    hours = lambda rows: max(r["hours"] for r in rows if "hours" in r)  # noqa: E731
    g, c = vals(grown)[-1]["val"], vals(control)[-1]["val"]
    NEW, KEY = ("instruct", "agent", "program"), ("fineweb_edu", "books", "nemotron_qa", "nemotron_code")
    checks = {}
    same_step = abs(step_size(grown) - step_size(control)) <= 2 * seq
    same_hours = abs(hours(grown) - hours(control)) <= 0.5
    checks["0"] = bool(same_step and same_hours)
    checks["1"] = g["_mix"] <= c["_mix"] + 0.02
    checks["3b"] = all(g[k] <= c[k] + 0.02 for k in KEY)
    for a, v in (("grown", vals(grown)), ("control", vals(control))):
        first, last = v[0]["val"], v[-1]["val"]
        checks[f"3a.{a}"] = all(last.get(k, 9) < first.get(k, 0) for k in NEW if k in first)
    return checks


def test_gates_0_1_3_match_the_notebooks_readout_on_fixture_metrics():
    grown = metrics(val(2.50, instruct=3.0, agent=3.1, program=2.9), val(2.41, instruct=2.6, agent=2.7, program=2.5))
    control = metrics(val(2.50, instruct=3.0, agent=3.1, program=2.9), val(2.40, instruct=2.7, agent=2.8, program=2.6))
    r = gr.gates(grown, control, [], SEQ)
    nb = notebook_checks(grown, control, SEQ)
    by = {k.split(".")[0] if not k.startswith("3a") else "3a." + k.split(". ")[1].split(":")[0]: v for k, v in r["checks"].items()}
    assert by["0"] == nb["0"] is True and by["1"] == nb["1"] is True and by["3b"] == nb["3b"] is True
    assert by["3a.grown"] == nb["3a.grown"] is True and by["3a.control"] == nb["3a.control"] is True
    assert r["sizes"] == {"grown": 131072, "control": 131072} and not any(r["kills"].values())


def test_growth_fails_and_kills_by_the_pre_registered_margins_and_unequal_hours_fail_gate_0():
    control = metrics(val(2.50), val(2.40))
    assert gr.gates(metrics(val(2.50), val(2.43)), control, [], SEQ)["checks"]["1. growth: grown mix <= control mix + 0.02"] is False
    r = gr.gates(metrics(val(2.50), val(2.46)), control, [], SEQ)
    assert r["kills"]["1. growth killed: grown mix above the control's by more than 0.05"] is True
    assert gr.gates(metrics(val(2.50), val(2.42)), control, [], SEQ)["checks"]["1. growth: grown mix <= control mix + 0.02"] is True
    r = gr.gates(metrics(val(2.5), val(2.4), hours=10.6), control, [], SEQ)
    assert r["checks"]["0. equal compute: the same tokens a step, the same card, hours within 0.5 h"] is False
    r = gr.gates(metrics(val(2.5), val(2.4)), control, [], SEQ, cards={"grown": "B200", "control": "H200"})
    assert r["checks"]["0. equal compute: the same tokens a step, the same card, hours within 0.5 h"] is False
    assert notebook_checks(metrics(val(2.50), val(2.43)), control, SEQ)["1"] is False


def test_gate_2_from_the_probes_and_not_yet_without_them():
    grown, control = metrics(val(2.5, copy=11.5), val(2.4, copy=11.4)), metrics(val(2.5), val(2.4))
    r = gr.gates(grown, control, [], SEQ)
    assert r["checks"]["2a. memory: grown exact on facts_text_far >= both floors + 0.10"] is None
    rec = lambda e: {"shards": {"facts_text_far": {"exact": e, "first_token": e}}}  # noqa: E731
    needle = lambda cells: {"lengths": [1024, 2048], "depths": [0.0, 0.5, 1.0], "distract": 7, "trained": {"1024": cells, "2048": cells}}  # noqa: E731
    probes = {"grown": rec(0.45), "control": rec(0.30), "base": rec(0.31), "needle_grown": needle([0.5, 0.4, 0.9]), "needle_control": needle([0.125, 0.2, 0.9])}
    r = gr.gates(grown, control, [], SEQ, probes)
    assert r["checks"]["2a. memory: grown exact on facts_text_far >= both floors + 0.10"] is True
    assert r["checks"]["2b. memory: grown _copy not above its first reading + 0.05"] is True
    assert r["checks"]["2c. memory: needle beyond-window cells > 0.25 where the control sits at chance"] is True
    assert r["probes"]["needle_beyond"]["grown"] == [0.5, 0.5, 0.4]           # (1024 at depth 0.5 sits in the window)
    assert r["probes"]["needle_beyond"]["chance"] == 0.125
    probes["grown"] = rec(0.39)
    assert gr.gates(grown, control, [], SEQ, probes)["checks"]["2a. memory: grown exact on facts_text_far >= both floors + 0.10"] is False
    probes["needle_control"] = needle([0.4, 0.3, 0.9])                    # the control is not at chance: nothing shown
    assert gr.gates(grown, control, [], SEQ, probes)["checks"]["2c. memory: needle beyond-window cells > 0.25 where the control sits at chance"] is False
    md = gr.markdown(r)
    assert "PASS  1. growth" in md and "| _mix |" in md
