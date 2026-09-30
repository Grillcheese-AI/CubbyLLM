"""Pins for validation/exp_he18_ab.py (H-E18's readout): a plan is correct by its SEED and HOP binds, a value
family by the VM's value against the gold, arithmetic is broken down by step count, and the verdict applies
the pre-registered gate (+10 pts held-out, no val family down > 2 pts, 0 copied / unknown slots) and kill
(< +3 pts). A fake VM; no cubelang needed. Run: python -m pytest standin/tests/test_exp_he18_ab.py -q
-p no:hypothesispytest"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation"), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_he18_ab as ab  # noqa: E402


def fake_vm(program):
    """'VALUE=<v>' in the program is what the VM returns; 'BROKEN' does not run."""
    if "BROKEN" in program:
        return False, None
    for tok in program.split():
        if tok.startswith("VALUE="):
            return True, tok[6:]
    return True, None


def plan(seed, *hops):
    lines = [f'bind frame, SEED, "{seed}";'] + [f'bind frame, HOP{i + 1}, "{h}";' for i, h in enumerate(hops)]
    return "\n".join(lines)


def arith(i, ok=True, steps=2):
    return {"task": "arithmetic", "subtype": f"pf:steps={steps}", "gold": "12", "reference": "",
            "generated": "VALUE=12" if ok else "VALUE=13", "slot_reason": ""}


def test_plan_is_correct_by_seed_and_hops_not_by_its_vm_value():
    ref = plan("Olaf Tzschoppe", "country of citizenship", "head of government")
    assert ab.plan_correct(plan("olaf tzschoppe", "country of citizenship", "head of government"), ref)
    assert not ab.plan_correct(plan("Olaf Tzschoppe", "country of citizenship"), ref)          # a hop short
    assert not ab.plan_correct(plan("Someone Else", "country of citizenship", "head of government"), ref)


def test_score_breaks_arithmetic_down_by_step_count_and_counts_slot_failures():
    outs = [arith(0), arith(1, ok=False), arith(2, steps=5),
            {"task": "plan", "gold": None, "reference": plan("X", "capital"), "generated": plan("x", "capital"),
             "slot_reason": ""},
            {"task": "chain", "gold": "Paris", "reference": "", "generated": "BROKEN", "slot_reason": "copied literal 'Paris'"}]
    s = ab.score(outs, vm=fake_vm)
    assert s["arithmetic"]["n"] == 3 and abs(s["arithmetic"]["correct"] - 2 / 3) < 1e-9
    assert s["arithmetic/steps=2"]["n"] == 2 and s["arithmetic/steps=5"]["correct"] == 1.0
    assert s["plan"]["correct"] == 1.0 and s["chain"]["executes"] == 0.0 and s["chain"]["copied"] == 1
    assert s["_all"]["n"] == 5 and abs(s["_all"]["correct"] - 3 / 5) < 1e-9


def _sc(correct, n=100, copied=0, unknown=0, **fams):
    d = {"_all": {"n": n, "executes": 1.0, "correct": correct, "copied": copied, "unknown_slot": unknown}}
    d.update({k: {"n": 50, "executes": 1.0, "correct": v, "copied": 0, "unknown_slot": 0} for k, v in fams.items()})
    return d


def test_verdict_applies_the_gate_and_the_kill():
    ctrl_val, ctrl_held = _sc(0.8, arithmetic=0.70, chain=0.95), _sc(0.40)
    ok = ab.verdict(ctrl_val, _sc(0.8, arithmetic=0.71, chain=0.94), ctrl_held, _sc(0.55))
    assert ok["verdict"].startswith("PASS") and abs(ok["delta_held"] - 0.15) < 1e-9
    drop = ab.verdict(ctrl_val, _sc(0.8, arithmetic=0.60, chain=0.95), ctrl_held, _sc(0.55))
    assert drop["verdict"].startswith("NEITHER") and drop["worst_val_drop"]["family"] == "arithmetic"
    copied = ab.verdict(ctrl_val, _sc(0.8, copied=1, arithmetic=0.70, chain=0.95), ctrl_held, _sc(0.55))
    assert copied["verdict"].startswith("NEITHER")
    kill = ab.verdict(ctrl_val, _sc(0.8, arithmetic=0.70, chain=0.95), ctrl_held, _sc(0.42))
    assert kill["verdict"].startswith("KILL")


def test_main_reads_four_files_and_writes_the_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "vm_run", fake_vm)
    monkeypatch.setattr(ab.score, "__defaults__", (fake_vm,))
    files = []
    for name, good in (("cv", 8), ("ch", 4), ("pv", 8), ("ph", 6)):
        outs = [arith(i, ok=i < good) for i in range(10)]
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps({"outputs": outs}), encoding="utf-8")
        files.append(str(f))
    out = tmp_path / "ab.json"
    ab.main(["--files", *files, "--out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert abs(r["delta_held"] - 0.2) < 1e-9 and r["verdict"].startswith("PASS")
