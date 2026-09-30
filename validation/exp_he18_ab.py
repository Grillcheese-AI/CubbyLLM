"""exp_he18_ab - H-E18's readout: the program-first arm against the control, both 450M emitter adapters trained
at equal steps on the same base (notebooks/emitter_450m.ipynb), every generated program run in the real VM.

Four generation files, two per arm (the notebook writes them beside each adapter):
    cubby450m_v12e_cont/val_generations_v12e_cont.json               control, v12e val
    cubby450m_v12e_cont/val_generations_v12e_cont_pfheld.json        control, program-first held-out eval
    cubby450m_v12e_pf_cont/val_generations_v12e_pf_cont.json         program-first arm, v12e val
    cubby450m_v12e_pf_cont/val_generations_v12e_pf_cont_pfheld.json  program-first arm, held-out eval

Correct means: the program passes the slot check (a copied literal or an unknown slot is a refusal the host
never runs, H-E15), the filled program runs, and its value matches the gold (arithmetic, chain, kernel,
role-binding) or -- a plan, whose VM value is only its seed -- its SEED and HOP binds match the reference plan
(normalized seed, `relation_matches` per hop). `vm_correct` is the same without the slot check (diagnosis only). The gate (pre-registered in CUBBYLLM_HYPOTHESES.md, H-E18): on the held-out eval
the program-first arm is correct on >= 10 pts more; no v12e val family drops > 2 pts; the program-first arm
copies 0 literals and names 0 unknown slots. Kill: under +3 pts.

    python validation/exp_he18_ab.py --drive "<Drive>/cubbyllm/emitter"
    python validation/exp_he18_ab.py --files ctrl_val.json ctrl_held.json pf_val.json pf_held.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GATE_PTS, KILL_PTS, DROP_PTS = 0.10, 0.03, 0.02
_BIND = re.compile(r'bind\s+\w+\s*,\s*(SEED|HOP(\d+))\s*,\s*"((?:[^"\\]|\\.)*)"\s*;')


def plan_of(text: str) -> tuple[str | None, list[str]]:
    seed, hops = None, {}
    for m in _BIND.finditer(text or ""):
        if m.group(1) == "SEED":
            seed = m.group(3)
        else:
            hops[int(m.group(2))] = m.group(3)
    return seed, [hops[k] for k in sorted(hops)]


def plan_correct(generated: str, reference: str) -> bool:
    from cubbyllm.reasoning.planner import normalize, relation_matches
    gs, gh = plan_of(generated)
    rs, rh = plan_of(reference)
    return (gs is not None and rs is not None and normalize(gs) == normalize(rs) and len(gh) == len(rh)
            and all(relation_matches(r, h) for r, h in zip(rh, gh)))


def vm_run(program: str) -> tuple[bool, object]:
    """(executes, result) from the real VM, with the harvest's ISolver shim and answer function."""
    from build_emitter_sft import answer_fn, shim_isolver
    from cubbyllm.bridges import cubelang_client as cc
    src = shim_isolver(program)
    try:
        out = cc.run_program_proto(src, fn=answer_fn(src))
        return bool(out.get("ok")), out.get("result")
    except Exception:
        return False, None


def value_correct(result, gold) -> bool:
    from build_emitter_sft import gold_matches
    from cubbyllm.reasoning.slots import _norm
    return bool(gold_matches(result, gold)) or _norm(str(result)) == _norm(str(gold))


_EMPTY_ROLE = re.compile(r"bind\s+\w+\s*,\s*,")


def stale_decode(outputs: list[dict]) -> int:
    """Role-binding programs with an empty role: generated before the decode kept the special role tokens."""
    return sum(bool(_EMPTY_ROLE.search(o.get("generated_slotted") or o.get("generated") or ""))
               for o in outputs if o.get("task") == "role_binding")


def steps_of(subtype: str) -> int | None:
    m = re.search(r"steps=(\d+)", subtype or "")
    return int(m.group(1)) if m else None


def score(outputs: list[dict], vm=vm_run) -> dict:
    """Per task: n, executes, correct, copied, unknown slot; arithmetic also by step count."""
    st: dict[str, Counter] = defaultdict(Counter)
    for o in outputs:
        task = o["task"]
        ok, res = vm(o["generated"])
        if task == "plan":
            good = ok and plan_correct(o["generated"], o.get("reference", ""))
        elif o.get("gold") is not None:
            good = ok and value_correct(res, o["gold"])
        else:
            good = ok
        reason = str(o.get("slot_reason") or "")
        slots_ok = bool(o.get("slots_ok", True))
        for key in (task, "_all") + ((f"arithmetic/steps={steps_of(o.get('subtype', ''))}",)
                                      if task == "arithmetic" and steps_of(o.get("subtype", "")) else ()):
            c = st[key]
            c["n"] += 1
            c["executes"] += int(ok)
            c["vm_correct"] += int(good)                    # what the VM would say if the host let it run
            c["correct"] += int(good and slots_ok)          # a copied literal / unknown slot is refused, never run
            c["copied"] += int(reason.startswith("copied"))
            c["unknown_slot"] += int(reason.startswith("unknown slot"))
    return {k: {"n": c["n"], "executes": c["executes"] / c["n"], "correct": c["correct"] / c["n"],
                "vm_correct": c["vm_correct"] / c["n"], "copied": c["copied"], "unknown_slot": c["unknown_slot"]}
            for k, c in sorted(st.items())}


def verdict(ctrl_val: dict, pf_val: dict, ctrl_held: dict, pf_held: dict) -> dict:
    delta = pf_held["_all"]["correct"] - ctrl_held["_all"]["correct"]
    drops = {t: ctrl_val[t]["correct"] - pf_val[t]["correct"] for t in ctrl_val
             if t != "_all" and "/" not in t and t in pf_val}
    worst = max(drops.items(), key=lambda kv: kv[1]) if drops else (None, 0.0)
    slots = {"copied": pf_val["_all"]["copied"] + pf_held["_all"]["copied"],
             "unknown_slot": pf_val["_all"]["unknown_slot"] + pf_held["_all"]["unknown_slot"]}
    checks = {f"held-out correct +{GATE_PTS * 100:.0f} pts": delta >= GATE_PTS,
              f"no v12e val family down > {DROP_PTS * 100:.0f} pts": worst[1] <= DROP_PTS,
              "0 copied literals, 0 unknown slots (program-first arm)": slots["copied"] == 0 and slots["unknown_slot"] == 0}
    if delta < KILL_PTS:
        call = f"KILL: +{delta * 100:.1f} pts < +{KILL_PTS * 100:.0f} -- binding, not wording, is the bottleneck"
    elif all(checks.values()):
        call = f"PASS: +{delta * 100:.1f} pts on the held-out eval"
    else:
        call = "NEITHER: " + "; ".join(k for k, v in checks.items() if not v) + f" (delta +{delta * 100:.1f} pts)"
    return {"delta_held": delta, "val_drops": drops, "worst_val_drop": {"family": worst[0], "pts": worst[1]},
            "pf_arm_slots": slots, "checks": checks, "verdict": call}


def _table(name: str, ctrl: dict, pf: dict) -> list[str]:
    lines = [f"\n{name}", f"  {'':22s} {'n':>5s}  {'control':>9s}  {'program-first':>13s}  {'delta':>7s}"]
    for k in sorted(set(ctrl) | set(pf), key=lambda k: (k == "_all", k)):
        c, p = ctrl.get(k), pf.get(k)
        if not c or not p:
            continue
        lines.append(f"  {k:22s} {c['n']:5d}  {c['correct']:9.3f}  {p['correct']:13.3f}  {(p['correct'] - c['correct']) * 100:+6.1f}")
    return lines


def find_files(drive: str) -> list[str]:
    names = [("cubby450m_v12e_cont", "val_generations_v12e_cont.json"),
             ("cubby450m_v12e_cont", "val_generations_v12e_cont_pfheld.json"),
             ("cubby450m_v12e_pf_cont", "val_generations_v12e_pf_cont.json"),
             ("cubby450m_v12e_pf_cont", "val_generations_v12e_pf_cont_pfheld.json")]
    return [os.path.join(drive, d, f) for d, f in names]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", default="", help="the Drive folder holding both adapter folders (cubbyllm/emitter)")
    ap.add_argument("--files", nargs=4, metavar=("CTRL_VAL", "CTRL_HELD", "PF_VAL", "PF_HELD"))
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "exp_he18_ab.json"))
    a = ap.parse_args(argv)
    files = a.files or find_files(a.drive)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    loaded = [json.load(open(f, encoding="utf-8"))["outputs"] for f in files]
    stale = [os.path.basename(f) for f, o in zip(files, loaded) if stale_decode(o)]
    if stale:
        raise SystemExit("decoded before the role-token fix (role-binding binds with an empty role): "
                         + ", ".join(stale) + " -- run the notebook's regenerate cell first")
    ctrl_val, ctrl_held, pf_val, pf_held = (score(o) for o in loaded)
    v = verdict(ctrl_val, pf_val, ctrl_held, pf_held)
    lines = _table("v12e val (the same records for both arms)", ctrl_val, pf_val)
    lines += _table("program-first held-out eval (the other writer; shapes and seeds never trained on)", ctrl_held, pf_held)
    lines += ["", f"program-first arm slots: {v['pf_arm_slots']}", *(f"  [{'x' if ok else ' '}] {k}" for k, ok in v["checks"].items()),
              f"VERDICT: {v['verdict']}"]
    print("\n".join(lines))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"files": [os.path.basename(f) for f in files], "control": {"val": ctrl_val, "held": ctrl_held},
               "program_first": {"val": pf_val, "held": pf_held}, **v}, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
