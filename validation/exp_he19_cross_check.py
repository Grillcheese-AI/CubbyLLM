"""H-E19 cross-model speak gate: the 450M step loop against the 2.6B one-pass stand-in.

Gate D (self-voting) cannot reach spoken-wrong <= 1%: the 450M is confidently wrong.
This read asks whether two independent routes through the VM - the 450M write-back
loop and the stand-in's one-pass program - agreeing on the value is a usable speak
gate: speak only when both VM results match, refuse otherwise.

Inputs (no model, no GPU): the loop's results json(s) from exp_he19_step_loop.py and
the stand-in's eval json on the same held-out questions.

    python validation/exp_he19_cross_check.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
STANDIN = ROOT / "standin" / "data" / "out" / "eval_emitter_vm_he19_standin_v12e_pfheld.json"


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def same(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def cross(loop_rows, standin):
    """Speak only when the loop's answer equals the stand-in's VM result."""
    n = len(loop_rows)
    agree = right = wrong = 0
    sb_right = sb_wrong = caught = 0
    for r in loop_rows:
        s = standin[r["id"]]
        a = num(r["answer"])
        sv = num(s["vm_result"]) if s.get("executes") else None
        gold = num(r["gold"])
        if s.get("gold_match"):
            sb_right += 1
        else:
            sb_wrong += 1
            caught += not same(a, sv)
        if same(a, sv):
            agree += 1
            if same(a, gold):
                right += 1
            else:
                wrong += 1
    return {
        "n": n,
        "spoken": agree / n,
        "right": right / n,
        "wrong": wrong / n,
        "precision": right / max(1, agree),
        "counts": {"agree": agree, "right": right, "wrong": wrong},
        "standin_alone_right": sb_right / n,
        "standin_wrong": sb_wrong,
        "standin_wrong_vetoed": caught,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=["step_held", "vote5_held"],
                    help="exp_he19_step_loop_<tag>.json files under validation/logs")
    ap.add_argument("--standin", type=Path, default=STANDIN)
    ap.add_argument("--out", type=Path, default=LOGS / "exp_he19_cross_check.json")
    a = ap.parse_args(argv)

    standin = {r["id"]: r for r in json.loads(a.standin.read_text(encoding="utf-8"))["rows"]}
    report = {}
    for tag in a.tags:
        rows = json.loads((LOGS / f"exp_he19_step_loop_{tag}.json").read_text(encoding="utf-8"))["results"]
        m = cross(rows, standin)
        report[tag] = m
        c = m["counts"]
        print(f"{tag}: n={m['n']} | 450M loop and 2.6B one-pass agree on {c['agree']} ({m['spoken']:.3f}): "
              f"right {c['right']} ({m['right']:.3f}), wrong {c['wrong']} ({m['wrong']:.3f}), "
              f"precision {m['precision']:.3f} | stand-in alone right {m['standin_alone_right']:.3f}")
        print(f"   stand-in wrong {m['standin_wrong']}; the 450M loop disagrees (veto) on {m['standin_wrong_vetoed']}")
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  -> {a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
