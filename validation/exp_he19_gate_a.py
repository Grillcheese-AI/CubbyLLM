"""exp_he19_gate_a - H-E19 Gate A readout: the v12e_w + TinyGSM adapter (cubby450m_v12e_w_tg_cont) against the
v12e control (cubby450m_v12e_cont) on arithmetic, every program run in the real VM (exp_he18_ab.score).

Gate A (pre-registered, CUBBYLLM_HYPOTHESES.md H-E19): arithmetic correct at one try >= 20% on the v12e val
arithmetic and on the program-first held-out arithmetic; copied literals <= control. Kill: < 10%, read on the
held-out eval (never trained on, another writer's wording).

Besides the score, the diagnosis a failed gate needs: how many programs never close (ran to the token cap),
how many repeat a step verbatim (a loop), how the generated step count compares with the reference's, and
(--stops) what a perfect stop would buy: each program cut after every step and run in the VM.

    python validation/exp_he19_gate_a.py --drive "<Drive>/cubbyllm/emitter" [--stops]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from exp_he18_ab import score, steps_of  # noqa: E402  (sets up the repo paths)

GATE, KILL = 0.20, 0.10
_STEP = re.compile(r"^\s*create\s+s\d+\s*:", re.M)
_END = re.compile(r"^\s*(?:sum|query|return)\b|^\s*\}", re.M)       # the closing ops end the last body
_BODY = re.compile(r"create\s+s(\d+)\s*:[^\n]*\n((?:(?!\s*create\s).*\n?)*)")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [o for o in (d["outputs"] if isinstance(d, dict) else d) if o.get("task") == "arithmetic"]


def closed(program: str) -> bool:
    """The program ends: balanced braces and a return."""
    p = program or ""
    return p.count("{") == p.count("}") and p.count("{") > 0 and "return" in p


def looped(program: str) -> bool:
    """Some step body (ops after its create, step index stripped) occurs 3+ times."""
    bodies = [re.sub(r"s\d+", "s", _END.split(m.group(2))[0]).strip() for m in _BODY.finditer(program or "")]
    return any(n >= 3 for n in Counter(b for b in bodies if b).values())


def shape(outputs: list[dict]) -> dict:
    n = len(outputs) or 1
    st = Counter()
    for o in outputs:
        g = o.get("generated_slotted") or o.get("generated") or ""
        ref, gen = len(_STEP.findall(o.get("reference") or "")), len(_STEP.findall(g))
        st["closed"] += closed(g)
        st["looped"] += looped(g)
        st["same_steps"] += gen == ref
        st["more_steps"] += gen > ref
        st["fewer_steps"] += gen < ref
    return {k: v / n for k, v in sorted(st.items())}


_CREATE = re.compile(r"^(\s*)create\s+(s\d+)\s*:", re.M)


def prefixes(program: str, limit: int = 12) -> list[tuple[int, str]]:
    """(k, program cut after its k-th complete step and closed on that step's variable), k = 1..; a block is
    complete when another step (or the closing ops) follows it, so a program cut at the token cap drops its last."""
    p = program or ""
    marks = list(_CREATE.finditer(p))
    out = []
    for k, m in enumerate(marks[:limit], 1):
        end = marks[k].start() if k < len(marks) else None
        if end is None:
            tail = re.search(r"^\s*(sum|query|return)\b", p[m.end():], re.M)
            if not tail:
                break
            end = m.end() + tail.start()
        ind, var = m.group(1).strip("\n"), m.group(2)
        out.append((k, p[:end].rstrip() + f"\n{ind}sum {var};\n{ind}query {var};\n{ind}return {var};\n    }}\n}}\n"))
    return out


def stops(outputs: list[dict]) -> dict:
    """If the host could stop the program at the right place: correct when cut at the reference's step count,
    and correct at ANY cut (an oracle stop: the answer was computed at some step and the program ran on)."""
    from exp_he18_ab import vm_run, value_correct
    st = Counter()
    for o in outputs:
        ref_k = len(_STEP.findall(o.get("reference") or ""))
        hits = [k for k, prog in prefixes(o.get("generated") or "")
                if (lambda r: r[0] and value_correct(r[1], o["gold"]))(vm_run(prog))]
        st["n"] += 1
        st["at_ref_len"] += ref_k in hits
        st["any_cut"] += bool(hits)
        st["first_step_right"] += any(k <= ref_k for k in hits)
    n = st.pop("n") or 1
    return {k: v / n for k, v in sorted(st.items())}


def read(outputs: list[dict]) -> dict:
    s = score(outputs)
    out = {"arithmetic": s.get("arithmetic", {}), "by_steps": {k.split("=", 1)[1]: v for k, v in s.items()
                                                              if k.startswith("arithmetic/steps=")}}
    out["shape"] = shape(outputs)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", required=True, help="<Drive>/cubbyllm/emitter")
    ap.add_argument("--stops", action="store_true", help="also cut each program at every step (oracle stop)")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "exp_he19_gate_a.json"))
    a = ap.parse_args(argv)
    f = lambda tag, suf="": os.path.join(a.drive, f"cubby450m_{tag}", f"val_generations_{tag}{suf}.json")
    tg_val, tg_held = load(f("v12e_w_tg_cont")), load(f("v12e_w_tg_cont", "_pfheld"))
    c_val, c_held = load(f("v12e_cont")), load(f("v12e_cont", "_pfheld"))
    ids = {o["id"] for o in c_val}
    cut = {}
    if a.stops:
        for name, outs in (("ctrl_held", c_held), ("tg_held", tg_held), ("ctrl_val", c_val), ("tg_val_all", tg_val)):
            cut[name] = stops(outs)
            print(f"stops {name:12s}", {k: round(v, 3) for k, v in cut[name].items()}, flush=True)
    res = {"stops": cut, "tg_val_all": read(tg_val), "tg_val_matched": read([o for o in tg_val if o["id"] in ids]),
           "ctrl_val": read(c_val), "tg_held": read(tg_held), "ctrl_held": read(c_held)}
    held_ids = {o["id"] for o in c_held}
    res["held_matched_ids"] = len(held_ids & {o["id"] for o in tg_held})
    acc = [res["tg_val_all"]["arithmetic"].get("correct", 0), res["tg_held"]["arithmetic"].get("correct", 0)]
    copied_ok = (res["tg_held"]["arithmetic"].get("copied", 0) <= res["ctrl_held"]["arithmetic"].get("copied", 0))
    # the kill reads the held-out eval: never trained on, written by another writer -- planning, not the look of GSM
    res["verdict"] = ("PASS" if min(acc) >= GATE and copied_ok else "KILL" if acc[1] < KILL else "FAIL")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for name in ("ctrl_val", "tg_val_matched", "tg_val_all", "ctrl_held", "tg_held"):
        r = res[name]
        ar, sh = r["arithmetic"], r["shape"]
        print(f"{name:15s} n={ar.get('n', 0):4d} correct={ar.get('correct', 0):.3f} vm={ar.get('vm_correct', 0):.3f} "
              f"exec={ar.get('executes', 0):.3f} copied={ar.get('copied', 0)} unknown={ar.get('unknown_slot', 0)} "
              f"closed={sh.get('closed', 0):.2f} looped={sh.get('looped', 0):.2f} same_steps={sh.get('same_steps', 0):.2f} "
              f"more={sh.get('more_steps', 0):.2f} fewer={sh.get('fewer_steps', 0):.2f}")
        print("   by steps: " + "  ".join(f"{k}:{v['correct']:.2f}/{v['n']}" for k, v in
                                          sorted(r["by_steps"].items(), key=lambda kv: int(kv[0]))))
    print(f"held matched ids {res['held_matched_ids']}  VERDICT {res['verdict']}  -> {a.out}")


if __name__ == "__main__":
    main()
