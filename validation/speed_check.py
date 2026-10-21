"""speed_check -- whether a run should take the fused MinGRU kernel (``CB_MINGRU_TRITON=1``).

On the card at hand, with the run's own environment (shape, micro, link, compile -- the
notebook passes its ``BASE``): the kernel's tests first (``tests/model/backbone/
test_mingru_triton.py``: values and gradients against the recurrence, the mixer under the
flag), then ``profile_base.py`` with the flag off and on. The run takes the kernel only if
the tests pass and it is faster; the verdict, both speeds and both profiles' summaries go
to ``CB_SPEED_OUT`` (JSON), so the control arm of a pair can follow the first arm's choice.

    <train_base env> CB_SPEED_OUT=<json> python validation/speed_check.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.environ.get("CB_SPEED_OUT", "")
NOISE = ("dynamo", "Graph break", "UserWarning", "warnings.warn", "_warn_once", "originates from",
         "_rope_tables", "_rope_inv_freq", "warn_once")


def stream(cmd: list, env: dict) -> tuple[int, list]:
    """Run ``cmd`` in the repo, printing its output as it comes (a captured subprocess shows
    nothing in Colab until it ends); returns (exit code, lines)."""
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not any(n in line for n in NOISE) and line.strip():
            print(line, end="", flush=True)
    return p.wait(), lines


def profile(flag: str) -> dict:
    env = dict(os.environ, CB_MINGRU_TRITON=flag, CB_PROF_STEPS=os.environ.get("CB_PROF_STEPS", "4"))
    env.pop("CB_PROF_TRACE", None)
    print(f"\n--- profile_base, CB_MINGRU_TRITON={flag} ---", flush=True)
    code, lines = stream([sys.executable, "-u", os.path.join(HERE, "profile_base.py")], env)
    step = next((ln for ln in lines if ln.startswith("step (")), "")
    m = re.search(r"([\d,]+) tok/s", step)
    kinds = [ln.strip() for ln in lines if re.match(r"\s+\S.*\d+\.\d%\s+\d", ln)][:8]
    return {"exit": code, "tok_s": float(m.group(1).replace(",", "")) if m else 0.0, "step": step.strip(), "kinds": kinds}


def main() -> None:
    print("--- the fused MinGRU kernel's tests on this card ---", flush=True)
    code, lines = stream([sys.executable, "-m", "pytest", "tests/model/backbone/test_mingru_triton.py", "-q",
                          "-p", "no:cacheprovider", "-rs"], dict(os.environ))
    tail = lines[-1] if lines else ""
    tests_ok = code == 0 and "passed" in tail and "failed" not in tail and "error" not in tail
    rep = {"tests": tests_ok, "tests_tail": [ln.rstrip() for ln in lines[-4:]]}
    if tests_ok:
        rep["off"], rep["on"] = profile("0"), profile("1")
        a, b = rep["off"]["tok_s"], rep["on"]["tok_s"]
        rep["speedup"] = round(b / a, 4) if a else None
        rep["use"] = bool(a and b > a and rep["on"]["exit"] == 0)
        print(f"\nfused MinGRU: {a:,.0f} -> {b:,.0f} tok/s ({100 * (b / a - 1) if a else float('nan'):+.1f}%) under the "
              f"profiler -> {'USE it' if rep['use'] else 'keep the torch path'}", flush=True)
    else:
        rep["use"] = False
        print("\nthe kernel's tests did not pass here: keep the torch path", flush=True)
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
