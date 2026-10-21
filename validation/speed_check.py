"""speed_check -- which kernels a run should take, measured on its own card at its own shape.

The notebook passes its ``BASE`` environment (shape, micro, link, compile, ``CB_NO_FLEX``). The
kernels' tests run first on this card (``tests/model/backbone/test_mingru_triton.py``,
``test_window_attn_triton.py``: values and gradients against the references, the mixers under
their flags); then ``profile_base.py`` once per variant whose tests passed:

  torch        the run's environment as given (the dense-mask attention where flex is off);
  mingru       + the fused MinGRU kernel (``CB_MINGRU_TRITON=1``);
  mingru+band  + the banded flash-attention kernels (``CB_ATTN_TRITON=1``);
  mingru+flex  + flex attention with tiles for a 99 KB card (``CB_FLEX_OPTS``) -- only where the
               run had turned flex off.

The fastest variant that ran is the run's; its environment overrides, every variant's speed and
the tests' verdicts go to ``CB_SPEED_OUT`` (JSON), so the control arm of a pair can take the
first arm's choice and a resumed session reuses it.

    <train_base env> CB_SPEED_OUT=<json> python validation/speed_check.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.environ.get("CB_SPEED_OUT", "")
NOISE = ("dynamo", "Graph break", "UserWarning", "warnings.warn", "_warn_once", "originates from",
         "_rope_tables", "_rope_inv_freq", "warn_once")
FLEX_SMALL = {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 1, "num_warps": 8,
              "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32}
TESTS = {"mingru": "tests/model/backbone/test_mingru_triton.py", "band": "tests/model/backbone/test_window_attn_triton.py"}


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


def tested(name: str) -> bool:
    """The test file passed on this card: read from pytest's JUnit XML, not its console (the repo's
    ``addopts = -q`` plus a ``-q`` here is ``-qq``, which prints no summary line at all; Colab colours
    what it does print; torch can write after it)."""
    print(f"\n--- {TESTS[name]} on this card ---", flush=True)
    xml = os.path.join(tempfile.gettempdir(), f"speed_check_{name}.xml")
    code, _ = stream([sys.executable, "-m", "pytest", TESTS[name], "-rs", "-p", "no:cacheprovider", f"--junitxml={xml}"],
                     dict(os.environ))
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(xml).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        n = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    except Exception as e:  # noqa: BLE001 -- no report is a failed check, not a crash
        print(f"    -> no JUnit report ({type(e).__name__}): NOT passed", flush=True)
        return False
    ok = code == 0 and n["failures"] == 0 and n["errors"] == 0 and n["tests"] > n["skipped"]
    print(f"    -> {'passed' if ok else 'NOT passed'} ({n})", flush=True)
    return ok


def profile(name: str, extra: dict) -> dict:
    env = dict(os.environ, CB_PROF_STEPS=os.environ.get("CB_PROF_STEPS", "4"), **extra)
    env.pop("CB_PROF_TRACE", None)
    print(f"\n--- profile_base: {name} {extra or ''} ---", flush=True)
    code, lines = stream([sys.executable, "-u", os.path.join(HERE, "profile_base.py")], env)
    step = next((ln for ln in lines if ln.startswith("step (")), "")
    m = re.search(r"([\d,]+) tok/s", step)
    kinds = [ln.strip() for ln in lines if re.match(r"\s+\S.*\d+\.\d%\s+\d", ln)][:8]
    return {"env": extra, "exit": code, "tok_s": float(m.group(1).replace(",", "")) if m and code == 0 else 0.0,
            "step": step.strip(), "kinds": kinds}


def main() -> None:
    ok = {name: tested(name) for name in TESTS}
    variants = {"torch": {}}
    if ok["mingru"]:
        variants["mingru"] = {"CB_MINGRU_TRITON": "1"}
    base = variants.get("mingru", {})
    if ok["band"]:
        variants["mingru+band" if base else "band"] = {**base, "CB_ATTN_TRITON": "1"}
    if os.environ.get("CB_NO_FLEX") == "1":
        variants["mingru+flex" if base else "flex"] = {**base, "CB_NO_FLEX": "0", "CB_FLEX_OPTS": json.dumps(FLEX_SMALL)}
    res = {name: profile(name, extra) for name, extra in variants.items()}
    ran = {n: r for n, r in res.items() if r["tok_s"] > 0}
    chosen = max(ran, key=lambda n: ran[n]["tok_s"]) if ran else "torch"
    a = res["torch"]["tok_s"]
    print("\nspeed check (tok/s under the profiler, the optimizer every micro):")
    for n, r in res.items():
        gain = f"{100 * (r['tok_s'] / a - 1):+.1f}%" if a and r["tok_s"] else "failed"
        print(f"  {n:12s} {r['tok_s']:>10,.0f}  {gain}{'   <- this run' if n == chosen else ''}")
    rep = {"tests": ok, "variants": res, "chosen": chosen, "env": res[chosen]["env"]}
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
