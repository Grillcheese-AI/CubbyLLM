"""Order 0078, step 3's two reads of `gsm8k.cube` (cubelang branch `agent-a/gsm8k`), at home, CPU only.

1. **The gold replay** on GSM8K train's 7,473 golds: `solve` replays each `<<expr=result>>` step; the share whose
   replay `verify`s against the gold's `#### n` (within 0.01), and every miss by kind: no annotation, an annotation
   whose stated result its own expression does not give (`steps_agree` 0: the dataset's slip, counted, never fixed),
   or a replay that is not the gold's answer.
2. **The checker against school.py's**, on the same (problem, proposed program) pairs: for each sampled problem, the
   gold written as a program of the arithmetic dialect (create / assign / add / sub / mul / div) and mutants of it
   (stopped early, the answer negated, scaled by 60, off by a stated number, off by 1 more than any of them). Each
   pair judged by `school.py` (`arith_step_values` of the program, then `classify_error` against the gold's step
   values and the question's numbers; right when the value is the gold's) and by `gsm8k.cube`'s `classify_error`
   (the same program's step values, written "v1;v2"). Both must give the same code on every pair (**kill** below
   100%).

Counts only in the log (no benchmark text). The VM binary and the .cube file are given as arguments (never a path in
this file).

  python validation/exp_gsm8k_cube.py --exe <cubelang.exe> --cube <gsm8k.cube> [--pairs 1000] --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.reasoning import school  # noqa: E402

__wiring__ = "STANDALONE"

KINDS = {"stopped early": 1, "reversed": 2, "unit conversion": 3, "one number off": 4, "other": 5}
ANN = re.compile(r"<<([^=>]*)=")
NUM = re.compile(r"\d(?:\d|,(?=\d{3}(?!\d)))*(?:\.\d+)?|\.\d+")


def call(exe: str, cube: str, fn: str, *args) -> float | None:
    cmd = [exe, "run", cube, "--fn", fn, "--json"]
    for a in args:
        cmd += ["--arg", str(a)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:                                    # noqa: BLE001
        return None
    return got.get("result") if got.get("ok") else None


def evaluate(expr: str) -> float:
    """An annotation's expression, by precedence, as `gsm8k.cube` reads it (spaces out, thousands commas)."""
    s = re.sub(r"\s", "", expr)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    if not re.fullmatch(r"[0-9.+\-*/()]*", s) or not s:
        raise ValueError(expr)
    return float(eval(s, {"__builtins__": {}}))           # the characters are checked above: numbers and + - * / ( )


def gold_values(answer: str) -> list[float]:
    return [evaluate(e) for e in ANN.findall(answer)]


def stated(question: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM.findall(question)]


def program(values: list[float]) -> str:
    """Values as a program of the arithmetic dialect: one register a step, the last the answer."""
    lines = []
    for i, v in enumerate(values):
        lines += [f"        create s{i}: number;", f"        assign s{i} = {plain(v)};"]
    return "program P implements ISolve {\n    public function solve(x: str): number {\n" + "\n".join(lines) + "\n    }\n}\n"


def plain(v: float) -> str:
    """A value as both readers parse it: no exponent, no trailing zeros."""
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def school_kind(values: list[float], right: list[float], st: list[float]) -> int | None:
    """school.py's verdict on a program with these step values: 0 right, else its error's code (None: not read)."""
    got = school.arith_step_values(program(values))
    if not got:
        return None
    gold = right[-1]
    if abs(got[-1] - gold) <= 1e-6 * max(1.0, abs(gold)):
        return 0
    return KINDS[school.classify_error(got[-1], gold, st, right, got)]


def mutants(rv: list[float], st: list[float], rng) -> list[list[float]]:
    out = [list(rv)]
    if len(rv) > 1:
        out.append(rv[:-1])                              # stopped early
    out.append(rv[:-1] + [-rv[-1]])                      # reversed
    out.append(rv[:-1] + [rv[-1] * 60])                  # unit conversion
    if st:
        out.append(rv[:-1] + [rv[-1] + rng.choice(st)])  # one number off
    out.append(rv[:-1] + [rv[-1] + max([abs(x) for x in st + rv] + [0]) + 1.5])   # other, by construction
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--pairs", type=int, default=1000, help="problems sampled for the checker's agreement")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in open(Path(local_paths.value("CUBBY_DATASETS") or ".") / "gsm8k" / "train.jsonl",
                                          encoding="utf-8")]

    def replay(r):
        ans = r["answer"]
        n = call(a.exe, a.cube, "parse", ans)
        if not n:
            return "no annotation"
        v = call(a.exe, a.cube, "solve", ans)
        if v is None:
            return "the VM stopped"
        if call(a.exe, a.cube, "verify", ans, v) == 1:
            return "replay equals the gold"
        if call(a.exe, a.cube, "steps_agree", ans) == 0:
            return "an annotation's result is not its expression's (the dataset's slip)"
        return "the last annotation is not the answer (the gold states it in prose after its steps)"
    with ThreadPoolExecutor(a.workers) as ex:
        kinds = Counter(ex.map(replay, rows))
    rng, judged, disagree = random.Random(78), Counter(), []
    sample = [r for r in rng.sample(rows, min(a.pairs, len(rows))) if ANN.search(r["answer"])]

    def judge(r):
        try:
            rv, st = gold_values(r["answer"]), stated(r["question"])
        except ValueError:
            return []
        out = []
        for m in mutants(rv, st, random.Random(r["id"])):
            m = [float(plain(x)) for x in m]
            want = school_kind(m, rv, st)
            if want is None:
                continue
            got = call(a.exe, a.cube, "classify_error", r["answer"], r["question"], ";".join(plain(x) for x in m))
            out.append((r["id"], want, got))
        return out
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(judge, sample):
            for rid, want, got in res:
                judged["pairs"] += 1
                judged[f"school: {want}"] += 1
                if got != want:
                    judged["disagree"] += 1
                    disagree.append({"id": rid, "school": want, "gsm8k.cube": got})
    report = {"order": "0078", "what": "step 3: gsm8k.cube's gold replay on GSM8K train, and its checker against school.py's",
              "replay": {"golds": len(rows), **dict(kinds.most_common()),
                         "share equal to the gold": round(kinds["replay equals the gold"] / len(rows), 4)},
              "checker": {**dict(sorted(judged.items())), "agreement": round(1 - judged["disagree"] / max(1, judged["pairs"]), 4),
                          "kill (below 100%)": judged["disagree"] > 0, "disagreements (ids)": disagree[:50]},
              "codes": {"0": "right", **{str(v): k for k, v in KINDS.items()}}}
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("replay", "checker")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
