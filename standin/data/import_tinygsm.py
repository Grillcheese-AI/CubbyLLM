"""import_tinygsm -- TinyGSM's Python solutions as CubeLang arithmetic programs (2026-09-30).

TinyGSM (arXiv 2312.09241, MIT): 12.3M grade-school problems, each solved by a short straight-line Python
function. A 125M model trained on it reached 63.1% on GSM8K; our 450M emitter saw ~6K GSM programs and
gets 3%. The plans are what the emitter lacks, and TinyGSM is 2,000x more of them.

Conversion, never by running the dataset's code:
  1. `ast.parse` the function; accept only assignments (and `x += y`) of numbers, names and + - * / --
     no call, loop, branch, list or attribute: anything else is a reject, counted by reason
  2. each binary operation becomes one step of v12e's dialect (`create sK`, `assign`, one op, the copy
     `assign 0; add sK, sJ` when the left side is an earlier step, `sum; query; return` on the last);
     steps the result does not use are pruned
  3. our own evaluator computes every value (floats, a zero division is a reject); a sample is re-run in
     the real VM, which must agree
  4. the literal filter: every number the program uses must be a number the question states (digits by
     value, or in words: "four", "twice", "a dozen") or a unit constant from a closed list (60, 24, 7, ...)
     -- a program that invents a number ("5.5 hours" for "5 hours and a 30 minute break") is dropped
  5. questions sharing a 10-word run with an eval question (v12e val, the program-first held-out) are dropped
The records are v12e's schema (task arithmetic, `gold` = the value), so `emitter_data.py` slots them
unchanged, and each carries its difficulty for the curriculum (steps, ops, word numbers, unit constants,
distractors).

    python standin/data/import_tinygsm.py convert --parquet <train-00000-of-00017.parquet> \
        --out standin/data/out/tinygsm/tinygsm_s0.jsonl [--limit N] [--vm-sample 2000] [--exclude <eval.jsonl> ...]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE, os.path.join(ROOT, "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cubbyllm.reasoning.slots import canon, extract, num_value  # noqa: E402

OPS = {ast.Add: ("+", "add"), ast.Sub: ("-", "sub"), ast.Mult: ("*", "mul"), ast.Div: ("/", "div")}
# the few conversion factors a question implies without stating them; each is a fact of the arithmetic world
UNIT_CONSTANTS = {1.0, 100.0, 60.0, 24.0, 7.0, 12.0, 52.0, 365.0, 30.0, 1000.0, 10.0, 16.0, 3600.0}
MAX_STEPS = 10


class Reject(Exception):
    pass


@dataclass(frozen=True)
class Operand:
    kind: str            # "lit" | "reg"
    v: float | int       # the literal, or the step index


@dataclass(frozen=True)
class Step:
    a: Operand
    op: str              # + - * /
    b: Operand


def _function(code: str) -> ast.FunctionDef:
    import warnings
    try:
        with warnings.catch_warnings():              # a docstring's "\$" is the dataset's, not ours to report
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        raise Reject("syntax")
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or len(tree.body) != 1:
        raise Reject("not one function")
    return fns[0]


def compile_steps(code: str) -> tuple[list[Step], list[float]]:
    """The function's arithmetic as steps (pruned to what the result uses) and each step's value."""
    fn = _function(code)
    env: dict[str, Operand] = {}
    steps: list[Step] = []
    vals: list[float] = []

    def value(o: Operand) -> float:
        return float(o.v) if o.kind == "lit" else vals[o.v]

    def emit(node) -> Operand:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Operand("lit", node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            raise Reject("negative literal")
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise Reject("unbound name")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = emit(node.left), emit(node.right)
            sym = OPS[type(node.op)][0]
            x, y = value(a), value(b)
            if sym == "/" and y == 0:
                raise Reject("zero division")
            v = x + y if sym == "+" else x - y if sym == "-" else x * y if sym == "*" else x / y
            if not math.isfinite(v) or abs(v) > 1e9:
                raise Reject("value out of range")
            steps.append(Step(a, sym, b)); vals.append(v)
            return Operand("reg", len(steps) - 1)
        raise Reject(f"node {type(node).__name__}")

    result = None
    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            continue                                              # the docstring
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = emit(st.value)
        elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and type(st.op) in OPS:
            env[st.target.id] = emit(ast.BinOp(ast.Name(st.target.id), st.op, st.value))
        elif isinstance(st, ast.Return) and st.value is not None:
            result = emit(st.value)
            break
        else:
            raise Reject(f"statement {type(st).__name__}")
    if result is None:
        raise Reject("no return")
    if result.kind == "lit":
        raise Reject("no computation")
    return _prune(steps, vals, result.v)


def _prune(steps: list[Step], vals: list[float], last: int) -> tuple[list[Step], list[float]]:
    """Keep the steps the result depends on, renumbered in order; the result is the last."""
    need, stack = set(), [last]
    while stack:
        k = stack.pop()
        if k in need:
            continue
        need.add(k)
        stack += [o.v for o in (steps[k].a, steps[k].b) if o.kind == "reg"]
    order = sorted(need)
    new = {old: i for i, old in enumerate(order)}

    def ren(o: Operand) -> Operand:
        return Operand("reg", new[o.v]) if o.kind == "reg" else o
    return [Step(ren(steps[k].a), steps[k].op, ren(steps[k].b)) for k in order], [vals[k] for k in order]


def fmt(x: float | int) -> str:
    return canon(float(x))


_HEAD = ("program {name} implements ISolver {{\n    type Input = str;\n    type Output = quantity;\n    @external\n"
         "    public function parse(raw: str): Input {{ return raw; }}\n\n"
         "    public pure function verify(input: Input, output: Output): bool {{ return true; }}\n\n\n\n"
         "    @external\n    public function solve(input: Input): Output {{\n")


def render(steps: list[Step], vals: list[float], question: str = "", name: str | None = None) -> str:
    """v12e's dialect, byte for byte the shape `build_program_first.arith_program` writes, with float values."""
    name = name or f"GSM{int(hashlib.sha1(repr(steps).encode()).hexdigest()[:6], 16) % 10000}"
    lines = []
    for k, st in enumerate(steps):
        a = fmt(st.a.v) if st.a.kind == "lit" else fmt(vals[st.a.v])
        b = fmt(st.b.v) if st.b.kind == "lit" else fmt(vals[st.b.v])
        lines.append(f"        create s{k} : quantity;   # step {k}: {a} {st.op} {b} = {fmt(vals[k])}")
        if st.a.kind == "lit":
            lines.append(f"        assign s{k} = {fmt(st.a.v)};")
        else:
            lines += [f"        assign s{k} = 0;", f"        add s{k}, s{st.a.v};"]
        rhs = f"s{st.b.v}" if st.b.kind == "reg" else fmt(st.b.v)
        lines.append(f"        {dict(OPS_BY_SYM)[st.op]} s{k}, {rhs};")
    last = len(steps) - 1
    lines += [f"        sum s{last};", f"        query s{last};", f"        return s{last};"]
    head = f"# {' '.join(question.split())}\n" if question else ""
    return head + _HEAD.format(name=name) + "\n".join(lines) + "\n    }\n}\n"


OPS_BY_SYM = [(sym, word) for sym, word in OPS.values()]


def question_numbers(question: str) -> set[float]:
    """The numbers a question states, by value: digits and the numbers it writes in words."""
    return {num_value(s.filled) for s in extract(question, numbers=True, words=True).spans
            if s.kind == "N" and num_value(s.filled) is not None}


def difficulty(steps: list[Step], question: str) -> dict:
    """What the curriculum orders by: steps, the ops used, numbers in words, unit constants, distractors."""
    stated = question_numbers(question)
    lits = [float(o.v) for st in steps for o in (st.a, st.b) if o.kind == "lit"]
    words = {num_value(s.filled) for s in extract(question).spans if s.kind == "N" and s.value is not None}
    return {"steps": len(steps), "ops": "".join(sorted({st.op for st in steps})),
            "word_numbers": sum(v in words for v in lits),
            "unit_constants": sum(v not in stated for v in lits),
            "distractors": len(stated - set(lits)),
            "decimals": any(v != int(v) for v in lits),
            "max_literal": max((abs(v) for v in lits), default=0.0)}


_WORD_RX = re.compile(r"[a-z0-9']+")


def shingles(text: str, n: int = 10) -> set[tuple]:
    w = _WORD_RX.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def eval_shingles(paths: list[str]) -> set[tuple]:
    out: set[tuple] = set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("split", "val") == "val":
                out |= shingles(r.get("question") or r.get("prompt", ""))
    return out


def to_record(question: str, code: str, idx: int, units: set[float] = UNIT_CONSTANTS) -> dict:
    """One TinyGSM row -> a v12e-schema arithmetic record, or Reject(reason)."""
    steps, vals = compile_steps(code)
    if len(steps) > MAX_STEPS:
        raise Reject("too many steps")
    stated = question_numbers(question)
    for st in steps:
        for o in (st.a, st.b):
            if o.kind == "lit" and float(o.v) not in stated and float(o.v) not in units:
                raise Reject("invented number")
    ans = vals[-1]
    if abs(ans * 100 - round(ans * 100)) > 1e-6:
        raise Reject("answer not whole or cents")
    q = " ".join(question.split())
    return {"id": f"tinygsm-{idx}", "task": "arithmetic", "subtype": f"steps={len(steps)}", "split": "train",
            "source": "tinygsm", "prompt": q, "program": render(steps, vals, q), "gold": fmt(ans),
            "difficulty": difficulty(steps, q)}


def vm_check(records: list[dict]) -> Counter:
    """Re-run programs in the real VM: its value must be ours."""
    from build_emitter_sft import answer_fn, shim_isolver
    from cubbyllm.bridges import cubelang_client as cc
    c = Counter()
    for r in records:
        src = shim_isolver(r["program"])
        try:
            out = cc.run_program_proto(src, fn=answer_fn(src))
            ok, res = bool(out.get("ok")), out.get("result")
        except Exception:
            ok, res = False, None
        v = num_value(str(res)) if ok else None
        c["runs"] += int(ok)
        c["agrees"] += int(v is not None and abs(v - float(r["gold"])) <= 1e-6 * max(1.0, abs(v)))
        c["n"] += 1
    return c


def convert(parquet: str, out: str, limit: int = 0, exclude: list[str] | None = None, vm_sample: int = 0,
            seed: int = 0, log=print) -> dict:
    t0 = time.time()
    if parquet.endswith(".jsonl"):                   # rows saved from the datasets-server rows API
        rows = [json.loads(line) for line in open(parquet, encoding="utf-8")]
        qs, codes = [r["question"] for r in rows], [r["code"] for r in rows]
    else:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet, columns=["question", "code"])
        qs, codes = table.column("question").to_pylist(), table.column("code").to_pylist()
    n = min(len(qs), limit) if limit else len(qs)
    banned = eval_shingles(exclude or [])
    why, kept, seen = Counter(), [], set()
    by_steps = Counter()
    base = int(hashlib.sha1(os.path.basename(parquet).encode()).hexdigest()[:6], 16) * 10_000_000
    for i in range(n):
        q, code = qs[i] or "", codes[i] or ""
        key = " ".join(_WORD_RX.findall(q.lower()))
        if key in seen:
            why["duplicate question"] += 1
            continue
        seen.add(key)
        if banned and shingles(q) & banned:
            why["overlaps an eval question"] += 1
            continue
        try:
            rec = to_record(q, code, base + i)
        except Reject as e:
            why[str(e)] += 1
            continue
        kept.append(rec)
        by_steps[rec["difficulty"]["steps"]] += 1
        if (i + 1) % 100_000 == 0:
            log(f"  {i + 1:,}/{n:,}: kept {len(kept):,} ({time.time() - t0:.0f}s)")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"parquet": os.path.basename(parquet), "rows": n, "kept": len(kept),
                "kept_rate": round(len(kept) / max(n, 1), 4), "rejects": dict(why.most_common()),
                "kept_by_steps": dict(sorted(by_steps.items())),
                "with_word_numbers": sum(r["difficulty"]["word_numbers"] > 0 for r in kept),
                "with_unit_constants": sum(r["difficulty"]["unit_constants"] > 0 for r in kept),
                "with_distractors": sum(r["difficulty"]["distractors"] > 0 for r in kept),
                "wall_s": round(time.time() - t0, 1), "license": "MIT (TinyGSM, arXiv 2312.09241)"}
    if vm_sample and kept:
        sample = random.Random(seed).sample(kept, min(vm_sample, len(kept)))
        manifest["vm_check"] = dict(vm_check(sample))
    json.dump(manifest, open(out.replace(".jsonl", ".manifest.json"), "w", encoding="utf-8"), indent=1)
    return manifest


def mix(base: str, tinygsm: str, out: str, cap: int = 0, keep_steps: int = 5, seed: int = 0) -> dict:
    """The training file: every record of `base` (v12e in slot form, all splits) + TinyGSM's slot records, at
    most `cap` of them (every one with >= `keep_steps` steps kept -- the long plans are the rare ones -- the rest
    sampled), without the `reference` / `question` copies a train record never reads (the file goes to Drive)."""
    rng = random.Random(seed)
    tg = [json.loads(line) for line in open(tinygsm, encoding="utf-8")]
    steps_of = [int(str(r.get("subtype", "steps=0")).split("=")[-1] or 0) for r in tg]
    long_ = [r for r, s in zip(tg, steps_of) if s >= keep_steps]
    short = [r for r, s in zip(tg, steps_of) if s < keep_steps]
    if cap and len(tg) > cap:
        rng.shuffle(short)
        tg = long_ + short[:max(0, cap - len(long_))]
    n_base = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(base, encoding="utf-8"):
            f.write(line if line.endswith("\n") else line + "\n")
            n_base += 1
        for r in tg:
            if r.get("split") == "train":
                r = {k: v for k, v in r.items() if k not in ("reference", "question")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    steps = Counter(str(r.get("subtype")) for r in tg)
    return {"base": n_base, "tinygsm": len(tg), "tinygsm_long": len(long_), "by_steps": dict(sorted(steps.items())),
            "out": os.path.basename(out), "bytes": os.path.getsize(out)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mx = sub.add_parser("mix", help="v12e slots + TinyGSM slots -> one training file")
    mx.add_argument("--base", required=True)
    mx.add_argument("--tinygsm", required=True)
    mx.add_argument("--out", required=True)
    mx.add_argument("--cap", type=int, default=0)
    c = sub.add_parser("convert")
    c.add_argument("--parquet", required=True, help="a TinyGSM parquet shard, or a .jsonl of {question, code} rows")
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--exclude", nargs="*", default=[], help="eval jsonl files whose questions must not leak in")
    c.add_argument("--vm-sample", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "mix":
        m = mix(a.base, a.tinygsm, a.out, a.cap)
    else:
        m = convert(a.parquet, a.out, a.limit, a.exclude, a.vm_sample)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()