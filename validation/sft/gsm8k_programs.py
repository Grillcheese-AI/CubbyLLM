"""gsm8k_programs -- order 0078, S1: GSM8K train's own gold solutions as step programs (the problems v12e lacks).

A gold's ``<<expr=result>>`` annotations become Python assignments, ``v<i> = expr``, where a number of the expression
equal to an earlier annotation's result is that result's name (the gold reuses its own results); the last one is
returned. Then `import_tinygsm.to_record` makes it a program as for TinyGSM (the same evaluator, literal filter and
dialect). Kept only when the program's value is the gold's ``#### n``; a miss is listed by id and kind (no
annotation, an annotation the converter rejects -- with its reason --, or a value that is not the gold's).

  python validation/sft/gsm8k_programs.py [--ids <file of ids>] [--out $CUBBY_DATA/s1_0078/gsm8k_extra.jsonl]

Wired: STANDALONE (order 0078's data build).
"""
from __future__ import annotations

import ast
import json
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validation.sft import import_tinygsm as tg  # noqa: E402

__wiring__ = "STANDALONE"

ANNOT = re.compile(r"<<([^=<>]+)=([^<>]*)>>")
FINAL = re.compile(r"####\s*([-\d,\.]+)")


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def to_code(answer: str) -> str | None:
    """The gold's annotations as a straight-line Python function, or None when it has none."""
    steps = ANNOT.findall(answer)
    if not steps:
        return None
    lines, results = ["def f():"], []
    for i, (expr, res) in enumerate(steps):
        expr = expr.replace(",", "")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return "def f():\n    return (\n"                       # the converter rejects it as syntax

        class Names(ast.NodeTransformer):
            def visit_Constant(self, node):                       # an earlier result, reused by value
                if isinstance(node.value, (int, float)):
                    for j in range(len(results) - 1, -1, -1):
                        if results[j] is not None and abs(results[j] - node.value) < 1e-9:
                            return ast.Name(id=f"v{j}", ctx=ast.Load())
                return node
        tree = Names().visit(tree)
        lines.append(f"    v{i} = {ast.unparse(tree.body)}")
        results.append(_num(res))
    lines.append(f"    return v{len(steps) - 1}")
    return "\n".join(lines) + "\n"


def program(row: dict, idx: int) -> tuple[dict | None, str]:
    """(the record, "kept") or (None, the miss's kind)."""
    code = to_code(row["answer"])
    if code is None:
        return None, "no annotation"
    m = FINAL.search(row["answer"])
    gold = _num(m.group(1)) if m else None
    try:
        rec = tg.to_record(row["question"], code, idx)
    except tg.Reject as e:
        return None, f"rejected: {e}"
    if gold is None or abs(float(rec["gold"]) - gold) > 1e-6:
        return None, "value is not the gold's"
    rec.update({"id": f"gsm8k-{row['id']}", "source": "gsm8k", "gsm8k_id": row["id"]})
    return rec, "kept"


def main() -> None:
    import argparse
    from collections import Counter
    from cubbyllm import local_paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="a json list of GSM8K train ids to convert (default: every non-dev one)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--misses", default=None, help="where the misses (id, kind) go")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(os.path.join(local_paths.value("CUBBY_DATASETS"), "gsm8k", "train.jsonl"),
                                        encoding="utf-8")]
    want = set(json.load(open(a.ids))) if a.ids else None
    kinds, misses = Counter(), []
    with open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r.get("dev") or (want is not None and r["id"] not in want):
                continue
            rec, kind = program(r, i)
            kinds[kind] += 1
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                misses.append({"id": r["id"], "kind": kind})
    if a.misses:
        json.dump(misses, open(a.misses, "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(kinds)))


if __name__ == "__main__":
    main()
