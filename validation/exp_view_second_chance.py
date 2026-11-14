"""Order 0067, the view reader's second chance (Nick, 2026-11-13: "we could give the model 2 chances: if it reverses the
condition and fails on gold it can reverse them to get right and record the change / pattern"; where: "train rows +
pattern table"). Never on the test splits: a pattern read there would leak into the grammar the test read measures.

For each answer the reader wrote on a dev (or train) row (``exp_view_reader.py --splits dev --keep``):
1. each of its parts is matched to the generator's part by span (its ``check`` segment's start and end);
2. both parts run alone on the row's replies in the VM; the generator's part is the truth (it is how the row was made);
3. where they differ, one reversal at a time is tried in the reader's part: a comparison negated (``<`` / ``>=``),
   mirrored (``<`` / ``>``), or the part's "not" toggled (``return c5`` / ``return c4``); the first that gives the
   generator's part on every reply is the second chance;
4. the change is recorded with the cue words of the part's own text ("at most", "fewer than", "au moins"...), the site
   (a count's filter, a count compared, a position's value, a view's size) and the comparison before and after;
5. a program whose every part is now right is written as a corrected row (the reader's own split and program, fixed)
   for the next adapter round; the pattern table says which cue reads which way and how often, so a pattern that always
   holds can become a grammar rule that needs no gold (Nick's call).

  python validation/exp_view_second_chance.py --answers <kept answers.jsonl> [--rows <rows.jsonl>]
      [--out validation/logs/view_second_chance_<date>.json] [--sft <corrected rows.jsonl, off-repo>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import exp_view_reader as E  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import view_reader as VR  # noqa: E402

__wiring__ = "STANDALONE"

NEG = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "=="}
MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
SITE = re.compile(r"if \((x\d+|k\d+|n_[a-z]+) (<=|>=|==|!=|<|>) ")
CUES = ["at most", "at least", "fewer than", "less than", "more than", "exactly", "no ", "none", "every", "all ",
        "not ", "only", "between", "au plus", "au moins", "moins de", "plus de", "exactement", "aucun", "chaque",
        "tous", "toutes", "pas ", "ne ", "seulement"]


def part_alone(program: str, k: str, args: str) -> str:
    """The program with ``solve`` returning part ``k`` alone (called with the reader's own arguments)."""
    head = program[:program.index("    public function solve")]
    return head + ("    public function solve(mention: str): number {\n"
                   f"        let q = part{k}({args});\n        return q;\n    }}\n}}\n")


def calls(program: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r"let p\d+ = part(\d+)\(([^)]*)\);", program)}


def body_of(program: str, k: str) -> tuple[int, int]:
    """Where part ``k``'s body is in the program (start and end of its text)."""
    s = program.index(f"    function part{k}(")
    return s, program.index("\n    }\n", s) + len("\n    }\n")


def variants(program: str, k: str):
    """(how, site text, op before, op after, program) for every single reversal inside part ``k``."""
    s, e = body_of(program, k)
    body = program[s:e]
    for m in SITE.finditer(body):
        op = m.group(2)
        for how, table in (("negated", NEG), ("mirrored", MIRROR)):
            if op in table:
                nb = body[:m.start(2)] + table[op] + body[m.end(2):]
                yield how, m.group(1)[0], op, table[op], program[:s] + nb + program[e:]
    ret = re.search(r"return (c\d+);", body)
    neg = re.search(r"if \((c\d+) == 0\) \{ assign (c\d+) = 1; \}", body)
    if ret and neg and neg.group(2) == ret.group(1):                    # a "not" at the end: drop it
        nb = body.replace(f"return {ret.group(1)};", f"return {neg.group(1)};")
        yield "not dropped", "c", "not", "", program[:s] + nb + program[e:]
    elif ret:                                                           # no "not": add one
        c = ret.group(1)
        nb = body.replace(f"        return {c};", f"        create z9: number; assign z9 = 0;\n"
                                                  f"        if ({c} == 0) {{ assign z9 = 1; }}\n        return z9;")
        yield "not added", "c", "", "not", program[:s] + nb + program[e:]


def results(vm, program: str, k: str, args: str, replies: list) -> list:
    src = part_alone(program, k, args)
    return [E.verdict(vm, src, r["text"])[0] for r in replies]


SITES = {"x": "a count's filter or a position's value", "k": "a count compared", "n": "a view's size", "c": "the part's not"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sft", default=None)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067" / "rows.jsonl")
    rows = {r["id"]: r for r in (json.loads(ln) for ln in open(rows_p, encoding="utf-8"))}
    vm, t, table, fixed_rows = cc.CubelangSession(), Counter(), defaultdict(Counter), []
    for ln in open(a.answers, encoding="utf-8"):
        k = json.loads(ln)
        r = rows[k["id"]]
        if r["split"].startswith("test"):
            raise SystemExit(f"{k['id']} is a test row: the second chance runs on dev and train rows only")
        t["answers"] += 1
        got = VR.parse(k["answer"], r["instruction"]) if "error" not in k else {"error": k["error"]}
        if "error" in got:
            t["answers not parsed"] += 1
            continue
        gold_prog, mine = r["join"], got["program"]
        gold_span = {(s["start"], s["end"]): str(s["part"] + 1) for s in r["segments"] if s["kind"] == "check"}
        gold_calls, my_calls = calls(gold_prog), calls(mine)
        all_right, seen = True, set()
        for seg in got["segments"]:
            if seg["kind"] != "check" or seg["part"] in seen:           # parts count from 0 in segments, 1 in names
                continue
            seen.add(seg["part"])
            mk, gk = str(seg["part"] + 1), gold_span.get((seg["start"], seg["end"]))
            t["parts"] += 1
            if gk is None or mk not in my_calls:
                t["parts not matched to a gold part (span)"] += 1
                all_right = False
                continue
            truth = results(vm, gold_prog, gk, gold_calls[gk], r["replies"])
            if results(vm, mine, mk, my_calls[mk], r["replies"]) == truth:
                t["parts right as written"] += 1
                continue
            text = r["instruction"][seg["start"]:seg["end"]].lower()
            cue = " + ".join(c.strip() for c in CUES if c in text) or "(no cue word)"
            fix = next((v for v in variants(mine, mk) if results(vm, v[4], mk, my_calls[mk], r["replies"]) == truth), None)
            if fix is None:
                t["parts wrong, no single reversal fixes them"] += 1
                all_right = False
                continue
            how, site, before, after, mine = fix
            t["parts fixed by one reversal"] += 1
            table[f"{cue} | {SITES[site]}"][f"{how}: {before} -> {after}".strip()] += 1
        if all_right:
            if mine != got["program"]:
                t["programs fixed whole"] += 1
                head = k["answer"].split(VR.SEP, 1)[0]
                fixed_rows.append({"id": "c" + r["id"], "split": "train", "family": f"second_chance_{r['n_parts']}",
                                   "lang": r["lang"], "prompt": VR.prompt(r["instruction"]), "answer": head + VR.SEP + mine})
            else:
                t["programs right as written"] += 1
    out = {"order": "0067", "what": "the view reader's second chance on dev/train rows: single reversals, the pattern table",
           "answers": a.answers.replace("\\", "/").split("/")[-1], "vm": vm_identity.identity(), "counts": dict(t),
           "patterns": {k: dict(v) for k, v in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))}}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.sft:
        Path(a.sft).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in fixed_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
