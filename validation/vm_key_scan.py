"""vm_key_scan -- cubelang's fix list P6-9, the cross-check (Cowork, 2026-11-13): with the OLD two-byte key (BLAKE3's
first two bytes), which CubeLang programs CubbyLLM has run and kept carry two spellings on one key, and did any of
them compare such a pair?

The spellings are what the compiler itself keys (`cubelang keys`, P6-9's build: names, string literals, the
compiler's own labels), so nothing is guessed from the text. A pair is classed by what its two spellings are:
  literal   a string literal, which the VM read back through one table keyed by the two bytes: a literal on a shared
            key could read as the other spelling; two literals on one key compared equal
  label     a jump target the compiler made (`_else_3`): two on one key send a jump to one place
  name      a register: two names on one key share it
"Compared" (a literal pair): both spellings are string literals of the program and one of them stands in a
comparison (`==`, `!=`, an `if` / `while` condition, a `for` over a list) -- where the old key let one equal the other.

Sets (each as kept; a set whose source cannot be rebuilt exactly says so):
  0046_b1q      the one fixed program R2's B1 / B1q ran (per-row values are runtime arguments and segment facts)
  0054_replay   the 450M's programs of the 0054 replay ($CUBBY_DRIVE/cubbyllm/system_eval/run_0054/programs.jsonl),
                a body fragment wrapped as the host wraps it (`step_loop.wrap_step`); a quoted slot ("$E2") as written
                (what it was filled with at run time is not kept); a bare number slot ($N1) as the number 1 (a
                number takes no key, so this changes no spelling); a bare entity slot ($E3 as a value) as the quoted
                placeholder, like a quoted one
  0067_turn     the rule programs of the turn runs (`_if_turn_*_rows.json` in the main checkout's logs: the plan
                from the rules, the spoken reply), rebuilt with `rule_plan.program`
  0067_plans    the plans the reader wrote (`_rule_reader_*_plans.json`, `_rule_ifbench_*_plans.json`), rebuilt with
                an empty reply: the plan's own spellings only (the replies were not kept)
  0067_views    the second reader's step-2 view programs (`$CUBBY_DATASETS/view_programs_0067`, built with the host
                guard that leaves a shared key out), each run on each of its kept replies (`view_compile.program`:
                the reply's views as list literals, then the body)
  0067_views_unguarded  the same build's first, unseeded draw without the guard (`$CUBBY_DATASETS/_discard/
                view_programs_0067_unseeded_2026-11-13`), where 0067 found 121 of the first 500 with a shared key
  logs          every program source kept in this checkout's `validation/logs` (any JSON string holding `implements I`)
  code          the programs the serve path and the tests carry as text (Python string constants holding a program,
                `.cube` files under cubbyllm/)

    python validation/vm_key_scan.py --cubelang <the P6-9 build> [--main-logs <the main checkout's validation/logs>]
-> validation/logs/vm_key_scan_<date>.json

Wired: STANDALONE (a cross-check).
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as _dt
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

__wiring__ = "STANDALONE"

PROGRAM = re.compile(r"\bprogram\s+\w+\s+implements\s+I\w+")
COND = re.compile(r"\b(?:if|while)\s*\((.*?)\)\s*\{|\bfor\s*\(\s*let\s+\w+\s+of\s+\[(.*?)\]\s*\)", re.S)
LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')


def key2(s: str) -> str:
    import blake3
    return blake3.blake3(s.encode("utf-8")).digest()[:2].hex()


def literals(src: str) -> set[str]:
    return {m.group(1).replace('\\"', '"').replace("\\\\", "\\") for m in LIT.finditer(src)}


def compared_literals(src: str) -> set[str]:
    out = set()
    for m in COND.finditer(src):
        out |= literals(m.group(1) or m.group(2) or "")
    return out


def kind(s: str, lits: set, labels: set) -> str:
    return "literal" if s in lits else "label" if s in labels else "name"


def collisions(src: str, spellings: list[str]) -> list[dict]:
    lits, comp = literals(src), compared_literals(src)
    labels = {s for s in spellings if re.fullmatch(r"_[a-z_]+_\d+", s) and s not in lits}
    by = collections.defaultdict(list)
    for s in spellings:
        by[key2(s)].append(s)
    out = []
    for k, ss in by.items():
        if len(ss) < 2:
            continue
        ks = sorted({kind(s, lits, labels) for s in ss})
        both_lit = sum(1 for s in ss if s in lits) >= 2
        out.append({"key": k, "spellings": ss, "kinds": ks,
                    "compared": bool(both_lit and any(s in comp for s in ss)),
                    "a_literal_in_a_comparison": any(s in comp for s in ss)})
    return out


def run_keys(exe: str, items: list[tuple[str, str]]) -> dict:
    lines = "\n".join(json.dumps({"id": i, "source": s}) for i, s in items) + "\n"
    p = subprocess.run([exe, "keys"], input=lines.encode("utf-8"), capture_output=True, check=True)
    return {d["id"]: d for d in map(json.loads, p.stdout.decode("utf-8").splitlines()) if d.get("id") is not None}


def sources_in(obj, out: list):
    if isinstance(obj, str):
        if PROGRAM.search(obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            sources_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            sources_in(v, out)


def set_0046() -> list:
    sys.path.insert(0, os.path.join(ROOT, "validation"))
    from retrieval_bench.r2_vm import PROGRAM as P
    return [("r2_vm.PROGRAM", P)]


def bare_slots(src: str) -> str:
    """Outside string literals only: `$N1` -> 1, `$E3` -> "$E3" (inside a literal a slot stays as written)."""
    parts = re.split(r'("(?:[^"\\]|\\.)*")', src)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\$N\d+", "1", parts[i])
        parts[i] = re.sub(r"(\$E\d+)", r'"\1"', parts[i])
    return "".join(parts)


def set_0054() -> list:
    from cubbyllm.local_paths import value
    from cubbyllm.reasoning.step_loop import wrap_step
    out = []
    for run in ("run_0054", "run_0055_after"):
        path = os.path.join(value("CUBBY_DRIVE") or "", "cubbyllm", "system_eval", run, "programs.jsonl")
        if not os.path.exists(path):
            continue
        for i, line in enumerate(open(path, encoding="utf-8")):
            d = json.loads(line)
            src = bare_slots(d.get("program") or "")
            if not PROGRAM.search(src):
                try:
                    src = wrap_step(src)
                except ValueError:
                    continue
            out.append((f"{run}#{i}", src))
    return out


def set_0067(main_logs: str) -> tuple[list, list]:
    from cubbyllm.speech import rule_plan
    sys.path.insert(0, os.path.join(ROOT, "validation"))
    from exp_if_turn import plan_of
    turn, plans = [], []
    for path in sorted(glob.glob(os.path.join(main_logs, "_if_turn_*_rows.json"))):
        for r in json.load(open(path, encoding="utf-8")):
            rules = ast.literal_eval(r["rules"]) if isinstance(r.get("rules"), str) else r.get("rules") or []
            ps = [plan_of(x) for x in rules]
            if not ps:
                continue
            plan = ps[0] if len(ps) == 1 else {"and": ps}
            if rule_plan.check(plan) is None:
                turn.append((f"{os.path.basename(path)}#{r['id']}", rule_plan.program(plan, r.get("reply") or "")[0]))
    for path in sorted(glob.glob(os.path.join(main_logs, "_rule_*_plans.json"))):
        d = json.load(open(path, encoding="utf-8"))
        for arm, rows in d.items():
            for rid, v in (rows or {}).items():
                plan = v.get("plan") if isinstance(v, dict) and "plan" in v else v
                if isinstance(plan, dict) and rule_plan.check(plan) is None:
                    plans.append((f"{os.path.basename(path)}#{arm}#{rid}", rule_plan.program(plan, "")[0]))
    return turn, plans


def set_views(folder: str) -> list:
    sys.path.insert(0, os.path.join(ROOT, "validation", "builders"))
    from view_compile import program
    path = os.path.join(folder, "rows.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        for j, r in enumerate(d.get("replies") or []):
            out.append((f"{os.path.basename(folder)}#{d['id']}#{j}", program(d["spec"], r["text"])))
    return out


def set_logs() -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "validation", "logs", "*.json*"))):
        if os.path.basename(path).startswith("_"):
            continue
        found = []
        try:
            if path.endswith(".jsonl"):
                for line in open(path, encoding="utf-8"):
                    if "implements I" in line:
                        sources_in(json.loads(line), found)
            else:
                text = open(path, encoding="utf-8").read()
                if "implements I" in text:
                    sources_in(json.loads(text), found)
        except (ValueError, UnicodeDecodeError):
            continue
        out += [(f"{os.path.basename(path)}#{i}", s) for i, s in enumerate(dict.fromkeys(found))]
    return out


def set_code() -> list:
    out = []
    for path in glob.glob(os.path.join(ROOT, "cubbyllm", "**", "*.cube"), recursive=True):
        out.append((os.path.relpath(path, ROOT), open(path, encoding="utf-8").read()))
    for base in ("cubbyllm", "cubby_serve", "tests"):
        for path in glob.glob(os.path.join(ROOT, base, "**", "*.py"), recursive=True):
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                      and PROGRAM.search(n.value) and "{" in n.value and "{{" not in n.value]
            out += [(f"{os.path.relpath(path, ROOT)}#{i}", s) for i, s in enumerate(consts)]
    return out


def scan(exe: str, items: list) -> dict:
    keys = run_keys(exe, items) if items else {}
    progs, errors, pairs = 0, 0, []
    for pid, src in items:
        d = keys.get(pid)
        if d is None or d.get("error"):
            errors += 1
            continue
        progs += 1
        for c in collisions(src, d["spellings"]):
            pairs.append({"program": pid, **c})
    by_kind = collections.Counter("+".join(p["kinds"]) for p in pairs)
    return {"programs": len(items), "compiled": progs, "parse_errors": errors,
            "programs_with_a_shared_key": len({p["program"] for p in pairs}), "pairs": len(pairs),
            "pairs_by_kind": dict(by_kind),
            "literal_pairs_compared": [p for p in pairs if p["compared"]],
            "a_literal_of_a_pair_in_a_comparison": sum(1 for p in pairs if p["a_literal_in_a_comparison"]),
            "label_pairs": [p for p in pairs if p["kinds"] == ["label"]][:50],
            "name_pairs": [p for p in pairs if p["kinds"] == ["name"]][:50],
            "examples": pairs[:20]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubelang", required=True, help="a build with `cubelang keys` (P6-9)")
    ap.add_argument("--main-logs", default=os.path.join(os.path.dirname(ROOT), "CubbyLLM", "validation", "logs"))
    a = ap.parse_args()
    turn, plans = set_0067(a.main_logs)
    from cubbyllm.local_paths import value
    data = value("CUBBY_DATASETS") or ""
    sets = {"0046_b1q": set_0046(), "0054_replay": set_0054(), "0067_turn": turn, "0067_plans": plans,
            "0067_views": set_views(os.path.join(data, "view_programs_0067")),
            "0067_views_unguarded": set_views(os.path.join(data, "_discard", "view_programs_0067_unseeded_2026-11-13")),
            "logs": set_logs(), "code": set_code()}
    rep = {"what": "cubelang P6-9's cross-check: programs CubbyLLM ran and kept, two spellings on one OLD 2-byte key",
           "date": _dt.date.today().isoformat(), "key": "BLAKE3 first 2 bytes (before P6-9)",
           "sets": {k: scan(a.cubelang, v) for k, v in sets.items()}}
    path = os.path.join(ROOT, "validation", "logs", f"vm_key_scan_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in rep["sets"].items():
        print(k, {x: v[x] for x in ("programs", "compiled", "parse_errors", "programs_with_a_shared_key", "pairs",
                                    "pairs_by_kind", "a_literal_of_a_pair_in_a_comparison")},
              "compared pairs:", len(v["literal_pairs_compared"]))
    print("->", path)


if __name__ == "__main__":
    main()
