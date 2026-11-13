"""Order 0067, the second reader's step 4: CubeLang code as a syntax curriculum. No benchmark.

Programs, each compiled on the VM build ``CUBELANG_EXE`` names before it is kept (``cubelang compile``; a program that
does not compile on today's build is left out and counted):
- ``example``: the cubelang repo's examples (``examples/**/*.cube``, Nick's own code);
- ``test``: the programs written inside the cubelang repo's tests (Rust string literals holding a ``program``);
- ``programs_seat``: the programs seat's earlier rows that are whole programs and not from a benchmark (role binding,
  reasoning kernels, the verified chain programs); the GSM8K-derived arithmetic rows and every TinyGSM row stay out;
- ``view_check``: 0067 step 2's checks, each bound to the shortest of its replies;
- ``verb_program``: order 0072's templates, one per verb and kind on a case of its own.
Duplicates (the same text, spaces folded) once; a program sharing 5 words with IFEval or IFBench is dropped.

  python validation/builders/build_cubelang_curriculum.py [--cubelang <repo>] [--out $CUBBY_DATASETS/cubelang_curriculum_0067]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import bench_overlap as BO  # noqa: E402
import verb_cases as VCASE  # noqa: E402
import view_compile as VC  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import verb_programs  # noqa: E402

__wiring__ = "STANDALONE"

SEAT_SOURCES = ("regen/svc", "regen/atier", "cubbyllm/cot_harvest_v3cf")


def from_repo(repo: Path) -> list[dict]:
    out = [{"kind": "example", "origin": str(p.relative_to(repo)).replace("\\", "/"), "text": p.read_text(encoding="utf-8")}
           for p in sorted(repo.glob("examples/**/*.cube"))]
    for p in sorted(repo.glob("tests/*.rs")):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(r'r(#*)"(.*?)"\1', src, re.S):
            body = m.group(2)
            if re.search(r"\bprogram\s+\w+", body) and "{" in body:
                out.append({"kind": "test", "origin": str(p.relative_to(repo)).replace("\\", "/"), "text": body})
    return out


def from_seat(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [{"kind": "programs_seat", "origin": r["source"], "text": r["program"]}
            for r in map(json.loads, open(path, encoding="utf-8")) if r.get("source") in SEAT_SOURCES and r.get("vm_ok")]


def from_view_checks(path: Path) -> list[dict]:
    out = []
    for r in map(json.loads, open(path, encoding="utf-8")):
        reply = min((x["text"] for x in r["replies"]), key=len)
        out.append({"kind": "view_check", "origin": r["id"], "text": VC.program(r["spec"], reply)})
    return out


def from_verbs(rng) -> list[dict]:
    return [{"kind": "verb_program", "origin": f"{verb}/{kind}",
             "text": verb_programs.program(kind, variant, VCASE.build(verb, variant, rng, "en", None))}
            for (verb, kind), variant in sorted(VCASE.VARIANT_OF.items())]


def compiles(exe: str, text: str, tmp: Path) -> tuple[bool, str]:
    f = tmp / "p.cube"
    f.write_text(text, encoding="utf-8")
    r = subprocess.run([exe, "compile", str(f), "-o", str(tmp / "p.cubebin")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-1][:200] if r.returncode else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubelang", default=None, help="the cubelang checkout (default: the one CUBELANG_EXE is built in)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exe = str(cc.find_cubelang_exe())
    repo = Path(a.cubelang) if a.cubelang else Path(exe).resolve().parents[2]
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = Path(a.out or ds / "cubelang_curriculum_0067")
    cands = from_repo(repo) + from_seat(ROOT / "standin" / "data" / "out" / "emitter_sft.jsonl") \
        + from_view_checks(ds / "view_programs_0067" / "rows.jsonl") + from_verbs(random.Random(0))
    bench, seen, kept, why = BO.benchmark_grams(ifbench=True), set(), [], Counter()
    with tempfile.TemporaryDirectory() as d:
        for c in cands:
            key = " ".join(c["text"].split())
            if key in seen:
                why[f"{c['kind']}: the same text again"] += 1
                continue
            seen.add(key)
            if BO.overlaps(c["text"], bench):
                why[f"{c['kind']}: shares 5 words with a benchmark"] += 1
                continue
            ok, err = compiles(exe, c["text"], Path(d))
            if not ok:
                why[f"{c['kind']}: does not compile on this build"] += 1
                continue
            kept.append(dict(c, id=f"c{len(kept):06d}", compiles=True))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "programs.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"order": "0067", "what": "the second reader's step 4: CubeLang programs as a syntax curriculum, each compiled",
           "candidates": dict(Counter(c["kind"] for c in cands)), "kept": dict(Counter(r["kind"] for r in kept)),
           "dropped": dict(why), "benchmarks_checked": sorted(bench), "excluded_sources": "GSM8K-derived arithmetic "
           "(regen/arith_xl), every TinyGSM row, the identity rows (text, not programs)",
           "cubelang_repo": repo.name, "vm": vm_identity.identity(),
           "sha256": hashlib.sha256((out / "programs.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "vm"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
