"""exp_r36 -- CubeLang helpers mined from the emitter's verified programs. The skill library's second kind.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHAT THIS IS
------------
The emitter's arithmetic programs (v14e_nochain's SFT set, every one verified: executes, result == gold) are
straight lines of binary steps. `cubbyllm/reasoning/helpers.py` reads two steps where the first feeds only the
second as one SHAPE -- (a / b) * c -- and a shape that recurs across the TRAIN programs becomes a candidate
CubeLang helper. The gate is the skill library's: a helper is adopted only if EVERY train program containing
its shape, rewritten to call it, runs in the real VM to exactly the result it ran to before; then all of them
at once.

THE CHECKS
----------
    held-out   every VAL program rewritten with the adopted helpers runs to its own result -- the gate was
               fit on train, so this is the test of it
    served     a sample of val programs written the way a taught emitter would write them -- the helper CALLS
               and no definitions -- run through `eval_emitter_vm.run_vm`, which must supply the definitions
    blind?     instrument check: every candidate whose inner op is not commutative, with its two inner
               operands SWAPPED in the function body, through the same gate on the real VM. It must reject
               every one; a swapped helper that gets in means the gate does not look
    size       statements and characters before and after, over the programs a helper touches

KILL: any val program whose result moves under the rewrite, or any served-path run that fails or moves.

    python validation/exp_r36_emitter_helpers.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
OUT = ROOT / "standin" / "data" / "out" / "exp_r36"
DATA = ROOT / "standin" / "data" / "out" / "emitter_sft_v14e_nochain.jsonl"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def statements(program: str) -> int:
    body = program.split("function solve", 1)[-1]
    return sum(1 for l in body.split("\n") if l.split("#")[0].strip().endswith(";"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="arithmetic")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--served-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=36)
    ap.add_argument("--exe", default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.helpers import candidates, gate, rewrite
    from cubbyllm.reasoning.skills import Library

    recs = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == a.task and str(r.get("vm_ok")) == "True" and str(r.get("gold_match")) == "True":
                recs.append(r)
    session = cc.CubelangSession(exe=a.exe)
    calls = collections.Counter()

    def run(source: str):
        calls["vm"] += 1
        try:
            out = session.run(source, fn="solve")
        except Exception as e:                                # noqa: BLE001 -- a failed run is a result that moved
            return f"ERROR {str(e)[:80]}"
        return out.get("result") if out.get("ok") else f"NOT OK {out}"

    progs = {"train": [], "val": []}
    for r in recs:
        progs["val" if r.get("split") == "val" else "train"].append({"id": r["id"], "program": r["program"], "rec": r})
    for split in progs:
        for p in progs[split]:
            p["result"] = run(p["program"])
    moved0 = sum(1 for s in progs for p in progs[s] if str(p["result"]) != str(p["rec"].get("vm_result")))
    log(f"{a.task}: {len(progs['train'])} train / {len(progs['val'])} val verified programs; "
        f"{moved0} whose VM result today differs from the one recorded (kept, compared against today's run)")

    shapes = candidates(progs["train"], min_support=a.min_support)
    log(f"candidates (>= {a.min_support} train programs): {len(shapes)} -- "
        + ", ".join(f"{s.name} {n}" for s, n in shapes[:12]))
    if OUT.exists():
        for f in sorted(OUT.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)
    lib = Library(OUT / "skills.jsonl")
    tg = time.perf_counter()
    rep = gate(progs["train"], shapes, run, "helpers-train", library=lib, min_support=a.min_support)
    log(f"gate: {len(rep['adopted'])} adopted, {len(rep['rejected'])} rejected, {len(rep['retired'])} retired by the "
        f"joint rewrite ({time.perf_counter() - tg:.0f}s, {calls['vm']} VM runs so far)")
    for name, why in list(rep["rejected"].items())[:8]:
        log(f"   rejected {name}: {why}")

    # ── the instrument: a broken helper must not get in ─────────────────────────────────────────────
    from cubbyllm.reasoning.helpers import Shape

    class Swapped(Shape):
        def source(self) -> str:                               # the inner op's operands the wrong way round
            src = super().source()
            return (src.replace("assign t = arg0;", "assign t = arg1;").replace(f"{self.op1} t, arg1;", f"{self.op1} t, arg0;")
                    if self.inner_left else
                    src.replace("assign u = arg1;", "assign u = arg2;").replace(f"{self.op1} u, arg2;", f"{self.op1} u, arg1;"))

    broken = [(Swapped(s.op1, s.op2, s.inner_left, s.consts), n) for s, n in shapes
              if s.op1 in ("sub", "div") and not s.consts]
    blind = gate(progs["train"], broken, run, "instrument", library=None, min_support=a.min_support)
    log(f"instrument: {len(broken)} helpers with swapped operands through the gate -> {len(blind['adopted'])} adopted "
        f"(must be 0), {len(blind['rejected'])} rejected")

    # ── held-out ────────────────────────────────────────────────────────────────────────────────────
    stats = {}
    rewritten = {}
    for split in ("train", "val"):
        touched = moved = 0
        st_before = st_after = ch_before = ch_after = 0
        per = collections.Counter()
        for p in progs[split]:
            new, used = rewrite(p["program"], rep["shapes"])
            rewritten[p["id"]] = (new, used)
            if not used:
                continue
            touched += 1
            per.update(used)
            got = run(new)
            if str(got) != str(p["result"]):
                moved += 1
                if moved <= 3:
                    log(f"   MOVED [{split}] {p['id']}: {p['result']} -> {got} via {used}")
            defs = re.sub(r"    function (?!solve)\w+\(.*?\n    \}\n?", "", new, flags=re.S)   # the calls, not the definitions
            st_before += statements(p["program"]); st_after += statements(defs)
            ch_before += len(p["program"].split("function solve", 1)[1]); ch_after += len(defs.split("function solve", 1)[1])
        stats[split] = {"programs": len(progs[split]), "touched": touched, "moved": moved,
                        "statements": [st_before, st_after], "chars": [ch_before, ch_after], "per_helper": dict(per)}
        log(f"[{split}] {touched}/{len(progs[split])} programs call a helper; {moved} moved; solve statements "
            f"{st_before} -> {st_after} ({100 * (st_before - st_after) / max(st_before, 1):.1f}% fewer), chars "
            f"{ch_before} -> {ch_after}")

    # ── served: calls only, the definitions supplied by run_vm ─────────────────────────────────────
    os.environ["CUBBY_SKILLS"] = str(OUT / "skills.jsonl")
    import eval_emitter_vm as ev
    ev._HELPERS = None
    rng = random.Random(a.seed)
    pool = [p for p in progs["val"] if rewritten[p["id"]][1]]
    served_bad = 0
    sample = rng.sample(pool, min(a.served_n, len(pool)))
    for p in sample:
        new, used = rewritten[p["id"]]
        calls_only = re.sub(r"    function (?!solve)\w+\(.*?\n    \}\n?", "", new, flags=re.S)
        assert not any(f"function {u}(" in calls_only for u in used)
        ok, result, err = ev.run_vm(calls_only)
        if not ok or str(result) != str(p["result"]):
            served_bad += 1
            log(f"   SERVED FAIL {p['id']}: {result} {err}")
    log(f"[served] {len(sample) - served_bad}/{len(sample)} val programs written with helper calls only ran to "
        f"their own result through eval_emitter_vm.run_vm")

    # ── the next round's data ──────────────────────────────────────────────────────────────────────
    with (OUT / f"emitter_sft_{a.task}_helpers.jsonl").open("w", encoding="utf-8") as f:
        for split in ("train", "val"):
            for p in progs[split]:
                new, used = rewritten[p["id"]]
                f.write(json.dumps(dict(p["rec"], program=new, helpers=used), ensure_ascii=False) + "\n")
    moved = stats["val"]["moved"] + stats["train"]["moved"]
    verdict = "KILLED" if moved or served_bad else ("INSTRUMENT BLIND" if blind["adopted"] else "survives")
    log(f"\nVERDICT: {verdict} -- {moved} programs moved under the rewrite, {served_bad} served-path failures; "
        f"{len(rep['adopted'])} helpers: {', '.join(rep['adopted'])}")
    log(f"{calls['vm']} VM runs, {time.perf_counter() - t0:.0f}s")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_r36_emitter_helpers.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / "exp_r36_emitter_helpers.json").write_text(json.dumps({
        "adopted": rep["adopted"], "rejected": rep["rejected"], "retired": rep["retired"], "stats": stats,
        "helpers": {n: h["source"] for n, h in lib.helpers.items()}}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
