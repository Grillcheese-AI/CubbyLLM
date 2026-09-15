"""score_val_generations -- re-score a training run's val_generations.json with the VM.

WHY THIS EXISTS
---------------
The Colab notebook reports `exact_match_by_task`: string equality between the
generated program and the reference program. On v13e that metric reported
arithmetic = 0.0 while the VM scored the same generations 6/8. The gap is not
model quality, it is two artifacts baked into the reference text:

  1. THE PROGRAM NAME. References are named `program GSM1669 implements ...`
     with an id drawn from the source corpus. Nothing in the prompt determines
     it, so the model cannot guess it, and every generation is a string
     mismatch no matter how correct the body is. 2/8 arithmetic generations
     were byte-identical to their reference once the name was normalized.

  2. UNDETERMINED CONSTANTS. The `kernel` decision template carries confidence
     literals (100 on the approve branch, 10 on reject) that the prompt never
     states. A generation that takes the correct branch but writes 70/20 is
     marked wrong while making the same decision.

Neither artifact is a wrong answer, and both dominate the reported number. An
arm-vs-arm comparison read off `exact_match_by_task` would mostly measure how
often each arm guessed an unguessable token.

WHAT THIS MEASURES INSTEAD
--------------------------
The program is run. The VM's return value is compared to `gold`. This is
scoreable only where `gold` IS the program's own return value:

  arithmetic, chain, kernel   -- scored here
  plan                        -- emits a walk skeleton; `gold` is the answer
                                 the disposer resolves downstream, not what
                                 `solve` returns. Reported, not scored.
  role_binding                -- `gold` is None. Reported, not scored.

Each reference program is run alongside its generation as a control. A
reference that does not itself reproduce `gold` means the harness is wrong,
not the model, and is reported separately -- no generation is counted against
a broken reference.

THREE OUTCOMES, NOT TWO
-----------------------
A refusal is a result (invariant 4), so the summary separates them:

  correct   -- ran, matched gold
  refused   -- did not compile or did not run (a truncated generation lands
               here: the token cap cuts the closing brace and the parse fails
               loudly). Not an answer, so not a wrong answer.
  WRONG     -- compiled, ran, returned a confident wrong value. This is the
               kill-line count. `verify()` in the emitted templates returns
               true unconditionally, so the VM cannot catch these; only the
               gold comparison can.

Usage:
    python validation/score_val_generations.py <val_generations.json> [-o NAME]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubbyllm.bridges.cubelang_client import CubelangSession, CubelangRunError  # noqa: E402

VM_TASKS = ("arithmetic", "chain", "kernel")
NAME_RE = re.compile(r"program\s+\w+\s+implements")


def agree(result, gold) -> bool:
    """Gold match. Numeric where both sides parse as numbers, else exact string."""
    if result is None or gold is None:
        return False
    a, b = str(result).strip(), str(gold).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def name_normalized_equal(generated: str, reference: str) -> bool:
    """String equality after neutralizing the unguessable program name. Reported
    to show how much of an `exact_match` miss is that artifact alone."""
    sub = lambda s: NAME_RE.sub("program X implements", s or "").strip()
    return sub(generated) == sub(reference)


def score(outputs: list[dict], timeout: float = 30.0) -> list[dict]:
    rows: list[dict] = []
    with CubelangSession(timeout=timeout) as sess:
        for o in outputs:
            task = o.get("task")
            if task not in VM_TASKS:
                continue
            for which in ("reference", "generated"):
                rec = {
                    "id": o.get("id"),
                    "task": task,
                    "which": which,
                    "gold": o.get("gold"),
                    "exact_match": o.get("exact_match"),
                }
                try:
                    r = sess.run(o.get(which) or "", fn="solve", args=[o.get("prompt", "")])
                    rec.update(ran=True, result=r.get("result"))
                    rec["correct"] = agree(r.get("result"), o.get("gold"))
                except CubelangRunError as e:
                    rec.update(ran=False, result=None, correct=False, error=str(e)[:400])
                rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="val_generations.json from a training run (off-repo)")
    ap.add_argument("-o", "--out", default=None,
                    help="log stem under validation/logs/ (default: from the file's version+arm)")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    src = pathlib.Path(args.path)
    if not src.is_file():
        print(f"not a file: {src}", file=sys.stderr)
        return 2
    doc = json.loads(src.read_text(encoding="utf-8"))
    outputs = doc.get("outputs", [])
    version, arm = doc.get("version", "?"), doc.get("arm", "?")
    stem = args.out or f"score_val_{version}_{arm}"

    rows = score(outputs, timeout=args.timeout)
    by = {(r["id"], r["which"]): r for r in rows}

    # -- reference control: a failing reference invalidates its pair -----------
    bad_refs = sorted({r["id"] for r in rows if r["which"] == "reference" and not r["correct"]})

    # -- three-way tally over generations with a sound reference --------------
    tally: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    wrong: list[dict] = []
    for r in rows:
        if r["which"] != "generated" or r["id"] in bad_refs:
            continue
        t = tally[r["task"]]
        t["n"] += 1
        if r["correct"]:
            t["correct"] += 1
        elif not r["ran"]:
            t["refused"] += 1
        else:
            t["wrong"] += 1
            wrong.append(r)

    # -- how much of the exact_match miss is the program-name artifact --------
    name_only = collections.Counter()
    for o in outputs:
        if not o.get("exact_match") and name_normalized_equal(o.get("generated"), o.get("reference")):
            name_only[o.get("task")] += 1

    print(f"{version} / {arm}  --  {src.name}")
    print(f"{'task':<12}{'n':>4}{'correct':>9}{'refused':>9}{'WRONG':>7}")
    print("-" * 41)
    tot = collections.Counter()
    for task in sorted(tally):
        t = tally[task]
        tot.update(t)
        print(f"{task:<12}{t['n']:>4}{t['correct']:>9}{t['refused']:>9}{t['wrong']:>7}")
    print("-" * 41)
    print(f"{'total':<12}{tot['n']:>4}{tot['correct']:>9}{tot['refused']:>9}{tot['wrong']:>7}")

    print(f"\nreported exact_match_by_task: {json.dumps(doc.get('exact_match_by_task', {}))}")
    if name_only:
        print(f"exact_match misses that are ONLY the program name: {dict(name_only)}")
    if bad_refs:
        print(f"\nreferences that do not reproduce gold (harness, excluded): {bad_refs}")

    if wrong:
        print(f"\n!! {len(wrong)} WRONG ANSWER(S) SPOKEN -- compiled, ran, returned a wrong value:")
        for r in wrong:
            print(f"   {r['id']:<28} gold={r['gold']!r:>14}  got={r['result']!r}")
    else:
        print("\n0 wrong answers spoken.")

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{stem}.json"
    out.write_text(json.dumps({
        "source": src.name,
        "version": version,
        "arm": arm,
        "reported_exact_match": doc.get("exact_match_by_task", {}),
        "tally": {k: dict(v) for k, v in tally.items()},
        "name_artifact_only": dict(name_only),
        "bad_references": bad_refs,
        "wrong_answers": wrong,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
