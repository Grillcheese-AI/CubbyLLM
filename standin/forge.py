"""forge — cubby-man writes arbitrary CubeLang through his trunk.

Wired: WIRED (stand-in; CubbyGhost forges tools in play; nothing in
cubbyllm/ imports this).

The shape is CubbyLLM's own, in miniature: a TASK (a prompt in the family
the trunk was trained on, plus an expectation cubby-man can check from his
own map or his own rule) -> the EMITTER writes a program -> the VM executes
it -> the result is CERTIFIED against the expectation -> the program is
kept as a tool in his library with the prompt, the source, the VM's verdict
and the reasoning, whether it passed or not. Acceptance per task kind is
therefore a MEASURED number (standin/scripts/forge_probe.py), not a claim.

Task kinds and how they are checked (phrasings are the trunk's OWN trained
templates with only the numbers swapped — the story words never reach the
VM; the comparison semantics do):
  decision   "A sensor reads A units; flag the sample if the reading is
             above the safety threshold B." -> the trained kernels return a
             confidence CLASS: 70/80/90/100 when A > B, 10/20/30 when not
             (measured over the SFT set, per template) -> checked as >= 50.
  compare    "Two trials measured A and B; report the higher measurement."
             -> returns max(A, B).
  chain      "question\nFacts:\n- …" over his own facts (level prefixes
             stripped for the prompt) -> the object his map holds; accepted
             on the coordinate tail (`0-5-1` for `level-1 cell 0-5-1`), the
             prefix being bookkeeping, not the answer.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "WIRED"

from build_emitter_sft import answer_fn, shim_isolver  # noqa: E402
from identity import EMITTER_SYSTEM  # noqa: E402
from serve import PROGRAM_PREFIX, _clean  # noqa: E402

from cubbyllm.reasoning.planner import normalize  # noqa: E402

def decision_true(got: str | None) -> bool | None:
    """The trained decision kernels answer with a confidence class: >= 50
    means the condition held. None when the answer is not a number."""
    try:
        return float(str(got)) >= 50
    except (TypeError, ValueError):
        return None


def coord_tail(text: str) -> str:
    """`level-1 cell 0-5-1` / `cell 0-5-1` / `0-5-1` -> `0-5-1`."""
    m = re.search(r"(-?\d+)-(-?\d+)-(-?\d+)\s*$", str(text).strip())
    return m.group(0).strip() if m else normalize(str(text))


@dataclass
class Task:
    name: str
    kind: str                                          # decision | compare | chain
    prompt: str
    expected: str
    why: str                                           # what in the game raised this task
    situation: dict = field(default_factory=dict)
    check: Callable[[str], bool] | None = None         # default: normalized equality with expected

    def ok(self, got: str | None) -> bool:
        if got is None:
            return False
        if self.check is not None:
            return bool(self.check(got))
        return normalize(str(got)) == normalize(self.expected)


class ToolForge:
    """emit -> execute -> certify -> keep. `stats[kind]` = (accepted, tried)."""

    def __init__(self, emitter, library, exe: str | None = None, trace=None,
                 max_new_tokens: int = 600, ledger=None) -> None:
        self.emitter = emitter
        self.library = library
        self.exe = exe
        self.trace = trace or (lambda kind, **d: None)
        self.max_new_tokens = max_new_tokens
        self.stats: dict[str, list[int]] = {}
        self.n = 0
        self.ledger = ledger if ledger is not None else getattr(library, "ledger", None)   # every decision hashed + signed (ledger.py)

    def forge(self, task: Task, step: int = 0) -> dict:
        from cubbyllm.bridges import cubelang_client as cc
        t0 = time.perf_counter()
        prefix = PROGRAM_PREFIX if task.kind == "chain" else ""
        try:
            raw = self.emitter.emit(task.prompt, context="programs", max_new_tokens=self.max_new_tokens,
                                    system=EMITTER_SYSTEM, prefix=prefix)
        except Exception as e:                           # the trunk must never stop the game
            raw, err = "", f"emitter error: {e}"[:200]
        else:
            err = None
        cleaned = _clean(raw) if raw else ""
        program = shim_isolver(cleaned) if cleaned else ""
        got = None
        if program and err is None:
            try:
                out = cc.run_program_proto(program, fn=answer_fn(program), exe=self.exe)
                got = None if out.get("result") is None else str(out["result"])
            except cc.CubelangRunError as e:
                err = str(e)[:200]
        elif err is None:
            err = "no program emitted"
        ok = task.ok(got)
        st = self.stats.setdefault(task.kind, [0, 0])
        st[1] += 1
        st[0] += int(ok)
        self.n += 1
        verdict = (f"certified: the VM returned {got!r}, matching the expectation {task.expected!r}" if ok
                   else f"REJECTED: the VM returned {got!r}, expected {task.expected!r}"
                   + (f" ({err})" if err else ""))
        name = f"{task.name}#{self.n}"
        cert = None
        if self.ledger is not None and program:          # the decision itself, certified or rejected, goes to the ledger
            cert = self.ledger.record(program, name, task.kind, answer_fn(program), task.prompt, task.expected, got, ok, verdict)
        self.library.add(name, None, program or "(no program)", "tool", step,
                         {"why": task.why, "because": f"{task.kind} task raised in play",
                          "situation": task.situation, "rationale": f"prompt: {task.prompt}",
                          "verdict": verdict, "expected": task.expected, "got": got, "ok": ok}, cert=cert)
        self.trace("forge", name=name, task_kind=task.kind, ok=ok, prompt=task.prompt, got=got,
                   expected=task.expected, program=program, error=err,
                   wall_s=round(time.perf_counter() - t0, 3))
        return {"name": name, "ok": ok, "got": got, "program": program, "error": err}

    def acceptance(self) -> dict[str, float]:
        return {k: (a / t if t else 0.0) for k, (a, t) in self.stats.items()}


# ── the game's task phrasings (the trained families) ────────────────────────
def flee_task(near: int, radius: int, situation: dict) -> Task:
    """Flee iff near <= radius: the sensor template with reading = radius + 1
    and threshold = the ghost distance, so 'above' <=> near <= radius."""
    want = near <= radius
    return Task(name="FLEE", kind="decision",
                prompt=(f"A sensor reads {radius + 1} units; flag the sample if the reading is "
                        f"above the safety threshold {near}."),
                expected="flag (>= 50)" if want else "clear (< 50)",
                why="a ghost inside the fear radius", situation=situation,
                check=lambda got, want=want: decision_true(got) is want)


def safer_exit_task(d1: int, d2: int, situation: dict) -> Task:
    return Task(name="SAFER-EXIT", kind="compare",
                prompt=f"Two trials measured {d1} and {d2}; report the higher measurement.",
                expected=str(max(d1, d2)), why="choosing where to run", situation=situation)


_LEVEL = re.compile(r"level-\d+ ")


def orientation_task(direction: str, cell: str, facts: list[str], expected: str,
                     situation: dict) -> Task:
    """Level prefixes are stripped for the prompt (bookkeeping the trunk never
    saw); the answer is accepted on its coordinate tail."""
    return Task(name="WHERE", kind="chain",
                prompt=f"What is the {direction} neighbor of {_LEVEL.sub('', cell)}?\nFacts:\n"
                       + "\n".join(f"- {_LEVEL.sub('', f)}" for f in facts),
                expected=expected, why="checking my bearings against my own map",
                situation=situation,
                check=lambda got, exp=expected: coord_tail(got) == coord_tail(exp))


_NUM = re.compile(r"-?\d+")


def numbers(text: str) -> list[int]:
    return [int(x) for x in _NUM.findall(text)]
