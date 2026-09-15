"""
Wired: WIRED (a stand-in; CubbyGhost opens world hypotheses in play).

A HYPOTHESIS is something he thinks might be true and has not yet earned the
right to assert. Nick, 2026-09-15:

    "if he is blind he might think the pellets are moving while they are not,
     its not a lie. If he waits long enough it will notice they dont move and
     needs to eat them."  ...  "hypothesis != wrong"

and, generalising it past the game:

    "same way with other problems outside the game, it can frame an hypothesis
     test it against the VM or a sandbox if code then the result is what is
     wrong or not"

So the shape is one shape, whatever the subject:

    frame a claim  ->  name the test that would settle it  ->  run the test
                   ->  THE RESULT is the verdict, not the model's confidence
                   ->  either way it becomes a fact he has earned

`ToolForge` (forge.py) already does this for one verifier and one kind of
expectation: a CubeLang program, executed on the VM, checked against something
the game knows. This module is that loop with the verifier as a parameter and
with the case forge.py cannot express — a claim that **resolves later**,
because the thing that settles it is time passing and nothing happening.

Three verifiers, and the honest description of each:

  VM        a CubeLang program whose result must match an expectation. This is
            what the trunk's reasoning is checked by everywhere else.
  RUNNER    code executed in a separate interpreter with a timeout, the result
            compared to an expectation. It is NOT a security sandbox: it is an
            isolation boundary for deciding right-or-wrong about code we wrote
            ourselves. Never point it at code from outside.
  WORLD     a predicate over the environment, re-checked each step. It may
            answer "not yet" — and after `patience` steps of not-yet, the
            world's silence IS the answer: the claim is refused.

The last one is the one that matters for an agent with senses. A wall is
learned by walking into it; "the pellets will come to me" is unlearned by
waiting and watching them not.

THE RULE, which is what keeps the kill line intact: an OPEN hypothesis may be
thought and said — as a guess, which is what it is — but it never enters the
world model as a fact. Only a verdict does. A guess spoken as a guess is not a
wrong answer; an untested claim asserted as fact is.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

__wiring__ = "WIRED"

OPEN, CONFIRMED, REFUSED = "open", "confirmed", "refused"


@dataclass
class Hypothesis:
    """A claim, the test that would settle it, and what either verdict teaches.

    `if_true` / `if_false` are the facts he EARNS — written in his own fact
    template so they go into the world model through the same gate every other
    fact passes. A hypothesis with no `if_false` teaches nothing by failing,
    which is usually a sign the claim was not worth framing."""

    claim: str                                           # what he thinks, in his words
    test: str                                            # what would settle it, in his words
    verifier: str                                        # "vm" | "runner" | "world"
    if_true: str = ""                                    # the fact a confirmation earns
    if_false: str = ""                                   # the fact a refusal earns
    payload: dict = field(default_factory=dict)          # verifier-specific
    made_at: int = 0
    patience: int = 0                                    # world only: steps before silence counts as no
    state: str = OPEN
    verdict_why: str = ""
    settled_at: int | None = None

    @property
    def open(self) -> bool:
        return self.state == OPEN

    def settle(self, ok: bool, why: str, now: int) -> str:
        self.state = CONFIRMED if ok else REFUSED
        self.verdict_why, self.settled_at = why, now
        return self.if_true if ok else self.if_false


def vm_verifier(exe: str | None = None):
    """A CubeLang program whose result must equal `expected`. The VM is the
    authority; a program that does not run is a refusal, not an error."""
    def check(h: Hypothesis, now: int):
        from cubbyllm.bridges import cubelang_client as cc
        from build_emitter_sft import answer_fn
        program, expected = h.payload.get("program", ""), str(h.payload.get("expected", ""))
        if not program:
            return False, "no program to run"
        try:
            out = cc.run_program_proto(program, fn=h.payload.get("fn") or answer_fn(program), exe=exe)
            got = None if out.get("result") is None else str(out["result"])
        except Exception as e:
            return False, f"the VM refused it: {str(e)[:120]}"
        return got == expected, f"the VM returned {got!r}, expected {expected!r}"
    return check


def runner_verifier(timeout: float = 5.0, python: str | None = None):
    """Run `payload['code']` in a separate interpreter and compare what it
    prints to `payload['expected']` (stripped).

    NOT a security boundary — see the module docstring. `-I` isolates it from
    the user's site-packages and environment, the cwd is a fresh temp dir, and
    it is killed at `timeout`; that is all. Point it only at code we generated
    ourselves."""
    def check(h: Hypothesis, now: int):
        code, expected = h.payload.get("code", ""), str(h.payload.get("expected", "")).strip()
        if not code.strip():
            return False, "no code to run"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "claim.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                p = subprocess.run([python or sys.executable, "-I", path], capture_output=True,
                                   text=True, timeout=timeout, cwd=tmp,
                                   env={"PATH": os.environ.get("PATH", ""), "SYSTEMROOT":
                                        os.environ.get("SYSTEMROOT", "")})
            except subprocess.TimeoutExpired:
                return False, f"it did not finish in {timeout:g}s"
            except OSError as e:
                return False, f"it would not start: {str(e)[:120]}"
        if p.returncode != 0:
            return False, f"it failed: {(p.stderr or '').strip().splitlines()[-1:] or ['no output']}"[:160]
        got = (p.stdout or "").strip()
        return got == expected, f"it printed {got!r}, expected {expected!r}"
    return check


def world_verifier(env):
    """A predicate over the world, re-checked every step.

    `payload['holds'](env)` returns True (settled yes), False (settled no), or
    None (not yet). After `patience` steps of None the claim is REFUSED and the
    reason says so — an agent that waits for the pellets to come to him has to
    be able to conclude, from nothing happening, that they do not."""
    def check(h: Hypothesis, now: int):
        holds = h.payload.get("holds")
        got = holds(env) if callable(holds) else None
        if got is None:
            if h.patience and now - h.made_at >= h.patience:
                return False, f"nothing happened in {now - h.made_at} steps, so it does not"
            return None, "not settled yet"
        return bool(got), ("the world did it" if got else "the world did not")
    return check


class Hypotheses:
    """His open claims, and the facts settling them earned him.

    `sweep(now)` runs every open claim's verifier and returns the facts the
    verdicts earned, for the caller to put through its own learning gate — this
    module never writes to a world model itself, because a fact that skipped
    the gate is exactly the kind of thing the gate exists for."""

    def __init__(self, verifiers: dict[str, Callable], trace=None, max_open: int = 32) -> None:
        self.verifiers = verifiers
        self.trace = trace or (lambda kind, **d: None)
        self.max_open = max_open
        self.all: list[Hypothesis] = []

    # ── framing ────────────────────────────────────────────────────────────
    def frame(self, h: Hypothesis) -> Hypothesis | None:
        """Open a claim. The same claim twice is one claim — he does not get
        to re-ask a question the world has already answered."""
        for prior in self.all:
            if prior.claim == h.claim:
                return None
        if len(self.open) >= self.max_open:
            return None
        self.all.append(h)
        self.trace("hypothesis", claim=h.claim, test=h.test, by=h.verifier, at=h.made_at)
        return h

    @property
    def open(self) -> list[Hypothesis]:
        return [h for h in self.all if h.open]

    def settled(self, state: str | None = None) -> list[Hypothesis]:
        return [h for h in self.all if not h.open and (state is None or h.state == state)]

    # ── testing ────────────────────────────────────────────────────────────
    def sweep(self, now: int) -> list[str]:
        """Test what can be tested. Returns the facts earned this sweep."""
        earned: list[str] = []
        for h in self.open:
            check = self.verifiers.get(h.verifier)
            if check is None:
                continue
            t0 = time.perf_counter()
            try:
                ok, why = check(h, now)
            except Exception as e:                       # a broken verifier is not a verdict
                self.trace("hypothesis_error", claim=h.claim, by=h.verifier, error=str(e)[:160])
                continue
            if ok is None:
                continue
            fact = h.settle(bool(ok), why, now)
            self.trace("hypothesis_settled", claim=h.claim, state=h.state, why=why,
                       learned=fact or None, after=now - h.made_at,
                       wall_s=round(time.perf_counter() - t0, 3))
            if fact:
                earned.append(fact)
        return earned

    # ── what it cost and bought ────────────────────────────────────────────
    def stats(self) -> dict:
        by = {}
        for h in self.all:
            s = by.setdefault(h.verifier, {"open": 0, CONFIRMED: 0, REFUSED: 0})
            s[h.state if not h.open else "open"] += 1
        return {"total": len(self.all), "open": len(self.open), "by_verifier": by,
                "earned": sum(1 for h in self.all
                              if not h.open and (h.if_true if h.state == CONFIRMED else h.if_false))}
