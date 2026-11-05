"""cubbyllm/selfimprove/mistakes.py -- practice's record: the mistake memory (order 0049, step 3).

For each practice problem the trunk attempts up to ``tries`` times (sampled); the VM marks each attempt against the
gold. Wrong programs are kept per problem with the VM's reason -- "never this" -- and keep growing until an attempt is
right. Then the problem moves to the right set, and its wrong attempts become (wrong, right) pairs: the training
signal T3 compares (arm A: right answers only; B: + its own pairs; C: + pairs shared from other gaps).
A wrong program already in the memory and tried again counts as a repeat (T3's "repeated identical wrong programs").

Wired: STANDALONE (order 0049).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__wiring__ = "STANDALONE"


@dataclass
class Problem:
    pid: str
    gap: str
    wrong: dict = field(default_factory=dict)      # program -> the VM's reason (first seen)
    right: str | None = None
    attempts: int = 0
    repeats: int = 0


@dataclass
class MistakeMemory:
    tries: int = 8
    problems: dict = field(default_factory=dict)

    def attempt(self, pid: str, gap: str, program: str, ok: bool, reason: str = "") -> Problem:
        """Record one attempt. A solved problem takes no more attempts (its record is final)."""
        p = self.problems.setdefault(pid, Problem(pid, gap))
        if p.right is not None or p.attempts >= self.tries:
            return p
        p.attempts += 1
        if ok:
            p.right = program
        elif program in p.wrong:
            p.repeats += 1
        else:
            p.wrong[program] = reason or "wrong"
        return p

    def open(self, pid: str) -> bool:
        p = self.problems.get(pid)
        return p is None or (p.right is None and p.attempts < self.tries)

    def negatives(self, pid: str) -> list[tuple[str, str]]:
        """The kept wrong programs and their reasons: what an attempt must not repeat."""
        p = self.problems.get(pid)
        return [] if p is None else list(p.wrong.items())

    def right_set(self, gap: str | None = None) -> list[tuple[str, str]]:
        return [(p.pid, p.right) for p in self.problems.values() if p.right is not None and (gap is None or p.gap == gap)]

    def pairs(self, gap: str | None = None, exclude: str | None = None) -> list[tuple[str, str, str]]:
        """(problem, wrong, right) for every solved problem with negatives; ``gap``: only its own, ``exclude``: every
        gap but this one (arm C's shared negatives)."""
        out = []
        for p in self.problems.values():
            if p.right is None or (gap is not None and p.gap != gap) or (exclude is not None and p.gap == exclude):
                continue
            out += [(p.pid, w, p.right) for w in p.wrong]
        return out

    def stats(self, gap: str | None = None) -> dict:
        ps = [p for p in self.problems.values() if gap is None or p.gap == gap]
        solved = [p for p in ps if p.right is not None]
        return {"problems": len(ps), "solved": len(solved), "repeats": sum(p.repeats for p in ps),
                "attempts_until_right": round(sum(p.attempts for p in solved) / len(solved), 3) if solved else None,
                "negatives": sum(len(p.wrong) for p in ps)}


__all__ = ["MistakeMemory", "Problem"]
