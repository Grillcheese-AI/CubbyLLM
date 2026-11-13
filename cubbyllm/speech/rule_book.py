"""cubbyllm/speech/rule_book.py -- the rules in force for a session (order 0067, step 4, the alpha's thin slice of
"rules live in a world, with both clocks").

A rule is a plan over the reply world (`rule_plan`), with who set it and at which level, the text it came from, the
turn it was set at and, once withdrawn, the turn it stopped at. Rules are never deleted: a withdrawn rule stays in the
book with its end, so the rules in force at any past turn can be read again (`in_force(turn)`).

Levels, highest first (Nick, 2026-11-12): ``global`` (the operator), ``company``, ``user`` (a user's system prompt
for their own session), ``conversation`` (an instruction given in the chat). Cubby's own lines are above all of them
and are not rules here: no rule can reach them. Text that is read (a document, a page, a tool's result) sets nothing.

Scope: ``session`` (until withdrawn) or ``turn`` (the turn it was set at only).

What the alpha leaves to step 1 and the addendum (E): who may set a level (the caller's role), and two rules that
cannot both hold. Today every rule in force is checked on every candidate; none is dropped for a higher one.

Wired: STANDALONE (`chat.CubbyChat` reads it behind ``CB_RULES=1``; step 1's API sets it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import rule_plan as rp

__wiring__ = "STANDALONE"

LEVELS = ("global", "company", "user", "conversation")         # highest first
SCOPES = ("session", "turn")


@dataclass
class Rule:
    id: str
    plan: dict
    level: str
    set_by: str
    set_at: int
    scope: str = "session"
    text: str = ""
    withdrawn_at: int | None = None
    meta: dict = field(default_factory=dict)

    def live(self, turn: int) -> bool:
        if turn < self.set_at or (self.withdrawn_at is not None and turn >= self.withdrawn_at):
            return False
        return self.scope == "session" or turn == self.set_at


class RuleBook:
    def __init__(self) -> None:
        self.rules: list[Rule] = []

    def set(self, plan: dict, level: str, set_by: str, at: int, scope: str = "session", text: str = "", **meta) -> Rule:
        """A rule from `at` on. Refused (ValueError) when the plan is not one the family runs, or the level or scope is
        unknown: a rule that cannot be checked is never filed as one (the caller keeps it as "asked")."""
        why = rp.check(plan)
        if why:
            raise ValueError(f"not a rule: {why}")
        if level not in LEVELS or scope not in SCOPES:
            raise ValueError(f"level {level!r} / scope {scope!r}: levels {LEVELS}, scopes {SCOPES}")
        r = Rule(id=f"rule{len(self.rules) + 1}", plan=plan, level=level, set_by=set_by, set_at=int(at), scope=scope,
                 text=text, meta=dict(meta))
        self.rules.append(r)
        return r

    def withdraw(self, rule_id: str, at: int) -> Rule:
        """The rule stops applying from turn `at` (it stays in the book)."""
        for r in self.rules:
            if r.id == rule_id and r.withdrawn_at is None:
                r.withdrawn_at = int(at)
                return r
        raise KeyError(rule_id)

    def in_force(self, turn: int) -> list[Rule]:
        """The rules that apply at `turn`, highest level first, then in the order they were set."""
        live = [r for r in self.rules if r.live(turn)]
        return sorted(live, key=lambda r: (LEVELS.index(r.level), self.rules.index(r)))

    def to_list(self) -> list[dict]:
        return [asdict(r) for r in self.rules]


__all__ = ["LEVELS", "SCOPES", "Rule", "RuleBook"]
