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

**Said back before it applies** (the addendum "say the rule back", Nick: "yes"): a rule applies only once it has been
shown, in plain words (`rule_say`), to whoever set it (`show`: a system prompt's rule book to its setter, or Cubby's
one line in the conversation). A rule shown and not corrected applies; `correct` replaces it from the next turn (the
old one withdrawn then, the two linked). A level in ``confirm`` (a company that asked "confirm before apply") holds its
new rules until `confirm`. `in_force` never returns a rule that was not shown to its setter, so no unseen rule is
ever applied (the addendum's kill line, by construction).

Wired: STANDALONE (`chat.CubbyChat` reads it behind ``CB_RULES=1``; step 1's API sets it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import rule_plan as rp
from . import rule_say

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
    shown_to: str | None = None                          # who saw it said back, and when
    shown_at: int | None = None
    said: str = ""
    held: bool = False                                   # waiting for "confirm before apply"
    corrects: str | None = None
    corrected_by: str | None = None

    def live(self, turn: int) -> bool:
        if turn < self.set_at or (self.withdrawn_at is not None and turn >= self.withdrawn_at):
            return False
        return self.scope == "session" or turn == self.set_at

    def shown(self, turn: int) -> bool:
        return self.shown_to == self.set_by and self.shown_at is not None and self.shown_at <= turn


class RuleBook:
    def __init__(self, confirm: tuple = (), prefix: str = "rule") -> None:
        self.rules: list[Rule] = []
        self.confirm_levels = set(confirm)
        self.prefix = prefix                             # rule ids: "<prefix>1", ... (books stacked in a session differ)

    def set(self, plan: dict, level: str, set_by: str, at: int, scope: str = "session", text: str = "", **meta) -> Rule:
        """A rule from `at` on. Refused (ValueError) when the plan is not one the family runs, or the level or scope is
        unknown: a rule that cannot be checked is never filed as one (the caller keeps it as "asked")."""
        why = rp.check(plan)
        if why:
            raise ValueError(f"not a rule: {why}")
        if level not in LEVELS or scope not in SCOPES:
            raise ValueError(f"level {level!r} / scope {scope!r}: levels {LEVELS}, scopes {SCOPES}")
        r = Rule(id=f"{self.prefix}{len(self.rules) + 1}", plan=plan, level=level, set_by=set_by, set_at=int(at), scope=scope,
                 text=text, meta=dict(meta), held=level in self.confirm_levels)
        self.rules.append(r)
        return r

    def withdraw(self, rule_id: str, at: int) -> Rule:
        """The rule stops applying from turn `at` (it stays in the book)."""
        for r in self.rules:
            if r.id == rule_id and r.withdrawn_at is None:
                r.withdrawn_at = int(at)
                return r
        raise KeyError(rule_id)

    def get(self, rule_id: str) -> Rule:
        return next(r for r in self.rules if r.id == rule_id)

    def show(self, rule_ids: list, to: str, at: int, lang: str = "en", kind: str = "set") -> str:
        """Say the rules back to `to` at turn `at` (Cubby's one line, `rule_say.said_back`); they may apply from then."""
        rules = [self.get(i) for i in rule_ids]
        said = rule_say.said_back(kind, [r.plan for r in rules], lang)
        for r in rules:
            r.shown_to, r.shown_at, r.said = to, int(at), said
        return said

    def show_book(self, entries: list[dict], to: str, at: int, lang: str = "en") -> list[str]:
        """A system prompt's rule book for its setter (`rule_say.book`); entries of kind "rule" carry "rule_id"."""
        lines = rule_say.book([dict(e, plan=self.get(e["rule_id"]).plan) if e["kind"] == "rule" else e for e in entries], lang)
        for e, line in zip(entries, lines):
            if e["kind"] == "rule":
                r = self.get(e["rule_id"])
                r.shown_to, r.shown_at, r.said = to, int(at), line
        return lines

    def unshown(self, turn: int) -> list[Rule]:
        """Rules set by `turn` and still live that nobody was shown yet."""
        return [r for r in self.rules if r.set_at <= turn and r.shown_at is None and r.withdrawn_at is None]

    def confirm(self, rule_ids: list, at: int) -> None:
        for i in rule_ids:
            r = self.get(i)
            r.held, r.meta["confirmed_at"] = False, int(at)

    def correct(self, rule_id: str, plan: dict, by: str, at: int, text: str = "", lang: str = "en") -> tuple[Rule, str]:
        """The setter corrects a rule at turn `at`: it is said back now and replaces the old one from turn ``at + 1``."""
        old = self.get(rule_id)
        new = self.set(plan, old.level, by, at + 1, old.scope, text, **{k: v for k, v in old.meta.items() if k != "confirmed_at"})
        new.corrects, old.corrected_by = old.id, new.id
        if old.withdrawn_at is None or old.withdrawn_at > at + 1:
            old.withdrawn_at = at + 1
        return new, self.show([new.id], by, at, lang, kind="changed")

    def in_force(self, turn: int) -> list[Rule]:
        """The rules that apply at `turn`, highest level first, then in the order they were set: live, shown to whoever
        set them, not held for a confirmation."""
        live = [r for r in self.rules if r.live(turn) and r.shown(turn) and not r.held]
        return sorted(live, key=lambda r: (LEVELS.index(r.level), self.rules.index(r)))

    def to_list(self) -> list[dict]:
        return [asdict(r) for r in self.rules]


__all__ = ["LEVELS", "SCOPES", "Rule", "RuleBook"]
