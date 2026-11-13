"""cubbyllm/speech/rule_turn.py -- a turn under rules (order 0067, step 4): steer the writing, check every candidate in
the VM, say the first that passes, or say that the form asked cannot be met. Off unless ``CB_RULES=1``; off, no turn
changes (the chat reads nothing here).

**How a rule acts, as the record says it** (order 0067, architecture 4):
- ``enforced``: the host wrote it in by construction. Today one shape: an opening (``opening has X``, ``starts with X``)
  becomes the reply's forced prefix. It is checked as well;
- ``checked``: the VM runs the rule on each candidate (`rule_plan.run`); one that fails is not offered;
- ``asked``: the rule could not be run (no measurement for it); it is recorded, never claimed.
A rule may also be **steered**: while the seat writes, the tokens that could only break it are masked through grilly2's
logits hook (order 0071): a banned word's own tokens, every token holding a capital under "all lowercase" (a small
letter under "all capitals"), every token holding a character the reply must not contain. Steering helps a candidate
pass; it is not a guarantee (a word can be written from other pieces), so a steered rule is still checked.

Steering reads leaf shapes over the reply world, never a kind of instruction: any plan whose top is a leaf or an
"and" of leaves is read leaf by leaf; inside "or" / "not" nothing is steered.

Wired: STANDALONE (`chat.CubbyChat.candidates` calls it behind ``CB_RULES=1``).
"""
from __future__ import annotations

import os

from . import rule_plan as rp

__wiring__ = "STANDALONE"

CANT = {"en": "I can't put that in the form you asked.", "fr": "Je ne peux pas le dire sous la forme demandée."}


def on() -> bool:
    return os.environ.get("CB_RULES", "0") == "1"


def _top_leaves(plan: dict) -> list[dict]:
    if "m" in plan:
        return [plan]
    if "and" in plan:
        return [q for p in plan["and"] for q in _top_leaves(p)]
    return []


def _never(lf: dict) -> bool:
    """The leaf says a count must be zero."""
    return (lf["op"] == "==" and int(lf["v"]) == 0) or (lf["op"] == "<" and int(lf["v"]) <= 1) or \
        (lf["op"] == "<=" and int(lf["v"]) == 0)


def steer(rules, vocab: list[str], keep=()) -> dict:
    """What the host does while the seat writes, for `rules` (rule_book.Rule, highest first): {"suppress": token ids,
    "prefix": text or None, "how": {rule id: "enforced" | "steered"}}. `vocab[i]` is token i as text; `keep` (the end
    token, the specials) is never masked."""
    masks, prefix, how = [], None, {}
    lower = any(lf["m"] == "case" and lf["op"] == "==" and lf["v"] == "lower" for r in rules for lf in _top_leaves(r.plan))
    for r in rules:
        for lf in _top_leaves(r.plan):
            m, op, v = lf["m"], lf["op"], lf.get("v")
            test = None
            if m == "word" and op == "lacks":
                w = str(v).lower()
                test = lambda s, w=w: s.strip().lower() == w                         # noqa: E731
            elif m == "count of the word" and _never(lf):
                w = lf["arg"].lower()
                test = lambda s, w=w: s.strip().lower() == w                         # noqa: E731
            elif m == "case" and op == "==" and v in ("lower", "upper"):
                test = (lambda s: any(c.isupper() for c in s)) if v == "lower" else (lambda s: any(c.islower() for c in s))
            elif m in ("count of the character", "count of the letter") and _never(lf):
                c = lf["arg"]
                test = (lambda s, c=c: c in s) if m.endswith("character") else (lambda s, c=c.lower(): c in s.lower())
            elif prefix is None and ((m == "opening" and op == "has") or (m == "starts with" and op == "==" and v == "yes")):
                text = str(v if m == "opening" else lf["arg"])
                prefix = text if lower or m == "starts with" else text[:1].upper() + text[1:]
                how[r.id] = "enforced"
                continue
            if test is not None:
                masks.append(test)
                how.setdefault(r.id, "steered")
    keep = set(keep)
    suppress = [i for i, s in enumerate(vocab) if i not in keep and s and any(t(s) for t in masks)] if masks else []
    return {"suppress": suppress, "prefix": prefix, "how": how}


def check(rules, reply: str, run_fn) -> list[dict]:
    """Each rule run by the VM on `reply`: [{"rule", "level", "held", "how", "why"}] (how: checked | asked)."""
    out = []
    for r in rules:
        got = rp.run(r.plan, reply, run_fn)
        out.append({"rule": r.id, "level": r.level, "held": got["held"], "how": got["how"], "why": got["why"]})
    return out


def passes(outcomes: list[dict]) -> bool:
    """Every rule that could be run held (an "asked" rule cannot block: it was never checked)."""
    return all(o["held"] for o in outcomes if o["how"] == "checked")


def record(rules, outcomes: list[dict] | None, steering: dict, said: str) -> list[dict]:
    """The turn's rules as its record keeps them: how each acted and its outcome on the reply said. ``said``:
    "model" (a candidate), "cant" (none passed: Cubby said so), or "safe" (Cubby's own line)."""
    by = {o["rule"]: o for o in outcomes or []}
    rows = []
    for r in rules:
        o = by.get(r.id, {})
        how = "asked" if o.get("how") == "asked" else ("enforced" if steering["how"].get(r.id) == "enforced" else "checked")
        rows.append({"rule": r.id, "level": r.level, "set_by": r.set_by, "text": r.text, "how": how,
                     "steered": steering["how"].get(r.id) == "steered",
                     "outcome": ("held" if o.get("held") else "broken" if o.get("held") is False else "not checked")
                     if said == "model" else "refused"})
    return rows


class TurnRules:
    """The rule half of one chat turn (`chat.CubbyChat.candidates`): the rules in force at this turn, the steering
    handed to the seat, each candidate's outcomes, and the record. `seat`: the talk seat (``vocab_strings``,
    ``specials``), or None (nothing is steered then; the rules are still checked)."""

    def __init__(self, book, turn: int, seat, run_fn) -> None:
        self.rules = book.in_force(turn) if book is not None else []
        self.run_fn, self.by_reply = run_fn, {}
        vocab = seat.vocab_strings() if self.rules and hasattr(seat, "vocab_strings") else None
        self.steering = steer(self.rules, vocab, seat.specials()) if vocab else {"suppress": [], "prefix": None, "how": {}}
        self.applied = False
        self.broke = False                                # a candidate was held back for a rule

    def context(self) -> dict:
        s = self.steering
        return {"steer": {"suppress": s["suppress"], "prefix": s["prefix"]}} if (s["suppress"] or s["prefix"]) else {}

    def why_not(self, reply: str, steered: bool = False) -> str | None:
        """None if every rule in force holds on `reply` (by the VM), else which rule failed."""
        if not self.rules:
            return None
        self.applied = self.applied or steered
        out = check(self.rules, reply, self.run_fn)
        self.by_reply[reply] = out
        if passes(out):
            return None
        self.broke = True
        return "rule " + ", ".join(o["rule"] for o in out if o["how"] == "checked" and not o["held"])

    def safe(self, dont_know: str, lang: str) -> str:
        """The line Cubby falls back on: "I can't put that in the form you asked" when a candidate was held back
        for a rule, else the don't-know line."""
        return CANT.get(lang, CANT["en"]) if self.broke else dont_know

    def record(self, reply: str) -> list[dict]:
        said = "model" if reply in self.by_reply and passes(self.by_reply[reply]) else \
            "cant" if reply in CANT.values() else "safe"
        st = dict(self.steering, how={k: v for k, v in self.steering["how"].items() if v == "enforced" or self.applied})
        return record(self.rules, self.by_reply.get(reply), st, said)


__all__ = ["CANT", "TurnRules", "check", "on", "passes", "record", "steer"]
