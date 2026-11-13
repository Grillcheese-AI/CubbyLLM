"""Order 0067, the addendum "say the rule back" (Nick, 2026-11-13: "yes"). Pinned: every rule applied was shown to
whoever set it (no rule shown to someone else, or not yet shown, ever applies); a correction replaces the rule from the
next turn, at both levels; the said-back text comes from templates only (no model on that path); a rule set in the
conversation is said back once, in the user's language; "confirm before apply" holds new rules until confirmed."""
from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.speech import chat
from cubbyllm.speech import rule_say as S
from cubbyllm.speech.identity import load_facts
from cubbyllm.speech.rule_book import LEVELS, RuleBook

W50 = {"m": "word count", "op": "<", "v": 50}
LOWER = {"m": "case", "op": "==", "v": "lower"}
F = load_facts()


def test_no_rule_applies_before_its_setter_saw_it():
    rng = random.Random(0)
    for _ in range(200):
        b, people = RuleBook(confirm=("company",)), ["op", "admin", "ana", "bo"]
        for turn in range(12):
            act = rng.random()
            if act < 0.4:
                r = b.set(rng.choice([W50, LOWER, {"m": "language", "op": "==", "v": "fr"}]), rng.choice(LEVELS),
                          rng.choice(people), at=turn)
                if rng.random() < 0.7:
                    b.show([r.id], rng.choice([r.set_by, r.set_by, rng.choice(people)]), turn + rng.choice([0, 0, 1]))
            elif act < 0.55 and b.rules:
                r = rng.choice(b.rules)
                if r.withdrawn_at is None:
                    b.correct(r.id, W50, r.set_by, turn)
            elif act < 0.65 and b.rules:
                b.confirm([rng.choice(b.rules).id], turn)
            elif act < 0.75 and b.rules:
                r = rng.choice(b.rules)
                if r.withdrawn_at is None:
                    b.withdraw(r.id, turn + 1)
            for r in b.in_force(turn):
                assert r.shown_to == r.set_by and r.shown_at <= turn and not r.held


def test_a_correction_replaces_the_rule_from_the_next_turn_at_both_levels():
    for level in ("user", "conversation"):
        b = RuleBook()
        r = b.set(W50, level, "ana", at=0)
        if level == "user":                               # a system prompt: its rule book shown to whoever set it
            lines = b.show_book([{"sentence": "Keep replies short.", "kind": "rule", "rule_id": r.id},
                                 {"sentence": "We sell bikes.", "kind": "fact"}], "ana", 0)
            assert lines == ['Rule 1: every reply under 50 words (from "Keep replies short.")',
                             '"We sell bikes.": kept as a stated fact']
        else:
            assert b.show([r.id], "ana", 0) == "Got it: under 50 words."
        new, said = b.correct(r.id, {"m": "word count", "op": "<=", "v": 50}, "ana", at=3)
        assert said == "Got it, from now on: at most 50 words."
        assert [x.id for x in b.in_force(3)] == [r.id] and [x.id for x in b.in_force(4)] == [new.id]
        assert new.corrects == r.id and r.corrected_by == new.id


def test_the_said_back_text_is_templates_only():
    tree = ast.parse(Path(S.__file__).read_text(encoding="utf-8"))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__"}                     # no seat, no model, no network on this path
    assert S.said_back("set", [W50, LOWER], "fr") == "C'est noté : moins de 50 mots, tout en minuscules."
    assert S.said_back("asked", ["be formal"]) == 'I\'ll keep "be formal" in mind; I can\'t check that one.'
    assert S.phrase({"m": "count of the word", "op": ">=", "v": 1, "arg": "paradox"}) == 'the word "paradox" at least once'


class Seat:
    """A talk seat that says its replies in turn."""
    name = "fake"

    def __init__(self, replies):
        self.replies, self.contexts, self.last = list(replies), [], {}

    def emit(self, prompt, max_new_tokens=768, system=None, context=None, **kw):
        self.contexts.append(context or {})
        self.last = {"hit_limit": False, "steered": False}
        return self.replies[(len(self.contexts) - 1) % len(self.replies)]

    def vocab_strings(self):
        return ["<EOS>", "fine"]

    def specials(self):
        return {0}


def test_a_conversation_rule_is_said_back_once_and_a_hidden_one_never_applies(monkeypatch):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    monkeypatch.setenv("CB_RULES", "1")
    seat = Seat(["fine.", "still fine."])
    c = chat.CubbyChat(seat, F)
    c.rules = RuleBook()
    c.rules.set(W50, "conversation", "user", at=0)
    hidden = c.rules.set(LOWER, "user", "the operator", at=0)        # a system rule never shown to its setter
    rec = c.turn("réponds en moins de 50 mots", mediate=False)
    assert rec["said_back"] == "C'est noté : moins de 50 mots."
    assert hidden.id not in [r["rule"] for r in rec["rules"]]
    rec2 = c.turn("ok", mediate=False)
    assert rec2["said_back"] is None


def test_confirm_before_apply_holds_new_rules():
    b = RuleBook(confirm=("company",))
    r = b.set(W50, "company", "admin", at=0)
    b.show([r.id], "admin", 0)
    assert b.in_force(1) == []
    b.confirm([r.id], 1)
    assert [x.id for x in b.in_force(1)] == [r.id]
