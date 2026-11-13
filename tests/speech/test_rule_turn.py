"""Order 0067, step 4: a turn under rules. The rule book reads the rules in force at a turn (set, withdrawn, a turn's
own rule); steering masks only the tokens that could break a rule and never a special; with ``CB_RULES`` off the chat
turn is the one before (nothing steered, nothing recorded); on, a candidate breaking a rule is not offered, the one
said is checked by the VM and recorded as held, and when none passes Cubby says it cannot put it in that form."""
from __future__ import annotations

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.speech import chat
from cubbyllm.speech import rule_turn as rt
from cubbyllm.speech.identity import load_facts, voice_ok
from cubbyllm.speech.rule_book import RuleBook


def put(b, plan, level, by, at, **k):
    """A rule set and said back to whoever set it (a rule applies only once shown: the addendum's kill line)."""
    r = b.set(plan, level, by, at=at, **k)
    b.show([r.id], by, at)
    return r

F = load_facts()
LOWER = {"m": "case", "op": "==", "v": "lower"}
NO_RIVER = {"m": "word", "op": "lacks", "v": "river"}
VOCAB = ["<EOS>", " River", "river", "Hello", "hello", "a,b", " short"]


class Seat:
    """A talk seat that says its replies in turn and keeps what it was handed."""
    name = "fake"

    def __init__(self, replies):
        self.replies, self.contexts, self.last = list(replies), [], {}

    def emit(self, prompt, max_new_tokens=768, system=None, context=None, **kw):
        self.contexts.append(context or {})
        steered = bool(((context or {}).get("steer") or {}).get("suppress"))
        self.last = {"hit_limit": False, "steered": steered}
        return self.replies[(len(self.contexts) - 1) % len(self.replies)]

    def vocab_strings(self):
        return VOCAB

    def specials(self):
        return {0}


def test_the_rule_book_reads_the_rules_in_force_at_a_turn():
    b = RuleBook()
    a = put(b, LOWER, "conversation", "user", at=0)
    g = put(b, NO_RIVER, "global", "operator", at=1)
    t = put(b, {"m": "json", "op": "==", "v": "yes"}, "user", "user", at=2, scope="turn")
    assert [r.id for r in b.in_force(0)] == [a.id]
    assert [r.id for r in b.in_force(2)] == [g.id, t.id, a.id]          # highest level first
    assert t.id not in [r.id for r in b.in_force(3)]                    # a turn's rule ends with its turn
    b.withdraw(a.id, at=3)
    assert [r.id for r in b.in_force(3)] == [g.id] and [r.id for r in b.in_force(2)] == [g.id, t.id, a.id]
    with pytest.raises(ValueError):
        put(b, {"m": "tone", "op": "==", "v": "warm"}, "user", "user", at=0)


def test_steering_masks_what_could_break_a_rule_and_never_a_special():
    b = RuleBook()
    put(b, LOWER, "user", "u", at=0)
    put(b, NO_RIVER, "user", "u", at=0)
    s = rt.steer(b.in_force(0), VOCAB, keep={0})
    assert s["suppress"] == [1, 2, 3] and s["prefix"] is None              # " River", "river", "Hello"; "<EOS>" kept
    b2 = RuleBook()
    r = put(b2, {"m": "opening", "op": "has", "v": "in short"}, "user", "u", at=0)
    assert rt.steer(b2.in_force(0), VOCAB)["prefix"] == "In short" and rt.steer(b2.in_force(0), VOCAB)["how"] == {r.id: "enforced"}
    put(b2, LOWER, "user", "u", at=0)
    assert rt.steer(b2.in_force(0), VOCAB, keep={0})["prefix"] == "in short"
    b3 = RuleBook()
    put(b3, {"or": [LOWER, NO_RIVER]}, "user", "u", at=0)
    assert rt.steer(b3.in_force(0), VOCAB) == {"suppress": [], "prefix": None, "how": {}}   # inside "or": nothing


def test_the_line_for_a_form_it_cannot_meet_is_in_cubbys_voice():
    assert all(voice_ok(line, F) for line in rt.CANT.values())


def test_off_the_turn_is_the_one_before(monkeypatch):
    monkeypatch.delenv("CB_RULES", raising=False)
    seat = Seat(["The River is wide."])
    c = chat.CubbyChat(seat, F)
    c.rules = RuleBook()
    put(c.rules, LOWER, "user", "u", at=0)
    rec = c.turn("tell me about rivers", mediate=False)
    assert "rules" not in rec and not any("steer" in ctx for ctx in seat.contexts) and rec["reply"] == "The River is wide."


@pytest.fixture
def vm_ready():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")


def test_on_a_candidate_breaking_a_rule_is_not_offered_and_the_one_said_held(monkeypatch, vm_ready):
    monkeypatch.setenv("CB_RULES", "1")
    seat = Seat(["The River is wide and slow.", "the water is wide and slow."])
    c = chat.CubbyChat(seat, F, n_candidates=2)
    c.rules = RuleBook()
    put(c.rules, LOWER, "user", "u", at=0, text="all lowercase, please")
    rec = c.turn("tell me about the water", mediate=False)
    assert "The River is wide and slow." in rec["rejected"] and seat.contexts[0]["steer"]["suppress"]
    if rec["reply"] == "the water is wide and slow.":
        assert rec["rules"][0]["outcome"] == "held" and rec["rules"][0]["how"] == "checked" and rec["rules"][0]["steered"]
    else:                                                # the body may still prefer its own line: then a refusal
        assert rec["rules"][0]["outcome"] == "refused"


def test_on_when_none_passes_cubby_says_so_and_the_record_says_which_rule(monkeypatch, vm_ready):
    monkeypatch.setenv("CB_RULES", "1")
    seat = Seat(["The River is wide.", "A River runs here."])
    c = chat.CubbyChat(seat, F, n_candidates=2)
    c.rules = RuleBook()
    r = put(c.rules, LOWER, "user", "u", at=0)
    rec = c.turn("tell me about rivers", mediate=False)
    assert rec["reply"] == rt.CANT["en"] and rec["spoke_safe"]
    assert rec["rules"] == [{"rule": r.id, "level": "user", "set_by": "u", "text": "", "how": "checked", "steered": True,
                             "outcome": "refused"}]
    assert c.last_rejection == f"rule {r.id}"


def test_a_withdrawn_rule_stops_at_the_next_turn(monkeypatch, vm_ready):
    monkeypatch.setenv("CB_RULES", "1")
    seat = Seat(["The River is wide."])
    c = chat.CubbyChat(seat, F)
    c.rules = RuleBook()
    r = put(c.rules, LOWER, "user", "u", at=0)
    assert c.turn("rivers?", mediate=False)["reply"] == rt.CANT["en"]
    c.rules.withdraw(r.id, at=len(c.history))
    rec = c.turn("rivers?", mediate=False)
    assert rec["reply"] == "The River is wide." and rec["rules"] == []
