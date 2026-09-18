"""Pins for the filter on the mouth (standin/saying.py + CubbyChat.candidates).

The claim being pinned: the host guards decide what MAY be said and the body
decides what IS said, and the second half cannot loosen the first.
Run: python -m pytest standin/tests -q
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import chat  # noqa: E402
import choosing  # noqa: E402
import identity as idn  # noqa: E402
import saying  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402

F = idn.load_facts()
DK_EN = idn.T(F, "dont_know_line", "en")

PLAIN = "I'm Cubby - glad to help!"
CLAIMY = "The Paleolithic era ended around 12000 years ago, when the Holocene began in Europe."
HEDGED = "I think the Paleolithic era ended around 12000 years ago, in Europe."

REST = None
MILD = {"dopamine": 0.28, "serotonin": 0.20, "cortisol": 0.25,
        "oxytocin": 0.08, "noradrenaline": 0.30}
ALARM = {"dopamine": 0.20, "serotonin": 0.12, "cortisol": 0.70,
         "oxytocin": 0.05, "noradrenaline": 0.75}


class FakeEmitter:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def emit(self, prompt, max_new_tokens=768, system=None, **kw):
        self.calls += 1
        return self.reply


def _chat(reply, state=None, history=True, **kw):
    c = chat.CubbyChat(FakeEmitter(reply), F, **kw)
    if state:
        c.set_state(state)
    if history:                                    # novelty is only defined from turn two
        c.history.append({"user": "hi", "reply": "Hey there, what are we doing today?"})
    return c


# ── claim surface: the term the whole thing turns on ────────────────────────
def test_one_commitment_is_counted_once():
    """"12000 years" is a number AND a quantity. The first version counted it
    twice and an ordinary sentence read as maximally reckless."""
    both = saying.claim_surface("It took 12000 years.")
    number_only = saying.claim_surface("It took 12000.")
    assert both == number_only > 0.0


def test_a_hedge_takes_back_part_of_the_surface_not_all_of_it():
    assert 0.0 < saying.claim_surface(HEDGED) < saying.claim_surface(CLAIMY)


def test_what_the_caller_can_back_is_not_claim_surface():
    assert saying.claim_surface(CLAIMY) > 0.0
    assert saying.claim_surface(CLAIMY, {"paleolithic", "12000 years", "holocene", "europe"}) == 0.0


def test_saying_his_own_name_is_not_a_claim():
    """The identity facts ARE a source. An alarmed body that will not
    introduce itself is the filter misreading its own grounding."""
    assert saying.claim_surface(PLAIN) > 0.0                       # a proper noun, ungrounded
    assert saying.claim_surface(PLAIN, chat._self_grounding(F)) == 0.0


def test_the_forbidden_lists_are_not_grounding():
    g = chat._self_grounding(F)
    assert "cubby" in g and "grillcheese" in g
    assert "chatgpt" not in g and "agi" not in g


# ── novelty: a repetition signal, not a freshness bonus ─────────────────────
def test_nothing_said_yet_means_novelty_is_zero_not_one():
    """`choosing`'s own convention: a dimension you cannot score is 0, so the
    filter has nothing to bend there. 1.0 let a frightened body punish the
    first reply of every conversation for being new."""
    assert saying.novelty_against("anything at all", []) == 0.0


def test_novelty_falls_toward_a_near_repeat():
    prior = ["the maze is quiet right now"]
    assert saying.novelty_against("the maze is quiet right now", prior) < 0.2
    assert saying.novelty_against("pears ripen slowly in cellars", prior) > 0.9


# ── the body sets the threshold; the guards set the ceiling ─────────────────
def test_a_resting_body_speaks_and_an_alarmed_one_takes_the_fallback():
    """Same words, same guards, same grounding — only the body differs."""
    assert _chat(CLAIMY, REST).candidates("tell me")[0][0] == CLAIMY
    assert _chat(CLAIMY, ALARM).candidates("tell me")[0][0] == DK_EN


def test_what_he_can_back_he_still_says_under_alarm():
    """The threshold moves, it does not gag him. A reply that reaches past
    nothing survives any state."""
    assert _chat(PLAIN, ALARM).candidates("who are you?")[0][0] == PLAIN


def test_a_hedge_buys_one_step_of_alarm_and_not_two():
    assert _chat(HEDGED, MILD).candidates("tell me")[0][0] == HEDGED
    assert _chat(CLAIMY, MILD).candidates("tell me")[0][0] == DK_EN
    assert _chat(HEDGED, ALARM).candidates("tell me")[0][0] == DK_EN


def test_the_fallback_is_always_offered_whatever_the_state():
    for st in (REST, MILD, ALARM):
        for reply in (PLAIN, CLAIMY, HEDGED):
            offered, _ = _chat(reply, st).candidates("tell me")
            assert DK_EN in offered, (st, reply)


def test_the_filter_cannot_rescue_a_reply_the_guards_rejected():
    """Rejected candidates are never scored, so no state can reach them. A
    body that can talk itself past a guard is not modulated, it is broken."""
    for st in (REST, MILD, ALARM):
        offered, rejected = _chat("To be honest, I'm Cubby.", st).candidates("qui es-tu ?")
        assert rejected == ["To be honest, I'm Cubby."]
        assert offered == [idn.T(F, "dont_know_line", "fr")]


def test_risk_only_ever_subtracts_across_the_whole_state_space():
    """`choosing` rule one, at the mouth: no mood may make the riskier of two
    otherwise identical options win."""
    safe_opt = choosing.Option("low", value=1.0, risk=0.10, novelty=0.5, cost=0.2, social=0.2)
    bold = choosing.Option("high", value=1.0, risk=0.60, novelty=0.5, cost=0.2, social=0.2)
    probe = Neurochemistry()
    seen = 0
    for da in (0.2, 0.5, 0.8):
        for ne in (0.06, 0.4, 0.85):
            for c in (0.06, 0.4, 0.75):
                probe.dopamine, probe.noradrenaline, probe.cortisol = da, ne, c
                ranked = choosing.weigh([bold, safe_opt], probe.modulation())
                assert ranked[0][0].key == "low", (da, ne, c)
                seen += 1
    assert seen == 27


# ── the sampling knob ───────────────────────────────────────────────────────
def test_one_draw_by_default_so_no_turn_got_slower():
    c = _chat(PLAIN, REST)
    c.candidates("tell me")
    assert c.emitter.calls == 1


def test_more_draws_when_asked_and_fewer_when_urgent():
    """Urgency narrows speech where narrowing is real — at how many replies
    get generated, not at which of them wins."""
    calm = _chat(PLAIN, REST, n_candidates=5)
    calm.candidates("tell me")
    assert calm.emitter.calls == 5
    pressed = _chat(PLAIN, ALARM, n_candidates=5)
    pressed.candidates("tell me")
    assert pressed.emitter.calls < 5


def test_the_choice_is_reported_in_numbers():
    c = _chat(CLAIMY, ALARM)
    c.candidates("tell me")
    d = c.last_choice
    assert d["gains"]["risk"] > choosing.BASE["risk"]           # alarm weighs risk harder
    assert d["deviation"]["caution"] > 0.0
    assert {s["safe"] for s in d["scored"]} == {True, False}


# ── the baseline every one of the above measures against ────────────────────
def test_the_first_body_in_a_process_rests_where_every_later_one_does():
    """A subprocess, because the point is what happens BEFORE anything else
    has touched the class. This was luck until `__init__` settled it: the
    first Cubby of a run sat at caution 0.255 against a resting 0.130."""
    src = (
        "import sys; sys.path[:0] = [r'%s', r'%s']\n"
        "from neurochem import Neurochemistry\n"
        "first = Neurochemistry().modulation()['caution']\n"
        "later = Neurochemistry().modulation()['caution']\n"
        "print(round(first, 6), round(later, 6))\n"
    ) % (str(ROOT / "standin"), str(ROOT / "standin" / "data"))
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, check=True)
    first, later = (float(x) for x in out.stdout.split())
    assert first == later
