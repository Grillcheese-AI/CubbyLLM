"""Pins for what he says BACK to the player (`CubbyPac.say_to_coach`).

He had a voice about what he DID and none about what was SAID TO HIM. Now the
model writes the reply, `reply_ok` guards it and `saying.rank` decides whether
it or the host's sanctioned line goes out.

The brain is faked. A real one needs the VM and the model, and none of what is
pinned here is about either: it is about which of two available sentences
leaves his mouth, and why.
Run: python -m pytest standin/tests -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import identity as idn  # noqa: E402
import pacman  # noqa: E402
import saying  # noqa: E402
from grounding import grounded_ok, reply_ok  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402

F = idn.load_facts()
WARN = "careful, ghost behind you!"


class _Chat:
    state = {"dopamine": 0.4, "serotonin": 0.4, "cortisol": 0.06,
             "oxytocin": 0.4, "noradrenaline": 0.11}


class _Emitter:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def emit(self, prompt, context=None, max_new_tokens=48, system=None, **kw):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _Brain:
    def __init__(self, reply):
        self.emitter = _Emitter(reply)
        self.facts = F
        self.chat = _Chat()


def _agent(reply=None):
    a = pacman.CubbyGhost(seed=7, blank=True)
    a.chem = Neurochemistry()
    if reply is not None:
        a.brain = _Brain(reply)
    a.log = []
    a._trace = lambda kind, **d: a.log.append((kind, d))
    return a


def _said(a):
    return [d for k, d in a.log if k == "said_back"][-1]


def _referent(a, heard=WARN):
    got = a.coach.hear(heard, step=a.env.steps, need=0.0)
    rec = a.percepts("heard", claim=got["kind"], why=got.get("why"))
    rec["they said"] = got["text"]
    rec["they have been right"] = round(a.coach.credibility, 2)
    return a.render_percepts(rec), got["text"]


# ── the host line: still there, still the fallback ──────────────────────────
def test_no_brain_means_the_host_line_and_nothing_else():
    a = _agent()
    got = a.hear(WARN)
    assert a.say_to_coach(got, "en") == a.coach_line(got, "en")


def test_the_host_line_still_moves_with_what_they_have_earned():
    """The behaviour the canned reply was written for, kept under its own
    name: the same words from a voice that has cried ghost read differently."""
    a = _agent()
    got = a.hear(WARN)
    a.coach.prior = 0.9
    trusted = a.coach_line(got, "en")
    a.coach.prior = 0.1
    doubted = a.coach_line(got, "en")
    assert trusted != doubted


# ── the model path ──────────────────────────────────────────────────────────
def test_a_groundable_reply_is_his_own_sentence():
    a = _agent("ok, watching for it now.")
    out = a.say_to_coach(a.hear(WARN), "en")
    assert out == "ok, watching for it now."
    assert _said(a)["verbalized"] is True


def test_an_invented_figure_is_refused_and_the_host_line_goes_out():
    """The half that does not relax. A made-up reassurance is worse than a
    made-up thought, because somebody acts on it."""
    a = _agent("there are 9999 of them out there.")
    got = a.hear(WARN)
    out = a.say_to_coach(got, "en")
    d = _said(a)
    assert out == a.coach_line(got, "en")
    assert d["verbalized"] is False and d["spoke_safe"] is True
    assert "9999" in d["refused"]                      # refused, not silently dropped


def test_a_verbatim_repeat_is_refused():
    """The novelty term is signed, so a frightened body prefers the familiar —
    right for the sanctioned line, wrong for his own last sentence twice. The
    broken record comes out as a guard, not as a tuned knob."""
    a = _agent("ok, watching for it now.")
    first = a.say_to_coach(a.hear(WARN), "en")
    second = a.say_to_coach(a.hear(WARN), "en")
    assert first == "ok, watching for it now."
    assert second != first and _said(a)["verbalized"] is False


def test_a_broken_emitter_is_traced_not_swallowed():
    a = _agent(RuntimeError("no model"))
    got = a.hear(WARN)
    assert a.say_to_coach(got, "en") == a.coach_line(got, "en")
    assert any(k == "reply_error" for k, _ in a.log)


def test_the_prompt_carries_what_they_said_and_asks_for_an_answer():
    a = _agent("noted, thanks.")
    a.say_to_coach(a.hear(WARN), "en")
    p = a.brain.emitter.prompts[-1]
    assert WARN in p and "Answer them" in p


# ── reply_ok vs grounded_ok: the differences, and the one that does not move ─
def test_second_person_is_a_defect_in_a_thought_and_the_point_of_a_reply():
    rec = "i am at: level-1 cell 0-0-0; claim: warn"
    text = "you were right about that."
    assert reply_ok(rec, WARN, text, F)
    assert not grounded_ok(rec, text, F)


def test_one_word_is_a_reply_and_not_a_thought():
    rec = "i am at: level-1 cell 0-0-0; claim: warn"
    assert reply_ok(rec, WARN, "noted.", F)
    assert not grounded_ok(rec, "noted.", F)


def test_quoting_the_player_is_not_inventing():
    """The referent gains what they said. Most of what an answer to a sentence
    consists of is the sentence."""
    rec = "i am at: level-1 cell 0-0-0; claim: warn"
    assert reply_ok(rec, "careful, 3 ghosts down there!", "3 of them, ok.", F)
    assert not reply_ok(rec, "careful!", "3 of them, ok.", F)


def test_invention_does_not_relax_for_a_reply():
    rec = "i am at: level-1 cell 0-0-0; claim: warn"
    assert not reply_ok(rec, WARN, "there are 9999 of them out there.", F)
    assert not reply_ok(rec, WARN, "watch the FLOOR-9 gate.", F)


def test_the_reply_prompts_own_framing_is_refused_in_both_languages():
    """From the coach box's first live run, which spoke one: *"Je ne vois pas
    quelqu'un qui vient de te dire ça."* — the model answering the
    instruction instead of the player. It invents no figure and names no
    cell, so only the echo guard can catch it, and that guard has to know the
    framing of BOTH prompts."""
    rec = "i am at: level-1 cell 0-0-0; claim: warn"
    for leak in ("Je ne vois pas quelqu'un qui vient de te dire ça.",
                 "Réponds-lui en une phrase très courte.",
                 "Somebody just said to you that.",
                 "I will answer them in one very short sentence.",
                 "Do not invent anything you do not perceive."):
        assert not reply_ok(rec, WARN, leak, F), leak
    assert reply_ok(rec, WARN, "ok, watching for it now.", F), "an actual answer still passes"


def test_handing_their_own_sentence_back_is_not_an_answer():
    """Also from the live box: 'you can do it!!' came back as 'You can do
    it!!'. Every word is grounded — they are the player's words — so no
    invention check can see it, and it is not a reply."""
    from grounding import parroting
    rec = "i am at: level-1 cell 0-0-0; claim: talk"
    assert parroting("you can do it!!", "You can do it!!")
    assert not reply_ok(rec, "you can do it!!", "You can do it!!", F)
    # quoting IS most of answering: keep their phrase, add your own words
    assert not parroting("lache pas la patate",
                         "il n'y a rien a cacher, alors je ne lache pas la patate")
    assert not parroting("careful, 3 ghosts down there!", "3 of them, ok.")
    assert reply_ok(rec, "you can do it!!", "i can, and i am nearly through this one.", F)


# ── grounding is one function on both sides ─────────────────────────────────
def test_a_source_always_backs_itself():
    """`claim_surface(t, specifics(t))` is zero for every t, which is what
    makes `specifics` the unit of grounding rather than two half-measures."""
    for t in ("The Paleolithic ended 12000 years ago in Europe.",
              "i am at: level-1 cell 0-0-0; standing 0.50", WARN):
        assert saying.claim_surface(t, saying.specifics(t)) == 0.0


def test_the_filter_sees_the_name_the_guard_does_not():
    """What the filter adds on this path, stated exactly. `reply_ok` blocks
    figures, ALLCAPS names and `level-N cell` forms; an ordinary capitalised
    name mid-sentence is how an invented character arrives, and only
    `claim_surface` sees it."""
    a = _agent("x")
    flat, heard = _referent(a)
    invented = "the one called Blinky is closing in."
    assert reply_ok(flat, heard, invented, F), "the guard lets it through"
    assert saying.claim_surface(invented, saying.specifics(f"{flat} {heard}")) > 0.0


# ── still wired ─────────────────────────────────────────────────────────────
def test_handle_still_answers_through_the_new_path():
    a = _agent("noted, thanks.")
    said = a.handle("you can do it!!")
    assert said["meta"]["heard"] == "talk"
    assert said["offered"] == ["noted, thanks."]
