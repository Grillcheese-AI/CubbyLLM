"""Pins for the VM-mediated chat turn (standin/chat.py). The rendering and
voice checks need no VM; the live turn skips without cubelang.exe.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import chat  # noqa: E402
import identity as idn  # noqa: E402

F = idn.load_facts()
DK_EN = idn.T(F, "dont_know_line", "en")
DK_FR = idn.T(F, "dont_know_line", "fr")


class FakeEmitter:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.systems = []

    def emit(self, prompt, max_new_tokens=768, system=None):
        self.systems.append(system)
        return self.reply


def test_voice_ok_and_guess_lang():
    assert idn.voice_ok("I'm Cubby, built by Grillcheese Research Lab. Happy to help!", F)
    assert not idn.voice_ok("I'll be honest with you: I'm Cubby.", F)             # forbidden word
    assert not idn.voice_ok("I'm ChatGPT, built by OpenAI.", F)                   # other model
    assert not idn.voice_ok("Yes, I am AGI.", F)
    assert idn.guess_lang("Comment tu t'appelles ?") == "fr"
    assert idn.guess_lang("What's your name?") == "en"
    assert idn.guess_lang("Salut !") == "fr"


def test_render_talk_program_embeds_candidates_and_is_an_iagent():
    src = chat.render_talk_program(['She said "hi"\nand left', DK_EN])
    assert "program CubbyTalk implements IAgent" in src
    assert 'assign c0 = "She said \\"hi\\" and left";' in src        # escaped quotes, newline collapsed
    assert f'assign c1 = "{DK_EN}";' in src
    assert 'ask "which reply", c0, c1;' in src
    for fn in ("think(input: str, ctx: str)", "act(decision: str, ctx: str)", "observe(result: str, ctx: str)"):
        assert fn in src
    with pytest.raises(ValueError):
        chat.render_talk_program([])


def test_candidates_filter_by_voice_and_always_offer_the_dont_know_line():
    c = chat.CubbyChat(FakeEmitter("<think>hmm</think>I'm Cubby — glad to help!"), F)
    offered, rejected = c.candidates("Who are you?")
    assert offered == ["I'm Cubby — glad to help!", DK_EN] and rejected == []
    assert "Hormonal state" in c.emitter.systems[-1]                 # the state block reached the model
    c = chat.CubbyChat(FakeEmitter("To be honest, I'm Cubby."), F)
    offered, rejected = c.candidates("Qui es-tu ?")
    assert offered == [DK_FR] and rejected == ["To be honest, I'm Cubby."]   # filtered; French fallback


def test_nudge_moves_state_within_bands_and_sets_register():
    c = chat.CubbyChat(FakeEmitter("x"), F)
    st = c.nudge("HELP ME NOW!!")
    assert st["noradrenaline"] >= 0.45 and idn.derived(st)["register"] == "cautious"
    c2 = chat.CubbyChat(FakeEmitter("x"), F)
    st2 = c2.nudge("Thanks so much, hello!")
    assert st2["oxytocin"] >= 0.35 and idn.derived(st2)["register"] == "warm"
    for h, (lo, hi) in idn.HORMONE_RANGE.items():
        assert lo <= st[h] <= hi and lo <= st2[h] <= hi


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_live_turn_through_the_vm_selects_the_model_reply_when_it_passes():
    _exe_or_skip()
    c = chat.CubbyChat(FakeEmitter("I'm Cubby, built by Grillcheese Research Lab. What can I do for you?"), F)
    rec = c.turn("Who are you?", feedback="good")
    assert rec["reply"] == "I'm Cubby, built by Grillcheese Research Lab. What can I do for you?"
    assert rec["offered"][1] == DK_EN and rec["question"] == "which reply" and rec["acted"] == rec["reply"]


def test_live_turn_falls_back_to_the_dont_know_line_when_the_voice_fails():
    _exe_or_skip()
    c = chat.CubbyChat(FakeEmitter("Honestly, I'm ChatGPT."), F)
    rec = c.turn("Qui t'a créé ?")
    assert rec["reply"] == DK_FR and rec["rejected"] == ["Honestly, I'm ChatGPT."]


def test_live_vm_rejects_a_selection_the_host_never_offered():
    """THE GUARANTEE through the whole stack: even a buggy host policy cannot
    make the VM say something that was not offered."""
    _exe_or_skip()
    from cubbyllm.bridges import cubelang_client as cc
    src = chat.render_talk_program(["offered one", DK_EN])
    with pytest.raises(cc.CubelangRunError, match="not among"):
        cc.resume_program_proto(src, fn="think", args=["hi", "calm"], answers=["invented reply"])
