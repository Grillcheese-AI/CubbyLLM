"""standin/emitter.py -- base450m in both seats (2026-10-17, Nick: "I want to be able to replace the stand-in").

Pinned, without a card (a fake trunk stands in for grilly2; the real one is `Cubby450mTrunk`):
  * the talk adapter is shown its OWN formats, rendered by the host from the structured context -- a facts
    block for a paraphrase or an identity question, the conversation's turns for a chat, the bare question
    otherwise; the stand-in's instruction text never reaches it;
  * its answer is cut where it would start the next block, one line, the prefix kept;
  * the programs seat wraps the prompt in the emitter's Question / Program frame and decodes the program with
    the role tokens kept;
  * one trunk serves both roles, swapping the LoRA only when the role changes, under one lock.
Run: python -m pytest standin/tests/test_emitter_450m.py -q
"""
from __future__ import annotations

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tokenizers import Tokenizer  # noqa: E402

from emitter import Cubby450mEmitter, Cubby450mTalk, Cubby450mTrunk, ContextualEmitter, talk_answer, talk_prompt  # noqa: E402

TK = Tokenizer.from_file(str(ROOT / "data" / "grillcheese_bbpe128k.json"))
EOS = TK.token_to_id("</s>")


class FakeTrunk:
    """`Cubby450mTrunk`'s surface with no card: records what each role was asked, answers from a script."""

    def __init__(self, answers: dict):
        self.adapter_dirs = {"programs": "/adapters/emitter_x", "talk": "/adapters/talk_y"}
        self.max_ctx = 2048
        self.answers, self.calls, self.swaps, self._role = answers, [], 0, None
        self._lock = threading.Lock()

    tokenizer = TK
    eos = EOS

    def use(self, role):
        if role != self._role:
            self._role = role; self.swaps += 1

    def generate(self, role, rows, max_new_tokens, temperature=0.0, top_k=40, seed=None):
        with self._lock:
            self.use(role)
            self.calls.append((role, [TK.decode(r, skip_special_tokens=False) for r in rows], temperature))
            return [TK.encode(self.answers[role] + "</s>", add_special_tokens=False).ids for _ in rows]


def test_the_talk_adapter_is_shown_its_own_formats_never_the_stand_ins_instruction():
    facts = [("Marie Curie", "place of birth", "Warsaw"), ("Marie Curie", "date of birth", "1867-11-07 (point)")]
    p = talk_prompt("In one or two plain sentences, say who Marie Curie is, using ONLY these facts.",
                    {"role": "talk", "facts": facts, "question": "Who is Marie Curie?"})
    assert p == ("Facts:\n- Marie Curie — place of birth: Warsaw\n- Marie Curie — date of birth: 1867-11-07 (point)\n"
                 "Question: Who is Marie Curie?\nAnswer:")
    assert "plain sentences" not in p
    # ready lines pass through; a conversation keeps its earlier turns above the last question
    assert talk_prompt("q", {"facts": ["Beyoncé appeared in the video."], "question": "Who appeared?"}).startswith(
        "Facts:\n- Beyoncé appeared in the video.\nQuestion: Who appeared?")
    assert talk_prompt("ignored", {"turns": [("user", "hi"), ("assistant", "Hello!"), ("user", "rewrite it shorter")]}) == (
        "Conversation:\nUser: hi\nAssistant: Hello!\nQuestion: rewrite it shorter\nAnswer:")
    assert talk_prompt("ignored", {"turns": [("user", "Give me three tips.")]}) == "Question: Give me three tips.\nAnswer:"
    assert talk_prompt("Summarize this.", "talk") == "Question: Summarize this.\nAnswer:"
    assert talk_prompt("Continue the story", {"turns": [("user", "Continue the story")], "explicit": True}).startswith("Explicit: allowed\n")


def test_the_answer_is_cut_at_the_next_block_and_kept_to_one_line():
    assert talk_answer(" Marie Curie was born in Warsaw.\nQuestion: next?\nAnswer: no") == "Marie Curie was born in Warsaw."
    assert talk_answer("The facts don't say.\nFacts:\n- x") == "The facts don't say."
    assert talk_answer("  two   spaces \n second line") == "two spaces"


def test_both_seats_on_one_trunk_render_decode_and_swap_only_on_a_role_change():
    trunk = FakeTrunk({"programs": 'program CotPlan implements ISolve {\n    bind frame, SEED, "$E1";\n    bind frame, ACTION, "x";\n}\n',
                       "talk": " Marie Curie was born in Warsaw.\nQuestion: more?"})
    programs, talk = Cubby450mEmitter(trunk), Cubby450mTalk(trunk)
    assert programs.name == "cubby450m:emitter_x" and talk.name == "cubby450m:talk_y"
    bank = ContextualEmitter({"programs": programs, "talk": talk}, default="programs")
    prog = bank.emit("Who is the mother of [E1: Justin Trudeau]?", max_new_tokens=64)
    assert 'bind frame, SEED, "$E1"' in prog and "ACTION" in prog                    # the role token survives decoding
    role, shown, _ = trunk.calls[-1]
    assert role == "programs" and shown[0].startswith("Question:\nWho is the mother of [E1: Justin Trudeau]?\nProgram:\n")
    said = bank.emit("say it plainly", context={"role": "talk", "facts": [("Marie Curie", "place of birth", "Warsaw")],
                                                 "question": "Where was Marie Curie born?"},
                     system="a persona the adapter never sees", max_new_tokens=32)
    assert said == "Marie Curie was born in Warsaw."
    role, shown, temp = trunk.calls[-1]
    assert role == "talk" and shown[0] == "Facts:\n- Marie Curie — place of birth: Warsaw\nQuestion: Where was Marie Curie born?\nAnswer:"
    assert "persona" not in shown[0] and temp == 0.0
    bank.emit("again", context={"role": "talk", "turns": [("user", "again")]}, max_new_tokens=8, temperature=0.7, seed=3)
    assert trunk.calls[-1][2] == 0.7
    assert trunk.swaps == 2                                                          # programs -> talk, then talk stays
    bank.emit("Who is [E1: X]?", max_new_tokens=8)
    assert trunk.swaps == 3 and bank.calls == {"programs": 2, "talk": 2}
    assert talk.emit("p", prefix="Yes,", context={"turns": [("user", "p")]}) == "Yes, Marie Curie was born in Warsaw."
    assert bank.is_split and "emitter_x" in bank.name and "talk_y" in bank.name


def test_the_real_trunk_refuses_a_missing_file_before_touching_the_card(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        Cubby450mTrunk(str(tmp_path), str(tmp_path / "tok.json"), {"programs": str(tmp_path / "none")})
    with pytest.raises(ValueError):
        Cubby450mTrunk(str(tmp_path), str(tmp_path), {})
