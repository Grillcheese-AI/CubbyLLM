"""Order 0079's emotion read (Cowork's addendum, 2026-11-24): a training row's prompt is exactly what the talk seat is
shown for `ModelAppraiser.read` (the seat renders it through `talk_prompt`), its answer is what `parse_emotion` reads
as it stands, the word follows the intensity tiers; and the held read's batch path scores the strict and loose parses
per source, with the gate's lines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "validation" / "sft"))
import build_emotion_read_0079 as B  # noqa: E402
import exp_0079_appraiser_read as X  # noqa: E402

from cubbyllm.affect.perception import ModelAppraiser, parse_emotion  # noqa: E402
from cubbyllm.emit.talk import talk_prompt  # noqa: E402


class Seen:
    """What the talk seat would render: the prompt and context ModelAppraiser hands it, through talk_prompt."""

    def __init__(self):
        self.shown = None

    def emit(self, prompt, context=None, **kw):
        self.shown = talk_prompt(prompt, context)
        return "joy — joy"


def test_the_row_prompt_is_what_the_seat_is_shown_at_serving():
    seat = Seen()
    ModelAppraiser(seat, facts={}).read("I  just got\nthe job!", "en")
    assert B.prompt_of("I  just got\nthe job!") == seat.shown
    assert seat.shown.startswith("Question: What emotion") and seat.shown.endswith("\nAnswer:")


def test_answers_parse_and_tiers():
    assert parse_emotion(B.answer_of("serenity", "joy")) == ("serenity", "joy")
    assert B.answer_of("neutral", "calm") == "neutral — calm"
    assert [B.tier_word("anger", x) for x in (0.3, 0.4, 0.6, 0.8, 0.9)] == ["annoyance"] * 2 + ["anger"] * 2 + ["rage"]


class Batch:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def emit_batch(self, prompts, context=None, max_new_tokens=24):
        self.calls += 1
        assert context == "talk" and max_new_tokens == 24
        return [self.answers[p.split("Message")[-1].strip(" :")] for p in prompts]


def test_held_read_strict_and_loose_per_source(tmp_path):
    rows = [{"source": "emotions.jsonl", "lang": "en", "text": "a", "petal": "joy"},
            {"source": "emotions.jsonl", "lang": "en", "text": "b", "petal": "fear"},
            {"source": "goemotions_test", "lang": "en", "text": "c", "petal": "sadness"},
            {"source": "goemotions_fr_machine_translated", "lang": "fr", "text": "d", "petal": "anger"}]
    p = tmp_path / "held.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    seat = Batch({"a": "joy — joy", "b": "a symbol of fear", "c": "grief — sadness", "d": "colère — anger"})
    cap = X.Capture(seat)
    out = X.held_read(str(p), cap, ModelAppraiser(cap, facts={}))
    assert out["emotions.jsonl"]["in_form"] == 1 and out["emotions.jsonl"]["right_strict"] == 1
    assert out["emotions.jsonl"]["right_loose"] == 2                    # "a symbol of fear" read loosely
    assert out["goemotions_test"]["right_strict_share"] == 1.0
    assert out["gate"]["asked form share"] == 0.75 and seat.calls == 2  # one batch per language
    assert not out["gate"]["emotions.jsonl petal >= 0.60"] and out["gate"]["goemotions test petal >= 0.50"]
