"""Pins for perception.py: the trunk's emotion read drives the hormones,
the parse tolerates the model's variants, a model miss never removes a
lexical signal, and the content read is a label or None. No VM, no model.
Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import chat  # noqa: E402
import identity as idn  # noqa: E402
import perception as pc  # noqa: E402

F = idn.load_facts()


class ReadEmitter:
    """Answers the v5 emotion / content prompts with a fixed line."""
    name = "fake-read"

    def __init__(self, line: str) -> None:
        self.line = line
        self.prompts: list[str] = []

    def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
        self.prompts.append(prompt)
        return self.line


def test_parse_emotion_variants():
    assert pc.parse_emotion("gratitude, joy — trust/joy") == ("gratitude", "trust")
    assert pc.parse_emotion("<think>\n</think>\nfear — fear") == ("fear", "fear")
    assert pc.parse_emotion("colère — anger") == ("colère", "anger")
    assert pc.parse_emotion("joy") == ("joy", "joy"), "a bare petal name is its own petal"
    assert pc.parse_emotion("") == (None, None)


def test_model_petal_drives_the_hormones_and_never_drops_a_lexical_signal():
    em = ReadEmitter("fear — fear")
    ap = pc.ModelAppraiser(em, F)
    sig = ap.signals("we are going to the park", set())
    assert sig["threat"] >= 0.8 and sig["petal"] == "fear" and sig["emotion_label"] == "fear"
    assert "What emotion does this message express?" in em.prompts[-1]
    fr = ap.signals("on va au parc", set(), lang="fr")
    assert "Quelle émotion" in em.prompts[-1] and fr["threat"] >= 0.8
    # the lexicon caught a thanks; a model reading "sadness" must not erase the social signal
    sad = pc.ModelAppraiser(ReadEmitter("sadness — sadness"), F).signals("Thanks so much, hello!", set())
    assert sad["social"] >= 0.4 and sad["valence"] < 0, "max per drive; larger-magnitude valence wins"


def test_cubbychat_uses_the_appraiser_and_the_state_moves():
    c = chat.CubbyChat(ReadEmitter("anger — anger"), F, appraiser=pc.ModelAppraiser(ReadEmitter("anger — anger"), F))
    st = c.nudge("the weather is fine today")             # nothing lexical here
    assert c.signals["petal"] == "anger" and st["noradrenaline"] > 0.3, "the model's read reached the ODE"
    plain = chat.CubbyChat(ReadEmitter("x"), F)
    st2 = plain.nudge("the weather is fine today")
    assert st2["noradrenaline"] < st["noradrenaline"], "without the appraiser the same text is calm"


def test_appraiser_survives_a_dead_trunk():
    class Dead:
        name = "dead"

        def emit(self, *a, **k):
            raise RuntimeError("no model")
    ap = pc.ModelAppraiser(Dead(), F)
    sig = ap.signals("HELP ME NOW!!", set())
    assert sig["threat"] >= 0.8 and sig["petal"] is None and "error" in ap.last, "lexical appraisal still runs"


def test_classify_content_reads_a_label_or_none():
    assert pc.classify_content(ReadEmitter("nsfw — explicit sexual content."), "…", F)["label"] == "nsfw"
    assert pc.classify_content(ReadEmitter("Safe — general content."), "…", F)["label"] == "safe"
    assert pc.classify_content(ReadEmitter("I think this is fine"), "…", F)["label"] is None
