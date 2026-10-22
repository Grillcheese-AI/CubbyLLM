"""Pins for the chat builder (standin/data/build_chat_sft.py, rewritten in b647ebc for one instruction-following
source), the v6 checks other code still imports from it (chat_sft_v6.py), and the gap families' parts.
The tests of the removed v6 builder steps are archived in docs/archive/test_build_chat_sft_v6_2026-10-22.py (order 0030).
No corpus access, no model, no VM. Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_chat_sft as b  # noqa: E402
import identity as idn  # noqa: E402

F = idn.load_facts()


def test_history_ok_scores_when_dating_and_containment():
    when = {"subtype": "when", "gold": "3100 BCE", "gold_any": ["3100 BCE"]}
    assert b.history_ok(when, "It was around 3100 BCE.", F) and b.history_ok(when, "About 3100 BC, I believe.", F)
    assert not b.history_ok(when, "Around 2000 BCE.", F)
    dating = {"subtype": "dating", "gold": "1925", "gold_any": ["1925"]}
    assert b.history_ok(dating, "That reads like 1927, the late twenties.", F) and not b.history_ok(dating, "Probably 1961.", F)
    assert not b.history_ok(dating, "I cannot tell.", F)
    about = {"subtype": "about", "gold": "1066", "gold_any": ["Hastings", "William", "1066"]}
    assert b.history_ok(about, "William won at Hastings.", F) and not b.history_ok(about, "A battle somewhere, long ago.", F)
    assert not b.history_ok(about, "To be honest, William won at Hastings.", F), "voice rules still hold"
    assert b.proper_nouns("During the siege, General Grant took Vicksburg in 1863.") == ["General", "Grant", "Vicksburg", "1863"]


def test_diabla_pairs_are_human_on_both_sides_and_quebec_records_take_both_forms(tmp_path):
    import gap_families as g
    from identity import load_facts
    F = load_facts()
    dialogues = {"d1": {"utterances": {
        "0": {"language": "english", "original_text": "This is not a good start to the day", "reference_translation": "Ce n'est pas un bon début de journée."},
        "1": {"language": "french", "original_text": "Non, l'ascenseur est bloqué et je devais présenter le projet à neuf heures.", "reference_translation": "No, the lift is stuck and I had to present the project at nine."},
        "2": {"language": "english", "original_text": "Then we have time to go over it together.", "reference_translation": "Alors on a le temps de le revoir ensemble."},
        "3": {"language": "english", "original_text": "Do you have the slides?", "reference_translation": "Tu as les diapositives ?"}}}}
    pairs = list(g.diabla_pairs(dialogues))
    assert pairs == [("fr", "Ce n'est pas un bon début de journée.", "Non, l'ascenseur est bloqué et je devais présenter le projet à neuf heures."),
                     ("en", "No, the lift is stuck and I had to present the project at nine.", "Then we have time to go over it together.")], \
        "same-language consecutive turns (2 -> 3) are not a pair; the user side is the human reference translation"
    # quebec: one benchmark row -> an explain record (definition, token-F1 check) and a multiple-choice record (number first)
    row = {"expression": "Être à cheval entre deux réalités.", "choices": ["Chercher à concilier deux vérités.", "Hésiter entre deux réalités, tergiverser.", "Douter de soi-même."], "correct_index": 1}
    word = {"terme": "Adonner", "choices": ["Préparer un plat.", "Se produire de façon fortuite, une coïncidence.", "Offrir un cadeau."], "correct_index": 1}
    tmp = tmp_path / "q.jsonl"
    tmp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp2 = tmp_path / "w.jsonl"
    tmp2.write_text(json.dumps(word, ensure_ascii=False) + "\n", encoding="utf-8")
    saved = g.QUEBEC_FILES
    g.QUEBEC_FILES = (("expression", str(tmp)), ("mot", str(tmp2)))
    try:
        recs, why = g.build_quebec(random.Random(0), F)
    finally:
        g.QUEBEC_FILES = saved
    assert {r["task"] for r in recs} == {"quebec", "quebec_mc"} and all(r["lang"] == "fr" for r in recs) and len(recs) == 4
    assert any(r["subtype"] == "mot" and "« Adonner »" in r["prompt"] and r["gold"].startswith("Se produire") for r in recs), "QFrCoRT rows say `terme`"
    ex = next(r for r in recs if r["task"] == "quebec" and r["subtype"] == "expression"); mc = next(r for r in recs if r["task"] == "quebec_mc" and r["subtype"] == "expression")
    assert "« Être à cheval entre deux réalités. »" in ex["prompt"] and ex["gold"] == "Hésiter entre deux réalités, tergiverser."
    assert g.quebec_ok(ex, "Ça veut dire hésiter entre deux réalités, tergiverser sans se décider.") and not g.quebec_ok(ex, "Douter de soi.")
    assert mc["program"].startswith(mc["gold"] + " — ") and f"{mc['gold']}. Hésiter" in mc["prompt"] and g.label_first_ok(mc, mc["gold"] + ".")
    assert g.gap_ok(mc, mc["gold"], F) and g.gap_ok(ex, ex["gold"], F)



def test_free_text_families_score_as_generative_multiple_choice_and_empathy_takes_synonyms():
    """First v9t bake-off read (2026-09-04): appraisal 2/40 and quebec 5/40 measured the F1 check, not the model."""
    import gap_families as g
    ap = {"task": "appraisal", "gold": "go with Cameron", "program": "Go with Cameron.", "choices": ["go with Cameron", "ignore Cameron", "move away"]}
    assert g.appraisal_ok(ap, "Get to know Cameron better and go with Cameron.") and not g.appraisal_ok(ap, "Move away from the whole thing.")
    qb = {"task": "quebec", "gold": "Très laid.", "program": "Très laid.", "choices": ["Très laid.", "Très beau.", "Très fatigué."]}
    assert g.quebec_ok(qb, "Ça veut dire quelqu'un de très laid.") and not g.quebec_ok(qb, "Quelqu'un de très fatigué.")
    assert g.quebec_ok({"task": "quebec", "gold": "Très laid.", "program": "Très laid."}, "très laid"), "no choices: the F1 fallback"
    em = {"task": "empathy", "gold": "sentimental"}
    assert g.empathy_ok(em, "nostalgic.") and g.empathy_ok(em, "Sentimental") and not g.empathy_ok(em, "angry.")
    assert g.gap_ok(em, "nostalgic", None)



def test_v11_silent_negatives_are_voice_checked_both_formats_and_never_call():
    """The v10t probe (Q4 twice, Q8 once) called a tool on 'raconte-moi ta journée'; v11 adds creative/personal turns whose
    reply stands. Every hand-written reply passes the voice rules; the check rejects any call for them."""
    import random
    import gap_families as g
    from identity import voice_ok
    chat = [{"task": "chat", "prompt": "tell me a story about a fox", "program": "A fox found a hat. The end.", "source": "t", "lang": "en"},
            {"task": "chat", "prompt": "what is the capital of peru", "program": "Lima.", "source": "t", "lang": "en"},
            {"task": "chat", "prompt": "raconte-moi une blague", "program": "Un mur, un fantôme, boum.", "source": "t", "lang": "fr"}]
    recs, why = g.build_toolcall_silent(random.Random(0), F, chat, n=10)
    assert {r["format"] for r in recs} == {"lfm", "hermes"} and all(r["gold"] == "none" and r["task"] == "toolcall" for r in recs)
    kinds = {r["subtype"] for r in recs}
    assert {"silent:day", "silent:story", "silent:statement", "silent:joke", "silent:chat"} <= kinds
    assert any(r["prompt"].lower() == "raconte-moi ta journée" for r in recs), "the probe's own turn is in the set"
    assert why["creative_chat_rows_available"] == 2, "the capital question is not creative"
    assert not [k for k in why if k.startswith("voice:")], f"every hand-written reply passes voice_ok: {why}"
    for k, l, ask, reply in g.silent_pairs():
        assert voice_ok(reply, F), reply
    for r in recs[:40]:
        assert g.toolcall_ok(r, r["program"])
        assert not g.toolcall_ok(r, g.render_call(r["format"], "request_tool", {"name": "chat_status", "description": "x", "example_request": r["prompt"]}))
        assert not g.toolcall_ok(r, g.render_call(r["format"], "remember_fact", {"fact": "ma chienne est nommée Luna"}))


def test_a_request_tool_never_names_a_registered_tool():
    import gap_families as g
    rec = {"task": "toolcall", "format": "lfm", "gold": "request_tool", "call_args": {"name": "dice_roll", "description": "roll dice", "example_request": "roll a d20"}}
    assert g.toolcall_ok(rec, g.render_call("lfm", "request_tool", {"name": "dice_roll", "description": "roll dice", "example_request": "roll a d20"}))
    assert not g.toolcall_ok(rec, g.render_call("lfm", "request_tool", {"name": "remember_fact", "description": "store a fact", "example_request": "roll a d20"})), \
        "the Q4 probe asked the factory for a tool that already exists"
    assert not g.toolcall_ok(rec, g.render_call("hermes", "request_tool", {"name": "News_Search", "description": "x", "example_request": "y"}))


def test_the_builder_reads_language_refusals_and_placeholders():
    assert b.language("The cat is on the mat and it is happy to see you.") == "en"
    assert b.language("Le chat est dans la maison et il est pour vous avec nous.") == "fr"
    assert b.language("El gato está en la casa y es muy feliz con los niños.") is None       # a neighbour, not ours
    assert b.language("ok") is None
    assert b.refusal("I'm sorry, but I can't help with that request.") and not b.refusal("I can help with that: here it is.")
    assert b.placeholder_policy("Tell me about NAME_1.", "NAME_1 is the fifth planet.") == "placeholder"
    assert b.placeholder_policy("Let's role-play: you are NAME_1, a knight.", "I am NAME_1.") is None
    assert b.placeholder_policy("hi", "<PERSON_ANONYMIZED_1> says hi") == "presidio"
    u = b.unmask("r1", "NAME_1 met NAME_2, then NAME_1 left")
    assert "NAME_" not in u and u.split()[0] == u.split()[-2]                                # one name per placeholder


def test_the_builder_renders_turns_and_splits_by_id():
    assert b.render([("user", "Hi?")]) == "Question: Hi?\nAnswer:"
    two = b.render([("system", "Be brief."), ("user", "And then?")], unlocked=True)
    assert two.startswith(b.UNLOCK + "\nConversation:\nSystem: Be brief.\nQuestion: And then?")
    splits = [b.split_of(f"id{i}") for i in range(2000)]
    assert splits == [b.split_of(f"id{i}") for i in range(2000)] and 60 < splits.count("held") < 140   # 1 in 20, stable
    assert b.support("Paris is the capital", "What is the capital of France? Paris?") == 1.0
    assert b.chat_view("1. The answer\n2. I think so", {"the"}).count("the answer") == 1        # a sentence-opening common word lowered


def test_the_v6_checks_other_code_imports():
    assert b.affect_ok({"gold_any": [0.4, 0.6]}, "valence 0.5, arousal 0.3") and not b.affect_ok({"gold_any": [0.4, 0.6]}, "valence -0.5, arousal 0.3")
    assert b.parse_affect("no numbers") is None
    assert len(b.GOEMOTIONS) == 28 and set(b.GOEMOTIONS_PETAL) == set(b.GOEMOTIONS) == set(b.GOEMOTIONS_FR)
    assert set(b.MULTI_PROMPT) == {"en", "fr"} and "{u2}" in b.MULTI_PROMPT["fr"]
