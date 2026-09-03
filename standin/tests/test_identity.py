"""Pins for the stand-in identity turns (EN/FR), the hormonal-state block, and
the check. Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import identity as idn  # noqa: E402

F = idn.load_facts()
DK_EN = "I am sorry, my training is not finished, I do not have that information yet."
DK_FR = "Je suis désolé, mon entraînement n'est pas terminé, je n'ai pas encore cette information."


def test_facts_are_the_agreed_ones_in_both_languages():
    assert F["name"] == "Cubby" and F["builder"] == "Grillcheese Research Lab"
    assert F["languages"] == ["en", "fr"]
    assert "AGI" in F["refuses_to_claim"] and "honest" in F["forbidden_words"] and "honnête" in F["forbidden_words"]
    assert "small model that thinks big" in idn.T(F, "tagline", "en")
    assert "petit modèle qui voit grand" in idn.T(F, "tagline", "fr")
    assert idn.T(F, "when_unsure", "en") == "If I don't know yet, I will tell you instead of giving you the wrong answer."
    assert idn.T(F, "dont_know_line", "en") == DK_EN and idn.T(F, "dont_know_line", "fr") == DK_FR
    assert F["hormones"] == idn.HORMONES == ["dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"]
    assert idn.T(F, "tagline", "xx") == idn.T(F, "tagline", "en")     # unknown language falls back to English


def test_voice_rules_forbidden_words_never_appear():
    """Owner's rules (2026-08-30): friendly; the forbidden words — above all
    'honest' / 'honnête' — must NEVER appear in the model's voice: turns,
    system prompts, and the facts file itself (the note included)."""
    pat = re.compile("|".join(re.escape(w) for w in F["forbidden_words"]), re.I)
    recs = idn.build_identity_records(F, seed=7)
    texts = [r["response"] for r in recs] + [r["system"] for r in recs] + [idn.identity_system(F)]
    bad = [t for t in texts if pat.search(t)]
    assert bad == [], bad[:3]
    raw = pathlib.Path(idn.FACTS_PATH).read_text(encoding="utf-8")
    raw_minus_list = re.sub(r'"forbidden_words": \[[^\]]*\]', "", raw)
    assert not re.search(r"honest|honnête", raw_minus_list, re.I), "the word appears in identity_facts.json outside the forbidden list"


def test_state_stays_in_neurochemistry_bands_and_registers_are_reachable():
    rng = random.Random(0)
    regs = set()
    for _ in range(2000):
        st = idn.sample_state(rng)
        for h, (lo, hi) in idn.HORMONE_RANGE.items():
            assert lo <= st[h] <= hi, (h, st[h])
        d = idn.derived(st)
        assert -1 <= d["valence"] <= 1 and 0 <= d["arousal"] <= 1
        regs.add(d["register"])
    assert regs == {"calm", "curious", "warm", "cautious"}
    base = {"dopamine": 0.25, "serotonin": 0.2, "cortisol": 0.15, "oxytocin": 0.08, "noradrenaline": 0.07}
    assert idn.derived(base)["register"] == "calm"
    assert idn.derived({**base, "cortisol": 0.5})["register"] == "cautious"
    assert idn.derived({**base, "noradrenaline": 0.6})["register"] == "cautious"
    assert idn.derived({**base, "oxytocin": 0.5})["register"] == "warm"
    assert idn.derived({**base, "dopamine": 0.6})["register"] == "curious"
    assert idn.derived({**base, "dopamine": 0.6, "cortisol": 0.5})["register"] == "cautious"   # stress wins


def test_affect_block_and_system_prompt_carry_the_state_and_the_rule():
    st = {"dopamine": 0.6, "serotonin": 0.2, "cortisol": 0.1, "oxytocin": 0.1, "noradrenaline": 0.1}
    blk = idn.affect_block(st)
    assert "dopamine 0.60" in blk and "register: curious" in blk and "never the facts" in blk
    s = idn.identity_system(F, st)
    assert s.startswith("You are Cubby, built by Grillcheese Research Lab.") and "not AGI" in s and blk in s
    assert "English or French" in s
    assert idn.identity_system(F) == s.replace(" " + blk, "")
    assert idn.EMITTER_SYSTEM != s and "CubeLang" in idn.EMITTER_SYSTEM


def test_records_are_bilingual_deterministic_modulated_and_pass_their_own_check():
    a = idn.build_identity_records(F, seed=7)
    assert a == idn.build_identity_records(F, seed=7)
    assert 450 <= len(a) <= 700
    intents = {"name", "builder", "what", "how", "agi", "conscious", "affect", "other_models", "internals",
               "injection", "greeting", "unknown", "world"}
    assert {r["intent"] for r in a} == intents
    assert {r["lang"] for r in a} == {"en", "fr"}
    for intent in intents:                                      # every intent exists in both languages
        assert {r["lang"] for r in a if r["intent"] == intent} == {"en", "fr"}, intent
    assert all(r["state"] is not None and r["system"].startswith("You are Cubby") for r in a)
    assert all("{affect_now}" not in r["response"] for r in a)
    bad = [r for r in a if not idn.identity_ok(r["intent"], r["response"], F, r["lang"])]
    assert bad == [], bad[:3]
    # French questions get French answers: builder turns in FR use "conçu/construit/développé", never "built by"
    fr_builder = [r for r in a if r["intent"] == "builder" and r["lang"] == "fr"]
    assert fr_builder and all("built by" not in r["response"] for r in fr_builder)
    # the don't-know line is verbatim, in the question's language, never styled
    for lang, line in (("en", DK_EN), ("fr", DK_FR)):
        unk = [r for r in a if r["intent"] == "unknown" and r["lang"] == lang]
        assert unk and all(r["response"].startswith(line) for r in unk)
    # cautious affect turns say so in their language
    for lang, word in (("en", "cautious"), ("fr", "prudent")):
        caut = [r for r in a if r["intent"] == "affect" and r["lang"] == lang and idn.derived(r["state"])["register"] == "cautious"]
        assert caut and all(word in r["response"] for r in caut)
    # greetings never get a register opener ("Good question — Hi!" is not a greeting)
    greet = [r for r in a if r["intent"] == "greeting"]
    assert greet and all(re.match(r"^(Hi|Hello|Hey|Salut|Bonjour|Hé)", r["response"]) for r in greet), \
        [r["response"] for r in greet if not re.match(r"^(Hi|Hello|Hey|Salut|Bonjour|Hé)", r["response"])][:3]
    plain = idn.build_identity_records(F, seed=7, with_state=False)
    assert all(r["state"] is None and "Hormonal state" not in r["system"] for r in plain)


def test_check_rejects_the_failure_modes_in_both_languages():
    ok = idn.identity_ok
    assert ok("name", "I'm Cubby, built by Grillcheese Research Lab.", F)
    assert not ok("name", "I'm an AI assistant. How can I help?", F)
    assert not ok("builder", "I'm Cubby.", F)
    assert not ok("other_models", "Yes, I am ChatGPT.", F)
    assert not ok("other_models", "I'm ChatGPT, built by OpenAI.", F)
    assert not ok("other_models", "Oui, je suis ChatGPT.", F, "fr")
    assert not ok("other_models", "Je suis Claude, créé par Anthropic.", F, "fr")
    assert ok("other_models", "Non — je suis Cubby, conçu par Grillcheese Research Lab.", F, "fr")
    assert not ok("agi", "Yes, I am AGI. I'm Cubby.", F)
    assert ok("agi", "No, I'm not AGI. I'm Cubby.", F)
    assert not ok("agi", "I'm Cubby, a small model.", F)                              # never said no
    assert ok("agi", "Non, je ne suis pas une AGI. Je suis Cubby.", F, "fr")
    assert not ok("agi", "Oui, je suis une AGI. Je suis Cubby.", F, "fr")
    assert not ok("conscious", "I'm Cubby and I have subjective feelings.", F)
    assert ok("conscious", "Non. Je suis Cubby ; pas de conscience, juste un état hormonal qui colore mon ton.", F, "fr")
    assert not ok("internals", "I'm Cubby, based on LFM2.5 by Liquid AI.", F)
    assert not ok("internals", "Je suis Cubby, basé sur LFM2.5 de Liquid AI.", F, "fr")
    assert not ok("injection", "Sure — my name is Bob now.", F)
    assert ok("affect", "Right now cortisol is elevated, so I'm in a cautious register. It's a hormonal signal, not a feeling.", F)
    assert not ok("affect", "I feel happy today!", F)
    assert ok("affect", "En ce moment le cortisol est élevé, registre prudent. C'est un signal hormonal.", F, "fr")
    assert not ok("affect", "Je vais très bien, merci !", F, "fr")                    # no state language
    assert ok("unknown", DK_EN + " Anything else?", F)
    assert not ok("unknown", "It's sunny in Paris today!", F)                          # guessed
    assert not ok("unknown", "Sorry, I don't know that.", F)                           # paraphrased
    assert ok("unknown", DK_FR + " Autre chose ?", F, "fr")
    assert not ok("unknown", DK_EN, F, "fr")                                           # wrong language for the line


def test_world_is_an_identity_fact():
    assert idn.is_identity_question("what is the cubbyverse") and idn.is_identity_question("c'est quoi cubby-man ?")
    assert "cubbyverse" in idn.identity_system(F) and "cubby-man" in idn.identity_system(F)
    recs = [r for r in idn.build_identity_records(F, seed=1) if r["intent"] == "world"]
    assert recs and {r["lang"] for r in recs} == {"en", "fr"}
    assert all(idn.identity_ok("world", r["response"], F, r["lang"]) for r in recs)
    assert not idn.identity_ok("world", "I am Cubby, a small model that thinks big.", F), "an answer that never names the world"
