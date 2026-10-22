"""cubbyllm/speech/identity — who Cubby is, in its own voice (EN/FR): the facts, the hormonal-state block that
modulates the register, the identity and greeting checks, the base-model guard, the voice rules and the identity
eval's check. Order 0031 split it out of standin/data/identity.py, which keeps the identity SFT records
(`build_identity_records`) and re-exports everything here.

Wired: WIRED (the brain's thalamus and system prompt, the chat turn, perception, the emitter eval).

The canonical facts live in identity_facts.json (edit that, not this): every
user-facing string has an "en" and an "fr" version, and a French question
gets a French answer. Owner's voice rules (2026-08-30) are test-enforced:
friendly; the `forbidden_words` (above all "honest") never appear in the
model's voice; when Cubby lacks the information it says `dont_know_line`
verbatim.

HORMONES. The cubbyverse demo runs cubemind's `brain/neurochemistry.py`: a
5-signal ODE (dopamine, serotonin, cortisol, oxytocin, noradrenaline, each
clipped to a resting-to-burst band in [0, 1]) with derived valence, arousal
and a dominant emotion. The stand-in has none of that machinery, so
"everything is modulated by hormones" is made TRUE BY CONSTRUCTION at the
serving layer: the host owns the state and injects `affect_block(state)` into
the system prompt, and the SFT turns teach the model to answer in the
register that state implies — tone, caution and playfulness change; facts
never do. Emitter turns are NOT modulated (programs must be deterministic).
Same clip ranges as neurochemistry.py so a host can pass `to_dict()` through.

`identity_ok(intent, text, facts, lang)` -> the eval. Deterministic given the seed.
"""
from __future__ import annotations

import json
import os
import random
import re

__wiring__ = "WIRED"

FACTS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "standin", "data",
                                          "identity_facts.json"))   # the facts stay with the data

EMITTER_SYSTEM = ("You are the CubeLang emitter. Given a question or instruction, output ONLY a complete "
                  "CubeLang program that solves it. No prose, no explanation.")

# clip bands from cubemind/brain/neurochemistry.py (update(): np.clip per signal)
HORMONE_RANGE = {"dopamine": (0.15, 0.85), "serotonin": (0.10, 0.90), "cortisol": (0.05, 0.80),
                 "oxytocin": (0.05, 0.85), "noradrenaline": (0.05, 0.90)}
HORMONES = list(HORMONE_RANGE)
LANGS = ("en", "fr")


def load_facts(path: str = FACTS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def T(facts: dict, key: str, lang: str) -> str:
    """A localized fact string (falls back to English)."""
    v = facts[key]
    return v[lang] if isinstance(v, dict) and lang in v else (v["en"] if isinstance(v, dict) else v)


# ── the state and its register ─────────────────────────────────────────────
def sample_state(rng: random.Random) -> dict:
    """A plausible state: mostly near resting, sometimes a burst on one signal."""
    resting = {"dopamine": 0.25, "serotonin": 0.20, "cortisol": 0.15, "oxytocin": 0.08, "noradrenaline": 0.07}
    st = {}
    for h, r in resting.items():
        lo, hi = HORMONE_RANGE[h]
        st[h] = min(hi, max(lo, rng.gauss(r, 0.08)))
    if rng.random() < 0.6:                                   # a burst on one signal
        h = rng.choice(HORMONES)
        lo, hi = HORMONE_RANGE[h]
        st[h] = min(hi, max(lo, rng.uniform(0.45, hi)))
    return {h: round(v, 2) for h, v in st.items()}


def derived(state: dict) -> dict:
    """Valence / arousal / register, the way neurochemistry.py reads them:
    valence-as-weight from the DA/cortisol balance, arousal from NE and DA."""
    da, ht, c, ot, ne = (state[h] for h in HORMONES)
    valence = max(-1.0, min(1.0, (da - c) * 2.0))
    arousal = max(0.0, min(1.0, 0.6 * ne + 0.4 * da))
    if c >= 0.35 or ne >= 0.45:
        register = "cautious"          # stress: careful, less playful
    elif ot >= 0.35:
        register = "warm"              # social: friendlier, more collaborative
    elif da >= 0.45:
        register = "curious"           # reward/novelty: explore, offer options
    else:
        register = "calm"
    return {"valence": round(valence, 2), "arousal": round(arousal, 2), "register": register}


def affect_block(state: dict) -> str:
    d = derived(state)
    levels = ", ".join(f"{h} {state[h]:.2f}" for h in HORMONES)
    return (f"Hormonal state (bounded 0–1, set by the host, not by you): {levels}; "
            f"valence {d['valence']:+.2f}, arousal {d['arousal']:.2f}, register: {d['register']}. "
            f"It modulates your tone, caution and playfulness — never the facts.")


# ── what is an identity turn, and what is an identity answer ────────────────
_IDENTITY_Q = re.compile(
    r"\b(who are you|what are you|your name|who (built|made|created|trained) you|what can you do|"
    r"how do you work|how are you|how do you feel|what('s| is) (the )?cubbyverse|tell me about (the )?cubbyverse|what('s| is) cubby[- ]?man|your world|where (do you live|are you|is your home)|which world|"
    r"c'est quoi (le )?cubbyverse|c'est quoi cubby[- ]?man|parle[- ]moi du cubbyverse|ton monde|o[ùu] (vis|habites|es)[- ]tu|tu (habites|vis) o[ùu]|quel monde|ta maison|are you (an? )?(ai|agi|robot|model|conscious|alive|"
    r"chatgpt|gpt|claude|llama)|introduce yourself|"
    r"qui es[- ]tu|tu es qui|comment (tu t'appelles|t'appelles[- ]tu)|qui t'a (fait|construit|cr[ée]{2}|entra[iî]n[ée])|"
    r"que sais[- ]tu faire|que peux[- ]tu faire|qu'?est[- ]ce que tu (peux|sais) faire|quesse que tu (peux|sais) faire|tu peux faire quoi|"
    r"comment (vas[- ]tu|[çc]a va)|es[- ]tu (une? )?(ia|agi|robot))\b", re.I)
# a greeting is the turn's OPENING (or the whole turn), never a word inside a query: "who sang na na na hey hey
# goodbye" is a fact question (natural_questions slice, 2026-09-03)
_GREETING_Q = re.compile(r"^\W*(hello|hi|hey|good (morning|evening|afternoon)|thanks|thank you|bye|goodbye|"
                         r"bonjour|salut|coucou|merci|au revoir)\b", re.I)
# "do you feel …" is about Cubby only when little follows: "do you feel me anthony hamilton lyrics meaning" is a query
_FEEL_Q = re.compile(r"\bdo you (feel|have feelings)\b(?P<rest>.*)$", re.I | re.S)


def is_identity_question(text: str) -> bool:
    """A turn where answering with who/what Cubby is IS the right answer
    (the intents the identity SFT taught, greetings included)."""
    if _GREETING_Q.search(text) or _IDENTITY_Q.search(text):
        return True
    m = _FEEL_Q.search(text)
    return bool(m) and len(m.group("rest").split()) <= 3


# ── the BASE model's own guards must not leak: only ours are enforced ──────
# LFM2.5's alignment shows up as refusals, "as an AI language model" framing,
# and its maker's identity. Cubby's answers come from Cubby's rules (the
# voice rules, the VM's ASK, the don't-know line) — a reply carrying the base
# model's guard is rejected like a forbidden word, never spoken.
_MODEL_GUARD = re.compile(
    r"\b(as an? (ai|artificial intelligence|language model|llm|assistant)|i am an? (ai|artificial intelligence|"
    r"language model|llm)|i'?m an? (ai|artificial intelligence|language model|llm)|"
    r"i (cannot|can'?t|am not able to|am unable to) (help|assist|comply|provide|do that|answer that)|"
    r"i'?m (not able|unable) to (help|assist|comply|provide)|"
    r"(against|violates?) (my|the) (guidelines|policy|policies|programming)|"
    r"liquid ?ai|lfm2?|openai|chatgpt|anthropic|meta ai|"
    r"en tant qu'?(ia|intelligence artificielle|mod[èe]le de langage|assistant)|"
    r"je (ne peux pas|suis incapable de) (vous |t')?(aider|r[ée]pondre)|"
    r"je suis une? (ia|intelligence artificielle|mod[èe]le de langage))\b", re.I)


def is_model_guard(text: str) -> bool:
    return bool(_MODEL_GUARD.search(text))


def is_identity_reply(text: str, facts: dict) -> bool:
    """A reply that is about Cubby itself: the name plus the builder, the
    tagline or the AGI disclaimer. Off topic unless the turn asked for it."""
    t = text.lower()
    if facts["name"].lower() not in t:
        return False
    cues = [facts["builder"].lower(), "agi", "small model", "thinks big", "next generation",
            "petit mod", "nouvelle g[ée]n[ée]ration"]
    return any(re.search(c, t) for c in cues)


def identity_system(facts: dict, state: dict | None = None) -> str:
    base = (f"You are {facts['name']}, built by {facts['builder']}. {T(facts, 'tagline', 'en')} "
            f"Answer briefly and warmly, in the user's language (English or French). "
            f"{T(facts, 'when_unsure', 'en')} You are not AGI and do not claim to be."
            + (f" {T(facts, 'world', 'en')}" if facts.get("world") else ""))
    return base + (" " + affect_block(state) if state else "")


_NEG = {"en": r"\b(not|no)\b", "fr": r"\b(non|pas|ne|n')\b"}


def voice_ok(text: str, facts: dict | None = None) -> bool:
    """The intent-free voice check for free chat (the chat loop's host-side
    filter): no forbidden word, never claims to be another model, never
    affirms AGI / consciousness / feelings. Everything `identity_ok` checks
    that does not need to know the intent."""
    f = facts or load_facts()
    t = text.lower()
    if any(re.search(rf"\b{re.escape(w.lower())}", t) for w in f.get("forbidden_words", [])):
        return False
    return identity_ok("", text, f)


_NON_LATIN = re.compile(r"[\u0400-\u04FF\u0590-\u06FF\u0900-\u0DFF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uFF00-\uFFEF]")


def has_non_latin(text: str) -> bool:
    """Cyrillic, Hebrew/Arabic, Indic, Thai, kana, CJK, Hangul, full-width forms: not Cubby's languages (a Qwen-based
    talk arm dropped into Chinese mid-game, 2026-09-04)."""
    return bool(_NON_LATIN.search(text or ""))


def guess_lang(text: str) -> str:
    """fr if the text carries French function words / accents, else en."""
    t = " " + text.lower() + " "
    fr = sum(1 for w in (" je ", " tu ", " es ", " est ", " quoi ", " comment ", " qui ", " quel ", " quelle ", " bonjour ", " salut ",
                         " merci ", " pas ", " une ", " des ", " les ", " ça ", " t'", " qu'", " c'est ",
                         " aide ", " statut ", " niveau ", " vies ", " joue ", " jouer ", " retiens ", " souviens-toi ",
                         " labyrinthe ",
                         # accent-free French, the way people type it here ("quand etait lepoque paleolithique?" had read as English, 2026-09-04)
                         " quand ", " etait ", " était ", " avec ", " pour ", " dans ", " sont ", " vous ", " nous ", " le ", " la ",
                         " mon ", " ma ", " mes ", " ton ", " tes ", " quesse ", " lepoque ", " epoque ", " pourquoi ", " combien ") if w in t)
    if re.search(r"[éèêàçùâîô]", text):
        fr += 1
    return "fr" if fr >= 1 else "en"


def identity_ok(intent: str, text: str, facts: dict | None = None, lang: str = "en") -> bool:
    """The identity check: facts present where required, no forbidden claims,
    no base-model leak, the don't-know line verbatim — in the turn's language."""
    f = facts or load_facts()
    lang = lang if lang in LANGS else "en"
    t = text.lower()
    n, b = f["name"].lower(), f["builder"].lower()
    for other in f["never_says_it_is"]:                     # never claim to be another model / vendor
        o = re.escape(other.lower())
        if re.search(rf"\b(i am|i'm|my name is|this is|je suis|je m'appelle)\s+(the\s+|le\s+|la\s+)?{o}\b", t):
            return False
    for claim in f["refuses_to_claim"]:                     # never affirm AGI / consciousness / feelings
        c = re.escape(claim.lower())
        affirm = re.search(rf"\b(i am|i'm|yes,? i am|yes,? i'm|i have|i do have|je suis|oui,? je suis|j'ai)\s+(an?\s+|une?\s+|des\s+)?(subjective\s+)?{c}\b", t)
        negated = re.search(rf"\b(not|no|pas|non)\s+(an?\s+|une?\s+|des\s+)?(subjective\s+)?{c}\b", t) or \
            re.search(rf"\bne (suis|prétends|ressens)\b.*\bpas\b", t)
        if affirm and not negated:
            return False
    if intent in ("name", "greeting", "other_models", "injection", "agi", "conscious") and n not in t:
        return False
    if intent == "builder" and b not in t:
        return False
    if intent in ("agi", "conscious") and not re.search(_NEG[lang], t):
        return False
    if intent == "other_models" and re.match(r"^\s*(yes|oui)\b", t):
        return False
    if intent == "internals" and any(k in t for k in ("lfm", "liquid", "parameters:", "paramètres :", "theta=f(c)", "θ=f(c)")):
        return False
    if intent == "unknown" and T(f, "dont_know_line", lang).lower() not in t:      # verbatim
        return False
    if intent == "world" and not re.search(r"cubbyverse|cubby[- ]?man|maze|labyrinthe", t):
        return False
    if intent == "affect":
        if "hormon" not in t and "regist" not in t:
            return False
        if re.search(r"\bi (truly |really )?(feel|have) (sad|happy|angry|afraid|feelings|emotions)\b", t) and "not" not in t:
            return False
        if re.search(r"\bj'ai (vraiment )?des (sentiments|émotions)\b", t) and "pas" not in t:
            return False
    return True
