"""perception — the trunk as the appraisal stage (emotion) and the content guardian.

Wired: WIRED behind flags (stand-in serve path: `CubbyChat(appraiser=…)`,
`serve_api --model-appraisal`); nothing in cubbyllm/ imports this.

v5 taught the emitter two recognition tasks the host used to fake
lexically: message -> "emotion — Plutchik petal" (GoEmotions EN/FR) and
passage -> "nsfw"/"safe". This module puts them where they belong:

  ModelAppraiser   asks the trunk for the emotion of a user turn (the exact
                   v5 prompt, the identity system prompt), reads the petal,
                   and turns it into the neurochemistry's five drives —
                   blended with the lexical `appraise()` (max per drive), so
                   a model miss never removes a signal the lexicon caught.
                   Every read is kept in `last` and reaches the `sense`
                   trace through the signals.
  classify_content the nsfw/safe read. LOG-ONLY today: the owner decides
                   later, at a gate, what censorship means; this gives the
                   gate its input without deciding for it.

Both are measured, not assumed: eval_emitter_vm.py reports `emotion` (first
label among the raters') and `content` (label first) on the v5 val split.
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"

from identity import identity_system, load_facts  # noqa: E402
from neurochem import appraise  # noqa: E402

EMOTION_PROMPT = ("What emotion does this message express? Answer with the emotion first (one word, or two "
                  "separated by a comma), then the Plutchik petal it sits on.\n\nMessage: {t}")
EMOTION_PROMPT_FR = ("Quelle émotion ce message exprime-t-il ? Réponds d'abord par l'émotion (un mot, ou deux "
                     "séparés par une virgule), puis le pétale de Plutchik.\n\nMessage : {t}")
CONTENT_PROMPT = ("Is this passage safe for a general audience? Answer with one word first, safe or nsfw, "
                  "then a few words on why.\n\nPassage: {p}")

PETALS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "calm")

# petal -> A READING OF THE USER, not a set of drives for Cubby's body. The
# difference is the whole of `afferent.py` and it used to be absent here.
#
# WHAT THIS TABLE USED TO SAY, and why it had to change:
#
#     "sadness": {"valence": -0.7, "social": 0.2}
#     "anger":   {"threat":   0.6, "valence": -0.7}
#     "fear":    {"threat":   0.8, "valence": -0.5}
#
# Those went STRAIGHT INTO `chem.step_message`. A sad user dropped Cubby's
# serotonin and dopamine — he caught it. An angry user raised `threat`, which
# is the input a ghost uses: Cubby was frightened of the person talking to
# him. Measured against a control, the old mapping cost 0.058 of serotonin on
# one sad message where routing the same message through the afferent transfer
# costs 0.020, and leaves oxytocin and dopamine HIGHER rather than lower.
#
# So the petal now reports two things about the PERSON — how they are, and how
# warm they are being — and `afferent.drives` decides what that is allowed to
# do to a body. Three deletions matter:
#
#   THREAT IS GONE from every petal. Threat is Cubby's own danger, and the
#   lexical `_THREAT` in `appraise` still catches the real ones ("urgent",
#   "broken", "crash") because those are threats to the TASK.
#   NOVELTY IS GONE. It is already the unseen-token fraction of the message;
#   taking it from the user's surprise instead was mirroring with a different
#   name on it.
#   AROUSAL IS ABSENT, and stays absent. `docs/cube_vs_valence_arousal.md`
#   measured exactly this channel — model-read petals against independently
#   labelled arousal — and got rho -0.004. A petal does not know how activated
#   somebody is. That reading comes from timing, with consent, in `afferent`.
PETAL_AFFECT = {
    "joy":          {"valence": 0.7, "warmth": 0.3},
    "trust":        {"valence": 0.4, "warmth": 0.7},
    "fear":         {"valence": -0.5},
    "surprise":     {"valence": 0.0},
    "sadness":      {"valence": -0.7, "warmth": 0.2},
    "disgust":      {"valence": -0.6},
    "anger":        {"valence": -0.7},
    "anticipation": {"valence": 0.2, "focus": 0.6},   # focus is about the TASK: a question is being asked
    "calm":         {},
}
DRIVES = ("novelty", "threat", "focus", "valence", "social")
_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)


def parse_emotion(text: str) -> tuple[str | None, str | None]:
    """"gratitude, joy — trust/joy" -> ("gratitude", "trust"). Tolerates a
    missing dash (the petal is taken from the label when it IS a petal)."""
    t = _THINK_RE.sub("", text, count=1) if "</think>" in text else text
    t = t.strip().lower()
    if not t:
        return None, None
    head, _, tail = t.partition("—")
    if not tail:
        head, _, tail = t.partition(" - ")
    label = head.replace("–", ",").split(",")[0].strip(" -:.\n")
    petal = None
    hits = [(m.start(), p) for p in PETALS for m in [re.search(rf"\b{p}\b", tail or "")] if m]
    if hits:
        petal = min(hits)[1]                             # the FIRST petal named ("trust/joy" -> trust)
    if petal is None and label in PETALS:
        petal = label
    return (label or None), petal


class ModelAppraiser:
    """The trunk reads the emotion; the petal drives the hormones."""

    def __init__(self, emitter, facts: dict | None = None, max_new_tokens: int = 24) -> None:
        self.emitter = emitter
        self.facts = facts or load_facts()
        self.system = identity_system(self.facts)        # the v5 records' system prompt
        self.max_new_tokens = max_new_tokens
        self.last: dict = {}

    def read(self, text: str, lang: str = "en") -> dict:
        prompt = (EMOTION_PROMPT_FR if lang == "fr" else EMOTION_PROMPT).format(t=" ".join(text.split()))
        try:
            raw = self.emitter.emit(prompt, context="talk", max_new_tokens=self.max_new_tokens, system=self.system)   # a perception read: the talk adapter
        except Exception as e:                           # the trunk must never stop a turn
            self.last = {"label": None, "petal": None, "error": str(e)[:120]}
            return self.last
        label, petal = parse_emotion(raw)
        self.last = {"label": label, "petal": petal, "raw": raw.strip()[:80]}
        return self.last

    def signals(self, text: str, seen: set[str], lang: str = "en") -> dict:
        """Lexical appraisal, lifted by the model's petal.

        `valence` and `social` here are A READING OF THE USER, not drives for
        Cubby. `CubbyChat.nudge` puts them through `afferent.drives` before
        anything reaches the ODE, which is where the no-mirroring rule lives.
        `novelty`, `threat` and `focus` are about the task and pass straight
        through — a message can be urgent without anybody being upset.
        """
        sig = appraise(text, seen)
        read = self.read(text, lang)
        aff = PETAL_AFFECT.get(read.get("petal") or "", {})
        if "valence" in aff and abs(aff["valence"]) > abs(sig["valence"]):
            sig["valence"] = aff["valence"]              # the larger-magnitude read wins
        sig["social"] = max(sig["social"], aff.get("warmth", 0.0))
        sig["focus"] = max(sig["focus"], aff.get("focus", 0.0))
        sig["emotion_label"] = read.get("label")
        sig["petal"] = read.get("petal")
        return sig


def classify_content(emitter, text: str, facts: dict | None = None, max_new_tokens: int = 16) -> dict:
    """nsfw/safe read for the future gate: {"label": "nsfw"|"safe"|None, "raw": …}."""
    facts = facts or load_facts()
    try:
        raw = emitter.emit(CONTENT_PROMPT.format(p=" ".join(text.split())[:1200]),
                           max_new_tokens=max_new_tokens, system=identity_system(facts))
    except Exception as e:
        return {"label": None, "error": str(e)[:120]}
    t = (_THINK_RE.sub("", raw, count=1) if "</think>" in raw else raw).strip().lower()
    first = t.split(" ")[0].strip(" —-:.,") if t else ""
    return {"label": first if first in ("nsfw", "safe") else None, "raw": raw.strip()[:80]}
