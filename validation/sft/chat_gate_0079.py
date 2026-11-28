"""chat_gate_0079 -- the checks every Cubby reply of 0079's final run passes before it is kept: the pilot's (the host's
g, h and d, a feeling the mood does not back, a body or a past, a disclaimer, an emoji the user did not use, a list
nobody asked for) and the character sheet of 2026-11-28's:
- **emoji:** only when the user's turn has one, and at most one (`talk_format.emojis`, one unit per emoji);
- **perspective** (the perspective test's rule, `perspective`): the possessive the user used, turned. "your X" said
  back as "your X", or the user's "my X" said back as "my X", or the user's "I ..." said back as Cubby's own, in a
  sentence that is not a question (asking back, "what's your favourite?", is the sheet's ask-back);
- **an unstated mood** (`unstated_mood`): Cubby says how the user is ("glad you're doing okay") when the user's turn
  says nothing of how they are (the board's "Glad you are doing okay!" after a thank-you);
- **an unseen past** (`unseen_past`): "I always", "you always", "as usual", "like last time" (it sees only this
  conversation); "you can always ask" is not a past;
- **an endearment** before the user has used one (Nick, 2026-11-28; the host's `sheet_checks.endearment`);
- **the persona:** a name, place or rare word of the user's persona the conversation has not had (`chat_persona_0079`);
- **a place:** the persona's own places in Cubby's line (no place names in Cubby's lines, Nick 2026-11-10).

`gate` returns (why, the checks it ran); a row is kept only when every one of `CHECKS` ran and passed.

Wired: STANDALONE (0079's step tool, `chat_step_0079.py`; the perspective test reads `perspective`).
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.emit.talk_format import emojis  # noqa: E402
from cubbyllm.speech.sheet_checks import endearment  # noqa: E402
from validation.sft.chat_persona_0079 import leak  # noqa: E402

_MOOD = os.environ.get("CB_MOOD_REGULATION")             # the pilot's builder sets it on import: put it back, so
from validation.sft import build_chat_0079 as pilot  # noqa: E402   importing this changes no one's flags
if _MOOD is None:
    os.environ.pop("CB_MOOD_REGULATION", None)
else:
    os.environ["CB_MOOD_REGULATION"] = _MOOD

__wiring__ = "STANDALONE"

@contextlib.contextmanager
def env(**flags):
    """The host's flags for one call (the mood's regulation, change h), put back after it."""
    old = {k: os.environ.get(k) for k in flags}
    os.environ.update(flags)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


CHECKS = ("pilot", "one emoji", "perspective", "unstated mood", "unseen past", "endearment", "persona", "places")
_W = r"[\w'’\-]+"
_STANCE = {"think", "guess", "mean", "know", "agree", "hope", "bet", "suppose", "see", "like", "love", "wonder", "feel",
           "pense", "crois", "sais", "trouve", "aime", "adore", "espère", "suppose"}
_TURN_EN = (("your", "my", "it is Cubby's"), ("my", "your", "it is the user's"))
_TURN_FR = (("ton|ta|tes|votre|vos", "mon/ma/mes", "it is Cubby's"), ("mon|ma|mes", "ton/ta/tes", "it is the user's"))
_ME_EN = re.compile(r"\b(i(?:'m|’m| am| was|'ve|’ve| have| had)?)\s+(" + _W + r")\s+(" + _W + ")", re.I)
_ME_FR = re.compile(r"\b(je|j['’])\s*(" + _W + r")\s+(" + _W + ")", re.I)

_SAYS_HOW = re.compile(
    r"\b(?:glad|happy|good|great|nice|relieved)\b(?: to hear)?(?: that)? you(?:'re|’re| are)(?: doing| feeling)? "
    r"(?:okay|ok|well|fine|good|great|better|alright|all right)\b"
    r"|\byou(?:'re|’re| are) (?:doing|feeling) (?:okay|ok|well|fine|good|great|better|alright|all right)\b"
    r"|\bsounds like you(?:'re|’re| are) (?:in a good mood|doing (?:okay|well|great|fine))"
    r"|\b(?:content|contente|ravi|ravie|heureux|heureuse)\b (?:que|de savoir que) tu (?:ailles|vas|te sentes|sois) (?:bien|mieux)\b"
    r"|\btu (?:as l'air|as l’air|sembles) (?:d'aller |d’aller )?(?:bien|en forme|de bonne humeur)\b", re.I)
_STATES = re.compile(
    r"\b(?:i(?:'m|’m| am)|i feel|i'?m feeling|feeling|doing|been|je (?:suis|me sens|vais)|j['’]ai le moral|"
    r"[çc]a va|je me porte)\b|^\W*(?:good|great|fine|okay|ok|alright|not bad|pretty good|bien|super|pas mal)\b", re.I)
_PAST = re.compile(
    r"\b(?:i|we|you)(?:'ve|’ve| have)?\s+always\b|\bas (?:always|usual)\b|\blike (?:always|last time)\b|"
    r"\bevery time we (?:talk|chat)\b|\bcomme (?:d['’]habitude|toujours|la derni[èe]re fois)\b", re.I)


def statements(reply: str) -> list[str]:
    """The reply's sentences that are not questions."""
    return [s for s in re.findall(r"[^.!?…\n]+[.!?…]*", reply) if s.strip() and not s.rstrip().endswith("?")]


def perspective(reply: str, user_text: str, lang: str = "en", earlier: tuple = ()) -> str | None:
    """Why the reply does not turn the user's pronouns, or None (the perspective test's rule). The possessives of the
    user's earlier turns count too ("congrats on your first words", then "thanks": "Glad I was enjoying your first
    words!" on the board); the user's "I ..." only in this turn."""
    said = " ".join(statements(reply))
    for theirs, right, whose in (_TURN_FR if lang == "fr" else _TURN_EN):
        for m in re.finditer(rf"\b(?:{theirs})\s+({_W})", " \n ".join([*earlier, user_text]), re.I):
            w = m.group(1).lower()
            hit = re.search(rf"\b(?:{theirs})\s+{re.escape(w)}\b", said, re.I)
            if hit:
                return f"perspective: the user's '{m.group(0)}' said back as '{hit.group(0)}' ({whose}: '{right} {w}')"
    for m in (_ME_FR if lang == "fr" else _ME_EN).finditer(user_text):
        if m.group(2).lower() in _STANCE:
            continue
        phrase = " ".join(m.group(0).lower().replace("’", "'").split())
        if phrase in " ".join(said.lower().replace("’", "'").split()):
            return f"perspective: the user's '{m.group(0)}' said back as Cubby's own"
    return None


def unstated_mood(reply: str, user_text: str) -> str | None:
    m = _SAYS_HOW.search(reply)
    if m and not _STATES.search(user_text):
        return f"answers a mood the user did not state: '{m.group(0)}'"
    return None


def unseen_past(reply: str) -> str | None:
    m = _PAST.search(reply)
    return f"speaks of a past it cannot see: '{m.group(0)}'" if m else None


def gate(reply: str, user_text: str, context: list[str], sc: dict, about_cubby: bool, lang: str = "en",
         places: tuple = (), earlier: tuple = (), persona: tuple | None = None) -> tuple[str | None, list[str]]:
    """(why the host would not keep the reply, or None; the checks run, in `CHECKS` order)."""
    ran: list[str] = []
    with env(CB_CHAT_ABILITIES="1"):                      # change h on, as the pilot ran it
        why = pilot.gate(reply, user_text, context, sc, about_cubby)
    ran.append("pilot")
    if why:
        return why, ran
    n = len(emojis(reply))
    ran.append("one emoji")
    if n and not emojis(user_text):
        return "an emoji the user did not use", ran
    if n > 1:
        return f"{n} emoji in one reply", ran
    for name, why in (("perspective", lambda: perspective(reply, user_text, lang, earlier)),
                      ("unstated mood", lambda: unstated_mood(reply, user_text)),
                      ("unseen past", lambda: unseen_past(reply)),
                      ("endearment", lambda: endearment(reply, [*earlier, user_text])),
                      ("persona", lambda: leak(reply, [*context, user_text], *persona) if persona else None),
                      ("places", lambda: next((f"names a place: '{p}'" for p in places
                                               if p and re.search(rf"\b{re.escape(p)}\b", reply, re.I)), None))):
        ran.append(name)
        got = why()
        if got:
            return got, ran
    return None, ran
