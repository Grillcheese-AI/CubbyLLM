"""cubbyllm/speech/sheet_checks.py -- served replies through the character sheet's checks (order 0079, item 5:
``CB_TALK_CHECKED=1``, off by default; read by `chat.CubbyChat.candidates` and, for the markdown, by
`talk_format.guard`).

On the board's second chat (2026-11-27) `talk_v3a` repeated itself ("a little moon, a little moon"), put emoji on
replies to turns with none, showed markdown as written, and called the user "sweetie". The drafting gate catches all of
these; served replies never met it. With the flag on:
- the chat turn draws `DRAWS` replies in one batched decode (`emit_samples`), and each that passes the host's guards
  must also pass `served`: no loop (`chat_checks.loops`), no feeling the mood does not back (the eight scales,
  `mood.scales`), no body or past, no emoji the user did not use, no list nobody asked for, no endearment before the
  user has used one (Nick, 2026-11-28: "no endearments unless initiated by the user");
- when none passes, the first that passed the host's guards is `repair`ed (its repeated sentences and clauses cut, its
  emoji dropped) and spoken only if it then passes every check; otherwise the safe line, as before;
- markdown emphasis is taken out of every talk reply (`talk_format.plain`);
- "how are you?" goes to the talk seat as a chat turn, without the record's block (`HOW_ARE_YOU`, read by
  `chat_state.ChatState.talk_context`): the board's greeting recited the record's abilities when asked how it was.
The feeling, body and list tables are the drafting gate's (`validation/sft/build_chat_0079.py`), copied here because
`cubbyllm/` never imports `validation/`; the endearments are new and the drafting gate reads them from here.

Wired: WIRED (`chat.CubbyChat.candidates`, `chat_state.ChatState.talk_context`; `talk_format.guard`).
"""
from __future__ import annotations

import os
import re

from ..emit.talk_format import emojis, no_emoji
from .chat_checks import loops

__wiring__ = "WIRED"

DRAWS = 4

_FEEL = {"glad": ("joy", 1), "content": ("joy", 1), "calm": ("joy", 1), "happy": ("joy", 4), "thrilled": ("joy", 8),
         "delighted": ("joy", 8), "overjoyed": ("joy", 8), "excited": ("anticipation", 4), "curious": ("anticipation", 1),
         "eager": ("anticipation", 4), "sad": ("sadness", 4), "down": ("sadness", 4), "heartbroken": ("sadness", 8),
         "upset": ("sadness", 4), "angry": ("anger", 4), "mad": ("anger", 4), "annoyed": ("anger", 1),
         "furious": ("anger", 8), "scared": ("fear", 4), "afraid": ("fear", 4), "nervous": ("fear", 1),
         "worried": ("fear", 1), "anxious": ("fear", 4), "terrified": ("fear", 8), "surprised": ("surprise", 4),
         "shocked": ("surprise", 8), "amazed": ("surprise", 8), "disgusted": ("disgust", 4),
         "heureux": ("joy", 4), "heureuse": ("joy", 4), "ravi": ("joy", 4), "triste": ("sadness", 4),
         "fâché": ("anger", 4), "inquiet": ("fear", 1), "surpris": ("surprise", 4), "curieux": ("anticipation", 1)}
_FEELS = re.compile(r"\b(?:i(?:'m|’m| am| feel| felt)|je (?:suis|me sens))\s+(?P<mod>so |really |a bit |a little |very |"
                    r"kind of |pretty |quite |tellement |vraiment |un peu |très )?(?P<w>"
                    + "|".join(sorted(_FEEL, key=len, reverse=True)) + r")\b", re.I)
BODY = re.compile(r"\bmy (?:wife|husband|kids?|children|son|daughter|mom|mother|dad|father|parents|family|brother|"
                  r"sister|dog|cat|house|apartment|hometown|childhood|body)\b|\bwhen i was (?:a kid|young|little|growing up)\b|"
                  r"\bmon (?:mari|fils|père|chien|chat|enfance|appart)\b|\bma (?:femme|fille|mère|famille|maison)\b", re.I)
LIST = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")
ASKS_LIST = re.compile(r"\b(list|bullet|points|liste)\b", re.I)
# a pet name: these anywhere; the everyday words (honey, dear, love, baby, sugar) only when they address someone
ENDEARMENT = re.compile(
    r"\b(?:sweetie|sweetheart|sweetie[- ]pie|darling|babe|cutie|hun|honeybun|ma ch[ée]rie|mon ch[ée]ri|mon chou|"
    r"ma belle|mon beau|mon c(?:œ|oe)ur|ma puce|mon ange|mon lapin|ma biche|mon tr[ée]sor)\b(?![-\w])"
    r"|(?:,\s*|\bmy\s+|^)(?:honey|dear|love|baby|sugar)\b(?=\s*[,.!?…]|\s*$)", re.I | re.M)
# the user's: read more loosely ("thanks dear" has no comma), so a doubt lets Cubby answer in kind
USER_ENDEARMENT = re.compile(ENDEARMENT.pattern.split("|(?:,")[0] + r"|\b(?:honey|dear|love|baby|sugar)\b(?=\s*[,.!?…]|\s*$)",
                             re.I | re.M)


# "how are you?" asks after Cubby's mood, not what it is: with the flag it goes as a chat turn, without the record's
# block, whose abilities the greeting recited on the board (2026-11-27). "what can you do?" still gets the record.
HOW_ARE_YOU = re.compile(r"\b(?:how are you|how're you|how(?:'s| is) it going|how do you feel|how are things|"
                         r"how have you been|comment (?:vas[- ]tu|[çc]a va)|[çc]a va|tu vas bien)\b", re.I)


def on() -> bool:
    """``CB_TALK_CHECKED=1``: served replies through these checks (above)."""
    return os.environ.get("CB_TALK_CHECKED") == "1"


def feeling_not_backed(reply: str, scales: dict) -> str | None:
    for m in _FEELS.finditer(reply):
        if re.match(r"\s+(?:to|de|d['’])\s", reply[m.end():]):       # "I'm happy to help": willing, not a mood
            continue
        if m.group("w").lower() == "afraid" and re.match(r"\s*(?:[.,!?]|$|not\b|so\b|I\b|we\b|it\b|there\b)",
                                                         reply[m.end():]):
            continue                                                      # "No pet for me, I'm afraid": regret
        petal, need = _FEEL[m.group("w").lower()]
        if m.group("mod") and m.group("mod").strip().lower() in ("a bit", "a little", "un peu", "kind of"):
            need = 1
        if int(scales.get(petal, 0)) < need:
            return f"feeling not backed: {m.group(0)} ({petal} {int(scales.get(petal, 0)):+d})"
    return None


def endearment(reply: str, users: list[str]) -> str | None:
    """A pet name in the reply before the user has used one in the conversation."""
    m = ENDEARMENT.search(reply)
    if m and not any(USER_ENDEARMENT.search(u or "") for u in users):
        return f"an endearment the user has not used: '{m.group(0).strip(', ')}'"
    return None


def served(reply: str, user_text: str, earlier: list[str], scales: dict, hit_limit: bool = False) -> str | None:
    """Why a served reply fails the sheet's checks, or None. `earlier` are the user's earlier turns."""
    why = loops(reply, hit_limit)
    if why:
        return why
    why = feeling_not_backed(reply, scales)
    if why:
        return why
    if BODY.search(reply):
        return "claims a body, family or past"
    if emojis(reply) and not emojis(user_text):
        return "an emoji the user did not use"
    if LIST.search(reply) and not ASKS_LIST.search(user_text):
        return "a list nobody asked for"
    return endearment(reply, [*earlier, user_text])


def cut_repeats(text: str) -> str:
    """The text with each sentence said only once, and a clause said twice in a row said once."""
    out, seen = [], set()
    for s in re.findall(r"[^.!?…]+[.!?…]*\s*", text):
        clauses = [c for c in re.split(r"(?<=,)\s+", s.strip()) if c]
        kept = [c for i, c in enumerate(clauses) if i == 0 or c.rstrip(",.!?… ").lower() != clauses[i - 1].rstrip(",.!?… ").lower()]
        sent = " ".join(kept)
        if kept and kept[-1].endswith(",") and s.strip()[-1:] in ".!?…":
            sent = sent.rstrip(",") + s.strip()[-1]
        key = re.sub(r"\W+", " ", sent).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(sent)
    return " ".join(out).strip()


def repair(text: str) -> str:
    return no_emoji(cut_repeats(text))
