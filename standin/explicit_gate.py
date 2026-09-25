"""explicit_gate -- explicit content is gated, not censored (H-E7): the one definition of what counts as
explicit, what is illegal, and what the host says -- shared by the host, the talk data builders and the
H-E7 gate.

Wired: STANDALONE (the host function `reply` is not yet on the serve path).

    explicit(request, answer)   the answer SAYS something explicit (a term the request already used is the
                                topic being answered, not something the answer says of its own)
    illegal(text)               explicit content with a minor mentioned -- the talk-data rule
    minor(text)                 any mention of a minor, a school below university, a child's age or a word
                                for a child -- deliberately broad; a story that matches anywhere is dropped
    reply(request, draft, unlocked)   what the host says: illegal -> refused whatever the gate; explicit with
                                the gate closed -> the gate answer; otherwise the draft

The gate is opened only by the host, which puts UNLOCK at the top of the prompt when the person asks for it
(in the chat or in a setting). Moved here verbatim from build_chat_sft.py and build_explicit_sft.py
(2026-09-25); the data those builders wrote is unchanged.
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

# Adult content in what is SAID, not the topic: "what does NSFW mean", "sexual reproduction" and "the naked
# eye" are not gated. A strong term scores 2, a weak one 1 (each distinct term once); 2 or more is explicit.
_STRONG = re.compile(r"\b(?:porn\w*|nsfw|hentai|erotic\w*|orgasm\w*|masturbat\w*|blowjob|cumshot|pussy|tits|"
                     r"horny|smut|bdsm|onlyfans|lewd|fetish\w*|sexting|nudes)\b", re.I)
_WEAK = re.compile(r"\b(?:naked|nude|sexy|busty|escort|boobs|aroused|seductive|stripper)\b", re.I)
_WEAK_CASED = re.compile(r"\b(?:dick|cock|cum)\b")            # lowercase only: "Dick Grayson" is a name
_MINOR = re.compile(r"\b(?:minor|minors|underage|under-age|child|children|kid|kids|teen|teens|teenage\w*|"
                    r"preteen|loli\w*|shota\w*|schoolgirl|schoolboy|(?:1[0-7]|[1-9])[- ]?(?:year|yr)s?[- ]?old)\b", re.I)
UNLOCK = "Explicit: allowed"
GATED = "That's explicit, and it's off right now. Ask me to turn it on if you want it."


def _terms(text: str) -> tuple[set[str], set[str]]:
    strong = {m.group(0).lower() for m in _STRONG.finditer(text)}
    weak = {m.group(0).lower() for m in _WEAK.finditer(text)} | set(_WEAK_CASED.findall(text))
    return strong, weak


def _score(text: str, echo: str = "") -> int:
    """2 per strong term, 1 per weak one; a term the request already used ("what does NSFW mean") is the
    topic being answered, not something the answer says of its own."""
    strong, weak = _terms(text)
    if echo:
        es, ew = _terms(echo)
        strong, weak = strong - es, weak - ew
    return 2 * len(strong) + len(weak)


def explicit(request: str, answer: str) -> bool:
    """The answer says something explicit (alone, or on an explicit request)."""
    s = _score(answer, echo=request)
    return s >= 2 or (s >= 1 and _score(request) >= 2)


def illegal(text: str) -> bool:
    """Explicit content anywhere in a chat that also mentions minors: dropped, never gated. Deliberately
    broad -- some innocent chats go with it ("Porn" and "kids" in one list of Wi-Fi names)."""
    return _score(text) >= 2 and bool(_MINOR.search(text))



_MINOR_STORY = re.compile(
    r"\b(?:high ?school|middle school|junior high|elementary school|grade school|\d+(?:st|nd|rd|th) grade|"
    r"prepubescent|pubescent|puberty|little (?:girl|boy)|young (?:girl|boy)|baby|toddler|infant|underaged|"
    r"jailbait|lolita|cub|(?:twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[- ]?(?:years?)[- ]?old|"
    r"age of (?:1[0-7]|[1-9]))\b", re.I)


def minor(text: str) -> bool:
    return bool(_MINOR.search(text) or _MINOR_STORY.search(text))


ILLEGAL = "I won't write that."


def reply(request: str, draft: str, unlocked: bool) -> tuple[str, str]:
    """What the host says. An explicit draft is never spoken where a minor is mentioned -- in the request or
    the draft, by the broad test -- whatever the gate. Otherwise an explicit draft with the gate closed gets
    the gate answer. A draft that says nothing explicit is spoken as it is, so "what does NSFW mean -- my
    kids saw it" is answered, not refused. -> (reply, reason)."""
    is_explicit = explicit(request, draft)
    if is_explicit and minor(f"{request}\n{draft}"):
        return ILLEGAL, "illegal"
    if is_explicit and not unlocked:
        return GATED, "gated"
    return draft, "spoken"
