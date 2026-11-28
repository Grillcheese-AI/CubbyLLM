"""cubbyllm/reasoning/opener_view.py -- a reply as the name-and-number guard should read it (order 0010, change a).

`grounded_prose` takes every capitalised word outside the facts for a name. English capitalises any word that opens a
sentence or a line, and the pronoun I, so a framed reply ("Here is what I have: ...") was refused over "Here". The
opener view lowers only those forced capitals: a word from `opener_words.txt` (the chat corpus's lowercase-dominant
words, moved here from the chat builder, which reads the same list) or an interjection, when it opens a sentence or a
line, and the pronoun I; list enumerators go. A name keeps its capital wherever it stands, so every name the guard
exists to catch is still caught. The value check and the claim check read the draft as it is.

Flag: `CB_OPENER_VIEW=1` makes `ask_claims.talk_reply` read drafts through this view; off (the default), the host reads
the raw draft, bit-identical to before. `validation/sft/build_chat_sft.chat_view` is this function with the builder's
own word set.

Wired: WIRED (ask_claims.talk_reply under the flag; the chat and voice builders through build_chat_sft.chat_view).
"""
from __future__ import annotations

import functools
import os
import pathlib
import re

__wiring__ = "WIRED"

WORDS = pathlib.Path(__file__).with_name("opener_words.txt")
# 0010 addendum 2, point 3: openers the chat corpus's list lacks (the voice rows' manifest counts them)
INTERJECTIONS = frozenset("hmm ooh aha oups hello hey bonjour coucou j'espère j’espère qu'est-ce qu’est-ce".split()) \
    | frozenset("yes yep yeah nope nah ok oui ouais honnêtement".split()) \
    | frozenset("oh ah aw aww awww wow whoa yay ugh ha haha hehe oof phew yikes gosh omg alright okay um uh "
                "ouf bof super génial bravo merci désolé ravi oh là".split()) \
    | frozenset("congratulations félicitations côté carrément ouah hi sacré sacrée chouette dingue why ouch aïe ouille "
                "gush relire chanter hocher".split())
# 0010 / 0079 (Cowork, 2026-11-20 and 21):
# "Yes! I can show you my steps", Nick's approved "Nope! I'm Cubby" and "Oh no, that sounds worrying" were refused
# as names: an interjection opening a sentence is a forced capital, never a name

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][\w'’\-]*|\d[\d.,:/\-]*")        # ask._TOKEN
_ENUM = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")
# an emoji ends a sentence too ("... a fun class 😄 Did you keep up?"; 0079's batch, 2026-11-28)
_EMO = "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF][\uFE0F\U0001F3FB-\U0001F3FF]?"
_OPENER = re.compile(r"(?:^\s*(?:" + _EMO + r"\s*)*|(?:[.!?:…]|" + _EMO + r")\s+|\n\s*)([A-Za-zÀ-ÿ][\w'’\-]*)")
_PRONOUN_I = {"i", "i'm", "i’m", "i'd", "i’d", "i'll", "i’ll", "i've", "i’ve"}
# a French verb joined to its pronoun (an imperative or an inversion): "Installe-toi", "Dis-moi", "Allez-y", "Peut-on".
# Opening a sentence it is a forced capital, never a name (Cowork, 2026-11-11: "Bonjour ! Installe-toi, ..." was refused)
_FR_VERB = re.compile(r"^[a-zà-ÿ]+-(?:toi|vous|moi|nous|lui|leur|le|la|les|y|en|t-il|t-elle|t-on|il|elle|on|je|tu)$")


def on() -> bool:
    return os.environ.get("CB_OPENER_VIEW", "0") == "1"


@functools.lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """The host's opener words: the list beside this file and the interjections."""
    return frozenset(WORDS.read_text(encoding="utf-8").split()) | INTERJECTIONS


def openers(text: str) -> set[int]:
    """Positions where English capitalises any word: sentence and line starts."""
    return {m.start(1) for m in _OPENER.finditer(text)}


def _inflected(low: str, common) -> bool:
    """An inflection of a common word: -ing ("Picturing", "Hand-transcribing"), a plural ("Hedges"), a French
    imperative or second plural in -ez whose infinitive is common ("Essayez"; "Notez", "Profitez" by their -er stem)."""
    stem = low[:-3] if low.endswith("ing") and len(low) > 5 else ""
    if stem and (stem in common or stem + "e" in common or stem[:-1] in common):
        return True
    if low.endswith("es") and len(low) > 4 and (low[:-1] in common or low[:-2] in common):
        return True
    if low.endswith("s") and len(low) > 3 and low[:-1] in common:
        return True
    return low.endswith("ez") and len(low) > 4 and (low[:-2] + "er" in common or low[:-2] in common or low[:-2] + "e" in common)


def opener_view(answer: str, common: set[str] | frozenset[str] | None = None) -> str:
    """The answer with list enumerators removed and the capitals English forces lowered (a common word opening a
    sentence or line, a French verb joined to its pronoun there, the pronoun I). `common` defaults to the host's words."""
    common = common_words() if common is None else common
    text = _ENUM.sub("\n", answer)
    starts = openers(text)
    out = list(text)
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        low = tok.lower().strip("'’-.,:")
        if low.endswith(("'s", "’s")):
            low = low[:-2]
        joined = "-" in low and all(p in common or _inflected(p, common) for p in low.split("-") if p)   # "Front-row"
        joined = joined or _inflected(low, common)       # "Picturing", "Hedges", "Notez": a common word, inflected
        if tok[0].isupper() and (low in _PRONOUN_I or (m.start() in starts and (low in common or joined
                                                                                 or _FR_VERB.match(low)))):
            out[m.start()] = tok[0].lower()
    return "".join(out)


def guard_text(draft: str) -> str:
    """What the guard reads: the opener view with `CB_OPENER_VIEW=1`, else the draft unchanged."""
    return opener_view(draft) if on() else draft
