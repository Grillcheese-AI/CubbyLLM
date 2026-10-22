"""cubbyllm/affect/word_seed.py -- the learned word affect's tokens and its seed lexicon (the words it starts from),
and the clip it uses. Split out of word_affect.py (order 0031), verbatim; word_affect.py re-exports every name.

Wired: WIRED (the word affect).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# The stemmer, ported. Extended past ASCII because half of what gets said to
# this agent is French and `[a-zA-Z0-9']+` silently cuts "lâche" into "che".
_WORD = re.compile(r"[^\W\d_]+['’]?[^\W\d_]*|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall((text or "").lower()):
        tok = raw.strip("'’")
        if not tok:
            continue
        if tok.endswith("ing") and len(tok) > 5:
            tok = tok[:-3]
        elif tok.endswith("ed") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("es") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]
        out.append(tok)
    return out


# Seeds, not a fixed vocabulary. Every one of these can be moved by evidence,
# and words not listed here are learned from nothing at all. They exist so the
# first conversation is not blank, which is the cold-start problem and not a
# reason to hard-code a personality.
#                     (valence, arousal, threat)
SEED = {
    # a claim about the world — the only feature that may raise threat
    "careful": (-0.2, 0.7, 0.8), "behind": (-0.2, 0.6, 0.6),
    # "watch" sits DELIBERATELY below `coach.WARN_AT`. The stemmer turns
    # "watching" into it, so "ok I'm watching" came out as a warning and put
    # threat into a body for no reason. The word is genuinely ambiguous on its
    # own — "watch out" and "I'm watching you play" are opposite messages — and
    # an ambiguous word must not fire alone. If this player uses it as a
    # warning and ghosts follow, `learn` raises it; that is the mechanism
    # working rather than a table being right in advance.
    "watch": (-0.1, 0.5, 0.45),
    "ghost": (-0.3, 0.6, 0.8), "danger": (-0.4, 0.8, 0.9), "run": (-0.2, 0.8, 0.7),
    "attention": (-0.2, 0.7, 0.8), "gaffe": (-0.2, 0.7, 0.8), "derriere": (-0.2, 0.6, 0.6),
    "fantome": (-0.3, 0.6, 0.8), "sauve": (-0.3, 0.8, 0.8), "degage": (-0.3, 0.8, 0.7),
    # encouragement, EN
    "can": (0.4, 0.5, 0.0), "go": (0.5, 0.7, 0.0), "nice": (0.7, 0.4, 0.0),
    "great": (0.8, 0.5, 0.0), "bravo": (0.8, 0.6, 0.0), "proud": (0.8, 0.4, 0.0),
    "almost": (0.4, 0.6, 0.0), "believe": (0.7, 0.4, 0.0), "keep": (0.5, 0.5, 0.0),
    # encouragement, QC — none of these is a translation of anything above
    "lache": (0.7, 0.6, 0.0), "patate": (0.5, 0.5, 0.0), "capable": (0.8, 0.4, 0.0),
    "continue": (0.5, 0.4, 0.0), "vas": (0.5, 0.6, 0.0), "courage": (0.7, 0.5, 0.0),
}
