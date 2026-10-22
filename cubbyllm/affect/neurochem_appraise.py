"""cubbyllm/affect/neurochem_appraise.py -- a turn's text appraised for the chemistry (threat, social, positive,
negative, focus), and the two curves the chemistry uses (a sigmoid, a clip). Split out of neurochem.py (order 0031),
verbatim; neurochem.py re-exports every name.

Wired: WIRED (the neurochemistry).
"""
from __future__ import annotations

import math
import re

__wiring__ = "WIRED"


def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
    """Receptor saturation curve. Prevents linear blow-up at extremes."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── the host's lexical appraisal (EN/FR) ────────────────────────────────────
_THREAT = re.compile(r"\b(urgent|emergency|danger|help|asap|hurry|now|broken|crash|"
                     r"urgence|danger|aide|vite|casse|panne|au secours)\b", re.I)


_SOCIAL = re.compile(r"\b(thanks|thank you|please|hi|hello|hey|welcome|friend|"
                     r"merci|s'il te pla[iî]t|bonjour|salut|coucou|bienvenue|ami)\b", re.I)


_POS = re.compile(r"\b(great|good|love|cool|awesome|nice|perfect|fun|happy|thanks?|thank you|"
                  r"super|g[ée]nial|bravo|parfait|chouette|content|heureux|merci)\b|:\)|<3", re.I)


_NEG = re.compile(r"\b(bad|wrong|hate|angry|terrible|awful|stupid|sad|"
                  r"nul|mauvais|faux|d[ée]teste|f[âa]ch[ée]|triste|horrible)\b|:\(", re.I)


_FOCUS = re.compile(r"\?|\b(how|what|why|when|where|which|explain|calculate|solve|"
                    r"comment|quoi|pourquoi|quand|o[ùu]|quel|explique|calcule|r[ée]sous)\b", re.I)


def _hits(rx: re.Pattern, text: str, per: float, cap: float) -> float:
    return min(cap, per * len(rx.findall(text)))


def appraise(text: str, seen: set[str]) -> dict:
    """One message -> the ODE's five drive inputs. Lexical and inspectable:
    novelty is the unseen-token fraction against the session vocabulary
    (halved, so first contact reads as new without drowning the rest);
    shouting (caps, !!) reads as threat/urgency alongside the lexicons."""
    tokens = re.findall(r"[\w']+", text.lower())
    unseen = sum(1 for t in tokens if t not in seen) / max(1, len(tokens))
    shouting = 0.4 if (text.isupper() and len(text) > 3) else 0.0
    bangs = min(0.3, 0.15 * text.count("!!"))
    valence = _hits(_POS, text, 0.4, 1.0) - _hits(_NEG, text, 0.4, 1.0)
    return {"novelty": round(0.5 * unseen, 3),
            "threat": round(min(1.0, _hits(_THREAT, text, 0.35, 0.8) + shouting + bangs), 3),
            "focus": round(_hits(_FOCUS, text, 0.3, 0.8), 3),
            "valence": round(_clip(valence, -1, 1), 3),
            "social": round(_hits(_SOCIAL, text, 0.4, 1.0), 3)}
