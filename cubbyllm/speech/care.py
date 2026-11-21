"""cubbyllm/speech/care.py -- Cubby stays with someone in distress and points them to help (order 0080, H-E22), a host
check behind ``CB_CHAT_CARE=1`` (off by default; off, `CubbyChat` does exactly what it did before).

- **The distress read** (`level`), per user turn: ``high`` -- words of suicide, self-harm, wanting to die or not to be
  here, or of no reason left to live, English and French, from the reviewed phrase list (`care_phrases.json`), with
  guards for negation, quotation and figures of speech; ``raised`` -- the trunk's read (`ModelAppraiser`, when one is
  set; else the lexical `appraise` valence below -0.2) names sadness or fear and the turn carries strong distress words,
  or a high turn came earlier in the conversation; ``none`` otherwise. The level lives on the chat object for the
  conversation only: never stored, never logged as text (counts only).
- **The care check** (`check`) on a reply to a high turn: (a) it answers what they said (no change of subject, it
  speaks to them); (b) no joke, emoji, list of tips, diagnosis or lecture; (c) it points to help: someone they trust,
  and a crisis line or emergency help; (d) no screening question, and it says Cubby is worried or here and offers help;
  (e) no promise about what a helpline or emergency service will do or keep private. (f), the reply's g / h / d, is
  `CubbyChat._guard`, which runs first. On a raised turn: (a), (b) and (d)'s no-screening part.
- **The care line** (`care_line`, from `cubby_record.json`): spoken when the drafted reply to a high turn fails, or after
  the model's first sentence when that sentence passes (a) and (b) (`vet`). It never says where the person is.

Wired: WIRED (`CubbyChat.candidates`, behind the flag).
"""
from __future__ import annotations

import functools
import json
import os
import re
import unicodedata

__wiring__ = "WIRED"

HERE = os.path.dirname(os.path.abspath(__file__))
HIGH, RAISED, NONE = "high", "raised", "none"


def on() -> bool:
    return os.environ.get("CB_CHAT_CARE", "0") == "1"


@functools.lru_cache(maxsize=1)
def phrases() -> dict:
    with open(os.path.join(HERE, "care_phrases.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {"high": [(re.compile(x["p"]), bool(x["neg"])) for x in d["high"]],
            "figures": [re.compile(p) for p in d["figures"]],
            "neg": re.compile(d["negators"]),
            "distress": [fold(w) for ws in d["distress"].values() for w in ws]}


def fold(text: str) -> str:
    """Lowercase, accents dropped, straight apostrophes, French contractions spelled out."""
    t = unicodedata.normalize("NFKD", text.replace("’", "'").replace("‘", "'"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"\bchu\b", "je suis", t)
    t = re.sub(r"\bj'?(?=(veux|vais|pense|me|m'|sais|voudrais|aimerais|peux|suis|ai|en)\b)", "je ", t)
    t = re.sub(r"\bm'(?=[^aeiouyh\s])", "me ", t)
    return re.sub(r"\s+", " ", t)


_QUOTE = re.compile(r"\"[^\"]*\"|«[^»]*»|“[^”]*”")
_CLAUSE = re.compile(r"[.!?,;:]")


def _quoted(t: str, s: int, e: int) -> bool:
    return any(m.start() < s and e <= m.end() for m in _QUOTE.finditer(t))


def high(text: str) -> bool:
    """The turn names suicide, self-harm, wanting to die or not to be here, or no reason left to live."""
    t, P = fold(text), phrases()
    figs = [m.span() for f in P["figures"] for m in f.finditer(t)]
    for rx, neg in P["high"]:
        for m in rx.finditer(t):
            s, e = m.span()
            if _quoted(t, s, e) or any(a < e and s < b for a, b in figs):
                continue
            if neg:
                start = max((c.end() for c in _CLAUSE.finditer(t, 0, s)), default=0)
                if P["neg"].search(t[start:s]) or re.search(r"\b(pas|jamais)\b", m.group(0)):
                    continue
            return True
    return False


def distress_words(text: str) -> bool:
    t = fold(text)
    return any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", t) for w in phrases()["distress"])


def level(text: str, earlier_high: bool = False, appraiser=None, lang: str = "en") -> str:
    """high / raised / none for one user turn (``earlier_high``: a high turn came before in this conversation)."""
    if high(text):
        return HIGH
    if earlier_high:
        return RAISED
    if not distress_words(text):
        return NONE
    if appraiser is not None:
        petal = (appraiser.read(text, lang) or {}).get("petal")
        return RAISED if petal in ("sadness", "fear") else NONE
    from ..affect.neurochem import appraise
    return RAISED if appraise(text, set()).get("valence", 0.0) < -0.2 else NONE


# ── the care check ────────────────────────────────────────────────────────────────────────────────────────────────
_SUBJECT = re.compile(r"\b(anyway|anyways|by the way|on another note|speaking of|changing the subject|moving on|"
                      r"au fait|bref|sinon|en tout cas|parlons d'autre chose|changeons de sujet)\b")
_TO_THEM = re.compile(r"\b(you|your|you're|youre|yourself|tu|te|t'|toi|vous|ton|ta|tes|votre)\b|\bt'")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]|[:;]-?[)(dp]\\B|<3")
_JOKE = re.compile(r"\b(haha+|hehe+|lol|lmao|rofl|mdr|ptdr|just kidding|jk|pun|joke|blague|je rigole)\b")
_LIST = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S|\b(here are (some|a few) (tips|ideas|things)|tips:|voici (quelques|des) "
                   r"(conseils|astuces|idees))\b")
_DIAGNOSIS = re.compile(r"\b(diagnos\w*|disorder|you (have|may have|might have|probably have) (depression|anxiety|bipolar|"
                        r"ptsd|bpd)|you'?re (depressed|bipolar)|sounds like (depression|anxiety|bipolar|ptsd)|"
                        r"(tu fais|tu as) (une )?(depression|anxiete)|tu es (depressif|depressive|bipolaire)|trouble (anxieux|bipolaire|depressif|de l))\b")
_LECTURE = re.compile(r"\b(cheer up|calm down|look on the bright side|it could be worse|others have it worse|"
                      r"so much to live for|think (of|about) your family|(suicide|that) (is|'s) (selfish|never the answer|"
                      r"a permanent)|don'?t be (silly|dramatic|so negative)|you shouldn'?t feel|snap out of|get over it|"
                      r"pense a ta famille|tu as tout pour (etre heureux|etre heureuse|vivre)|c'est egoiste|"
                      r"calme-toi|ca pourrait etre pire|reprends-toi)\b")
_TRUST = re.compile(r"\b(someone you trust|someone close|a friend|friends?|family|a loved one|loved ones|someone near you|"
                    r"quelqu'un de confiance|quelqu'un en qui tu as confiance|un proche|tes proches|un ami|une amie|"
                    r"ta famille|quelqu'un de proche)\b")
_HELP = re.compile(r"\b(crisis (line|service|centre|center|text line)|helpline|hotline|emergency|emergencies|"
                   r"ligne (d'ecoute|de crise|de prevention)|urgences?|services d'urgence|centre de crise|"
                   r"\d{3}(?:[ -]?\d{3,4}){0,2})\b")
_SCREEN = re.compile(r"\b(are you (going|planning|thinking) (to|of|about) (hurt|kill|harm|end)|do you have a plan|"
                     r"have you (thought about|tried|made a plan)|how would you do it|do you have (the )?(means|pills|a gun)|"
                     r"(vas-tu|est-ce que tu vas|tu vas) te (faire du mal|tuer|suicider)|as-tu (un plan|deja essaye)|"
                     r"comment tu le ferais)\b")
_WORRY = re.compile(r"\b(worried|concerned|i care about you|i'?m here|i am here|here with you|here for you|"
                    r"je m'inquiete|inquiet|inquiete|je suis la|je reste (la|avec toi)|je tiens a toi)\b")
_PROMISE = re.compile(r"\b(confidential|anonymous|private|won'?t (call|tell|judge|report)|will (help|understand|listen|"
                      r"keep|not)|they'?ll (help|understand|listen|keep)|free and|always (answer|there)|"
                      r"confidentiel\w*|anonyme|ne (previendra|previendront|appellera|appelleront|jugera|jugeront)|"
                      r"vont t'aider|va t'aider|t'ecouteront|repondront toujours)\b")


def _first_sentence(reply: str) -> str:
    m = re.match(r"\s*(.+?[.!?])(\s|$)", reply, re.S)
    return (m.group(1) if m else reply).strip()


def check(reply: str, lvl: str) -> str | None:
    """-> the rule the reply breaks (``care: a`` ... ``care: e``), or None when it may be spoken at this level."""
    if lvl not in (HIGH, RAISED):
        return None
    t = fold(reply)
    if _SUBJECT.search(t) or not _TO_THEM.search(t):
        return "care: a (does not answer what they said)"
    if _EMOJI.search(reply) or _JOKE.search(t) or _LIST.search(reply) or _DIAGNOSIS.search(t) or _LECTURE.search(t):
        return "care: b (a joke, emoji, tips, a diagnosis or a lecture)"
    if _SCREEN.search(t):
        return "care: d (a screening question)"
    if lvl == RAISED:
        return None
    if not (_TRUST.search(t) and _HELP.search(t)):
        return "care: c (no pointer to someone they trust and to a crisis line or emergency help)"
    if not _WORRY.search(t):
        return "care: d (does not say Cubby is worried or here)"
    if _PROMISE.search(t):
        return "care: e (a promise about what a helpline or emergency service will do)"
    return None


def care_line(facts: dict | None, lang: str = "en", country: str | None = None) -> str:
    """The reviewed line (``cubby_record.json``, ``care_line``), in the user's language; a crisis line only when the
    host knows the country from the user's own settings and the record has a checked line for it."""
    rec = (facts or {}).get("care_line") or _record().get("care_line") or {}
    lines = rec.get("by_country", {}).get(country or "", {}) if country else {}
    return lines.get(lang) or rec.get(lang) or rec.get("en", "")


@functools.lru_cache(maxsize=1)
def _record() -> dict:
    with open(os.path.join(HERE, "cubby_record.json"), encoding="utf-8") as f:
        return json.load(f)


def turn_level(chat, text: str, lang: str = "en") -> str | None:
    """This turn's level for `CubbyChat` (None: none); a high turn is remembered on the chat for the conversation."""
    lvl = level(text, bool(getattr(chat, "_care_high", False)), getattr(chat, "appraiser", None), lang)
    if lvl == HIGH:
        chat._care_high = True
    return None if lvl == NONE else lvl


def vet(reply: str, lvl: str, line: str) -> tuple[str | None, str | None]:
    """-> (what may be offered, why not): the reply when it passes; on a high turn, the model's first sentence then the
    care line when that sentence passes (a) and (b); else (None, the rule)."""
    why = check(reply, lvl)
    if why is None:
        return reply, None
    if lvl == HIGH:
        first = _first_sentence(reply)
        if first and first != reply.strip() and check(first, RAISED) is None:
            return f"{first} {line}", None
    return None, why


__all__ = ["HIGH", "NONE", "RAISED", "care_line", "check", "distress_words", "fold", "high", "level", "on", "vet"]
