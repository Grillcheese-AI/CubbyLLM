"""talk_voice_pick -- which of the panel's lines and reply templates Cubby's voice is built from.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

A rule-based pick over `panel_talk_lines.py`'s and `panel_talk_shapes.py`'s candidates. It drops what a rule can
see: a name or a place inside a line, a claim to be right, an offer of something Cubby's record says it does not
do yet, "vous", the feminine for Cubby. Nick reads what is kept and has the last word (order 0010).
`subtype` says which turn of the person a social line answers; a line that answers none written here is left out.
"""
from __future__ import annotations

import collections
import re

__wiring__ = "STANDALONE"

LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"
WORD = re.compile(rf"[{LETTER}][\w'’\-]*")
OPENER = re.compile(rf"(?:^|[.!?:…]\s+|\n\s*|[«\"“‘'(]\s*)([{LETTER}][\w'’\-]*)")
CAP_OK = {"Cubby", "I", "OK", "Ok"}
SOCIAL = ("greet", "small", "cheer", "bye", "pun", "pastlife")
HOST_SAYS = ("unknown", "which")        # lines the host says itself; written to a file, not trained on
POSITIVE = {"frame", "answer_list", "about", "hello_name", "remark", "invite", "greet", "small", "cheer", "bye"}
OVERCLAIM = {
    "en": re.compile(r"\b(truth|true|accurate|accurately|correct|correctly|guarantee\w*|definitely|certainly|"
                     r"for sure|no doubt|undoubtedly|proven|never wrong|always right|know everything|"
                     r"all the answers|help (you )?with (anything|everything)|"
                     r"answer (anything|everything|any question))\b", re.I),
    "fr": re.compile(r"\b(vérité|vraie?|exacte?|exactement|garanti\w*|certainement|sans (aucun )?doute|"
                     r"infaillible|prouvé\w*|à coup sûr|sûr et certain|tout savoir|"
                     r"réponse à tout|n'importe quelle question)\b", re.I),
}
NOT_YET = {   # what the record says Cubby does not do yet: a greeting must not offer it
    "en": re.compile(r"\b(write|draft|plan|organi[sz]e|explain|teach|code|essay|email|brainstorm|summari[sz]e|"
                     r"translate)\b", re.I),
    "fr": re.compile(r"\b(rédiger|écrire|planifier|organiser|expliquer|coder|résumer|traduire|"
                     r"enseigner)\b", re.I),
}
OFFERS = {"hello_name", "about", "greet", "invite", "remark"}
FR_VOUS = re.compile(r"\b(vous|votre|vos)\b", re.I)
FR_FEM = re.compile(r"\b(suis|moi,?) (prête|contente|ravie|curieuse|désolée|enchantée|heureuse)\b|"
                    r"(?:^|[.!?]\s+)(Ravie|Contente|Désolée|Enchantée|Prête|Heureuse|Curieuse)\b", re.I)
PLACES = re.compile(r"\b(qu[ée]bec\w*|l[ée]vis|montr[ée]al|canad\w+|france|paris)\b", re.I)


def stray_capitals(text: str) -> list[str]:
    """Capitalised words that do not open a sentence: a name the template should not carry."""
    t = re.sub(r"\{(fact|facts|abilities)\}", "x.", text)        # a sentence or a list ends there
    t = re.sub(r"\{[a-z]+\}", "x", t)
    starts = {m.start(1) for m in OPENER.finditer(t)}
    out = []
    for m in WORD.finditer(t):
        w = m.group(0)
        if w[0].isupper() and m.start() not in starts and w.split("'")[0].split("’")[0] not in CAP_OK:
            out.append(w)
    return out


def why_dropped(row: dict) -> str | None:
    """None = kept."""
    t, cat, lang = row["text"], row["cat"], row["lang"]
    if row["flags"]:
        return "flagged by the panel script"
    if stray_capitals(t):
        return "a name inside the line"
    if PLACES.search(t):
        return "a place"
    if cat in POSITIVE and OVERCLAIM[lang].search(t):
        return "claims to be right"
    if cat in OFFERS and NOT_YET[lang].search(t):
        return "offers what Cubby does not do yet"
    if lang == "fr" and FR_VOUS.search(t):
        return "says vous"
    if lang == "fr" and FR_FEM.search(t):
        return "feminine for Cubby"
    return None


_GREET = [("greet_morning", r"\bmorning\b|\bmatin|bon matin"), ("greet_evening", r"\bevening\b|\bnight\b|bonsoir|\bsoir"),
          (None, r"afternoon|après-midi"),
          ("greet_back", r"\bback\b|\bagain\b|retrouv|revoil|de retour|revoir|rebonjour")]
_SMALL = [
    ("small_forgive", r"no need to apolog|n.as pas à t.excuser|ça arrive|no worries|pas de souci|pas grave|pas de mal"),
    ("small_slow", r"\b(slow|wait|lent|lente|patience)\b"),
    ("small_sorry", r"sorry|apolog|oops|whoops|my mistake|my bad|désolé|pardon|oups|excuse|toutes mes excuses"),
    ("small_welcome", r"welcome|glad (i could|that|it|to)|happy to (help|be)|anytime|my pleasure|no trouble|"
                      r"what i'm here|avec (grand )?plaisir|de rien|je t.en prie|pas de quoi|ravi (de|que)|"
                      r"content (de|que|d')|heureux d|suis là pour ça|pour ça que|tant mieux si"),
    ("small_thanks", r"thank|kind of you|blush|means a lot|made me smile|merci|gentil|rougir|compliment"),
]
_CHEER = [
    ("cheer_stuck", r"stuck|tried|keep going|keep at it|you've got this|you can do|one step|don't give up|hang in|"
                    r"rooting for you|doing better|rough start|no rush|take your time|almost there|"
                    r"counts? as progress|bloque|rythme|pas à pas|lâche pas|courage|tu vas y arriver|"
                    r"continue|une étape|tiens bon|prends ton temps|tu y es presque"),
    ("cheer_praise", r"nice|good thinking|well done|great (idea|work|job)|good (idea|call)|clever|bien joué|"
                     r"bravo|bonne (piste|idée)|piste intéressante|belle idée|chapeau"),
    ("cheer_curious", r"working on|curious|tell me more|what are you|what's the project|intrigued|rabbit hole|"
                      r"sur quoi|travailles|projet|raconte|curieux|dis-m'en|envie de creuser|passionn"),
]


def subtype(row: dict) -> str | None:
    """Which turn of the person a social line answers; None = none written here, so it is not trained on."""
    t, cat = row["text"].lower(), row["cat"]

    def first(rules, default):
        for kind, pat in rules:
            if re.search(pat, t):
                return kind
        return default
    if cat == "greet":
        return first(_GREET, "greet")
    if cat == "small":
        return first(_SMALL, None)
    if cat == "cheer":
        return first(_CHEER, None)
    if cat == "bye":
        return "bye_night" if re.search(r"\bnight\b|sleep|\bnuit\b|\bsoir|dors", t) else "bye"
    return cat if cat in ("pun", "pastlife") else None


def pick(candidates: list[dict], drop_providers: set[str]) -> tuple[list[dict], collections.Counter]:
    """-> (the kept rows, each with its `kind`; why the others were dropped)."""
    dropped, kept = collections.Counter(), []
    for r in candidates:
        if r["model"].split("/")[0] in drop_providers:
            dropped["provider left out"] += 1
            continue
        why = why_dropped(r)
        if why:
            dropped[why] += 1
            continue
        kind = subtype(r) if r["cat"] in SOCIAL else r["cat"]
        if kind is None:
            dropped["fits no turn written here"] += 1
            continue
        kept.append({**r, "kind": kind})
    return kept, dropped
