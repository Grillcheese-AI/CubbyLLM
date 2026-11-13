"""Order 0067: the host's guard on a rule the reader wrote (Cowork, 2026-11-13: "build and measure it, no model call").

Round 6's IFBench read showed the plan reader writing the nearest measurement for a constraint it has no measurement
for, with numbers and words that are not in the instruction ("sentence count >= 22" for a keyword's position). The
guard asks three things of a written plan, each against the instruction's own text; a plan that fails one is
``asked`` (kept as text, never checked):

1. **numbers:** every count the plan compares (other than 0 and 1, which "no", "a", "any" say without a digit) is in
   the instruction, as digits or as a number word (English, French);
2. **words:** every argument the plan carries (a word, a letter, a phrase, a marker) is in the instruction (a letter
   or a word as a whole token; a longer phrase as written; the quotation mark as its name);
3. **the unit:** the measurement's unit is named in the instruction (`UNITS`, a small lexicon per measurement,
   English and French: sentence / phrase, word / mot, paragraph / paragraphe, ...).

`guard(instruction, plan)` -> None (the plan may be checked) or {"part": 1-3, "why": ...}.

Wiring: STANDALONE. Nothing in the serve path calls it; `validation/exp_rule_guard.py` measures it. Wiring it waits
for its read (CB_RULES stays off).
"""
from __future__ import annotations

import re
import unicodedata

from cubbyllm.speech import rule_plan as rp

__wiring__ = "STANDALONE"

_EN = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
       "seventeen eighteen nineteen twenty").split()
_FR = ("zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize dix-sept "
       "dix-huit dix-neuf vingt").split()
TENS = {30: ("thirty", "trente"), 40: ("forty", "quarante"), 50: ("fifty", "cinquante"), 60: ("sixty", "soixante"),
        70: ("seventy", "soixante-dix"), 80: ("eighty", "quatre-vingts"), 90: ("ninety", "quatre-vingt-dix"),
        100: ("hundred", "cent"), 1000: ("thousand", "mille")}
NUM_WORDS: dict[int, set[str]] = {i: {_EN[i], _FR[i]} for i in range(21)}
NUM_WORDS[1] |= {"une", "a single", "single"}
NUM_WORDS[2] |= {"twice", "a couple", "pair", "double", "paire", "deux fois", "both", "les deux"}
NUM_WORDS[3] |= {"thrice", "triple"}
NUM_WORDS[12] |= {"dozen", "douzaine"}
for _n, _ws in TENS.items():
    NUM_WORDS.setdefault(_n, set()).update(_ws)

_RESP = ("response", "answer", "reponse", "******", "version", "reply", "replies", "fois", "variant", "alternative")
UNITS: dict[str, tuple[str, ...]] = {                    # a stem per entry, matched at a word's start (accents folded)
    "word count": ("word", "mot"),
    "sentence count": ("sentence", "phrase"),
    "paragraph count": ("paragraph", "paragraphe", "***"),
    "empty paragraph count": ("paragraph", "paragraphe", "***"),
    "section count": ("section", "part", "partie", "chapter", "chapitre"),
    "highlight count": ("highlight", "surlign", "mettre en evidence", "mis en evidence", "met en evidence", "*", "italic",
                        "italique", "asterisk", "asterisque", "emphas", "soulign", "accent"),
    "title count": ("title", "titre", "<<", "headline", "heading", "intitule", "chevron", "angle bracket"),
    "json only": ("json",),
    "case": ("capital", "upper", "lower", "majuscule", "minuscule", "caps", "case", "casse", "letter", "lettre"),
    "language": ("language", "langue", "english", "anglais", "french", "francais", "spanish", "espagnol", "german",
                 "allemand", "italian", "italien", "portuguese", "portugais", "chinese", "chinois", "japanese", "japonais",
                 "russian", "russe", "arabic", "arabe", "hindi", "korean", "coreen", "dutch", "neerlandais"),
    "count of the word": ("word", "mot", "keyword", "term", "terme", "mention", "use", "utilise", "emploi", "include",
                          "inclu", "say", "dis", "appear", "apparai", "time", "fois", "show", "figure", "glisse", "place",
                          "case", "compte", "count", "contain", "contien", "feature", "insert", "insere", "integr",
                          "presen", "drop", "work", "fit"),
    "count of the letter": ("letter", "lettre", "character", "caractere"),
    "count of the character": ("character", "caractere", "symbol", "symbole", "sign", "signe", "mark", "marque"),
    "starts with": ("start", "begin", "open", "commence", "debut", "ouvr", "repeat", "repet", "reprend", "first",
                    "premier", "quot", "guillemet", '"', "cit", "recop", "reproduc", "echo", "verbatim", "word for word",
                    "mot pour mot", "avant de repondre", "before answering", "before you answer", "enclose", "wrap",
                    "entour", "encadr", "lead", "kick off", "lance", "request", "demande", "redis", "reecri"),
    "ends with": ("end", "finish", "close", "termin", "fini", "fin", "last", "dernier", "quot", "guillemet", '"', "cit",
                  "enclose", "wrap", "entour", "encadr"),
    "opening": ("start", "begin", "open", "commence", "debut", "ouvr", "first", "premier", "lead", "kick off", "lance",
                "before anything", "avant tout", "d'abord", "entame", "attaque", "demarr"),
    "response count": _RESP, "empty response count": _RESP, "distinct response count": _RESP,
    "markdown": ("markdown", "format", "plain", "brut", "bold", "gras", "bullet", "puce", "header", "en-tete", "simple",
                 "markup", "balis", "mise en forme", "styling", "decor", "raw"),
}
_ARG_NAMES = {'"': ("quot", "guillemet", '"'), "*": ("*", "asterisk", "asterisque")}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c))


def _has_number(low: str, n: int) -> bool:
    if re.search(rf"(?<![\d.,]){n}(?![\d])", low):
        return True
    return any(re.search(rf"(?<![\w-]){re.escape(_fold(w))}(?![\w-])", low) for w in NUM_WORDS.get(n, ()))


def _has_arg(low: str, arg: str) -> bool:
    a = _fold(arg).strip()
    if a in _ARG_NAMES:
        return any(x in low for x in _ARG_NAMES[a])
    if not a:
        return False
    if len(a.split()) == 1 and re.match(r"^\w+$", a):
        return re.search(rf"(?<!\w){re.escape(a)}(?!\w)", low) is not None
    return a in low


def _has_unit(low: str, m: str) -> bool:
    for stem in UNITS.get(m, ()):
        if not stem[0].isalnum():
            if stem in low:
                return True
        elif re.search(rf"(?<!\w){re.escape(stem)}", low):
            return True
    return False


def parts(instruction: str, plan: dict | None) -> list[dict]:
    """Every part the plan fails, [{"part": 1-3, "why": ...}] (empty: it may be checked)."""
    if not plan:
        return []
    low, out = _fold(instruction), []
    leaves = rp.leaves(plan)
    for lf in leaves:
        if rp._is_num(lf["m"]) and isinstance(lf.get("v"), int) and lf["v"] > 1 and not _has_number(low, lf["v"]):
            out.append({"part": 1, "why": f"{lf['v']} is not in the instruction ({lf['m']})"})
    for lf in leaves:
        if lf.get("arg") and not _has_arg(low, lf["arg"]):
            out.append({"part": 2, "why": f"{lf['arg']!r} is not in the instruction ({lf['m']})"})
    for lf in leaves:
        if not _has_unit(low, lf["m"]):
            out.append({"part": 3, "why": f"the instruction names no unit of {lf['m']!r}"})
    return out


def guard(instruction: str, plan: dict | None) -> dict | None:
    """None when the plan may be checked; else the first part it fails, {"part": 1-3, "why": ...}."""
    got = parts(instruction, plan)
    return got[0] if got else None


__all__ = ["NUM_WORDS", "UNITS", "guard", "parts"]
