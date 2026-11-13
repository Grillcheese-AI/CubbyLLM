"""cubbyllm/speech/reply_world.py -- the reply world (order 0067, step 2): a reply in, statements about it out.

A rule an operator or a user sets ("in two sentences", "never use the word X", "réponds sans chiffres") is a program
over these statements, written by the program seat and run by the VM (step 3). No kind of instruction has code of
its own: a kind of instruction is a plan over these relations, and a new measurement is a new relation here (a
helper in `RELATIONS`, with its own tests).

Each relation is a helper: the reply's text -> its values (strings: a count is said in digits; a relation may have
several values, one statement each, e.g. every word the reply contains). The statements read like any fact the
worlds hand a run: ``"<value> is the <relation> of the reply"``.

The starter relations are the measurements behind `validation/exp_if_follow.py`'s ten checks (the same splits of
words, sentences and list items, and the same stop-word reading of the language, so a rule over them agrees with
that measure), and M-IFEval's French ones (accented letters, the oe ligature, the cedilla, digits, the form of
address). Restoring accents needs a lexicon and is not a measurement here: a rule asking for it stays "asked".
Then the measurements the seen kinds of 0067's line-3 split need (paragraphs between ``***``, responses between
``******``, highlighted spans, ``<<titles>>``), written before any held-out kind was read for a helper.

**Relations with an argument** (`PARAMETRIC`): "count of the word" X, "count of the letter" X, "count of the
character" X, "starts with" X, "ends with" X, "section count" X (sections opened by X and a number). A plan's leaf
names the argument (``{"m": "count of the word", "arg": "river", "op": ">=", "v": 2}``); the statement says it:
``"2 is the count of the word "river" of the reply"``.

Wired: STANDALONE (0067's rule plan and turn read it once they are built; until then its tests).
"""
from __future__ import annotations

import json
import re

__wiring__ = "STANDALONE"

SUBJ = "the reply"
OPENING = 5                                              # "first N words" for N up to this
STOP = {"en": {"the", "and", "is", "of", "to", "in", "that", "it", "with", "for", "you", "are", "this", "was", "not"},
        "fr": {"le", "la", "les", "et", "est", "de", "des", "en", "que", "une", "un", "pour", "dans", "pas", "vous", "nous"},
        "es": {"el", "la", "los", "las", "y", "es", "de", "en", "que", "una", "un", "para", "con", "por", "no", "se"}}
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_MARKDOWN = re.compile(r"(^|\n)\s*#|\*\*|`|\[[^\]]+\]\([^)]+\)|(^|\n)\s*[-*]\s")
_TU = {"tu", "toi", "te", "ton", "ta", "tes", "t'"}
_VOUS = {"vous", "votre", "vos"}


def words_of(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text or "", re.U)


def sentences_of(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]


def _language(text: str) -> list[str]:
    """The language whose commonest words the reply uses most, strictly more than any other's; else "none"."""
    ws = [w.lower() for w in words_of(text)]
    counts = {lang: sum(1 for w in ws if w in stop) for lang, stop in STOP.items()}
    best = max(counts, key=counts.get)
    rest = max(v for lang, v in counts.items() if lang != best)
    return [best if counts[best] > 0 and counts[best] > rest else "none"]


def _case(text: str) -> list[str]:
    if not any(c.isalpha() for c in text or ""):
        return ["none"]
    return ["lower" if text == text.lower() else "upper" if text == text.upper() else "mixed"]


def _json(text: str) -> tuple[bool, list]:
    body = (text or "").strip()
    m = re.search(r"\{.*\}", body, re.S)
    try:
        obj = json.loads(m.group(0) if m else body)
    except ValueError:
        return False, []
    return isinstance(obj, dict), sorted(obj) if isinstance(obj, dict) else []


def _opening(text: str) -> list[str]:
    ws = [w.lower() for w in words_of(text)]
    return [" ".join(ws[:n]) for n in range(1, min(OPENING, len(ws)) + 1)]


def _address(text: str) -> list[str]:
    ws = {w.lower().replace("’", "'") for w in words_of(text)}
    tu, vous = bool(ws & _TU), bool(ws & _VOUS)
    return ["both" if tu and vous else "tu" if tu else "vous" if vous else "none"]


RELATIONS = {                                            # relation -> helper (text -> its values)
    "word count": lambda t: [str(len(words_of(t)))],
    "sentence count": lambda t: [str(len(sentences_of(t)))],
    "line count": lambda t: [str(len([ln for ln in (t or "").splitlines() if ln.strip()]))],
    "list item count": lambda t: [str(sum(1 for ln in (t or "").splitlines() if _ITEM.match(ln)))],
    "language": _language,
    "case": _case,
    "word": lambda t: sorted({w.lower() for w in words_of(t)}),        # each word it contains, once
    "opening": _opening,                                 # its first 1 .. OPENING words, lowercased
    "last word": lambda t: [w.lower() for w in words_of(t)[-1:]] or ["none"],
    "json": lambda t: ["yes" if _json(t)[0] else "no"],
    "json key": lambda t: _json(t)[1],
    "markdown": lambda t: ["yes" if _MARKDOWN.search(t or "") else "no"],
    "accented letter count": lambda t: [str(sum(1 for c in t or "" if c.lower() in "àâäéèêëîïôöùûüÿ"))],
    "oe ligature count": lambda t: [str(sum(1 for c in t or "" if c in "œŒ"))],
    "cedilla count": lambda t: [str(sum(1 for c in t or "" if c in "çÇ"))],
    "digit count": lambda t: [str(sum(1 for c in t or "" if c.isdigit()))],
    "form of address": _address,
}


def _split_count(text: str, pattern: str) -> tuple[int, int]:
    """(parts with text, empty parts between two others) of `text` split on `pattern`."""
    parts = re.split(pattern, text or "")
    full = sum(1 for p in parts if p.strip())
    empty_inside = sum(1 for i, p in enumerate(parts) if not p.strip() and 0 < i < len(parts) - 1)
    return full, empty_inside


def _highlights(text: str) -> int:
    one = [h for h in re.findall(r"\*[^\n\*]*\*", text or "") if h.strip("*").strip()]
    two = [h for h in re.findall(r"\*\*[^\n\*]*\*\*", text or "") if h[2:-2].strip()]
    return len(one) + len(two)


def _json_only(text: str) -> bool:
    """The whole reply parses as JSON once a code fence around it is taken off (``json`` above: it contains an object)."""
    body = (text or "").strip()
    for fence in ("```json", "```Json", "```JSON", "```"):
        if body.startswith(fence):
            body = body[len(fence):]
            break
    body = body[:-3] if body.endswith("```") else body
    try:
        json.loads(body.strip())
    except ValueError:
        return False
    return True


def _responses(text: str) -> list[str]:
    return [p.strip() for p in (text or "").split("******") if p.strip()]


RELATIONS.update({
    "paragraph count": lambda t: [str(_split_count(t, r"\s?\*\*\*\s?")[0])],          # paragraphs between ***
    "empty paragraph count": lambda t: [str(_split_count(t, r"\s?\*\*\*\s?")[1])],
    "response count": lambda t: [str(len(_responses(t)))],                              # responses between ******
    "empty response count": lambda t: [str(_split_count(t, r"\*\*\*\*\*\*")[1])],
    "distinct response count": lambda t: [str(len(set(_responses(t))))],
    "highlight count": lambda t: [str(_highlights(t))],                                 # *spans* and **spans**
    "json only": lambda t: ["yes" if _json_only(t) else "no"],                          # the whole reply, fences off
    "title count": lambda t:[str(sum(1 for x in re.findall(r"<<[^\n]+>>", t or "") if x.strip("<>").strip()))],
})

PARAMETRIC = {                                           # relation -> helper (text, argument -> its values)
    "count of the word": lambda t, a: [str(len(re.findall(r"\b" + re.escape(a) + r"\b", t or "", re.I)))],
    "count of the letter": lambda t, a: [str((t or "").lower().count(a.lower()))],
    "count of the character": lambda t, a: [str((t or "").count(a))],
    "starts with": lambda t, a: ["yes" if (t or "").strip().lower().startswith(a.strip().lower()) else "no"],
    "ends with": lambda t, a: ["yes" if (t or "").strip().lower().endswith(a.strip().lower()) else "no"],
    "section count": lambda t, a: [str(len(re.split(r"\s?" + re.escape(a) + r"\s?\d+\s?", t or "")) - 1)],
}


def measure_arg(text: str, relation: str, arg: str) -> list[tuple[str, str, str]]:
    """A relation with an argument, measured: [(subject, 'relation "arg"', value)]."""
    return [(SUBJ, f'{relation} "{arg}"', v) for v in PARAMETRIC[relation](text or "", str(arg))]


def measure(text: str, relations=None) -> list[tuple[str, str, str]]:
    """The reply's statements as (subject, relation, value), in `RELATIONS`' order (`relations`: only these)."""
    out = []
    for rel, fn in RELATIONS.items():
        if relations is None or rel in relations:
            out += [(SUBJ, rel, v) for v in fn(text or "")]
    return out


def statements(text: str, relations=None) -> list[str]:
    """The reply's statements, said as the worlds say a fact: "<value> is the <relation> of the reply"."""
    return [f"{v} is the {r} of {s}" for s, r, v in measure(text, relations)]


__all__ = ["PARAMETRIC", "RELATIONS", "SUBJ", "measure", "measure_arg", "sentences_of", "statements", "words_of"]
