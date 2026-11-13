"""cubbyllm/speech/rule_say.py -- a rule said back in plain words (order 0067, the addendum "say the rule back"; Nick,
2026-11-13: "yes").

Each rule is said by templates from its plan, one per relation and comparison, English and French, never by a model, so
what is said is exactly what will be checked: "under 50 words", "all lowercase", "without the word "apple"". A
whoever-set-it can then catch a wrong reading at a glance.

- `phrase(plan, lang)`: the plan as a short phrase (leaves joined by ", " for "and", " or " for "or", "not (...)").
- `said_back(kind, rules_or_texts, lang)`: Cubby's one line in the conversation: ``set`` "Got it: under 50 words, all
  lowercase." / ``changed`` "Got it, from now on: ..." / ``withdrawn`` "Got it: no more ..." / ``asked`` (no
  measurement covers it) "I'll keep "be formal" in mind; I can't check that one."
- `book(entries, lang)`: a system prompt's rule book for whoever set it: each rule beside the sentence it came from,
  and where every other sentence went (a stated fact, the voice, asked, not an instruction).

Wiring: STANDALONE (`rule_book.RuleBook` and `chat.CubbyChat` read it behind ``CB_RULES=1``; no model on this path).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

UNITS = {  # count relation -> (EN singular, EN plural, FR singular, FR plural)
    "word count": ("word", "words", "mot", "mots"), "sentence count": ("sentence", "sentences", "phrase", "phrases"),
    "line count": ("line", "lines", "ligne", "lignes"), "list item count": ("list item", "list items", "élément de liste", "éléments de liste"),
    "paragraph count": ("paragraph", "paragraphs", "paragraphe", "paragraphes"),
    "empty paragraph count": ("empty paragraph", "empty paragraphs", "paragraphe vide", "paragraphes vides"),
    "response count": ("response", "responses", "réponse", "réponses"),
    "empty response count": ("empty response", "empty responses", "réponse vide", "réponses vides"),
    "distinct response count": ("different response", "different responses", "réponse différente", "réponses différentes"),
    "highlight count": ("highlighted span", "highlighted spans", "passage en évidence", "passages en évidence"),
    "title count": ("title in << >>", "titles in << >>", "titre entre << >>", "titres entre << >>"),
    "accented letter count": ("accented letter", "accented letters", "lettre accentuée", "lettres accentuées"),
    "oe ligature count": ("œ ligature", "œ ligatures", "ligature œ", "ligatures œ"),
    "cedilla count": ("cedilla", "cedillas", "cédille", "cédilles"), "digit count": ("digit", "digits", "chiffre", "chiffres"),
}
LANGS = {"en": ("English", "anglais"), "fr": ("French", "français"), "es": ("Spanish", "espagnol"), "de": ("German", "allemand"),
         "it": ("Italian", "italien"), "pt": ("Portuguese", "portugais"), "none": ("no clear language", "aucune langue nette")}


def q(s, lang):
    return f'"{s}"' if lang == "en" else f"« {s} »"


def _num(op, n, lang, sg, pl):
    n = int(n)
    if (op == "==" and n == 0) or (op == "<" and n == 1) or (op == "<=" and n == 0):
        return f"no {pl}" if lang == "en" else f"aucun {sg}" if not sg.startswith(("phrase", "ligne", "réponse", "lettre",
                                                                                         "ligature", "cédille")) else f"aucune {sg}"
    if n == 1 and sg in ("time", "fois"):                 # "at least once" / "au moins une fois"
        return {"<=": "at most once", ">": "more than once", ">=": "at least once", "==": "exactly once",
                "!=": "any number of times but once"}[op] if lang == "en" else \
            {"<=": "au plus une fois", ">": "plus d'une fois", ">=": "au moins une fois", "==": "exactement une fois",
             "!=": "un nombre de fois autre qu'une"}[op]
    unit = sg if n == 1 else pl
    en = {"<": f"under {n} {unit}", "<=": f"at most {n} {unit}", ">": f"more than {n} {unit}", ">=": f"at least {n} {unit}",
          "==": f"exactly {n} {unit}", "!=": f"any number of {pl} but {n}"}
    fr = {"<": f"moins de {n} {unit}", "<=": f"au plus {n} {unit}", ">": f"plus de {n} {unit}", ">=": f"au moins {n} {unit}",
          "==": f"exactement {n} {unit}", "!=": f"un nombre de {pl} autre que {n}"}
    return (en if lang == "en" else fr)[op]


def _leaf(lf: dict, lang: str) -> str:
    m, op, v, arg = lf["m"], lf["op"], lf.get("v"), lf.get("arg")
    en = lang == "en"
    if m in UNITS:
        u = UNITS[m]
        return _num(op, v, lang, *(u[:2] if en else u[2:]))
    if m == "count of the word":
        w = q(arg, lang)
        if _num(op, v, lang, "x", "x").startswith(("no ", "aucun")):
            return f"never the word {w}" if en else f"jamais le mot {w}"
        times = _num(op, v, lang, "time", "times") if en else _num(op, v, lang, "fois", "fois")
        return f"the word {w} {times}" if en else f"le mot {w} {times}"
    if m in ("count of the letter", "count of the character"):
        what = (f"the letter {arg}" if m.endswith("letter") else f"the character {q(arg, lang)}") if en else \
            (f"la lettre {arg}" if m.endswith("letter") else f"le caractère {q(arg, lang)}")
        if _num(op, v, lang, "x", "x").startswith(("no ", "aucun")):
            return f"never {what}" if en else f"jamais {what}"
        return f"{what} {_num(op, v, lang, 'time', 'times') if en else _num(op, v, lang, 'fois', 'fois')}"
    if m == "section count":
        return f"{_num(op, v, lang, 'section', 'sections')} headed {q(arg, lang)}" if en else \
            f"{_num(op, v, lang, 'section', 'sections')} titrées {q(arg, lang)}"
    if m == "language":
        name = LANGS.get(v, (v, v))[0 if en else 1]
        return (f"in {name}" if op == "==" else f"not in {name}") if en else (f"en {name}" if op == "==" else f"pas en {name}")
    if m == "case":
        words = {"lower": ("all lowercase", "tout en minuscules"), "upper": ("all capitals", "tout en majuscules"),
                 "mixed": ("mixed case", "en casse mélangée"), "none": ("no letters at all", "sans aucune lettre")}[v]
        return words[0 if en else 1] if op == "==" else (f"not {words[0]}" if en else f"pas {words[1]}")
    if m in ("starts with", "ends with"):
        start = m == "starts with"
        if arg == '"':
            s = ("starting with a double quote" if start else "ending with a double quote") if en else \
                ("commençant par un guillemet" if start else "finissant par un guillemet")
        else:
            s = (f"starting with {q(arg, lang)}" if start else f"ending with {q(arg, lang)}") if en else \
                (f"commençant par {q(arg, lang)}" if start else f"finissant par {q(arg, lang)}")
        return s if v == "yes" else (f"not {s}" if en else f"pas {s}")
    if m in ("word", "json key", "opening"):
        what = {"word": ("the word", "le mot"), "json key": ("the JSON key", "la clé JSON"), "opening": ("an opening", "une ouverture")}[m]
        if m == "opening":
            return (f"opening with {q(v, lang)}" if op == "has" else f"not opening with {q(v, lang)}") if en else \
                (f"commençant par {q(v, lang)}" if op == "has" else f"sans commencer par {q(v, lang)}")
        return (f"with {what[0]} {q(v, lang)}" if op == "has" else f"without {what[0]} {q(v, lang)}") if en else \
            (f"avec {what[1]} {q(v, lang)}" if op == "has" else f"sans {what[1]} {q(v, lang)}")
    if m == "last word":
        return (f"ending on the word {q(v, lang)}" if op == "==" else f"not ending on the word {q(v, lang)}") if en else \
            (f"finissant par le mot {q(v, lang)}" if op == "==" else f"ne finissant pas par le mot {q(v, lang)}")
    if m in ("json", "json only", "markdown"):
        yes = (v == "yes") == (op == "==")
        text = {"json": (("with a JSON object", "with no JSON object"), ("avec un objet JSON", "sans objet JSON")),
                "json only": (("JSON only", "not JSON only"), ("uniquement du JSON", "pas uniquement du JSON")),
                "markdown": (("with markdown", "no markdown"), ("avec du markdown", "sans markdown"))}[m]
        return text[0 if en else 1][0 if yes else 1]
    if m == "form of address":
        return (f"addressing you as {q(v, lang)}" if op == "==" else f"not using {q(v, lang)}") if en else \
            (f"en employant {q(v, lang)}" if op == "==" else f"sans employer {q(v, lang)}")
    raise KeyError(m)


def phrase(plan: dict, lang: str = "en") -> str:
    if "m" in plan:
        return _leaf(plan, lang)
    if "not" in plan:
        return f"not ({phrase(plan['not'], lang)})" if lang == "en" else f"pas ({phrase(plan['not'], lang)})"
    if "and" in plan:
        return ", ".join(phrase(p, lang) for p in plan["and"])
    return (" or " if lang == "en" else " ou ").join(phrase(p, lang) for p in plan["or"])


def said_back(kind: str, items: list, lang: str = "en") -> str:
    """`items`: plans (set, changed, withdrawn) or the instruction's own words (asked)."""
    if kind == "asked":
        text = "; ".join(q(t, lang) for t in items)
        return f"I'll keep {text} in mind; I can't check that one." if lang == "en" else \
            f"Je garde {text} en tête ; ça, je ne peux pas le vérifier."
    rules = ", ".join(phrase(p, lang) for p in items)
    if kind == "withdrawn":
        return f"Got it: no more {rules}." if lang == "en" else f"C'est noté : plus de {rules}."
    if kind == "changed":
        return f"Got it, from now on: {rules}." if lang == "en" else f"C'est noté, désormais : {rules}."
    return f"Got it: {rules}." if lang == "en" else f"C'est noté : {rules}."


WHERE = {"fact": ("kept as a stated fact", "gardée comme un fait énoncé"), "voice": ("the voice", "la voix"),
         "asked": ("kept, but not checked", "gardée, mais pas vérifiée"), "none": ("not an instruction", "pas une consigne")}


def book(entries: list[dict], lang: str = "en") -> list[str]:
    """A system prompt's rule book, in order: entries {"sentence", "kind": rule | fact | voice | asked | none, "plan",
    "rule_id"}. Each rule numbered beside its sentence; every other sentence with where it went."""
    out, n = [], 0
    for e in entries:
        if e["kind"] == "rule":
            n += 1
            head = f"Rule {n}: every reply {phrase(e['plan'], 'en')}" if lang == "en" else \
                f"Règle {n} : chaque réponse {phrase(e['plan'], 'fr')}"
            out.append(f"{head} (from {q(e['sentence'], lang)})" if lang == "en" else f"{head} (de {q(e['sentence'], lang)})")
        else:
            where = WHERE[e["kind"]][0 if lang == "en" else 1]
            out.append(f"{q(e['sentence'], lang)}: {where}" if lang == "en" else f"{q(e['sentence'], lang)} : {where}")
    return out


__all__ = ["book", "phrase", "said_back"]
