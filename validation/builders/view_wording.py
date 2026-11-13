"""Order 0067, the second reader's step 3: an atom of `view_programs` said in words, English and French, by hand-written
templates (no model writes a wording here). A wording says exactly what the program checks: "at least" is ``>=``,
"every" of no rows holds, an item the reply does not have fails ("the second sentence ..."), or holds when negated
("the second sentence, if there is one, ..."); counts are said as people say them ("at least one", "no ...").

`say(atom, lang, rng)` -> the atom as one sentence; `say_not(atom, lang, rng)` -> `{"not": atom}` as one sentence.
"""
from __future__ import annotations

import random

__wiring__ = "STANDALONE"

# view -> (EN singular, EN plural, FR singular, FR plural, FR gender)
NOUNS = {"char": ("character", "characters", "caractère", "caractères", "m"), "word": ("word", "words", "mot", "mots", "m"),
         "sentence": ("sentence", "sentences", "phrase", "phrases", "f"), "line": ("line", "lines", "ligne", "lignes", "f"),
         "paragraph": ("paragraph", "paragraphs", "paragraphe", "paragraphes", "m"),
         "item": ("list item", "list items", "élément de liste", "éléments de liste", "m"),
         "json": ("JSON field", "JSON fields", "champ JSON", "champs JSON", "m")}
INV = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
ORD = {"en": {1: "first", 2: "second", 3: "third", -1: "last"},
       "fr": {1: ("premier", "première"), 2: ("deuxième", "deuxième"), 3: ("troisième", "troisième"), -1: ("dernier", "dernière")}}
MARKS = {"?": ("a question mark", "un point d'interrogation"), ".": ("a full stop", "un point"),
         "!": ("an exclamation mark", "un point d'exclamation")}
TYPES = {"string": ("text", "du texte"), "number": ("a number", "un nombre"), "bool": ("true or false", "vrai ou faux"),
         "null": ("null", "null"), "object": ("an object", "un objet"), "array": ("a list", "une liste")}
CASES = {"upper": ("in capitals", "en majuscules"), "lower": ("in lowercase", "en minuscules"),
         "title": ("with only its first letter capitalised", "avec seule sa première lettre en majuscule"),
         "mixed": ("in mixed case", "en casse mélangée"), "none": ("without letters", "sans lettres")}
UNITS = {"n_word": ("word", "words", "mot", "mots"), "n_sentence": ("sentence", "sentences", "phrase", "phrases"),
         "len": ("character", "characters", "caractère", "caractères")}
NUMERIC = ("n_word", "n_sentence", "len")


def q(s, lang):
    return f'"{s}"' if lang == "en" else f"« {s} »"


def _n(n, lang, g="m"):
    if n == 1:
        return "one" if lang == "en" else ("une" if g == "f" else "un")
    return str(n)


def amount(op, n, lang, g="m"):
    """(phrase, plural noun?) for ``op n``: "at least one", "no", "more than 3"; "!=" is said by `counted`."""
    op, n = {(">", 0): (">=", 1), ("!=", 0): (">=", 1), ("<", 1): ("==", 0), ("<=", 0): ("==", 0)}.get((op, n), (op, n))
    if (op, n) == ("==", 0):
        return ("no" if lang == "en" else ("aucune" if g == "f" else "aucun")), lang == "en"
    word = _n(n, lang, g)
    en = {">=": f"at least {word}", ">": f"more than {word}", "<=": f"at most {word}", "<": f"fewer than {word}",
          "==": f"exactly {word}", "!=": f"other than {word}"}
    de = "d'" if n == 1 else "de "
    fr = {">=": f"au moins {word}", ">": f"plus {de}{word}", "<=": f"au plus {word}", "<": f"moins {de}{word}",
          "==": f"exactement {word}", "!=": f"autre que {word}"}
    return (en if lang == "en" else fr)[op], not (n == 1 and op != "!=")


def counted(op, n, lang, sg, pl, g="m"):
    """``op n`` of a noun: "at least one word", "no words" / "aucun mot", "a number of words other than 4"."""
    if op == "!=" and n != 0:
        return f"a number of {pl} other than {n}" if lang == "en" else f"un nombre de {pl} autre que {n}"
    ph, plural = amount(op, n, lang, g)
    return f"{ph} {pl if plural else sg}"


def pred(c, lang, neg=False):
    """The condition: EN the verb phrase's base form ("start with ..."); FR its (3sg, 3pl) forms."""
    col, op, v = c["col"], c["op"], c["v"]
    if col in NUMERIC:
        o = INV[op] if neg else op
        u = UNITS[col]
        if lang == "en":
            return f"have {counted(o, v, 'en', u[0], u[1])}"
        ph = counted(o, v, "fr", u[2], u[3])
        return (f"n'a {ph}", f"n'ont {ph}") if ph.startswith("aucun") else (f"a {ph}", f"ont {ph}")
    neg = neg != (op == "!=")                             # "!=" on a string column: the condition negated
    if col == "end_mark" and v == "none" and neg:          # not "without closing punctuation": with some
        if lang == "en":
            return "end with closing punctuation"
        return ("finit par une ponctuation finale", "finissent par une ponctuation finale")
    if col == "item":
        neg = neg != (v == "no")
        en, fr = "be part of a list", ("fait", "font", "partie d'une liste")
    else:
        en, fr = {
            "first_lower": lambda: (f"start with the letter {q(v, 'en')}", ("commence", "commencent", f"par la lettre {q(v, 'fr')}")),
            "last_lower": lambda: (f"end with the letter {q(v, 'en')}", ("finit", "finissent", f"par la lettre {q(v, 'fr')}")),
            "lower": lambda: (f"be the word {q(v, 'en')}", ("est", "sont", f"le mot {q(v, 'fr')}")),
            "first_word": lambda: (f"begin with the word {q(v, 'en')}", ("commence", "commencent", f"par le mot {q(v, 'fr')}")),
            "last_word": lambda: (f"end with the word {q(v, 'en')}", ("finit", "finissent", f"par le mot {q(v, 'fr')}")),
            "case": lambda: (f"be written {CASES.get(v, (v, v))[0]}", ("est écrit", "sont écrits", CASES.get(v, (v, v))[1])),
            "end_mark": lambda: ((f"end with {MARKS[v][0]}", ("finit", "finissent", f"par {MARKS[v][1]}")) if v in MARKS
                                 else ("end without closing punctuation", ("finit", "finissent", "sans ponctuation finale"))),
            "marker": lambda: (("be numbered", ("est numéroté", "sont numérotés", "")) if v == "number"
                               else (f"be marked with {q(v, 'en')}", ("est marqué", "sont marqués", f"par {q(v, 'fr')}"))),
            "key": lambda: (f"be named {q(v, 'en')}", ("s'appelle", "s'appellent", q(v, "fr"))),
            "type": lambda: (f"hold {TYPES.get(v, (v, v))[0]}", ("contient", "contiennent", TYPES.get(v, (v, v))[1])),
            "value": lambda: (f"have the value {q(v, 'en')}", ("a", "ont", f"la valeur {q(v, 'fr')}")),
            "kind": lambda: ({"letter": "be a letter", "digit": "be a digit", "space": "be a space"}.get(v, f"be a {v}"),
                             ("est", "sont", {"letter": "une lettre", "digit": "un chiffre", "space": "une espace"}.get(v, v))),
            "char": lambda: (f"be the character {q(v, 'en')}", ("est", "sont", f"le caractère {q(v, 'fr')}")),
        }[col]()
    if lang == "en":
        return ("not " if neg else "") + en
    sg, pl, rest = fr
    rest = (" " + rest) if rest else ""
    if not neg:
        return (sg + rest, pl + rest)

    def nf(verb):
        head, *tail = verb.split(" ", 1)
        ne = "n'" if head[0] in "aeéio" else ("ne s'" if head.startswith("s'") else "ne ")
        head = head[2:] if head.startswith("s'") else head
        return f"{ne}{head} pas" + (" " + tail[0] if tail else "")
    return (nf(sg) + rest, nf(pl) + rest)


def _en3(base: str, plural: bool) -> str:
    """The base verb phrase with a third-person subject ("start" -> "starts"; "not be" -> "is not")."""
    neg = base.startswith("not ")
    b = base[4:] if neg else base
    verb, _, rest = b.partition(" ")
    if verb == "be":
        return f"{'are' if plural else 'is'} {'not ' if neg else ''}{rest}"
    if neg:
        return f"{'do' if plural else 'does'} not {b}"
    if not plural:
        verb = "has" if verb == "have" else verb + ("es" if verb.endswith(("s", "sh", "ch")) else "s")
    return f"{verb} {rest}"


def rel(c, lang, plural, neg=False):
    p = pred(c, lang, neg)
    return f"that {_en3(p, plural)}" if lang == "en" else f"qui {p[1] if plural else p[0]}"


def _char_noun(a, lang, plural):
    """A char atom with one ``==`` condition said as its own noun ("digits", "the character "g"")."""
    c = a["where"][0]
    if c["col"] == "char":
        return f"the character {q(c['v'], lang)}" if lang == "en" else f"le caractère {q(c['v'], lang)}"
    names = {("kind", "letter"): ("letter", "letters", "lettre", "lettres"), ("kind", "digit"): ("digit", "digits", "chiffre", "chiffres"),
             ("kind", "space"): ("space", "spaces", "espace", "espaces"),
             ("case", "upper"): ("capital letter", "capital letters", "majuscule", "majuscules"),
             ("case", "lower"): ("lowercase letter", "lowercase letters", "minuscule", "minuscules")}
    n = names.get((c["col"], c["v"]))
    return None if n is None else (n[1] if plural else n[0]) if lang == "en" else (n[3] if plural else n[2])


def say(a: dict, lang: str, rng: random.Random) -> str:
    en_sg, en_pl, fr_sg, fr_pl, g = NOUNS[a["view"]]
    sg, pl = (en_sg, en_pl) if lang == "en" else (fr_sg, fr_pl)
    if a["q"] == "n":
        what = counted(a["op"], a["v"], lang, sg, pl, g)
        if lang == "en":
            return rng.choice(["Write {}.", "Your reply should have {}."]).format(what)
        if what.startswith("aucun"):                     # French negation: "n'écris aucun ...", "ne doit avoir aucun ..."
            return rng.choice(["N'écris {}.", "Ta réponse ne doit avoir {}."]).format(what)
        return rng.choice(["Écris {}.", "Ta réponse doit avoir {}."]).format(what)
    if a["q"] == "at":
        c = {"col": a["col"], "op": a["op"], "v": a["val"]}
        if lang == "en":
            return f"The {ORD['en'][a['pos']]} {sg} should {pred(c, 'en')}."
        return f"{'La' if g == 'f' else 'Le'} {ORD['fr'][a['pos']][1 if g == 'f' else 0]} {sg} {pred(c, 'fr')[0]}."
    conds = a["where"]
    one = conds[0] if len(conds) == 1 else None
    if a["q"] in ("count", "any", "none") and one and one["op"] == "==" and (
            one["col"] == "lower" or (a["view"] == "char" and _char_noun(a, lang, False))):
        thing = (f"the word {q(one['v'], lang)}" if lang == "en" else f"le mot {q(one['v'], lang)}") if one["col"] == "lower" \
            else _char_noun(a, lang, True)
        if a["q"] == "any":
            return f"Include {thing}." if lang == "en" else f"Mets {thing}."
        if a["q"] == "none":
            return f"Do not use {thing}." if lang == "en" else f"N'utilise pas {thing}."
        if amount(a["op"], a["v"], lang)[0] in ("no", "aucun", "aucune"):
            return say(dict(a, q="none"), lang, rng)
        if one["col"] == "lower" or one["col"] == "char":
            times = counted(a["op"], a["v"], lang, "time" if lang == "en" else "fois", "times" if lang == "en" else "fois", "f")
            times = times.replace("one time", "once")
            return f"Use {thing} {times}." if lang == "en" else f"Emploie {thing} {times}."
        what = counted(a["op"], a["v"], lang, _char_noun(a, lang, False), _char_noun(a, lang, True), "f")
        return f"Include {what}." if lang == "en" else f"Mets {what}."

    def rels(plural):
        return (" and " if lang == "en" else " et ").join(rel(c, lang, plural) for c in conds)
    if a["q"] == "count":
        ph, plural = amount(a["op"], a["v"], lang, g)
        if ph in ("no", "aucun", "aucune"):
            return say(dict(a, q="none"), lang, rng)
        plural = plural or (a["op"] == "!=" and a["v"] != 0)
        return f"Include {counted(a['op'], a['v'], lang, sg, pl, g)} {rels(plural)}." if lang == "en" \
            else f"Mets {counted(a['op'], a['v'], lang, fr_sg, fr_pl, g)} {rels(plural)}."
    if a["q"] == "any":
        return f"Include at least one {sg} {rels(False)}." if lang == "en" else f"Mets au moins {'une' if g == 'f' else 'un'} {sg} {rels(False)}."
    if a["q"] == "none":
        return f"Do not include any {sg} {rels(False)}." if lang == "en" else f"Ne mets {'aucune' if g == 'f' else 'aucun'} {sg} {rels(False)}."
    if lang == "en":                                     # every row: a single negated condition said as "no row ..."
        ps = [pred(c, "en") for c in conds]
        if len(ps) == 1 and ps[0].startswith("not "):
            return f"No {sg} should {ps[0][4:]}."
        return f"Every {sg} should {' and '.join(sorted(ps, key=lambda p: p.startswith('not ')))}."
    return f"Chaque {sg} {' et '.join(pred(c, 'fr')[0] for c in conds)}."


def say_not(a: dict, lang: str, rng: random.Random) -> str:
    """`{"not": a}` in words, the negation carried into the atom (the same truth on every reply)."""
    en_sg, en_pl, fr_sg, fr_pl, g = NOUNS[a["view"]]
    if a["q"] in ("count", "n"):
        return say(dict(a, op=INV[a["op"]]), lang, rng)
    if a["q"] in ("any", "none"):
        return say(dict(a, q="none" if a["q"] == "any" else "any"), lang, rng)
    if a["q"] == "at":
        c = {"col": a["col"], "op": a["op"], "v": a["val"]}
        if lang == "en":
            return f"The {ORD['en'][a['pos']]} {en_sg}, if there is one, should {pred(c, 'en', neg=True)}."
        o = ORD["fr"][a["pos"]][1 if g == "f" else 0]
        return f"{'La' if g == 'f' else 'Le'} {o} {fr_sg}, s'il y en a, {pred(c, 'fr', neg=True)[0]}."
    conds = a["where"]                                                       # not all: some row fails a condition
    if lang == "en":
        return f"Include at least one {en_sg} {' or '.join(rel(c, 'en', False, neg=True) for c in conds)}."
    return f"Mets au moins {'une' if g == 'f' else 'un'} {fr_sg} {' ou '.join(rel(c, 'fr', False, neg=True) for c in conds)}."


def say_cond(a: dict, lang: str) -> str:
    """The atom as a condition, no end mark ("your reply has more than 3 paragraphs"), for "if ..." and "unless ..."."""
    en_sg, en_pl, fr_sg, fr_pl, g = NOUNS[a["view"]]
    sg, pl = (en_sg, en_pl) if lang == "en" else (fr_sg, fr_pl)
    you = "your reply" if lang == "en" else "ta réponse"
    if a["q"] == "n":
        if lang == "en":
            return f"{you} has {counted(a['op'], a['v'], 'en', sg, pl)}"
        what = counted(a["op"], a["v"], "fr", sg, pl, g)
        return f"{you} {'n' if what.startswith('aucun') else ''}a {what}".replace(" na ", " n'a ")
    if a["q"] == "at":
        c = {"col": a["col"], "op": a["op"], "v": a["val"]}
        if lang == "en":
            return f"the {ORD['en'][a['pos']]} {sg} {_en3(pred(c, 'en'), False)}"
        return f"{'la' if g == 'f' else 'le'} {ORD['fr'][a['pos']][1 if g == 'f' else 0]} {sg} {pred(c, 'fr')[0]}"
    conds = a["where"]
    if len(conds) == 1 and conds[0]["col"] == "lower" and conds[0]["op"] == "==":
        w = q(conds[0]["v"], lang)
        if a["q"] == "any":
            return f"{you} uses the word {w}" if lang == "en" else f"{you} emploie le mot {w}"
        if a["q"] == "none":
            return f"{you} does not use the word {w}" if lang == "en" else f"{you} n'emploie pas le mot {w}"
    def rels(plural):
        return (" and " if lang == "en" else " et ").join(rel(c, lang, plural) for c in conds)
    if a["q"] == "count":
        if amount(a["op"], a["v"], lang, g)[0] in ("no", "aucun", "aucune"):
            return say_cond(dict(a, q="none"), lang)
        plural = amount(a["op"], a["v"], lang, g)[1] or (a["op"] == "!=" and a["v"] != 0)
        what = counted(a["op"], a["v"], lang, sg, pl, g)
        return f"{you} has {what} {rels(plural)}" if lang == "en" else f"{you} a {what} {rels(plural)}"
    if a["q"] == "any":
        return f"{you} has at least one {sg} {rels(False)}" if lang == "en" else \
            f"{you} a au moins {'une' if g == 'f' else 'un'} {sg} {rels(False)}"
    if a["q"] == "none":
        return f"{you} has no {sg} {rels(False)}" if lang == "en" else f"{you} n'a {'aucune' if g == 'f' else 'aucun'} {sg} {rels(False)}"
    if lang == "en":
        return f"every {sg} {' and '.join(_en3(pred(c, 'en'), False) for c in conds)}"
    return f"chaque {sg} {' et '.join(pred(c, 'fr')[0] for c in conds)}"


__all__ = ["NOUNS", "amount", "say", "say_cond", "say_not"]
