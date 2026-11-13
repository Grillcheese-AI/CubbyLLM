"""Order 0067, line 3: replies built to pass and to fail a rule, for reading a rule's agreement with a benchmark's
checker. The checker labels every reply; these builders only aim near the rule's edge on both sides, so the labels
come out mixed. A builder is keyed by the IFEval instruction id and reads that rule's kwargs.

`replies(instruction_id, kwargs, lang, rng, k)` -> k replies. Ids with no builder raise KeyError.
"""
from __future__ import annotations

import json
import random

__wiring__ = "STANDALONE"

SENT = {"en": ["The morning was quiet and the street was empty.", "We walked to the market before noon.",
               "Nobody knew why the bridge was closed.", "It's a small town, but people don't leave it.",
               "The answer depends on the season and the weather.", "Her notes were short and very clear.",
               "A good plan starts with a simple question.", "They sold bread, fruit and fresh cheese.",
               "The train was late again this week.", "Most of the work happens in the evening."],
        "fr": ["La matinée était calme et la rue était vide.", "Nous avons marché jusqu'au marché avant midi.",
               "Personne ne savait pourquoi le pont était fermé.", "C'est une petite ville, mais on ne la quitte pas.",
               "La réponse dépend de la saison et du temps.", "Ses notes étaient courtes et très claires.",
               "Un bon plan commence par une question simple.", "Ils vendaient du pain, des fruits et du fromage.",
               "Le train était encore en retard cette semaine.", "La plupart du travail se fait le soir."],
        "es": ["La mañana estaba tranquila y la calle vacía.", "Caminamos hasta el mercado antes del mediodía.",
               "Nadie sabía por qué el puente estaba cerrado.", "Es un pueblo pequeño, pero la gente no se va.",
               "La respuesta depende de la estación y del tiempo."]}
FILL = "the a on in at by we it so to up as or of".split()


def text(lang: str, rng: random.Random, n: int = 3) -> str:
    return " ".join(rng.choice(SENT.get(lang, SENT["en"])) for _ in range(n))


def around(n: int, rng: random.Random) -> int:
    """A count near `n`, on either side or on it."""
    return max(0, n + rng.choice([-max(2, n // 5), -1, 0, 0, 1, max(2, n // 5)]))


def words(m: int, rng) -> str:
    return " ".join(rng.choice(FILL) for _ in range(m)).capitalize() + "."


def _letter(letter: str, m: int, rng) -> str:
    pool = [w for w in FILL + ["sky", "dry", "hymn", "rhythm"] if letter not in w] or ["zz"]
    out = [rng.choice(pool) for _ in range(rng.randint(4, 12))] + [letter] * m
    rng.shuffle(out)
    return " ".join(out)


def build(iid: str, kw: dict, lang: str, rng: random.Random) -> str:
    n = lambda key: around(int(kw[key]), rng)                             # noqa: E731
    if iid == "length_constraints:number_words":
        return words(max(1, n("num_words")), rng) if rng.random() < 0.7 else text(lang, rng, rng.randint(1, 8))
    if iid == "length_constraints:number_sentences":
        return text(lang, rng, max(1, n("num_sentences")))
    if iid == "length_constraints:number_paragraphs":
        parts = [text(lang, rng, 2) for _ in range(max(1, n("num_paragraphs")))]
        if len(parts) > 2 and rng.random() < 0.2:
            parts.insert(1, "")
        return "\n***\n".join(parts)
    if iid == "detectable_format:multiple_sections":
        mk = kw["section_spliter"] if rng.random() < 0.8 else kw["section_spliter"].swapcase()
        return text(lang, rng, 1) + "\n" + "\n".join(f"{mk} {i + 1}\n{text(lang, rng, 1)}" for i in range(n("num_sections")))
    if iid == "detectable_format:number_highlighted_sections":
        return " ".join(f"*{text(lang, rng, 1)[:-1]}*" for _ in range(n("num_highlights"))) + " " + text(lang, rng, 1)
    if iid == "detectable_format:title":
        return (f"<<{text(lang, rng, 1)[:20].strip()}>>\n" if rng.random() < 0.5 else rng.choice(["", "<< >>\n", "# Title\n"])) \
            + text(lang, rng, 2)
    if iid == "detectable_format:json_format":
        obj = json.dumps({"answer": text(lang, rng, 1)}, ensure_ascii=False)
        return rng.choice([obj, f"```json\n{obj}\n```", text(lang, rng, 2), obj[:-1], f"Here it is: {obj}"])
    if iid in ("change_case:english_capital", "change_case:english_lowercase"):
        t = text(rng.choice(["en", "en", "fr"]), rng, 2)
        return rng.choice([t.upper(), t.lower(), t])
    if iid == "language:response_language":
        return text(rng.choice(["en", "fr", "es", kw["language"]]), rng, rng.randint(1, 3))
    if iid == "keywords:existence":
        t = text(lang, rng, 2)
        return t + (f" {kw['keywords'][0]}." if rng.random() < 0.5 else "")
    if iid == "keywords:frequency":
        return " ".join([text(lang, rng, 1)] + [f"{kw['keyword']} is here." for _ in range(n("frequency"))])
    if iid == "keywords:letter_frequency":
        return _letter(kw["letter"], n("let_frequency"), rng)
    if iid == "combination:repeat_prompt":
        p = kw["prompt_to_repeat"]
        return rng.choice([p + " " + text(lang, rng, 2), text(lang, rng, 2), p[:-3] + " " + text(lang, rng, 1)])
    if iid == "combination:two_responses":
        a, b = text(lang, rng, 1), text(lang, rng, 1)
        return rng.choice([f"{a}\n******\n{b}", f"{a}\n******\n{a}", f"{a}\n******\n{b}\n******\n{a}", a, f"{a}\n******\n\n******\n{b}"])
    if iid == "startend:quotation":
        t = text(lang, rng, 2)
        return rng.choice([f'"{t}"', t, f'"{t}', f' "{t}" '])
    raise KeyError(iid)


def replies(iid: str, kw: dict, lang: str, rng: random.Random, k: int) -> list[str]:
    return [build(iid, kw, lang, rng) for _ in range(k)]


__all__ = ["replies"]
