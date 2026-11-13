"""Order 0067, the second reader's step 2: replies built for the program generator (`view_programs.py`).

Replies of several shapes, seeded: prose (sentences of varied length, end marks and case, in paragraphs between blank
lines or ``***``), lists (``-`` / ``*`` / numbered items, some indented), mixed prose and list, and JSON objects (in a
fence or not). English and French words. The generator draws a program's thresholds and values from a pool of these,
so a program meets replies on both sides of its edge; nothing here knows any program.

`reply(rng, shape=None)` -> one reply; `pool(rng, n, shapes=None)` -> n replies.
"""
from __future__ import annotations

import json
import random

__wiring__ = "STANDALONE"

WORDS = {
    "en": ("the a of and to in is it that with for on as at by this we they you he she was are be have not but or "
           "time year people way day man thing woman life child world school state family student group country problem "
           "hand part place case week company system program question work government number night point home water "
           "room mother area money story fact month lot right study book eye job word business issue side kind head "
           "house service friend father power hour game line end member law car city community name team minute idea "
           "kid body back parent face others level office door health person art war history party result change "
           "morning reason research girl guy moment air teacher force education river bread travel planet winter "
           "bridge garden table window music light sound tree stone fire road quiet simple bright slow quick green "
           "small large old new good great high long little early young important few public bad same able").split(),
    "fr": ("le la les un une de des et à en est que qui dans pour pas sur avec il elle nous vous ils on ce cette "
           "temps année gens jour homme chose femme vie enfant monde école état famille groupe pays problème main "
           "partie lieu semaine maison eau ville ami père mère livre route arbre pierre feu lumière musique jardin "
           "table fenêtre pain rivière hiver pont voyage planète calme simple clair lent rapide vert petit grand "
           "vieux nouveau bon haut long jeune important même").split(),
}
KEYS = ("name", "title", "city", "count", "items", "answer", "score", "tags", "date", "id", "status", "notes")


def _sentence(rng: random.Random, lang: str) -> str:
    ws = [rng.choice(WORDS[lang]) for _ in range(rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20]))]
    if rng.random() < 0.15:
        ws.insert(rng.randrange(len(ws) + 1), str(rng.choice([2, 3, 7, 12, 40, 1999])))
    if rng.random() < 0.1:
        k = rng.randrange(len(ws))
        ws[k] = ws[k].upper()
    s = " ".join(ws)
    s = s[0].upper() + s[1:]
    if rng.random() < 0.1:
        s = s.upper()
    elif rng.random() < 0.07:
        s = s.lower()
    if rng.random() < 0.2:
        cut = rng.randrange(1, len(s.split()) + 1)
        s = " ".join(s.split()[:cut]) + ", " + " ".join(s.split()[cut:]) if cut < len(s.split()) else s
    return s + rng.choice([".", ".", ".", ".", "!", "?", "?", ""])


def _paragraph(rng, lang) -> str:
    return " ".join(_sentence(rng, lang) for _ in range(rng.choice([1, 1, 2, 3, 4, 6])))


def _list(rng, lang) -> str:
    marker = rng.choice(["-", "*", "number"])
    out = []
    for i in range(rng.choice([1, 2, 3, 4, 5, 7, 9])):
        mk = f"{i + 1}." if marker == "number" else marker
        indent = "  " if rng.random() < 0.2 else ""
        out.append(f"{indent}{mk} " + " ".join(rng.choice(WORDS[lang]) for _ in range(rng.choice([1, 2, 3, 5, 8]))))
    return "\n".join(out)


def _json(rng, lang) -> str:
    obj = {}
    for k in rng.sample(KEYS, rng.choice([1, 2, 3, 4, 6])):
        obj[k] = rng.choice([lambda: rng.choice(WORDS[lang]), lambda: rng.randint(0, 99), lambda: rng.random() < 0.5,
                             lambda: None, lambda: [rng.choice(WORDS[lang]) for _ in range(rng.randint(0, 3))],
                             lambda: {"x": rng.randint(0, 9)}])()
    text = json.dumps(obj, ensure_ascii=False, indent=rng.choice([None, 2]))
    return f"```json\n{text}\n```" if rng.random() < 0.3 else text


def reply(rng: random.Random, shape: str | None = None, lang: str | None = None) -> str:
    lang = lang or rng.choice(["en", "en", "fr"])
    shape = shape or rng.choice(["prose", "prose", "list", "mixed", "json"])
    if shape == "json":
        return _json(rng, lang)
    if shape == "list":
        return _list(rng, lang)
    paras = [_paragraph(rng, lang) for _ in range(rng.choice([1, 1, 2, 3, 4, 5]))]
    if shape == "mixed":
        paras.insert(rng.randrange(len(paras) + 1), _list(rng, lang))
    sep = rng.choice(["\n\n", "\n\n", "\n***\n"])
    return sep.join(paras)


def pool(rng: random.Random, n: int, shapes=None) -> list[str]:
    return [reply(rng, rng.choice(shapes) if shapes else None) for _ in range(n)]


__all__ = ["KEYS", "WORDS", "pool", "reply"]
