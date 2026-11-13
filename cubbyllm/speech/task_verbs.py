"""cubbyllm/speech/task_verbs.py -- the task an instruction sets (order 0067, the second reader's split).

An instruction's clauses are of several kinds: a check on the reply's form (the reply world measures it), a process
step, something no measurement covers ("asked"), or the **task**: what the answer has to do with its subject
("compare", "explain", "summarise"). A task is never checked by the VM and never read as a rule; it is handed to the
talk seat as the shape of the answer, and recorded so a user can see how their instruction was understood.

`TASKS`: the task words of written assignments (Nick's list, 2026-11-13), English and French, each with a gloss in
this file's own words and the shape of answer it asks for. `tasks_in(text)` -> the tasks a text sets, in order.

Wiring: STANDALONE (the second reader's split, step 3, reads it; until then its tests).
"""
from __future__ import annotations

import re
import unicodedata

__wiring__ = "STANDALONE"

# key: (English cues, French cues, gloss, the answer's shape)
TASKS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str, str]] = {
    "account_for": (("account for",), ("rends compte de", "rendez compte de", "justifie l'existence"),
                    "say how or why something came about", "causes and the way they led to it"),
    "analyse": (("analyse", "analyze", "analysis of"), ("analyse", "analysez"),
                "take it apart and name its important parts", "the parts, each with why it matters"),
    "comment_on": (("comment on",), ("commente", "commentez"),
                   "pick out the main issues and say something grounded about each", "issues with grounded remarks"),
    "compare": (("compare", "comparison"), ("compare", "comparez", "comparaison"),
                "show where things are alike, and what that likeness means", "likenesses and their weight"),
    "contrast": (("contrast",), ("oppose", "opposez", "mets en contraste", "contraste"),
                 "show where things differ, and whether the differences matter", "differences and their weight"),
    "critically_evaluate": (("critically evaluate", "critically assess", "critique"), ("evalue de facon critique",
                            "evaluez de facon critique", "critique"),
                            "weigh the case for and against and judge which holds better", "both sides, then a verdict"),
    "define": (("define", "definition of"), ("definis", "definissez", "donne la definition", "definition de"),
               "give the precise meaning a field uses", "the field's meaning, exactly"),
    "describe": (("describe", "description of"), ("decris", "decrivez", "description de"),
                 "say what something is like: its main features", "the main features"),
    "discuss": (("discuss",), ("discute", "discutez", "debats de"),
                "give arguments on several sides and reach a considered view", "arguments, then a view"),
    "distinguish": (("distinguish", "differentiate", "tell apart"), ("distingue", "distinguez", "differencie"),
                    "set out what separates two or more things", "the dividing differences"),
    "evaluate": (("evaluate", "assess", "appraise"), ("evalue", "evaluez", "apprecie", "estime la valeur"),
                 "judge worth or usefulness from evidence", "evidence, then a judgement"),
    "examine": (("examine", "investigate", "look closely at"), ("examine", "examinez", "etudie en detail"),
                "look at it closely, part by part", "a close look, part by part"),
    "explain": (("explain", "why does", "how does"), ("explique", "expliquez", "pourquoi", "comment fonctionne"),
                "make clear how or why", "the how or the why"),
    "illustrate": (("illustrate", "give examples of", "exemplify"), ("illustre", "illustrez", "donne des exemples"),
                   "make it concrete with examples", "examples that make it clear"),
    "interpret": (("interpret", "what does this mean", "make sense of"), ("interprete", "interpretez"),
                  "say what given material means or implies", "the meaning drawn from the material"),
    "justify": (("justify", "defend", "make the case for"), ("justifie", "justifiez", "defends"),
                "give the evidence behind a claim or a choice", "the claim and its support"),
    "narrate": (("narrate", "tell the story of", "recount"), ("raconte", "racontez", "narre"),
                "tell what happened, as a story", "events in order"),
    "outline": (("outline", "sketch out", "main points of"), ("esquisse", "donne les grandes lignes",
                "presente les grandes lignes"), "give the main points only", "main points, no detail"),
    "relate": (("relate", "connect", "link between"), ("relie", "reliez", "mets en relation", "lien entre"),
               "show how things connect", "the connections"),
    "state": (("state", "list the", "name the"), ("enonce", "enoncez", "indique", "cite les"),
              "give the main points plainly, in full sentences", "plain points in sentences"),
    "summarise": (("summarise", "summarize", "summary of", "sum up"), ("resume", "resumez", "fais un resume"),
                  "give the main points, leaving out detail and examples", "main points only"),
    "to_what_extent": (("to what extent", "how far", "how much does"), ("dans quelle mesure", "jusqu'a quel point"),
                       "judge how far something holds, between not at all and fully", "a degree, argued"),
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower().replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c))


_AFTER = r"(?=$|[^\w-]|-(?:le|la|les|moi|toi|lui|nous|vous|leur|en|y)\b)"   # "résume-le", "explique-moi"
_CUES = sorted(((c, key) for key, (en, fr, _, _) in TASKS.items() for c in en + fr), key=lambda x: -len(x[0]))


def tasks_in(text: str, lang: str | None = None) -> list[dict]:
    """The tasks `text` sets, in reading order: [{"task", "cue", "start", "gloss", "shape"}]; a longer cue wins over
    a shorter one it contains ("critically evaluate" over "evaluate"); the same task twice in a row is said once
    ("explique-moi pourquoi"). `lang` ("en" / "fr") keeps to that language's cues (English "resume" is not "résume")."""
    low, taken, out = _fold(text), [], []
    cues = _CUES if lang is None else [(c, k) for c, k in _CUES if c in TASKS[k][0 if lang == "en" else 1]]
    for cue, key in cues:
        for m in re.finditer(rf"(?<![\w-]){re.escape(cue)}{_AFTER}", low):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append({"task": key, "cue": cue, "start": m.start(), "gloss": TASKS[key][2], "shape": TASKS[key][3]})
    out.sort(key=lambda t: t["start"])
    return [t for i, t in enumerate(out) if i == 0 or t["task"] != out[i - 1]["task"]]


__all__ = ["TASKS", "tasks_in"]
