"""talk_voice_text -- the facts block read and written, and a template with its slot filled.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

The block is `cubbyllm/emit/talk.py`'s (`- entity — relation: value`). A reply shows one fact inline and several
as a numbered list with bold labels (Nick's target replies, order 0010). Nothing here words a fact: a fact is
shown as the block gives it, its label and its value.
"""
from __future__ import annotations

import hashlib
import re

from validation.sft.talk_voice_inputs import FR_LABEL

__wiring__ = "STANDALONE"

DASH = " — "
ABSENT = "The facts don't say."
PREMISE = {"contradicted": "Premise: contradicted", "absent": "Premise: not on record"}   # the host's check of a premise
LINE = re.compile(r"^- (.+?) — ([^:—]{1,60}): (.+)$")
VOWEL = "aeiouyhàâéèêëîïôöùûüœ"
LOWER_AFTER_COMMA = {"The", "A", "An", "Its", "In", "On", "At", "During", "His", "Her", "Their", "It", "There"}
ENDS = ".!?:…"


def parse(prompt: str):
    """A facts-block prompt -> ([(entity, relation, value)], question), or None when it is another shape."""
    if not prompt.startswith("Facts:\n") or not prompt.endswith("\nAnswer:"):
        return None
    body, sep, q = prompt[len("Facts:\n"):-len("\nAnswer:")].rpartition("\nQuestion: ")
    if not sep:
        return None
    triples = []
    for ln in body.split("\n"):
        m = LINE.match(ln)
        if not m:
            return None
        triples.append((m.group(1), m.group(2), m.group(3)))
    return triples, q


def render(triples, question: str, premise: str | None = None) -> str:
    head = (PREMISE[premise] + "\n") if premise else ""
    return head + "Facts:\n" + "\n".join(f"- {e}{DASH}{r}: {v}" for e, r, v in triples) + f"\nQuestion: {question}\nAnswer:"


def bare(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def after(turns, question: str) -> str:
    return "Conversation:\n" + "\n".join(f"{who}: {text}" for who, text in turns) + f"\nQuestion: {question}\nAnswer:"


def fact_lines(triples) -> list[str]:
    return [f"{e}{DASH}{r}: {v}" for e, r, v in triples]


def to_french(triples):
    """The block with French labels, or None when a label has none in the table."""
    out = []
    for e, r, v in triples:
        if r not in FR_LABEL:
            return None
        out.append((e, FR_LABEL[r][0], v))
    return out


def grouped(triples, entity: str):
    """[(label, [values])] for one entity, in the block's order."""
    seen: dict[str, list[str]] = {}
    for e, r, v in triples:
        if e == entity and v not in seen.setdefault(r, []):
            seen[r].append(v)
    return list(seen.items())


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def join(values: list[str], lang: str) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + (" and " if lang == "en" else " et ") + values[-1]


def render_items(items, lang: str) -> tuple[str, bool]:
    """The facts as a reply shows them: one item inline, several as a numbered list. -> (text, is a list)."""
    colon = ":" if lang == "en" else " :"
    if len(items) == 1:
        r, vs = items[0]
        return f"**{cap(r)}{colon}** {join(vs, lang)}", False
    return "\n".join(f"{i}. **{cap(r)}{colon}** {join(vs, lang)}" for i, (r, vs) in enumerate(items, 1)), True


def fill_list(tmpl: str, slot: str, text: str, is_list: bool, lang: str) -> str:
    """A template with its {facts} or {abilities} slot filled: a list goes on its own lines."""
    head, _, tail = tmpl.partition("{" + slot + "}")
    head, tail = " ".join(head.split()), " ".join(tail.split())
    if is_list:
        if head and head[-1] not in ENDS:
            head = head.rstrip(",;") + (":" if lang == "en" else " :")
        return (head + "\n" if head else "") + text + ("\n\n" + tail if tail else "")
    body = text if text.endswith((".", "!", "?")) else text + "."
    return " ".join(x for x in (head, body, tail) if x)


def fill_frame(tmpl: str, fact: str) -> str:
    """A checked sentence behind a frame. The frame's own words after the slot are dropped."""
    head = " ".join(tmpl.partition("{fact}")[0].split())
    if head and head[-1] not in ENDS and fact.split(" ", 1)[0] in LOWER_AFTER_COMMA:
        fact = fact[0].lower() + fact[1:]
    return (head + " " + fact).strip()


def frame_one(tmpl: str, label: str, vs: list[str], lang: str) -> str | None:
    """One fact, shown as its label and value, behind a frame. None when the frame runs on into the fact
    ("my notes say that ..."): a label and a value are not a clause."""
    head = " ".join(tmpl.partition("{fact}")[0].split())
    colon = ":" if lang == "en" else " :"
    if not head or head[-1] in ENDS:
        return f"{head} **{cap(label)}{colon}** {join(vs, lang)}.".strip()
    if head[-1] == ",":
        return f"{head} {label}{colon} {join(vs, lang)}."
    return None


def de(e: str) -> str:
    return ("d'" if e[:1].lower() in VOWEL else "de ") + e


def article(label: str, gender: str) -> str:
    return "l'" if label[:1].lower() in VOWEL else ("le " if gender == "m" else "la ")


def split_of(key: str, held_one_in: int = 25) -> str:
    return "held" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % held_one_in == 0 else "train"
