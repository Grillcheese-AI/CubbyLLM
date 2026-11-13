"""cubbyllm/speech/view_reader.py -- the second reader's form (order 0067, step 5): an instruction in, its split and its
join out, as the program seat writes them.

The reader writes, after `HEAD` and the instruction:
    one line per sentence or clause, in order: ``<kind>[ <part>][ condition] | <the text, copied exactly>``
        kinds: check (a part, numbered as the text states them), task, asked, process, none
    ``---``
    the join: a CubeLang program, a function per part and a ``solve`` that joins them (`validation/builders/view_join.py`)

`target(row)` -> the answer a training row teaches; `parse(answer, instruction)` -> {"segments": [{kind, part, role,
start, end}], "program": str} or {"error": why}: each segment's text must be found in the instruction, in order (its
span is where it is found); a segment the instruction does not hold makes the read "asked", never a guessed span.

Wiring: STANDALONE (the reader's SFT set and its read call it; nothing in the serve path yet: CB_RULES stays off).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

HEAD = ("Read the instruction: list each sentence or clause in order with its kind (check, task, asked, process, none), "
        "then write one CubeLang program that checks the reply: a function per check, a solve that joins them.\n")
SEP = "\n---\n"
KINDS = ("check", "task", "asked", "process", "none")
_LINE = re.compile(r"^(check|task|asked|process|none)(?: (\d+))?( condition)? \| (.*)$")


def prompt(instruction: str) -> str:
    return HEAD + f"Instruction: {instruction}\nRead:"


def target(row: dict) -> str:
    text, lines = row["instruction"], []
    for s in row["segments"]:
        label = s["kind"] + (f" {s['part'] + 1}" if s["kind"] == "check" else "") + (" condition" if s.get("role") == "condition" else "")
        lines.append(f"{label} | {text[s['start']:s['end']]}")
    return "\n" + "\n".join(lines) + SEP + row["join"].rstrip("\n")


def parse(answer: str, instruction: str) -> dict:
    if SEP not in answer:
        return {"error": "no separator between the split and the join"}
    head, program = answer.split(SEP, 1)
    segs, at = [], 0
    for line in [ln for ln in head.strip("\n").split("\n") if ln.strip()]:
        m = _LINE.match(line)
        if not m:
            return {"error": f"a split line not in the form: {line[:80]!r}"}
        kind, part, cond, text = m.group(1), m.group(2), m.group(3), m.group(4)
        k = instruction.find(text, at)
        if not text or k < 0:
            return {"error": f"a segment the instruction does not hold: {text[:80]!r}"}
        segs.append({"kind": kind, "part": int(part) - 1 if part else None, "role": "condition" if cond else None,
                     "start": k, "end": k + len(text)})
        at = k + len(text)
    return {"segments": segs, "program": program.strip() + "\n"}


__all__ = ["HEAD", "KINDS", "SEP", "parse", "prompt", "target"]
