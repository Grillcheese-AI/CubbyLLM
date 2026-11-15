"""cubbyllm/speech/view_split.py -- the reader's form for order 0073: an instruction in, its split out, each check
with what it checks. No program: the host compiles the descriptions (`view_account`).

The reader writes, after `HEAD` and the instruction, a newline and then one line per sentence or clause, in order:

    <kind>[ <join>][ <description>] | <the text, copied exactly>

- **kind:** check, task, asked, process, none;
- **join** (a check that carries a connective): ``either next``, ``if next``, ``if prev``, ``unless next``,
  ``unless prev``: the tagged check is the condition, or the first of an "either", and ``prev`` / ``next`` is the
  neighbouring check it ties to;
- **description** (every check): `view_desc`, in arm M's spelling (folded, no ``not``) or arm W's (as said);
- **text:** copied with no gap. A line's text starts at the first character after the last line's that is not white
  space, never crosses a line break of the instruction, does not end on white space; the last line ends where the
  instruction does. So no sentence can be left out and an answer cannot stop early.

`target(row, arm)` -> the answer a row of `long_rows_0073` teaches; `parse(answer, instruction, arm)` ->
{"lines": [{kind, join, form, start, end}]} or {"error": why}. ``strict=False`` reads an answer that did not come
through the grammar: each text is found going forward and gaps are allowed (the host flags them).

Wiring: STANDALONE (the reader's SFT sets and its read call it; nothing in the serve path: CB_RULES stays off).
"""
from __future__ import annotations

from cubbyllm.speech import view_desc as D
from cubbyllm.speech import view_own as O

__wiring__ = "STANDALONE"

HEAD = ("Read the instruction: list each sentence or clause in order with its kind (check, task, asked, process, none); "
        "after a check, write what it checks.\n")
KINDS = ("check", "task", "asked", "process", "none")
JOINS = ("either next", "if next", "if prev", "unless next", "unless prev")
ARMS = ("m", "w")
BAR = " | "


def prompt(instruction: str) -> str:
    return HEAD + f"Instruction: {instruction}\nRead:"


def span(text: str, start: int, end: int) -> tuple[int, int]:
    """A span without its outer white space."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def target(row: dict, arm: str) -> str:
    text, lines = row["instruction"], []
    for s in row["segments"]:
        a, b = span(text, s["start"], s["end"])
        label = s["kind"]
        if s["kind"] == "check":
            label += (" " + " ".join(s["join"]) if s.get("join") else "") + " " + D.write(row["said"][s["part"]][arm])
        lines.append(label + BAR + text[a:b])
    return "\n" + "\n".join(lines)


def _line(line: str, arm: str) -> dict:
    """A line's label read from the left; the text is what follows the bar."""
    kind = next((k for k in KINDS if line.startswith(k + " ")), None)
    if kind is None:
        raise ValueError("a kind not in its set")
    i, join, form = len(kind), None, None
    if kind == "check":
        join = next((j for j in JOINS if line.startswith(j + " ", i + 1)), None)
        i += len(join) + 1 if join else 0
        form, i = D.read_at(line, i + 1, arm)
    if not line.startswith(BAR, i):
        raise ValueError("no bar after the label" if kind == "check" else "a join tag or a description on a line that is not a check")
    return {"kind": kind, "join": join.split() if join else None, "form": form, "text": line[i + len(BAR):]}


def parse(answer: str, instruction: str, arm: str = "m", strict: bool = True, own_text: bool = False) -> dict:
    """``own_text`` (point 7's variant): a description says only what its line's text says (`view_own.allows`)."""
    if not answer.startswith("\n"):
        return {"error": "the answer does not start on a new line"}
    out, at = [], 0
    for line in answer[1:].split("\n") if strict else [ln for ln in answer.split("\n") if ln.strip()]:
        try:
            got = _line(line, arm)
        except ValueError as e:
            return {"error": f"{e}: {line[:80]!r}"}
        text = got.pop("text") if strict else got.pop("text").strip()
        if strict:
            k = next((j for j in range(at, len(instruction)) if not instruction[j].isspace()), -1)
            ok = bool(text) and text == text.strip() and k >= 0 and instruction.startswith(text, k)
        else:
            k = instruction.find(text, at) if text else -1
            ok = k >= 0
        if not ok:
            return {"error": f"a text that is not the instruction's next: {text[:80]!r}"}
        if own_text and got["form"] is not None and not O.allows(got["form"], text):
            return {"error": f"a description that says what its text does not: {line[:80]!r}"}
        at = k + len(text)
        out.append(dict(got, start=k, end=at))
    if strict and instruction[at:].strip():
        return {"error": "the instruction is not used up"}
    return {"lines": out}


__all__ = ["ARMS", "BAR", "HEAD", "JOINS", "KINDS", "parse", "prompt", "span", "target"]
