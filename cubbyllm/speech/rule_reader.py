"""cubbyllm/speech/rule_reader.py -- the program seat reads an instruction into a rule plan, one slot at a time, each
slot constrained to what the plan family allows (order 0067, step 3: "measured with the seat as it is, its slots
constrained to the plan's shape").

The seat writes a short surface form, one leaf after another, joined by " ;" (and):

    word count < 50 ; count of the word "river" >= 2 ; case == lower

For each slot the host hands the decoder the token sequences of the slot's options (a trie) through grilly2's
``prefix_allowed_tokens_fn`` (order 0071), so the seat can only write a plan of the family's shape:
- the measurement: a `reply_world` relation (or ``asked``, first slot only: no rule written);
- the argument, for a relation that takes one: a span of the instruction's own words, or ``"`` (one character of the
  instruction for a letter or character count);
- the comparison: the ops `rule_plan` allows for that relation;
- the value: a number (0..MAX_N) for a count; the relation's known values (``lower``, ``en``, ``yes`` ...); a span of
  the instruction for a word, an opening, a JSON key or a last word;
- then `` ;`` (another leaf, up to MAX_LEAVES) or a newline (the end, as each example ends).
The seat chooses inside each slot; the host only says what is allowed. The result passes `rule_plan.check` by
construction; "asked" is a result, never a guess.

`read(instruction, gen, encode, eos, examples)`: ``gen(ids, allowed, max_new) -> new ids`` runs the seat with
``allowed(new ids so far) -> allowed next ids``; ``encode(text) -> ids``. No grilly here: the caller binds the seat.

Wired: STANDALONE (`validation/exp_rule_reader.py` measures it; the turn, step 4, calls it once built).
"""
from __future__ import annotations

import re

from . import reply_world as rw
from . import rule_plan as rp

__wiring__ = "STANDALONE"

MAX_N = 400
MAX_LEAVES = 4
MAX_SPAN = 12
VALUES = {"case": ["lower", "upper", "mixed", "none"], "language": ["en", "fr", "es", "none"], "json": ["yes", "no"],
          "markdown": ["yes", "no"], "starts with": ["yes", "no"], "ends with": ["yes", "no"],
          "form of address": ["tu", "vous", "both", "none"]}
SPAN_VALUED = ("word", "json key", "opening", "last word")
ONE_CHAR = ("count of the letter", "count of the character")       # their argument: one character of the instruction
NL = "\n"
HEAD = "Each instruction becomes a rule over measurements of the reply.\n\n"


def surface(plan: dict) -> str:
    """A plan of leaves joined by "and", in the reader's surface form (the rows' targets, the few-shot examples)."""
    parts = []
    for lf in rp.leaves(plan):
        arg = f' "{lf["arg"]}"' if "arg" in lf else ""
        parts.append(f'{lf["m"]}{arg} {lf["op"]} {lf["v"]}')
    return " ; ".join(parts)


def spans(text: str) -> list[str]:
    """Every run of 1..MAX_SPAN of the instruction's words, with and without the punctuation around them."""
    ws = text.split()
    out = {'"'}
    for i in range(len(ws)):
        for j in range(i + 1, min(len(ws), i + MAX_SPAN) + 1):
            s = " ".join(ws[i:j])
            out.add(s)
            out.add(s.strip(".,;:!?«»\"'()"))
    return sorted(x for x in out if x.strip())


class _Trie:
    def __init__(self, options: dict, eos: int) -> None:          # option text -> its ids
        self.root, self.eos = {}, eos
        for text, ids in options.items():
            node = self.root
            for t in ids:
                node = node.setdefault(t, {})
            node[None] = text                                    # an option ends here

    def allowed(self, prefix: list) -> list:
        node = self.root
        for t in prefix:
            if t not in node:
                return [self.eos]
            node = node[t]
        nxt = [t for t in node if t is not None]
        return nxt + ([self.eos] if None in node else [])

    def which(self, ids: list):
        node = self.root
        for t in ids:
            if t == self.eos:
                break
            node = node.get(t)
            if node is None:
                return None
        return node.get(None)


def _choose(gen, encode, eos, text: str, options: list[str]):
    """The option the seat writes after `text` (each written with a leading space, except a newline), or None."""
    opts = {o: encode(o if o == NL else " " + o) for o in dict.fromkeys(options)}
    opts = {o: ids for o, ids in opts.items() if ids}
    trie = _Trie(opts, eos)
    out = gen(encode(text), trie.allowed, max(len(v) for v in opts.values()) + 1)
    return trie.which(out)


def _ops(rel: str) -> list[str]:
    if rel in rp.MULTI:
        return list(rp.SET_OPS)
    return list(rp.NUM_OPS) if rp._is_num(rel) else ["==", "!="]


def _values(rel: str, instruction: str) -> list[str]:
    if rp._is_num(rel):
        return [str(n) for n in range(MAX_N + 1)]
    if rel in SPAN_VALUED:
        return [s for s in spans(instruction) if s != '"']
    return VALUES.get(rel, ["yes", "no"])


def read(instruction: str, gen, encode, eos: int, examples=()) -> dict:
    """{"plan": dict | None, "how": "written" | "asked", "surface": str, "why"}. `examples`: (instruction, plan)
    pairs shown before it (few-shot), in the surface form."""
    text = HEAD + "".join(f"Instruction: {i}\nRule: {surface(p)}\n\n" for i, p in examples)
    text += f"Instruction: {instruction}\nRule:"
    rels = list(rw.RELATIONS) + list(rw.PARAMETRIC)
    leaves, written = [], ""
    for k in range(MAX_LEAVES):
        rel = _choose(gen, encode, eos, text + written, rels + (["asked"] if k == 0 else []))
        if rel is None or rel == "asked":
            return {"plan": None, "how": "asked", "surface": "asked", "why": "the seat wrote asked" if rel else "no option"}
        written += " " + rel
        lf = {"m": rel}
        if rel in rw.PARAMETRIC:
            pool = sorted(set(instruction) - set(" \n")) if rel in ONE_CHAR else spans(instruction)
            arg = _choose(gen, encode, eos, text + written, [f'"{s}"' for s in pool])
            if arg is None:
                return {"plan": None, "how": "asked", "surface": written, "why": "no argument"}
            written += " " + arg
            lf["arg"] = arg[1:-1]
        op = _choose(gen, encode, eos, text + written, _ops(rel))
        val = None if op is None else _choose(gen, encode, eos, text + written + " " + op, _values(rel, instruction))
        if op is None or val is None:
            return {"plan": None, "how": "asked", "surface": written, "why": "no comparison or value"}
        written += f" {op} {val}"
        lf.update(op=op, v=int(val) if rp._is_num(rel) else val)
        leaves.append(lf)
        if k == MAX_LEAVES - 1 or _choose(gen, encode, eos, text + written, [";", NL]) != ";":
            break
        written += " ;"
    plan = leaves[0] if len(leaves) == 1 else {"and": leaves}
    why = rp.check(plan)
    return {"plan": None if why else plan, "how": "asked" if why else "written", "surface": written.strip(), "why": why}


def same(a: dict | None, b: dict | None) -> bool:
    """Two plans of leaves joined by "and" hold the same leaves (in any order)."""
    if a is None or b is None:
        return a is b
    key = lambda lf: (lf["m"], lf.get("arg"), lf["op"], str(lf["v"]).lower())      # noqa: E731
    return sorted(map(key, rp.leaves(a))) == sorted(map(key, rp.leaves(b)))


__all__ = ["HEAD", "read", "same", "spans", "surface"]
