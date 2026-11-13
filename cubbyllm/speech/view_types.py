"""cubbyllm/speech/view_types.py -- the second reader's grammar, typed (order 0067, step 5; Nick, 2026-11-13: "we will do
the fix").

The first read of the view reader (`view_grammar.Grammar`) showed the VM refusing half of the long rows' programs with a
type error (cubelang P6-1: a number compared with a string or a list is an error, so a refusal and never a verdict):
``line_n_word[i] == "none"``, ``word_case[i] > 9``, a part called with its arguments in another order (a list where a
number was declared). The line shapes cannot see these; `TypedGrammar` adds, as each token arrives:

1. **a comparison's literal has its column's type:** a number column (`NUMBER`) is compared with a number; a string
   column with a quoted string, and only by ``==`` or ``!=``;
2. **a part's names are real and its own:** each parameter is ``n_<view>`` or ``<view>_<column>`` of `reply_views`;
   inside a part, every such name used is one of its parameters (no name the compiler would not know);
3. **a call passes the part's own parameters, in order:** ``let pK = partN(...)`` names a part already written, and
   its arguments are that part's parameters, the same names in the same order (``partN(a, b)`` for ``partN(a, b)``).

4. **"and" by default; "or" only when the instruction says it** (Nick, 2026-11-13: "we keep AND as a default and or
   only if its part of the instruction text"): ``solve`` joins parts with "and" unless the instruction's own words
   call for another join: an "or" group (``if (pA == 1) {...} if (pB == 1) {...}``) at most as many times as the
   instruction says "or", "either", "unless" (French "ou", "soit", "sauf"); an "if" group (not A, or B:
   ``if (pA == 0) ...``) at most as many times as it says "if" ("si"). The ``== 0`` that opens an "if" group is
   refused as it is written, so the reader always has "and" to go on with.

5. **every part is joined, in the generator's form:** first any "or" / "if" groups, each joining two parts' results;
   then one "and" line that reads every value not joined yet; then ``return`` of that last group. Each value is read
   once; a group is made only after the one before it is written; never more groups than parts. Checked as each token
   arrives (a value named only if it is still to be joined, an "and" line closed only over all of them), so the reader
   is never left with no way on. Why: the first typed probe showed a program that joined two of its four parts and
   returned (a part nothing joins cannot fail a reply, so the reply passes when it should not), and, with ``return``
   held back, a reader looping on ``if (g6 == 1) { if (g6 == 1) ...`` to the end of its budget. All 3,900 gold
   answers have this form.

Rules 1-3 read the answer's text only; rule 4 reads the instruction's words for its cues. The gold answers of every row
pass (`tests/speech/test_view_types.py`).

Wiring: STANDALONE (`validation/exp_view_reader.py` reads with it; nothing in the serve path).
"""
from __future__ import annotations

import re

from cubbyllm.speech import reply_views as RV
from cubbyllm.speech.view_grammar import Grammar, chooser

__wiring__ = "STANDALONE"

NUMBER = {"char": {"start", "end"}, "word": {"start", "end", "len", "sentence", "line", "paragraph"},
          "sentence": {"start", "end", "n_word"}, "line": {"start", "end", "n_word", "indent"},
          "paragraph": {"start", "end", "n_word", "n_sentence"}, "item": {"start", "end", "level", "n_word"},
          "json": set()}
_FUNC = re.compile(r"    function part(\d+)\(([^)]*)(\))?")
_CALL = re.compile(r"\s+let p\d+ = part(\d+)\(([^)]*)(\))?")
_NAME = re.compile(r"\b(n_[a-z]+|[a-z]+_[a-z_]+)(?=[^a-z_])")
_LET = re.compile(r"let (x\d+) = ([a-z]+_[a-z_]+)\[")
_IF = re.compile(r"if \((x\d+) (<=|>=|==|!=|<|>)(?: (.))?")
_QUOTED = re.compile(r'"(?:[^"\\\n]|\\.)*"?')
_OPEN_IF = re.compile(r"\s+if \(p\d+ == 0")                         # an "if" group, from its first test
_OPEN_OR = re.compile(r"\s+if \(p\d+ == 1\) \{ assign g\d+ = 1; \} if")   # an "or" group, from its second test
_RETURN = re.compile(r"        r")                                   # "return", from its first letter
_READ = re.compile(r"if \(([pg]\d+) ==")
CUES = {"or": re.compile(r"\b(?:or|either|unless|ou|soit|sauf)\b", re.I), "if": re.compile(r"\b(?:if|si)\b", re.I)}


def kind(name: str) -> str | None:
    """``number`` / ``string`` for a view name the host binds (``n_<view>`` is a number), None for any other name."""
    if name.startswith("n_"):
        return "number" if name[2:] in RV.COLUMNS else None
    view, _, col = name.partition("_")
    if view not in RV.COLUMNS or col not in RV.COLUMNS[view]:
        return None
    return "number" if col in NUMBER[view] else "string"


def _done_params(text: str) -> list[str]:
    return re.findall(r"([a-z_]+): number", text)


def typed_ok(prog: str) -> bool:
    """Whether the program written so far (whole lines and the line being written) keeps rules 1-3."""
    sigs: dict = {}
    params = None
    for ln in prog.split("\n"):
        m = _FUNC.match(ln)
        if m:
            done = _done_params(m.group(2))
            if any(kind(p) is None for p in done) or len(set(done)) < len(done):
                return False
            params = set(done)
            if m.group(3):
                sigs[m.group(1)] = done
            continue
        if ln.startswith("    public function solve"):
            params = None
            continue
        if params is not None:                                   # inside a part
            bare = _QUOTED.sub('""', ln)
            if any(n not in params for n in _NAME.findall(bare)):   # a name still being written (at the end) waits
                return False
            lets = dict(_LET.findall(bare))
            for var, op, lit in _IF.findall(ln):
                col = lets.get(var)
                if col is None:
                    return False
                if kind(col) == "string" and op not in ("==", "!="):
                    return False
                if lit and (lit == '"') != (kind(col) == "string"):
                    return False
            continue
        c = _CALL.match(ln)
        if c:
            sig = sigs.get(c.group(1))
            if sig is None:
                return False
            args = [a.strip() for a in c.group(2).split(",")] if c.group(2) else []
            closed = bool(c.group(3))
            whole, last = (args, None) if closed else (args[:-1], args[-1] if args else "")
            if whole != sig[:len(whole)] or (closed and whole != sig):
                return False
            if last is not None and (len(whole) >= len(sig) or not sig[len(whole)].startswith(last)):
                return False
    return True


def _group_ok(ln: str, loose: list, or_left: int, whole: bool) -> bool:
    """One join line, as far as it is written: it reads values not joined yet, each once; an "or" / "if" group joins
    two parts; an "and" line closes over every value not joined yet (unless it is becoming an "or" group)."""
    reads = _READ.findall(ln)
    if len(set(reads)) < len(reads) or any(v not in loose for v in reads):
        return False
    tail = re.search(r"if \(([pg]\d*)$", ln)                     # a value being named
    if tail and not any(v.startswith(tail.group(1)) and v not in reads for v in loose):
        return False
    if "} if" in ln or _OPEN_IF.match(ln):                        # an "or" / "if" group: two parts
        return all(v.startswith("p") for v in reads) and not (tail and tail.group(1)[:1] == "g")
    if "{ a" in ln and set(reads) != set(loose):
        can_or = (not whole and len(reads) == 1 and reads[0].startswith("p") and or_left > 0
                  and any(v.startswith("p") and v != reads[0] for v in loose))
        return can_or
    return True


def joins_ok(prog: str, instruction: str) -> bool:
    """Rules 4 and 5 on ``solve`` as far as it is written (see the module's text)."""
    lines = prog.split("\n")
    at = next((i for i, ln in enumerate(lines) if ln.startswith("    public function solve")), None)
    if at is None:
        return True
    body = lines[at + 1:]
    n_if = sum(1 for ln in body if _OPEN_IF.match(ln))
    n_or = sum(1 for ln in body if _OPEN_OR.match(ln))
    cues_or = len(CUES["or"].findall(instruction))
    if n_if > len(CUES["if"].findall(instruction)) or n_or > cues_or:
        return False
    parts, read, joined, group, n_g, n_or_done = [], set(), [], None, 0, 0
    for k, ln in enumerate(body):
        whole = k < len(body) - 1
        loose = [v for v in parts + joined if v not in read]   # a part's result or a group nothing joins yet
        if ln.startswith("        let p"):
            parts += re.findall(r"let (p\d+) =", ln)
        elif ln.startswith("        c"):                         # create gN: number; assign gN = 0;
            if group is not None and group not in joined:        # the last group made is not written yet
                return False
            m = re.match(r"\s+create (g\d+):", ln)
            if m:
                if m.group(1) in joined or n_g + 1 > len(parts):  # no more groups than parts
                    return False
                a = re.search(r"assign (g\d*)", ln)
                if a and not (m.group(1).startswith(a.group(1)) and (" =" not in ln[a.end():] or a.group(1) == m.group(1))):
                    return False
                if whole:
                    group, n_g = m.group(1), n_g + 1
        elif ln.startswith("        if"):
            if not _group_ok(ln, loose, cues_or - n_or_done, whole):
                return False
            a = re.search(r"assign (g\d*)", ln)
            if a and (group is None or group in joined or not group.startswith(a.group(1))):
                return False
            if whole:
                read |= set(_READ.findall(ln))
                joined.append(group)
                n_or_done += int(bool(_OPEN_OR.match(ln)))
        elif _RETURN.match(ln):
            if len(loose) != 1:
                return False
            var = ln.strip()[len("return "):].rstrip(";")
            return loose[0].startswith(var) and (not ln.rstrip().endswith(";") or var == loose[0])
    return True


class TypedGrammar(Grammar):
    """`view_grammar.Grammar` with rules 1-4 on the program."""

    def _program(self, text: str) -> tuple[bool, bool]:
        ok, done = super()._program(text)
        if not ok:
            return False, False
        return (True, done) if typed_ok(text) and joins_ok(text, self.instruction) else (False, False)


__all__ = ["CUES", "NUMBER", "TypedGrammar", "chooser", "joins_ok", "kind", "typed_ok"]
