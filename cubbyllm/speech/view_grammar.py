"""cubbyllm/speech/view_grammar.py -- what the second reader may write next (order 0067, step 5).

The reader's answer (`view_reader.py`'s form) is held, token by token, to:
- **the split:** lines ``<label> | <text>``, the label from the closed set, the text copied from the instruction going
  forward from where the last segment ended; then ``---`` once at least one segment is written;
- **the join:** the lines the generator writes (`validation/builders/view_compile.py`, `view_join.py`), in order: the
  program head, part functions (their parameters, the statement shapes of a check), ``solve`` (the calls, the groups'
  flags, the return), the close. Anything else does not parse, so it is never written.
What the line shapes cannot see (a name used but never made, a view column a part does not take) is left to the
compiler: a program that does not compile is "asked", never "checked".

`Grammar(instruction).viable(text)` -> whether ``text`` can still become a whole answer; `.complete(text)` -> whether
it is one. `chooser(grammar, piece, eos, prompt_len)` -> the callback `cubbyllm.ops.grilly_seats.ranked_processor`
takes: the best-scored token that keeps the answer viable (``eos`` only once the answer is complete).

Wiring: STANDALONE (the reader's read calls it; nothing in the serve path).
"""
from __future__ import annotations

from cubbyllm.speech.view_reader import SEP

__wiring__ = "STANDALONE"

V = r"(?:char|word|sentence|line|paragraph|item|json)"
OP = r"(?:<|<=|>|>=|==|!=)"
LIT = r'(?:\d+|"(?:[^"\\\n]|\\.)*")'
N = r"\d+"
I8 = " " * 8
BODY = [
    rf"{I8}create [cki]{N}: number; assign [cki]{N} = 0;",
    rf"{I8}create p{N}: number; assign p{N} = {N};",
    rf"{I8}create p{N}: number; assign p{N} = n_{V}; sub p{N}, 1;",
    rf"{I8}while \(i{N} < n_{V}\) \{{ (?:let x{N} = {V}_[a-z_]+\[i{N}\]; )+(?:if \(x{N} {OP} {LIT}\) \{{ )+add k{N}, 1;(?: \}})+ add i{N}, 1; \}}",
    rf"{I8}if \(n_{V} {OP} {N}\) \{{ assign c{N} = 1; \}}",
    rf"{I8}if \(p{N} >= 0\) \{{ if \(p{N} < n_{V}\) \{{ let x{N} = {V}_[a-z_]+\[p{N}\]; if \(x{N} {OP} {LIT}\) \{{ assign c{N} = 1; \}} \}} \}}",
    rf"{I8}if \(k{N} (?:{OP} {N}|== n_{V}|> 0)\) \{{ assign c{N} = 1; \}}",
    rf"{I8}if \(c{N} == 0\) \{{ assign c{N} = 1; \}}",
]
MAIN = [
    rf"{I8}let p{N} = part{N}\((?:[a-z_]+(?:, [a-z_]+)*)\);",
    rf"{I8}create g{N}: number; assign g{N} = 0;",
    rf"{I8}if \(p{N} == [01]\) \{{ assign g{N} = 1; \}} if \(p{N} == 1\) \{{ assign g{N} = 1; \}}",
    rf"{I8}(?:if \([pg]{N} == 1\) \{{ )+assign g{N} = 1;(?: \}})+",
]
LINES = {  # state -> [(pattern, next state)]
    "start": [(r"program Instruction implements ISolve \{", "funcs")],
    "funcs": [(rf"    function part{N}\((?:[a-z_]+: number(?:, [a-z_]+: number)*)\): number \{{", "body")],
    "body": [(p, "body") for p in BODY] + [(rf"{I8}return c{N};", "body_end")],
    "body_end": [(r"    \}", "after_func")],
    "after_func": [(rf"    function part{N}\((?:[a-z_]+: number(?:, [a-z_]+: number)*)\): number \{{", "body"),
                   (r"    public function solve\(mention: str\): number \{", "main")],
    "main": [(p, "main") for p in MAIN] + [(rf"{I8}return [gp]{N};", "main_end")],
    "main_end": [(r"    \}", "close")],
    "close": [(r"\}", "done")],
    "done": [],
}
_RX: dict = {}


def _rx() -> dict:
    """The compiled lines and label, made on first use (the `regex` module, for partial matches, is imported here)."""
    if not _RX:
        import regex
        _RX["lines"] = {s: [(regex.compile(p), nxt) for p, nxt in opts] for s, opts in LINES.items()}
        _RX["label"] = regex.compile(r"(?:check \d+(?: condition)?|task|asked|process|none) \| ")
    return _RX


class Grammar:
    def __init__(self, instruction: str):
        self.instruction = instruction

    def _split(self, text: str) -> tuple[bool, bool]:
        """(viable, complete) for the split section (no separator yet)."""
        LABEL = _rx()["label"]                            # noqa: N806
        lines = text.split("\n")
        if lines[0] != "":
            return False, False
        at, n = 0, 0
        for ln in lines[1:-1]:
            m = LABEL.match(ln)
            if not m or len(ln) == m.end():
                return False, False
            k = self.instruction.find(ln[m.end():], at)
            if k < 0:
                return False, False
            at, n = k + len(ln) - m.end(), n + 1
        last = lines[-1] if len(lines) > 1 else ""
        if len(lines) == 1:
            return True, False
        if n and "---".startswith(last):
            return True, False
        m = LABEL.match(last)
        if m:
            return self.instruction.find(last[m.end():], at) >= 0, False
        return bool(LABEL.fullmatch(last, partial=True)), False

    def _program(self, text: str) -> tuple[bool, bool]:
        _COMPILED = _rx()["lines"]                        # noqa: N806
        state, lines = "start", text.split("\n")
        for ln in lines[:-1]:
            nxt = next((n for rx, n in _COMPILED[state] if rx.fullmatch(ln)), None)
            if nxt is None:
                return False, False
            state = nxt
        last = lines[-1]
        if state == "done":
            return last == "", True
        if last == "":
            return True, False
        if any(rx.fullmatch(last) for rx, _ in _COMPILED[state]):
            return True, any(rx.fullmatch(last) and n == "done" for rx, n in _COMPILED[state])
        return any(rx.fullmatch(last, partial=True) for rx, _ in _COMPILED[state]), False

    def _check(self, text: str) -> tuple[bool, bool]:
        if SEP in text:
            head, prog = text.split(SEP, 1)
            ok, _ = self._split(head + "\n---")
            if not ok or head.count("\n") < 1:
                return False, False
            return self._program(prog)
        return self._split(text)

    def viable(self, text: str) -> bool:
        return self._check(text)[0]

    def complete(self, text: str) -> bool:
        return self._check(text)[1]


def chooser(grammar: Grammar, piece, eos: int, prompt_len: int):
    """`piece(token_id) -> str`. The callback for `ranked_processor`: the first of the ranked ids that keeps the answer
    viable (``eos`` when the answer is complete and ranked)."""
    cache = {"n": -1, "text": ""}

    def choose(ids: list, ranked: list) -> list:
        gen = ids[prompt_len:]
        if len(gen) == cache["n"] + 1 and cache["n"] >= 0:
            cache["text"] += piece(gen[-1])
        elif len(gen) != cache["n"]:
            cache["text"] = "".join(piece(t) for t in gen)
        cache["n"] = len(gen)
        text = cache["text"]
        done = grammar.complete(text)
        heads: dict = {}                                   # a piece's first 1 and 2 characters, judged once a step

        def head_ok(p: str) -> bool:
            for k in (1, 2):
                if len(p) > k:
                    if p[:k] not in heads:
                        heads[p[:k]] = grammar.viable(text + p[:k])
                    if not heads[p[:k]]:
                        return False
            return True
        screen = len(ranked) > SCREEN_AT
        for t in ranked:
            if t == eos:
                if done:
                    return [t]
                continue
            if not done and (not screen or head_ok(piece(t))) and grammar.viable(text + piece(t)):
                return [t]
        return []
    return choose


# Above this many ranked ids (the whole vocabulary, when none of the top 512 fits) a piece is first judged by its first
# one and two characters: a refused prefix has no way on, so a piece that starts with one is refused without being
# judged whole. The same pick as judging every piece (the grammar's refusals are prefix-closed: pinned in
# tests/speech/test_view_grammar_screen.py); a whole-vocabulary scan went from about 13 s to well under one.
SCREEN_AT = 4096


__all__ = ["Grammar", "LINES", "chooser"]
