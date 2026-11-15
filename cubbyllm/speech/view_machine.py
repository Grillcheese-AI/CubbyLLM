"""cubbyllm/speech/view_machine.py -- the reader's grammar as a state machine over characters (order 0073, "The
grammar is a state machine"): what may be written next in an answer of `view_split`'s form, built from the
instruction itself.

A state is one of:
- ``("pick", key, typed)``: a slot whose choices are a closed set (`_options(key)`: the kind, the join tag, the
  quantifier, the view, a column of that view, a comparison of that column's type, a value). Every choice is
  written with the delimiter that follows it, so no choice is the start of another;
- ``("num", ...)`` / ``("quote", ...)`` / ``("str", ...)``: a free number (at most `view_desc.MAX_DIGITS` digits) or a
  free string (at most `MAX_STRING` characters), where the variant leaves values free;
- ``("text", start, at, ends)``: the line's text, copied from the instruction one character at a time.

**No gap:** a line's text starts at the first character after the last line's that is not white space, never
crosses a line break, does not end on white space; a new line starts only while text is left, and the answer is
whole (`accepting`) only when none is. **Arm W** may write ``not `` before a quantifier and before a column.

**``own_text``** (point 7's variant): a description uses only values its line's text holds (`view_own`). The
description comes first, so each value it takes carries the place where the text has it, and the line may not end
before that place (``ends``: how far it must run; where it must stop to name no view, or quote nothing, when its
description counts on that). A choice is offered only if some end is left, and the text is copied no further than
the last end left, so **no state is a dead end**: a line can always be finished and the instruction used up.

`Machine(instruction, arm, own_text)`: `step(state, ch)` -> the next state or None; `accepting(state)`;
`allows(text)` / `accepts(text)` for a whole prefix or answer.

Wiring: STANDALONE (`view_tokens` walks it for the reader's read; nothing in the serve path: CB_RULES stays off).
"""
from __future__ import annotations

import json

from cubbyllm.speech import view_desc as D
from cubbyllm.speech import view_own as O
from cubbyllm.speech import view_split as V
from cubbyllm.speech.view_spec import FILTERS, NUM_OPS, STR_OPS

__wiring__ = "STANDALONE"

START = ("start",)
NEVER = 10 ** 9
DIGITS = "0123456789"


class Machine:
    def __init__(self, instruction: str, arm: str = "m", own_text: bool = False):
        self.text, self.arm, self.own_text = instruction, arm, own_text
        self._opts: dict = {}
        self._owns: dict = {}
        self._last: dict = {}

    # -- places -----------------------------------------------------------------------------------------------------
    def _next(self, p: int) -> int | None:
        """The first character at or after `p` that is not white space."""
        return next((j for j in range(p, len(self.text)) if not self.text[j].isspace()), None)

    def _own(self, s: int) -> O.Own:
        if s not in self._owns:
            self._owns[s] = O.Own(self.text, s)
        return self._owns[s]

    def _may_end(self, s: int, p: int, ends) -> bool:
        """Whether a line that started at `s` may end at `p`. ``ends`` (own text): (the place it must reach, where a
        view is first named, where its own view is named, where a string is first quoted, where its free strings
        are all quoted)."""
        if p <= s or self.text[p - 1].isspace():
            return False
        if ends is None:
            return True
        need, vcap, vfloor, qcap, qfloor = ends
        return (p >= need and p not in self._own(s).inside
                and (vcap is None or p < vcap or (vfloor is not None and p >= vfloor))
                and (qcap is None or p < qcap or p >= qfloor))

    def _last_end(self, s: int, ends) -> int | None:
        """The last place a line that started at `s` may end, or None when it has no end left."""
        if (s, ends) not in self._last:
            stop = self._own(s).stop
            self._last[(s, ends)] = next((p for p in range(stop, s, -1) if self._may_end(s, p, ends)), None)
        return self._last[(s, ends)]

    def _reach(self, s: int, ends, at: int):
        """``ends`` once the line must also reach `at`; None when no end is left."""
        got = (max(ends[0], at), *ends[1:])
        return got if self._last_end(s, got) is not None else None

    # -- the slots ----------------------------------------------------------------------------------------------------
    def _options(self, key: tuple) -> dict:
        if key not in self._opts:
            self._opts[key] = getattr(self, "_o_" + key[0])(*key[1:])
        return self._opts[key]

    def _o_kind(self, s):
        out = {f"{k}{V.BAR}": ("text", s, s, None) for k in V.KINDS if k != "check"}
        return dict(out, **{"check ": ("pick", ("head", s), "")})

    def _o_head(self, s):
        return dict({f"{j} ": ("pick", ("quant", s), "") for j in V.JOINS}, **self._options(("quant", s)))

    def _o_quant(self, s):
        out = {f"{q} ": ("pick", ("view", s, q), "") for q in D.QUANT}
        if self.arm == "w":
            out.update({f"not {q} ": ("pick", ("view", s, q), "") for q in D.QUANT if q != "at"})
        return out

    def _o_view(self, s, q):
        out = {}
        for v in D.VIEWS:
            ends = None
            if self.own_text:                              # a view the text names, or any when the line names none
                own = self._own(s)
                ends = self._reach(s, (s, own.view_cap, own.views.get(v), None, 0), s)
                if ends is None:
                    continue
            if q == "at":
                out[f"{v} "] = ("pick", ("pos", s, v, ends), "")
            elif q in ("n", "count"):
                out[f"{v} "] = ("pick", ("op", s, v, "num", (q,), ends), "")
            else:
                out[f"{v} where "] = ("pick", ("col", s, v, ("cond", 1), ends), "")
        return out

    def _o_pos(self, s, v, ends):
        out = {"last ": ("pick", ("col", s, v, ("at",), ends), "")}
        if self.own_text:                                  # free positions are read digit by digit (`step`)
            for n, at in self._own(s).positions.items():
                got = self._reach(s, ends, at)
                if got is not None:
                    out[f"{n} "] = ("pick", ("col", s, v, ("at",), got), "")
        return out

    def _o_posend(self, s, v):
        return {" ": ("pick", ("col", s, v, ("at",), None), "")}

    def _o_col(self, s, v, tail, ends):
        out = {}
        for c, kind in FILTERS[v].items():
            if kind == "str" and self.own_text and not any(self._strings(s, c, ends)):
                continue                                   # no value the line's text would let it take
            nxt = ("pick", ("op", s, v, (kind, c), tail, ends), "")
            out[f"{c} "] = nxt
            if self.arm == "w":
                out[f"not {c} "] = nxt
        return out

    def _o_op(self, s, v, typ, tail, ends):
        if typ == "num" or typ[0] == "num":
            nxt = ("pick", ("numval", s, v, tail, ends), "") if self.own_text else ("num", ("after", s, v, tail, None), "")
            return {f"{op} ": nxt for op in NUM_OPS}
        free, _ = self._strings(s, typ[1], ends) if self.own_text else (True, {})
        nxt = ("quote", (s, v, typ[1], tail, ends)) if free else ("pick", ("strval", s, v, typ[1], tail, ends), "")
        return {f"{op} ": nxt for op in STR_OPS}

    def _o_numval(self, s, v, tail, ends):
        out = {}
        for n, at in self._own(s).numbers.items():
            got = self._reach(s, ends, at)
            if got is not None:
                out.update({f"{n}{d}": nxt for d, nxt in self._options(("after", s, v, tail, got)).items()})
        return out

    def _strings(self, s, col, ends) -> tuple[bool, dict]:
        """Own text: (whether the column's value is free, {a value the text holds: ``ends`` after it})."""
        own = self._own(s)
        held = {x: self._reach(s, ends, at) for x, at in own.strings(col).items() if len(x) <= D.MAX_STRING}
        free = not O.FIXED.get(col) and self._reach(s, ends[:3] + ((own.quote_cap, NEVER) if own.quote_cap else ends[3:]), s)
        return bool(free), {x: e for x, e in held.items() if e is not None}

    def _o_strval(self, s, v, col, tail, ends):
        out = {}
        for x, got in self._strings(s, col, ends)[1].items():
            lit = json.dumps(x, ensure_ascii=False)
            out.update({f"{lit}{d}": nxt for d, nxt in self._options(("after", s, v, tail, got)).items()})
        return out

    def _o_after(self, s, v, tail, ends):
        text = ("text", s, s, ends)
        if tail == ("n",):
            return {V.BAR: text}
        if tail == ("count",):
            return {" where ": ("pick", ("col", s, v, ("cond", 1), ends), "")}
        if tail[0] == "cond":
            more = {" and ": ("pick", ("col", s, v, ("cond", tail[1] + 1), ends), "")} if tail[1] < D.MAX_CONDS else {}
            return dict(more, **{V.BAR: text})
        return {f" absent pass{V.BAR}": text, V.BAR: text}

    # -- the moves ----------------------------------------------------------------------------------------------------
    def step(self, st: tuple, ch: str):
        kind = st[0]
        if kind == "pick":
            _, key, typed = st
            if key[0] == "pos" and not self.own_text and not typed and ch in DIGITS:
                return self.step(("num", ("posend", key[1], key[2]), ""), ch)
            typed += ch
            opts = self._options(key)
            if typed in opts:
                return opts[typed]
            return ("pick", key, typed) if any(o.startswith(typed) for o in opts) else None
        if kind == "text":
            _, s, p, ends = st
            if p < self._last_end(s, ends) and ch == self.text[p]:
                return ("text", s, p + 1, ends)
            if ch == "\n" and self._may_end(s, p, ends):
                nxt = self._next(p)
                return None if nxt is None else ("pick", ("kind", nxt), "")
            return None
        if kind == "num":
            _, after, digits = st
            if ch in DIGITS:
                ok = len(digits) < D.MAX_DIGITS and digits != "0" and not (after[0] == "posend" and not digits and ch == "0")
                return ("num", after, digits + ch) if ok else None
            return self.step(("pick", after, ""), ch) if digits else None
        if kind == "quote":
            return ("str", st[1], "", False) if ch == '"' else None
        if kind == "str":
            _, (s, v, col, tail, ends), typed, esc = st
            if esc:
                return ("str", st[1], typed + ch, False) if ch in '"\\' else None
            if ch == '"':
                if ends is not None and self._own(s).quote_cap is not None:   # quoted by the text, or free of any quote
                    at = self._own(s).strings(col).get(typed, NEVER)
                    ends = ends[:3] + (self._own(s).quote_cap, max(ends[4], at))
                return ("pick", ("after", s, v, tail, ends), "")
            if ch < " " or len(typed) >= D.MAX_STRING:
                return None
            return ("str", st[1], typed, True) if ch == "\\" else ("str", st[1], typed + ch, False)
        if kind == "start":
            first = self._next(0)
            return ("pick", ("kind", first), "") if ch == "\n" and first is not None else None
        return None

    def accepting(self, st: tuple) -> bool:
        return st[0] == "text" and self._may_end(st[1], st[2], st[3]) and self._next(st[2]) is None

    def run(self, text: str, st: tuple = START):
        for ch in text:
            st = self.step(st, ch)
            if st is None:
                return None
        return st

    def allows(self, text: str) -> bool:
        return self.run(text) is not None

    def accepts(self, text: str) -> bool:
        st = self.run(text)
        return st is not None and self.accepting(st)


__all__ = ["START", "Machine"]
