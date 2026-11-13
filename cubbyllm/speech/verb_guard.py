"""cubbyllm/speech/verb_guard.py -- the static guard a verb program passes before it runs (order 0072, build step 2).

A program is not run unless (cubelang P6-1, P6-8 and P6-9 are on the build since `676df1d`: the VM itself now refuses a
missing value, an index past the end, a list over 253 items and a name-key collision; these checks stay, cheap and
before the VM):
1. every name a condition compares is bound before it (``create``, ``let``, ``for (let x of``, a parameter);
2. every index ``L[I]`` sits inside a block whose condition is ``I < n_<g>`` where ``L`` is ``<g>_<col>`` (a group of
   parallel lists the host binds with its length ``n_<g>``), and an index ever decreased (``sub I``) also inside
   ``I >= 0``;
3. no list literal holds more than 250 items.
(Until P6-9, also: no two strings sharing the VM's 2-byte name key; the build keys by 8 bytes now.)

`check(source)` -> [reasons] (empty: it may run).

Wiring: STANDALONE (0072's measurement calls it before every run; 0067's second reader will).
"""
from __future__ import annotations

import re

from cubbyllm.bridges import vm_names

__wiring__ = "STANDALONE"

MAX_ITEMS = 250
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|(?P<cond>(?:if|while|for))\s*\((?P<c>[^)]*)\)\s*\{|(?P<else>else\s*\{)|(?P<open>\{)'
                    r"|(?P<close>\})|(?P<idx>\b(?P<lst>\w+)\[(?P<i>\w+)\])")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")


def _items(body: str) -> int:
    depth, n, in_str, esc, seen = 0, 0, False, False, False
    for ch in body:
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str, seen = True, True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
        elif not ch.isspace():
            seen = True
    return n + 1 if seen else 0


def _lists(src: str) -> list[tuple[str, int]]:
    out = []
    for m in re.finditer(r"let (\w+) = \[", src):
        i, depth, in_str, esc = m.end(), 1, False, False
        while i < len(src) and depth:
            ch = src[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            i += 1
        out.append((m.group(1), _items(src[m.end():i - 1])))
    return out


def check(src: str) -> list[str]:
    why = [f"the list {name} has {n} items (the VM takes at most {MAX_ITEMS})" for name, n in _lists(src) if n > MAX_ITEMS]
    bound: set[str] = set()
    decreased = set(re.findall(r"\bsub (\w+),", src))
    stack: list[str | None] = []
    for m in _TOKEN.finditer(src):
        before = src[:m.start()]
        bound |= set(re.findall(r"\bcreate (\w+):", before)) | set(re.findall(r"\blet (\w+) =", before))
        bound |= set(re.findall(r"\bfor \(let (\w+) of", before)) | set(re.findall(r"\((?:[^()]*?,\s*)?(\w+): \w+", before))
        if m.group("cond"):
            c = m.group("c")
            if m.group("cond") != "for":
                names = [n for n in _IDENT.findall(re.sub(r'"(?:[^"\\]|\\.)*"', " ", c)) if not n.isdigit()]
                why += [f"compares {n!r}, never bound before" for n in names if n not in bound and n not in vm_names.KEYWORDS]
            stack.append(" ".join(c.split()) if m.group("cond") != "for" else None)
        elif m.group("else") or m.group("open"):
            stack.append(None)
        elif m.group("close"):
            if stack:
                stack.pop()
        elif m.group("idx"):
            lst, i = m.group("lst"), m.group("i")
            g = lst.split("_", 1)[0] if "_" in lst else None
            if not g or not any(s == f"{i} < n_{g}" for s in stack if s):
                why.append(f"{lst}[{i}] is read outside a check that {i} < n_{g or '?'}")
            elif i in decreased and not any(s == f"{i} >= 0" for s in stack if s):
                why.append(f"{lst}[{i}] is read with {i} decreased somewhere and no check that {i} >= 0")
    return sorted(set(why))


__all__ = ["MAX_ITEMS", "check"]
