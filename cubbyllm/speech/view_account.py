"""cubbyllm/speech/view_account.py -- the host's accounting of a read (order 0073, step 4): from the reader's lines
(`view_split.parse`) to what can be checked, what is only asked, and what no line holds. Nothing is dropped
silently.

- **Parts:** the check lines, numbered in the order of their lines. Each description is folded (`view_spec.fold`:
  arm W's ``not`` taken into the comparison; arm M's is already folded) and compiled (`view_compile`). One that
  does not compile makes its part **asked**.
- **Groups:** a join tag ties its check to the neighbouring line on the side it names. A tag whose neighbour is
  not a check, or whose pair would take a check that is already in one, cannot be resolved: its part is asked, and
  so is the check it points at (checked alone it would fail replies the instruction allows).
- **A pair with an asked member is asked whole**, for the same reason: one half of "A unless B" is not a check.
- **Flagged:** every stretch of the instruction that is in no line and is not white space (none, when the lines
  came through the grammar: its text has no gap).

`account(lines, instruction)` -> {"checks": [{line, part, form (folded, or None), asked (why, or None)}],
"pairs": the groups the tags make, in the reader's part numbers; "parts" and "groups": what can be checked,
renumbered; "program": their join (`view_join`), or None when nothing can be checked; "asked": {part: why};
"flagged": [[start, end]]}. The host runs ``view_join.program(program, parts, reply)``.

Wiring: STANDALONE (the reader's read and its scoring call it; nothing in the serve path: CB_RULES stays off).
"""
from __future__ import annotations

from cubbyllm.speech import view_compile as VC
from cubbyllm.speech import view_join as VJ
from cubbyllm.speech import view_spec as VS

__wiring__ = "STANDALONE"


def _folded(form: dict) -> tuple[dict | None, str | None]:
    try:
        m = VS.fold(form)
        VC.body(m)
        return m, None
    except (ValueError, KeyError, TypeError) as e:
        return None, f"a description that does not compile ({e})"


def _pairs(lines: list[dict], checks: list[int], asked: dict) -> dict:
    """{part: its group} from the join tags; what cannot be resolved goes into `asked`."""
    number, pair = {i: n for n, i in enumerate(checks)}, {}
    for n, i in enumerate(checks):
        tag = lines[i].get("join")
        if not tag:
            continue
        kind, side = tag
        other = number.get(i + (1 if side == "next" else -1))
        if other is None or (kind, side) == ("either", "prev"):
            asked.setdefault(n, "a join tag with no check beside it")
        elif n in pair or other in pair:
            asked.setdefault(n, "a join tag on a check that is already in a pair")
            asked.setdefault(other, "beside a join tag that cannot be resolved")
        else:                                              # the tagged check is the condition, or the first of an "either"
            pair[n] = pair[other] = {"if": ("if", n, other), "unless": ("unless", other, n)}.get(kind, ("either", n, other))
    for g in set(pair.values()):
        if any(x in asked for x in g[1:]):
            for x in g[1:]:
                asked.setdefault(x, "tied to a part that is asked")
    return pair


def flagged(lines: list[dict], instruction: str) -> list[list[int]]:
    """The stretches of the instruction that no line holds, without their outer white space."""
    out, at = [], 0
    for a, b in sorted((x["start"], x["end"]) for x in lines) + [(len(instruction), len(instruction))]:
        gap = instruction[at:a]
        if gap.strip():
            lead = len(gap) - len(gap.lstrip())
            out.append([at + lead, at + len(gap.rstrip())])
        at = max(at, b)
    return out


def account(lines: list[dict], instruction: str) -> dict:
    checks = [i for i, x in enumerate(lines) if x["kind"] == "check"]
    forms, asked = [], {}
    for n, i in enumerate(checks):
        m, why = _folded(lines[i]["form"])
        forms.append(m)
        if why:
            asked[n] = why
    pair = _pairs(lines, checks, asked)
    kept = [n for n in range(len(checks)) if n not in asked]
    new, groups, done = {n: k for k, n in enumerate(kept)}, [], set()
    for n in kept:
        g = pair.get(n)
        if g is None:
            groups.append(("one", new[n]))
        elif g not in done:
            done.add(g)
            groups.append((g[0], new[g[1]], new[g[2]]))
    parts = [forms[n] for n in kept]
    return {"checks": [{"line": i, "part": n, "form": forms[n], "asked": asked.get(n)} for n, i in enumerate(checks)],
            "pairs": sorted(set(pair.values()), key=lambda g: min(g[1:])), "parts": parts, "groups": groups,
            "program": VJ.join(parts, groups) if parts else None, "asked": dict(sorted(asked.items())),
            "flagged": flagged(lines, instruction)}


__all__ = ["account", "flagged"]
