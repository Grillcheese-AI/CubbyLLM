"""Order 0073, step 4: probes for the reader's grammar (`cubbyllm/speech/view_machine.py`), used by
`exp_view_split_gold.py` and its test.

- `mutants(lines, instruction, rng)`: one-slot changes of a parsed answer, as answer strings, each with the name
  of what was changed: a comparison, a view, a column, a number, a string, a position, a ``not``, ``absent``, the
  quantifier, a third condition, a join tag, a kind, a line cut in two, a line left out, the last character left
  out. Some stay inside the grammar and some leave it; the machine and `view_split.parse` must agree on each.
- `way_on(machine, state)`: whether the plainest way on from a state reaches a whole answer (finish the choice in
  hand, close a number or a string, copy the text while it may be copied, end the line).
- `walk_ids(machine, vocab, rng, steps, ids)`: a random walk in token ids from the start, each id drawn from the
  ones allowed; returns the states it reached and how many had no id allowed and were not a whole answer.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.speech import view_desc as D  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402
from cubbyllm.speech.view_machine import START, Machine  # noqa: E402
from cubbyllm.speech.view_spec import FILTERS, NUM_OPS  # noqa: E402

__wiring__ = "STANDALONE"


def _answer(lines: list[dict], text: str) -> str:
    out = []
    for x in lines:
        label = x["kind"] + (" " + " ".join(x["join"]) if x.get("join") else "")
        if x.get("form") is not None:
            label += " " + (x["form"] if isinstance(x["form"], str) else D.write(x["form"]))
        out.append(label + V.BAR + text[x["start"]:x["end"]])
    return "\n" + "\n".join(out)


def _forms(form: dict, rng, numbers: list, strings: list) -> list[tuple[str, dict]]:
    """A form with one slot changed, each with the slot's name."""
    out, view = [], form["view"]
    conds = [("where", i) for i in range(len(form.get("where", [])))] + ([("at", None)] if form["q"] == "at" else [])

    def put(where, **kw):
        if where[0] == "at":
            return dict(form, **{{"v": "val"}.get(k, k): v for k, v in kw.items()})
        return dict(form, where=[dict(c, **kw) if j == where[1] else c for j, c in enumerate(form["where"])])
    if "v" in form:
        out += [("a comparison", dict(form, op=o)) for o in NUM_OPS if o != form["op"]]
        out += [("a number", dict(form, v=n)) for n in {form["v"] + 1, max(0, form["v"] - 1), 0, 1, 12345, *numbers} - {form["v"]}]
    out += [("a view", dict(form, view=v)) for v in rng.sample(D.VIEWS, 3) if v != view] + [("a view", dict(form, view="words"))]
    for where in conds:
        c = {"col": form["col"], "op": form["op"], "v": form["val"]} if where[0] == "at" else form["where"][where[1]]
        out += [("a comparison", put(where, op=o)) for o in NUM_OPS if o != c["op"]]
        out += [("a column", put(where, col=x)) for x in list(FILTERS[view]) + ["lower", "n_word", "kind"] if x != c["col"]]
        vals = ({c["v"] + 1, max(0, c["v"] - 1), 0, 1, *numbers} if isinstance(c["v"], int)
                else {c["v"].upper(), c["v"] + "x", "zz", ".", "yes", *strings, 7})
        out += [("a number" if isinstance(c["v"], int) else "a string", put(where, v=x)) for x in vals - {c["v"]}]
        out.append(("a not", put(where, **{"not": not c.get("not")}) if where[0] == "where"
                    else dict(form, **{"not": not form.get("not")})))
    if form["q"] == "at":
        out += [("a position", dict(form, pos=p)) for p in {-1, 0, 1, 2, 3, form["pos"] + 1, *numbers} - {form["pos"]}]
        out.append(("absent", {k: v for k, v in form.items() if k != "absent"} if form.get("absent") else dict(form, absent="pass")))
    else:
        out.append(("a not", dict(form, **{"not": not form.get("not")})))
        out += [("the quantifier", dict(form, q=q)) for q in ("any", "none", "all", "some") if q != form["q"] and "where" in form]
        if "where" in form:
            out.append(("a third condition", dict(form, where=(form["where"] * 3)[:3])))
            out.append(("a condition left out", dict(form, where=form["where"][:-1])))
    return out


def mutants(lines: list[dict], text: str, rng, per_row: int = 40) -> list[tuple[str, str]]:
    """[(what was changed, the answer)], at most `per_row` of them, drawn evenly over the kinds of change."""
    numbers = sorted({int(x) for x in re.findall(r"\d{1,4}", text)})[:6]
    strings = sorted({a or b for a, b in re.findall(r'"([^"]*)"|« ([^»]*) »', text)})[:4]
    out = []
    for i, x in enumerate(lines):
        def with_line(new, i=i):
            return _answer(lines[:i] + new + lines[i + 1:], text)
        if x["kind"] == "check":
            for what, form in _forms(x["form"], rng, numbers, strings):
                try:
                    out.append((what, with_line([dict(x, form=D.write(form))])))
                except (KeyError, TypeError):
                    pass
            tags = [t.split() for t in V.JOINS] + [None, ["either", "prev"]]
            out += [("a join tag", with_line([dict(x, join=t)])) for t in tags if t != x["join"]]
            out.append(("a kind", with_line([dict(x, kind="task", join=None)])))                # a description on a task
            out.append(("a kind", with_line([dict(x, kind="none", join=None, form=None)])))
        else:
            out += [("a kind", with_line([dict(x, kind=k)])) for k in V.KINDS[1:] + ("note",) if k != x["kind"]]
            out.append(("a join tag", with_line([dict(x, join=["if", "next"])])))                # a tag on a line that is not a check
        cuts = [m.start() for m in re.finditer(r" ", text[x["start"]:x["end"]])]
        for k in rng.sample(cuts, min(2, len(cuts))):
            a = x["start"] + k
            out.append(("a line cut in two", with_line([dict(x, end=a), {"kind": "none", "start": a + 1, "end": x["end"]}])))
        out.append(("a line left out", with_line([])))
    whole = _answer(lines, text)
    out += [("the last character left out", whole[:-1]), ("a line more", whole + "\nnone | x"), ("no change", whole)]
    kinds = sorted({w for w, _ in out})
    rng.shuffle(out)
    keep = [next(m for m in out if m[0] == w) for w in kinds]                                    # one of each kind first
    return (keep + [m for m in out if m not in keep])[:max(per_row, len(kinds))]


def way_on(m: Machine, st: tuple, limit: int = 20000) -> bool:
    for _ in range(limit):
        if m.accepting(st):
            return True
        if st[0] == "pick":
            ch = min(o for o in m._options(st[1]) if o.startswith(st[2]))[len(st[2])]
        elif st[0] == "text":
            ch = m.text[st[2]] if st[2] < len(m.text) and m.step(st, m.text[st[2]]) else "\n"
        elif st[0] == "num":
            ch = " " if st[2] else "1"
        else:
            ch = {"start": "\n", "quote": '"', "str": '"'}[st[0]]
        st = m.step(st, ch)
        if st is None:
            return False
    return False


def walk_ids(m: Machine, vocab: T.Vocab, rng, steps: int, ids: dict) -> tuple[list, int]:
    """(the states reached, how many of them had no id allowed and were not a whole answer). `ids`: a cache."""
    seen, stuck, st = [START], 0, START
    for _ in range(steps):
        if st not in ids:
            ids[st] = sorted(T.allowed(m, vocab, st))
        if not ids[st]:
            stuck += not m.accepting(st)
            break
        st = m.run(vocab.pieces[rng.choice(ids[st])], st)
        seen.append(st)
    return seen, stuck


__all__ = ["mutants", "walk_ids", "way_on"]
