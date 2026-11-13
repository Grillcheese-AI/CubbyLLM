"""Order 0072, build step 3: cases for every verb and kind, built before the programs run (at least 20 each, half
failing, each failing case broken one named way: the order's list), text cases in English and French.

`VARIANT_OF[(verb, kind)]` -> the program variant (`cubbyllm/speech/verb_programs.py`); `cases(verb, kind, rng, n)` ->
[{"verb", "kind", "variant", "lang", "data", "breakage" (None when it should hold)}]; the act kind's cases are in
`verb_cases_act.py`.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_replies as VR  # noqa: E402

__wiring__ = "STANDALONE"

VARIANT_OF = {}
for _v in "extract filter find get identify list search select".split():
    VARIANT_OF[(_v, "pick")] = "filter"
VARIANT_OF[("read", "pick")] = "read"
VARIANT_OF.update({("sort", "reorder"): "sort", ("move", "reorder"): "move"})
for _v in "edit modify update replace rename".split():
    VARIANT_OF[(_v, "edit")] = "replace"
VARIANT_OF.update({("add", "edit"): "add", ("delete", "edit"): "delete", ("remove", "edit"): "delete",
                   ("set", "edit"): "set", ("reset", "edit"): "set", ("clear", "edit"): "clear", ("copy", "edit"): "copy",
                   ("merge", "edit"): "merge", ("apply", "edit"): "apply", ("format", "edit"): "same",
                   ("calculate", "compute"): "aggregate", ("evaluate", "compute"): "expression",
                   ("compare", "compute"): "larger", ("export", "compute"): "mapped", ("parse", "compute"): "mapped",
                   ("print", "compute"): "mapped", ("process", "compute"): "mapped",
                   ("check", "verdict"): "required", ("validate", "verdict"): "required", ("verify", "verdict"): "required",
                   ("test", "verdict"): "tests", ("compare", "verdict"): "equal"})
for _v in "analyze compare create define describe evaluate explain generate summarize translate write".split():
    VARIANT_OF[(_v, "write")] = "form"
ALL_GIVEN = {"filter", "list", "extract", "select"}       # complete: every item that meets the condition
BREAKS = {"filter": ["an item not in the input", "an item that fails the condition", "a missing item"],
          "read": ["another value", "two values", "nothing"], "sort": ["one out of order", "one dropped", "one doubled"],
          "move": ["the wrong place", "one dropped", "the rest reordered"],
          "replace": ["a place missed", "a change outside the ask", "an item lost"],
          "add": ["the wrong place", "a change outside the ask", "an item lost"],
          "delete": ["a place missed", "an item lost", "a change outside the ask"],
          "set": ["the value not set", "a change outside the ask", "an item lost"], "clear": ["one item left", "all left"],
          "copy": ["the original changed", "the copy differs", "an item lost"],
          "merge": ["an item lost", "an item doubled", "an item added"],
          "apply": ["a place missed", "a change outside the ask", "off by one"], "same": ["a word changed", "an item lost"],
          "aggregate": ["off by one", "the wrong unit", "another function's result"],
          "expression": ["off by one", "the wrong unit"], "larger": ["the other one", "equal when not"],
          "mapped": ["one field differs", "one missing", "one extra"], "required": ["one field missing"],
          "tests": ["one failed", "none ran"], "equal": ["one item differs", "one missing"],
          "form": ["too long", "empty", "the wrong language", "the subject missing"]}


def g(name: str, **cols) -> dict:
    """A group of parallel lists and its length."""
    out = {f"{name}_{c}": list(v) for c, v in cols.items()}
    out[f"n_{name}"] = len(next(iter(cols.values()))) if cols else 0
    return out


def words(rng, lang, n, unique=False):
    pool = sorted(set(VR.WORDS[lang]))
    if unique and n > len(pool):                          # long lists (the timing): numbered words past the vocabulary
        pool += [f"{w}{i}" for i in range(2, 2 + n // len(pool) + 1) for w in sorted(set(VR.WORDS[lang]))]
    return rng.sample(pool, n) if unique else [rng.choice(pool) for _ in range(n)]


def _other(rng, lang, avoid):
    while True:
        w = rng.choice(VR.WORDS[lang])
        if w not in avoid:
            return w


def build(verb, variant, rng, lang, brk, size=None):
    """One case's data; `brk` a breakage name or None."""
    n = size or rng.randint(4, 12)
    from verb_cases_more import build_more
    if variant == "filter":
        xs = [rng.randint(0, 99) for _ in range(n)]
        op, t = rng.choice(["<", "<=", ">", ">="]), rng.randint(20, 80)
        ok = lambda x: {"<": x < t, "<=": x <= t, ">": x > t, ">=": x >= t}[op]   # noqa: E731
        meets = [x for x in xs if ok(x)]
        full = verb in ALL_GIVEN
        out = list(dict.fromkeys(meets)) if full else list(dict.fromkeys(meets))[:max(1, len(meets) // 2)]
        if brk == "an item not in the input":
            out.append(next(v for v in range(0, 200) if ok(v) and v not in xs))
        elif brk == "an item that fails the condition":
            bad = [x for x in xs if x not in meets] or [{">": t, ">=": t - 1, "<": t, "<=": t + 1}[op]]
            if bad[0] not in xs:
                xs.append(bad[0])
            out.append(bad[0])
        elif brk == "a missing item" and out:
            out = out[1:] if full else out
        return {**g("in", val=xs, key=xs), **g("out", val=out, key=out), "arg_op": op, "arg_t": t, "arg_all": int(full)}
    if variant == "read":
        want = rng.choice(VR.WORDS[lang])
        out = {None: [want], "another value": [_other(rng, lang, {want})], "two values": [want, want], "nothing": []}[brk]
        return {**g("out", val=out), "arg_want": want}
    if variant == "sort":
        xs = words(rng, lang, n, unique=True)
        rank = {w: i for i, w in enumerate(sorted(xs))}
        d = rng.choice(["asc", "desc"])
        ys = sorted(xs, reverse=(d == "desc"))
        if brk == "one out of order":
            i = rng.randrange(len(ys) - 1)
            ys[i], ys[i + 1] = ys[i + 1], ys[i]
        elif brk == "one dropped":
            ys.pop(rng.randrange(len(ys)))
        elif brk == "one doubled":
            i = rng.randrange(len(ys))
            ys.insert(i, ys[i])
        return {**g("in", val=xs, key=[rank[w] for w in xs]), **g("out", val=ys, key=[rank[w] for w in ys]), "arg_dir": d}
    if variant == "move":
        xs = words(rng, lang, n, unique=True)
        item, p = rng.choice(xs), rng.randrange(n)
        rest = [w for w in xs if w != item]
        ys = rest[:p] + [item] + rest[p:]
        if brk == "the wrong place":
            q = (p + 1) % n
            ys = rest[:q] + [item] + rest[q:]
        elif brk == "one dropped":
            ys.pop(rng.choice([i for i in range(n) if i != p]))
        elif brk == "the rest reordered":
            i = rng.choice([i for i in range(len(rest) - 1)])
            rest[i], rest[i + 1] = rest[i + 1], rest[i]
            ys = rest[:p] + [item] + rest[p:]
        return {**g("in", val=xs), **g("out", val=ys), "arg_item": item, "arg_place": p}
    return build_more(verb, variant, rng, lang, brk, n)


def cases(verb: str, kind: str, rng: random.Random, n: int = 20) -> list[dict]:
    variant = VARIANT_OF[(verb, kind)]
    breaks = BREAKS[variant]
    if variant == "filter" and verb not in ALL_GIVEN:
        breaks = [b for b in breaks if b != "a missing item"]
    out = []
    for i in range(n):
        brk = None if i % 2 == 0 else breaks[(i // 2) % len(breaks)]
        lang = "fr" if i % 4 in (2, 3) else "en"
        out.append({"verb": verb, "kind": kind, "variant": variant, "lang": lang, "breakage": brk,
                    "data": build(verb, variant, rng, lang, brk)})
    return out


__all__ = ["BREAKS", "VARIANT_OF", "build", "cases", "g", "words"]
