"""Order 0072, build step 3: cases for the edit, compute, verdict and write kinds (`verb_cases.py` has the rest and the
breakage names). Write cases are real replies in English or French; the host's own measures give the word count and
the language (`cubbyllm/speech/reply_world.py`), as the serve path would.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import view_replies as VR  # noqa: E402
from verb_cases import _other, g, words  # noqa: E402

from cubbyllm.speech import reply_world as rw  # noqa: E402

__wiring__ = "STANDALONE"


def _edit(variant, rng, lang, brk, n):
    xs = words(rng, lang, n)
    if variant == "replace":
        old = rng.choice(xs)
        new = _other(rng, lang, set(xs))
        ys = [new if w == old else w for w in xs]
        hits = [i for i, w in enumerate(xs) if w == old]
        if brk == "a place missed":
            ys[hits[-1]] = old if len(hits) > 1 else ys[hits[-1]]
            if len(hits) == 1:                            # one place only: the change not made at all
                ys[hits[0]] = old
        elif brk == "a change outside the ask":
            i = next(i for i in range(len(xs)) if i not in hits) if len(hits) < len(xs) else None
            if i is None:
                xs.append(_other(rng, lang, {old, new}))
                ys.append(_other(rng, lang, {old, new, xs[-1]}))
            else:
                ys[i] = _other(rng, lang, {xs[i], new})
        elif brk == "an item lost":
            ys.pop(rng.randrange(len(ys)))
        return {**g("in", val=xs), **g("out", val=ys), "arg_old": old, "arg_new": new}
    if variant == "add":
        item, p = _other(rng, lang, set(xs)), rng.randrange(len(xs) + 1)
        ys = xs[:p] + [item] + xs[p:]
        if brk == "the wrong place":
            q = (p + 1) % (len(xs) + 1)
            ys = xs[:q] + [item] + xs[q:]
            if q == p or ys == xs[:p] + [item] + xs[p:]:
                ys = [item] + xs if p != 0 else xs + [item]
        elif brk == "a change outside the ask":
            i = rng.choice([i for i in range(len(ys)) if i != p])
            ys[i] = _other(rng, lang, {ys[i], item})
        elif brk == "an item lost":
            ys.pop(rng.choice([i for i in range(len(ys)) if i != p]))
        return {**g("in", val=xs), **g("out", val=ys), "arg_item": item, "arg_place": p}
    if variant == "delete":
        item = rng.choice(xs)
        xs.insert(rng.randrange(len(xs) + 1), item)       # the item at least twice
        ys = [w for w in xs if w != item]
        if brk == "a place missed":
            ys.insert(0, item)
        elif brk == "an item lost" and ys:
            ys.pop(rng.randrange(len(ys)))
        elif brk == "a change outside the ask" and ys:
            i = rng.randrange(len(ys))
            ys[i] = _other(rng, lang, {ys[i], item})
        elif ys == [] and brk:
            ys = [item]
        return {**g("in", val=xs), **g("out", val=ys), "arg_item": item}
    if variant == "set":
        keys = words(rng, lang, len(xs), unique=True)
        name, value = rng.choice(keys), _other(rng, lang, set(xs))
        ys = [value if k == name else v for k, v in zip(keys, xs)]
        ks = list(keys)
        if brk == "the value not set":
            ys = list(xs)
        elif brk == "a change outside the ask":
            i = next(i for i, k in enumerate(keys) if k != name)
            ys[i] = _other(rng, lang, {xs[i], value})
        elif brk == "an item lost":
            i = next(i for i, k in enumerate(keys) if k != name)
            ys.pop(i)
            ks.pop(i)
        return {**g("in", key=keys, val=xs), **g("out", key=ks, val=ys), "arg_name": name, "arg_value": value}
    if variant == "clear":
        ys = {None: [], "one item left": xs[:1], "all left": list(xs)}[brk]
        return {**g("in", val=xs), **(g("out", val=ys) if ys else {"out_val": [], "n_out": 0})}
    if variant in ("copy", "same"):
        ys, orig = list(xs), list(xs)
        if brk == "the original changed":
            orig[0] = _other(rng, lang, {orig[0]})
        elif brk in ("the copy differs", "a word changed"):
            ys[-1] = _other(rng, lang, {ys[-1]})
        elif brk == "an item lost":
            ys.pop()
        d = {**g("in", val=xs), **g("out", val=ys)}
        return {**d, **g("orig", val=orig)} if variant == "copy" else d
    if variant == "merge":
        b = words(rng, lang, rng.randint(2, 6))
        ys = xs + b
        if brk == "an item lost":
            ys.pop(rng.randrange(len(ys)))
        elif brk == "an item doubled":
            ys[rng.randrange(len(ys))] = ys[0] if ys[0] != ys[-1] else _other(rng, lang, set(ys))
            if sorted(ys) == sorted(xs + b):
                ys.append(ys[0])
        elif brk == "an item added":
            ys.append(_other(rng, lang, set(ys)))
        rng.shuffle(ys)
        return {**g("in", val=xs), **g("in2", val=b), **g("out", val=ys)}
    nums = [rng.randint(0, 99) for _ in range(len(xs))]   # apply: add k to every item above / below t
    op, t, k = rng.choice(["<", ">"]), rng.randint(20, 80), rng.randint(1, 20)
    if not any((x < t) if op == "<" else (x > t) for x in nums):
        nums[0] = t - 1 if op == "<" else t + 1
    ys = [x + k if ((x < t) if op == "<" else (x > t)) else x for x in nums]
    covered = [i for i, x in enumerate(nums) if ((x < t) if op == "<" else (x > t))]
    if brk == "a place missed":
        ys[covered[0]] = nums[covered[0]]
    elif brk == "a change outside the ask":
        rest = [i for i in range(len(nums)) if i not in covered]
        i = rest[0] if rest else None
        if i is None:
            nums.append(t)
            ys.append(t + k)
        else:
            ys[i] += k
    elif brk == "off by one":
        ys[covered[-1]] += 1
    return {**g("in", val=nums), **g("out", val=ys), "arg_op": op, "arg_t": t, "arg_k": k}


def _compute(variant, rng, lang, brk, n):
    if variant == "aggregate":
        xs, fn = [rng.randint(1, 99) for _ in range(n)], rng.choice(["sum", "max", "min", "count"])
        want = {"sum": sum, "max": max, "min": min, "count": len}[fn](xs)
        got = {None: want, "off by one": want + 1, "the wrong unit": want * 100,
               "another function's result": sum(xs) + 1 if fn != "sum" else max(xs)}[brk]
        return {**g("in", val=xs), "arg_fn": fn, "arg_got": got}
    if variant == "expression":
        a, b, fn = rng.randint(1, 99), rng.randint(1, 99), rng.choice(["add", "sub", "mul"])
        want = {"add": a + b, "sub": a - b, "mul": a * b}[fn]
        return {"arg_a": a, "arg_b": b, "arg_fn": fn, "arg_got": {None: want, "off by one": want + 1,
                                                                   "the wrong unit": want * 1000}[brk]}
    if variant == "larger":
        a, b = rng.sample(range(1, 99), 2)
        want = "a" if a > b else "b"
        return {"arg_a": a, "arg_b": b, "arg_got": {None: want, "the other one": "b" if want == "a" else "a",
                                                     "equal when not": "equal"}[brk]}
    exp = words(rng, lang, n)                             # mapped: the host function's own result, item by item
    out = list(exp)
    if brk == "one field differs":
        out[0] = _other(rng, lang, {out[0]})
    elif brk == "one missing":
        out.pop()
    elif brk == "one extra":
        out.append(rng.choice(exp))
    return {**g("exp", val=exp), **g("out", val=out)}


def _verdict(variant, rng, lang, brk, n):
    if variant == "required":
        req = words(rng, lang, rng.randint(2, 5), unique=True)
        have = req + words(rng, lang, rng.randint(0, 3))
        if brk:
            have = [h for h in have if h != req[rng.randrange(len(req))]]
        rng.shuffle(have)
        return {**g("in", key=req), **g("out", key=have)}
    if variant == "tests":
        ran = rng.randint(1, 60)
        return {"arg_ran": 0 if brk == "none ran" else ran, "arg_failed": 1 if brk == "one failed" else 0}
    a = words(rng, lang, n)
    b = list(a)
    if brk == "one item differs":
        b[0] = _other(rng, lang, {b[0]})
    elif brk == "one missing":
        b.pop()
    return {**g("a", val=a), **g("b", val=b)}


def _write(verb, rng, lang, brk, n):
    text = " ".join(VR.reply(rng, "prose", lang).split()[:rng.randint(30, 80)])
    other = "fr" if lang == "en" else "en"
    if brk == "the wrong language":
        text = " ".join(VR.reply(rng, "prose", other).split()[:rng.randint(30, 80)])
    elif brk == "empty":
        text = ""
    ws = [w.lower() for w in rw.words_of(text)]
    want = lang
    d = {"arg_words": len(ws), "arg_min": 1, "arg_max": 120, "arg_lang": rw.RELATIONS["language"](text)[0],
         "arg_want_lang": want}
    if verb == "summarize":
        d.update(arg_share=30, arg_src_words=max(1, len(ws)) * 4)
    if brk == "too long":
        d["arg_max"] = max(1, len(ws) - 5)
    if ws and verb != "translate":
        subj = rng.choice(ws)
        if brk == "the subject missing":
            subj = _other(rng, lang, set(ws))
        d.update({**g("out", lower=ws), "arg_subject": subj})
    return d


def build_more(verb, variant, rng, lang, brk, n):
    if variant in ("replace", "add", "delete", "set", "clear", "copy", "same", "merge", "apply"):
        return _edit(variant, rng, lang, brk, n)
    if variant in ("aggregate", "expression", "larger", "mapped"):
        return _compute(variant, rng, lang, brk, n)
    if variant in ("required", "tests", "equal"):
        return _verdict(variant, rng, lang, brk, n)
    return _write(verb, rng, lang, brk, n)


__all__ = ["build_more"]
