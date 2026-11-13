"""cubbyllm/speech/reply_views.py -- the reply's primitive views (order 0067, the second reader's step 1).

The plan reader fills one of about twenty fixed measurements (`reply_world.RELATIONS`). The second reader writes the
check in CubeLang instead, composing a few views of the reply: its characters, words, sentences, lines, paragraphs,
list items and JSON keys and values, each a list in reading order with positions. The existing measurements stay.

**Every value is computed here, in Python.** The VM build cannot index a string (``"hello"[1]`` is null) and compares
null equal to any string (cubelang's P6, fixed there later): a program never takes a string apart; it reads lists of
values computed here. A view is a table: one row per item, its columns said as parallel lists (`literals`), so a
program reads ``word_lower[i]``, ``word_first[i]``, ``word_len[i]`` with an index it never takes past ``n_word``.

Views (`views(text)` -> {view: [row, ...]}), each row with ``i`` (its place) and ``start`` / ``end`` (its characters
in the reply, so ``text[start:end]`` is the item as written):
- ``char``: every character; ``kind`` (letter, digit, space, punct), ``case`` (upper, lower, none);
- ``word``: `reply_world.words_of`'s words; ``lower``, ``len``, ``first``, ``last`` (characters, as written, and
  ``first_lower`` / ``last_lower``),
  ``case`` (upper, lower, title, mixed, none), ``sentence`` / ``line`` / ``paragraph`` (the row it sits in);
- ``sentence``: `reply_world.sentences_of`'s sentences; ``n_word``, ``first_word``, ``last_word`` (lowercased),
  ``end_mark`` (``.``, ``!``, ``?``, or none);
- ``line``: the non-empty lines; ``n_word``, ``first_word``, ``indent`` (leading spaces), ``item`` (a list marker);
- ``paragraph``: the parts between blank lines or ``***``; ``n_word``, ``n_sentence``, ``first_word``;
- ``item``: the lines a list marker opens; ``marker`` (``-``, ``*``, ``•``, or ``1.`` / ``1)`` as ``number``),
  ``level`` (indent), ``text`` (after the marker), ``n_word``;
- ``json``: when the whole reply (fences off) is JSON: one row per key and value, ``path`` (``a.b[0]``), ``key``,
  ``value`` (as a string), ``type`` (string, number, bool, null, object, array).

`literals(text, names)` -> CubeLang ``let`` lines binding the views named (each column a list, plus ``n_<view>``);
`statements(text, names)` -> the record's statements ("<value> is the <column> of <view> <i> of the reply").

Wiring: STANDALONE (the second reader's generator, step 2, reads it; until then its tests).
"""
from __future__ import annotations

import json
import re

from cubbyllm.speech import reply_world as rw

__wiring__ = "STANDALONE"

_WORD = re.compile(r"[\w'’-]+", re.U)
_ITEM = re.compile(r"^(\s*)([-*•]|\d+[.)])\s+(\S.*)$")
_PARA = re.compile(r"\n\s*\n|\s*\*\*\*\s*")
COLUMNS = {                                              # every view's columns, bound even when it has no rows
    "char": ("start", "end", "char", "kind", "case"),
    "word": ("start", "end", "word", "lower", "len", "first", "last", "first_lower", "last_lower", "case", "sentence",
             "line", "paragraph"),
    "sentence": ("start", "end", "sentence", "n_word", "first_word", "last_word", "end_mark"),
    "line": ("start", "end", "line", "n_word", "first_word", "indent", "item"),
    "paragraph": ("start", "end", "paragraph", "n_word", "n_sentence", "first_word"),
    "item": ("start", "end", "marker", "level", "text", "n_word"),
    "json": ("path", "key", "value", "type"),
}


def _kind(c: str) -> str:
    return "letter" if c.isalpha() else "digit" if c.isdigit() else "space" if c.isspace() else "punct"


def _case(s: str) -> str:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper" if len(letters) > 1 or not s[:1].isalpha() else "title"
    if all(c.islower() for c in letters):
        return "lower"
    return "title" if letters[0].isupper() and all(c.islower() for c in letters[1:]) else "mixed"


def _spans(text: str, parts: list[str], start: int = 0) -> list[tuple[int, int]]:
    """Each part's (start, end) in `text`, found in order."""
    out, at = [], start
    for p in parts:
        k = text.find(p, at)
        k = at if k < 0 else k
        out.append((k, k + len(p)))
        at = k + len(p)
    return out


def _inside(spans: list[tuple[int, int]], pos: int) -> int:
    return next((i for i, (s, e) in enumerate(spans) if s <= pos < e), -1)


def _words_in(text: str) -> list[str]:
    return [w.lower() for w in rw.words_of(text)]


def _json_rows(text: str) -> list[dict]:
    body = (text or "").strip()
    body = re.sub(r"^```(?:json|Json|JSON)?\s*|\s*```$", "", body)
    try:
        obj = json.loads(body)
    except ValueError:
        return []
    rows: list[dict] = []

    def walk(v, path, key):
        typ = ("bool" if isinstance(v, bool) else "number" if isinstance(v, (int, float)) else "string" if isinstance(v, str)
               else "null" if v is None else "object" if isinstance(v, dict) else "array")
        if path:
            rows.append({"path": path, "key": key, "type": typ,
                         "value": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if typ != "object" and typ != "array" else ""})
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else str(k), str(k))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]", key)
    walk(obj, "", "")
    for i, r in enumerate(rows):
        r["i"] = i
    return rows


def views(text: str) -> dict[str, list[dict]]:
    text = text or ""
    out: dict[str, list[dict]] = {"char": [{"i": i, "start": i, "end": i + 1, "char": c, "kind": _kind(c),
                                            "case": "upper" if c.isupper() else "lower" if c.islower() else "none"}
                                           for i, c in enumerate(text)]}
    sents = rw.sentences_of(text)
    s_spans = _spans(text, sents)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    l_spans = _spans(text, lines)
    paras = [p.strip() for p in _PARA.split(text) if p.strip()]
    p_spans = _spans(text, paras)
    out["word"] = []
    for i, m in enumerate(_WORD.finditer(text)):
        w = m.group(0)
        out["word"].append({"i": i, "start": m.start(), "end": m.end(), "word": w, "lower": w.lower(), "len": len(w),
                            "first": w[0], "last": w[-1], "first_lower": w[0].lower(), "last_lower": w[-1].lower(), "case": _case(w), "sentence": _inside(s_spans, m.start()),
                            "line": _inside(l_spans, m.start()), "paragraph": _inside(p_spans, m.start())})
    out["sentence"] = [{"i": i, "start": s, "end": e, "sentence": t, "n_word": len(_words_in(t)),
                        "first_word": (_words_in(t) or ["none"])[0], "last_word": (_words_in(t) or ["none"])[-1],
                        "end_mark": t.rstrip()[-1] if t.rstrip()[-1:] in ".!?" else "none"}
                       for i, (t, (s, e)) in enumerate(zip(sents, s_spans))]
    out["line"] = [{"i": i, "start": s, "end": e, "line": t, "n_word": len(_words_in(t)),
                    "first_word": (_words_in(t) or ["none"])[0], "indent": len(t) - len(t.lstrip(" ")),
                    "item": "yes" if _ITEM.match(t) else "no"} for i, (t, (s, e)) in enumerate(zip(lines, l_spans))]
    out["paragraph"] = [{"i": i, "start": s, "end": e, "paragraph": t, "n_word": len(_words_in(t)),
                         "n_sentence": len(rw.sentences_of(t)), "first_word": (_words_in(t) or ["none"])[0]}
                        for i, (t, (s, e)) in enumerate(zip(paras, p_spans))]
    out["item"] = []
    for t, (s, e) in zip(lines, l_spans):
        m = _ITEM.match(t)
        if m:
            mk = m.group(2)
            out["item"].append({"i": len(out["item"]), "start": s, "end": e, "marker": "number" if mk[0].isdigit() else mk,
                                "level": len(m.group(1)), "text": m.group(3), "n_word": len(_words_in(m.group(3)))})
    out["json"] = _json_rows(text)
    return out


def _lit(v) -> str:
    if isinstance(v, bool):
        return '"yes"' if v else '"no"'
    if isinstance(v, int):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"'


def literals(text: str, names=None) -> list[str]:
    """CubeLang lines binding the views named (all when None): ``let <view>_<column> = [...];`` and ``n_<view>``."""
    vs = views(text)
    lines = []
    for name, rows in vs.items():
        if names is not None and name not in names:
            continue
        lines.append(f"create n_{name}: number; assign n_{name} = {len(rows)};")
        for c in COLUMNS[name]:
            lines.append(f"let {name}_{c} = [{', '.join(_lit(r[c]) for r in rows)}];")
    return lines


def statements(text: str, names=None) -> list[str]:
    """The record's statements: "<value> is the <column> of <view> <i> of the reply"."""
    out = []
    for name, rows in views(text).items():
        if names is None or name in names:
            out += [f"{json.dumps(r[c], ensure_ascii=False)} is the {c} of {name} {r['i']} of {rw.SUBJ}"
                    for r in rows for c in r if c not in ("i", "start", "end")]
    return out


__all__ = ["COLUMNS", "literals", "statements", "views"]
