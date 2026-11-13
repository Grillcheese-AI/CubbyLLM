"""Order 0067, the second reader's step 1: the reply's primitive views agree with the reply world's measurements, give
each item's place in the text, bind every column even on an empty reply, and run in the VM as lists a program reads
(the VM itself never takes a string apart: cubelang's P6)."""
from __future__ import annotations

import re

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.speech import reply_views as V
from cubbyllm.speech import reply_world as rw

REPLIES = [
    "",
    "Hello WORLD. Is it \"fine\"?\n\n- one item\n  2) two\n***\nLast para!",
    "Réponds vite. L'été est là !\nÇa marche ?",
    "```json\n{\"a\": {\"b\": [1, \"x\"]}, \"c\": null, \"d\": true}\n```",
    "ALL CAPS HERE\n\n\nthen lower\n* a\n* b\n10. ten",
]


def m(text, rel):
    return int(rw.RELATIONS[rel](text)[0])


@pytest.mark.parametrize("text", REPLIES)
def test_the_views_agree_with_the_reply_worlds_counts(text):
    v = V.views(text)
    assert len(v["word"]) == m(text, "word count")
    assert len(v["sentence"]) == m(text, "sentence count")
    assert len(v["line"]) == m(text, "line count")
    assert len(v["item"]) == m(text, "list item count")
    assert [w["lower"] for w in v["word"]] == [w.lower() for w in rw.words_of(text)]


@pytest.mark.parametrize("text", REPLIES)
def test_every_item_is_found_where_it_says(text):
    v = V.views(text)
    assert "".join(c["char"] for c in v["char"]) == text
    for w in v["word"]:
        assert text[w["start"]:w["end"]] == w["word"] and w["first"] == w["word"][0] and w["len"] == len(w["word"])
    for name, key in (("sentence", "sentence"), ("line", "line"), ("paragraph", "paragraph")):
        for r in v[name]:
            assert text[r["start"]:r["end"]] == r[key]
    for w in v["word"]:                                  # a word sits inside the sentence and the paragraph it names
        for name in ("sentence", "paragraph", "line"):
            if w[name] >= 0:
                r = v[name][w[name]]
                assert r["start"] <= w["start"] < r["end"]


def test_lists_json_and_case():
    v = V.views(REPLIES[1])
    assert [(r["marker"], r["level"], r["text"]) for r in v["item"]] == [("-", 0, "one item"), ("number", 2, "two")]
    assert [r["end_mark"] for r in v["sentence"]][:2] == [".", "?"]
    assert [w["case"] for w in v["word"][:3]] == ["title", "upper", "title"]
    j = {r["path"]: (r["type"], r["value"]) for r in V.views(REPLIES[3])["json"]}
    assert j == {"a": ("object", ""), "a.b": ("array", ""), "a.b[0]": ("number", "1"), "a.b[1]": ("string", "x"),
                 "c": ("null", "null"), "d": ("bool", "true")}
    assert V.views("plain text")["json"] == []


def test_an_empty_reply_binds_every_column():
    bound = " ".join(V.literals(""))
    for name, cols in V.COLUMNS.items():
        assert f"n_{name} = 0" in bound
        for c in cols:
            assert f"let {name}_{c} = [];" in bound


@pytest.fixture(scope="module")
def vm():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    return cc.CubelangSession()


def run(vm, text, body, names):
    lets = "\n        ".join(V.literals(text, names))
    src = (f"program P implements ISolve {{\n    public function solve(mention: str): number {{\n        {lets}\n"
           f"        create r: number; assign r = 0;\n        {body}\n    }}\n}}\n")
    return int(float(vm.run(src, fn="solve")["result"]))


@pytest.mark.parametrize("text", REPLIES)
def test_a_program_reads_the_views_in_the_vm(vm, text):
    upper = sum(1 for w in V.views(text)["word"] if w["case"] == "upper")
    body = "for (let c of word_case) { if (c == \"upper\") { add r, 1; } } return r;"
    assert run(vm, text, body, ["word"]) == upper
    t_words = sum(1 for w in V.views(text)["word"] if w["first_lower"] == "t")
    body = ("create i: number; assign i = 0; while (i < n_word) { let f = word_first_lower[i]; "
            "if (f == \"t\") { add r, 1; } add i, 1; } return r;")
    assert run(vm, text, body, ["word"]) == t_words
    questions = sum(1 for s in V.views(text)["sentence"] if s["end_mark"] == "?")
    body = "for (let e of sentence_end_mark) { if (e == \"?\") { add r, 1; } } return r;"
    assert run(vm, text, body, ["sentence"]) == questions


def test_quotes_backslashes_and_newlines_survive_the_literal(vm):
    text = 'He said "no" \\ then\nleft.'
    body = 'let s = line_line[0]; if (s == "He said \\"no\\" \\\\ then") { assign r = 1; } return r;'
    assert re.search(r"\\\\", " ".join(V.literals(text, ["line"])))
    assert run(vm, text, body, ["line"]) == 1
