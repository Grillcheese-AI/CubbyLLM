"""Order 0073, step 4 (cubbyllm/speech/view_tokens.py): the machine in token ids. The ids allowed from a state are
the ones whose whole piece the machine takes, looked up in a prefix tree; a piece that prints nothing is never
allowed; the end token only on a whole answer; inside a free string the pieces kept by length give the same set as
walking every piece; an answer records how it ended."""
from __future__ import annotations

import pytest

from cubbyllm.speech import view_tokens as T
from cubbyllm.speech.view_machine import START, Machine

TEXT = 'Write a note. Use fewer than 40 words.\nDo not use the word "so".'
GOLD = ('\ntask | Write a note.\ncheck n word < 40 | Use fewer than 40 words.\n'
        'check none word where lower == "so" | Do not use the word "so".')
PIECES = ["", "task", " | ", "Write a note.", "check ", "check n", " word", " < ", "40", " |", " Use", " fewer than",
          " 40 words.", "none ", "word where ", "lower == ", 'so"', ' | Do not', " use the word ", '"so".', "\nnone",
          "so", '" | Do', "then", '\\', ' "', "Use fewer than 40 words.\ncheck "] + sorted(set(TEXT + GOLD))
EOS = len(PIECES)
IDS = {p: i for i, p in enumerate(PIECES)}


def vocab() -> T.Vocab:
    return T.Vocab(dict(enumerate(PIECES)))


def spell(text: str) -> list[int]:
    """`text` in the longest pieces, left to right."""
    out, at = [], 0
    while at < len(text):
        p = max((p for p in PIECES if p and text.startswith(p, at)), key=len)
        out.append(IDS[p])
        at += len(p)
    return out


def test_an_answer_is_written_in_ids_and_ends_finished():
    v = vocab()
    for own in (False, True):
        r = T.Reader(Machine(TEXT, "m", own), v, EOS)
        for i in spell(GOLD):
            got = r.allowed()
            assert i in got and IDS[""] not in got and EOS not in got      # nothing printed is never allowed; no early end
            r.push(i)
        assert r.allowed() == {EOS} and r.end is None                      # the instruction is used up: only the end
        r.push(EOS)
        assert r.end == "finished" and r.n == len(spell(GOLD)) + 1 and r.allowed() == frozenset()


def test_allowed_ids_are_the_pieces_the_machine_takes_whole():
    v, m = vocab(), Machine(TEXT, "m")
    assert T.allowed(m, v, START) == {IDS["\n"], IDS["\nnone"], IDS[" "]}      # the trainer's one space may come first
    line = m.run("\ntask | Write a note.\n")
    assert T.allowed(m, v, line) == {IDS[p] for p in ("check ", "check n", "c", "t", "task", "a", "n", "none ")}
    text = m.run("\ntask | Write a note.\ncheck n word < 40 | ")
    assert T.allowed(m, v, text) == {IDS["U"], IDS["Use fewer than 40 words.\ncheck "]}   # a piece may run into the next line
    for st in (START, line, text):
        assert T.allowed(m, v, st) == {i for i, p in enumerate(PIECES) if p and m.run(p, st) is not None}


def test_inside_a_free_string_the_pieces_kept_by_length_give_the_same_ids():
    v, m = vocab(), Machine(TEXT, "m")
    for typed in ("", "s", "x" * 38, "x" * 40):
        st = m.run('\ntask | Write a note.\ncheck any word where lower == "' + typed)
        assert st[0] == "str" and T.allowed(m, v, st) == frozenset(T.walk(m, st, v.root))
    assert IDS["then"] in T.allowed(m, v, m.run('\ntask | Write a note.\ncheck any word where lower == "' + "x" * 36))
    assert IDS["then"] not in T.allowed(m, v, m.run('\ntask | Write a note.\ncheck any word where lower == "' + "x" * 37))
    after = m.run('\ntask | Write a note.\ncheck any word where lower == "a\\')
    assert T.allowed(m, v, after) == {IDS['"'], IDS["\\"], IDS['" | Do']}   # after a backslash: a quote or a backslash, in the string


def test_an_answer_with_no_id_left_ends_in_error_and_a_stopped_one_at_its_limit():
    poor = T.Vocab({i: p for i, p in enumerate(PIECES) if p != "W" and not p.startswith("Write")})
    r = T.Reader(Machine(TEXT, "m"), poor, EOS)
    for i in spell("\ntask | "):
        r.push(i)
    assert r.allowed() == frozenset() and r.end == "error"                 # no piece prints the "W" the text needs
    with pytest.raises(ValueError):
        T.Reader(Machine(TEXT, "m"), vocab(), EOS).push(IDS["task"])       # not allowed at the start
    r = T.Reader(Machine(TEXT, "m"), vocab(), EOS)
    r.push(IDS["\n"])
    r.stop("limit")
    assert r.end == "limit" and r.allowed() == frozenset()
