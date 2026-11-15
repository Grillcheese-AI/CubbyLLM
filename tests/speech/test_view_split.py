"""Order 0073, step 4 (cubbyllm/speech/view_split.py): the reader's form. A row's target is one line a segment with
the check's description in the arm's spelling and the join tag on the check that carries the connective; it parses
back to the row's kinds, tags, forms and spans (without outer white space). Read strictly, the text is copied with
no gap and the instruction is used up; a tag or a description on a line that is not a check is refused."""
from __future__ import annotations

from cubbyllm.speech import view_split as V

TEXT = ('Write a note. If the reply has at least 3 sentences, use fewer than 40 words.\n'
        'Do not use the word "so". Stay kind.')
PIECES = [("task", "Write a note."), ("check", "If the reply has at least 3 sentences, "), ("check", "use fewer than 40 words."),
          ("check", 'Do not use the word "so".'), ("asked", "Stay kind.")]
SO = [{"col": "lower", "op": "==", "v": "so"}]
SAID = [{"how": "cond", "w": {"q": "n", "view": "sentence", "op": ">=", "v": 3}, "m": {"q": "n", "view": "sentence", "op": ">=", "v": 3}},
        {"how": "imp", "w": {"q": "n", "view": "word", "op": "<", "v": 40}, "m": {"q": "n", "view": "word", "op": "<", "v": 40}},
        {"how": "imp", "w": {"q": "any", "view": "word", "where": SO, "not": True}, "m": {"q": "none", "view": "word", "where": SO}}]


def row() -> dict:
    segs, at, part = [], 0, 0
    for kind, piece in PIECES:
        k = TEXT.index(piece, at)
        seg = {"kind": kind, "start": k, "end": k + len(piece)}
        if kind == "check":
            seg.update(part=part, how=SAID[part]["how"])
            if part == 0:
                seg.update(role="condition", join=["if", "next"])
            part += 1
        segs.append(seg)
        at = k + len(piece)
    return {"instruction": TEXT, "segments": segs, "said": SAID}


W = ('\ntask | Write a note.\ncheck if next n sentence >= 3 | If the reply has at least 3 sentences,\n'
     'check n word < 40 | use fewer than 40 words.\ncheck not any word where lower == "so" | Do not use the word "so".\n'
     'asked | Stay kind.')


def test_the_target_is_one_line_a_segment_in_the_arms_spelling():
    assert V.target(row(), "w") == W
    assert V.target(row(), "m") == W.replace("not any word", "none word")
    assert V.prompt(TEXT).endswith(f"Instruction: {TEXT}\nRead:")


def test_a_target_parses_back_to_the_rows_kinds_tags_forms_and_spans():
    r = row()
    for arm in V.ARMS:
        got = V.parse(V.target(r, arm), TEXT, arm)["lines"]
        assert [x["kind"] for x in got] == [k for k, _ in PIECES]
        assert [x["join"] for x in got] == [None, ["if", "next"], None, None, None]
        assert [x["form"] for x in got if x["kind"] == "check"] == [s[arm] for s in SAID]
        assert [(x["start"], x["end"]) for x in got] == [V.span(TEXT, s["start"], s["end"]) for s in r["segments"]]
        assert [TEXT[x["start"]:x["end"]] for x in got][1] == "If the reply has at least 3 sentences,"
    assert "error" in V.parse(W, TEXT, "m")                  # arm M's grammar has no `not`


def lines(*, drop=None, swap=None, edit=None) -> str:
    ls = W[1:].split("\n")
    if edit:
        ls[edit[0]] = edit[1]
    if swap:
        ls[swap[0]], ls[swap[1]] = ls[swap[1]], ls[swap[0]]
    return "\n" + "\n".join(x for i, x in enumerate(ls) if i != drop)


def test_read_strictly_the_text_has_no_gap_and_the_instruction_is_used_up():
    for bad, why in [(lines(drop=2), "not the instruction's next"),            # a sentence left out
                     (lines(swap=(1, 2)), "not the instruction's next"),       # text out of order
                     (lines(drop=4), "not used up"),                           # stopped early
                     (lines(edit=(1, "check if next n sentence >= 3 | If the reply has at least 3 sentences, ")),
                      "not the instruction's next"),                           # a text that ends on white space
                     (lines(edit=(0, "tasks | Write a note.")), "kind not in its set"),   # a line that is not in the form
                     (W[1:], "new line"), (W + "\n", "kind not in its set")]:
        got = V.parse(bad, TEXT, "w")
        assert why in got.get("error", ""), (bad, got)
    cut = lines(edit=(2, "check n word < 40 | use fewer than")).replace("asked | Stay", "none | 40 words.\nasked | Stay")
    assert "error" in V.parse(cut, TEXT, "w")                # the cut is fine; the lines after it are out of order
    two = W.replace('check not any word where lower == "so" | Do not use the word "so".',
                    'check not any word where lower == "so" | Do not use\nnone | the word "so".')
    assert [x["kind"] for x in V.parse(two, TEXT, "w")["lines"]] == ["task", "check", "check", "check", "none", "asked"]


def test_a_tag_or_a_description_belongs_to_a_check():
    for line in ("task if next | Write a note.", "task n word < 40 | Write a note.", "check | Write a note.",
                 "check if next | Write a note.", "check either prev n word < 40 | Write a note.", "note | Write a note."):
        assert "error" in V.parse(lines(edit=(0, line)), TEXT, "w"), line
    tagged = V.parse(lines(edit=(0, "check unless prev n word < 40 | Write a note.")), TEXT, "w")["lines"][0]
    assert tagged["join"] == ["unless", "prev"]             # the grammar takes it; the host finds no check before it


def test_read_loosely_a_gap_is_allowed_and_found():
    got = V.parse(lines(drop=2) + "\n", TEXT, "w", strict=False)["lines"]
    assert [x["kind"] for x in got] == ["task", "check", "check", "asked"]
    assert TEXT[got[1]["end"]:got[2]["start"]].strip() == "use fewer than 40 words."
    assert "error" in V.parse(lines(swap=(1, 2)), TEXT, "w", strict=False)
