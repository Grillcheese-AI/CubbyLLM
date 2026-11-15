"""Order 0073, step 4 (cubbyllm/speech/view_account.py): the host's accounting. A read's checks are numbered in line
order, folded and joined into the program by their tags; a sentence no line holds is flagged; a tag with no check
beside it, or a description that does not compile, makes its part asked; a pair with an asked member is asked
whole, and so is a check a tag that cannot be resolved points at. Nothing is dropped silently."""
from __future__ import annotations

from cubbyllm.speech import view_account as A
from cubbyllm.speech import view_join as VJ
from cubbyllm.speech import view_split as V

TEXT = ('Write a note. If the reply has at least 3 sentences, use fewer than 40 words.\n'
        'Do not use the word "so". Stay kind. End every sentence with a period, unless it is the last.')
LINES = ["task | Write a note.",
         "check if next n sentence >= 3 | If the reply has at least 3 sentences,",
         "check n word < 40 | use fewer than 40 words.",
         'check not any word where lower == "so" | Do not use the word "so".',
         "asked | Stay kind.",
         'check all sentence where end_mark == "." | End every sentence with a period,',
         "check unless prev n sentence == 1 | unless it is the last."]
SENTENCES = {"q": "n", "view": "sentence", "op": ">=", "v": 3}
WORDS = {"q": "n", "view": "word", "op": "<", "v": 40}
NO_SO = {"q": "none", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}
PERIODS = {"q": "all", "view": "sentence", "where": [{"col": "end_mark", "op": "==", "v": "."}]}
ONE = {"q": "n", "view": "sentence", "op": "==", "v": 1}


def read(lines=LINES, strict=True) -> dict:
    got = V.parse("\n" + "\n".join(lines), TEXT, "w", strict=strict)
    assert "lines" in got, got
    return A.account(got["lines"], TEXT)


def edit(i: int, line: str | None) -> list[str]:
    return [x for j, x in enumerate(LINES[:i] + [line] + LINES[i + 1:]) if x is not None]


def test_a_read_is_numbered_folded_and_joined_by_its_tags():
    got = read()
    assert [c["form"] for c in got["checks"]] == got["parts"] == [SENTENCES, WORDS, NO_SO, PERIODS, ONE]   # arm W's `not` folded
    assert [c["line"] for c in got["checks"]] == [1, 2, 3, 5, 6] and [c["part"] for c in got["checks"]] == [0, 1, 2, 3, 4]
    assert got["pairs"] == [("if", 0, 1), ("unless", 3, 4)] and got["groups"] == [("if", 0, 1), ("one", 2), ("unless", 3, 4)]
    assert got["program"] == VJ.join(got["parts"], got["groups"]) and not got["asked"] and not got["flagged"]


def test_a_sentence_no_line_holds_is_flagged():
    got = read(edit(4, None), strict=False)
    assert [TEXT[a:b] for a, b in got["flagged"]] == ["Stay kind."] and len(got["parts"]) == 5
    assert [TEXT[a:b] for a, b in read(LINES[:3], strict=False)["flagged"]] == [TEXT[TEXT.index("Do not"):]]
    assert [TEXT[a:b] for a, b in A.flagged([], TEXT)] == [TEXT]


def test_a_tag_with_no_check_beside_it_makes_its_part_asked():
    got = read(edit(3, 'check if next not any word where lower == "so" | Do not use the word "so".'))   # the next line is "asked"
    assert list(got["asked"]) == [2] and "no check beside it" in got["asked"][2]
    assert got["parts"] == [SENTENCES, WORDS, PERIODS, ONE] and got["groups"] == [("if", 0, 1), ("unless", 2, 3)]
    assert got["checks"][2]["asked"] and got["checks"][2]["form"] == NO_SO                 # kept, and said to be asked
    first = read(edit(1, "check if prev n sentence >= 3 | If the reply has at least 3 sentences,"))       # a task before it
    assert list(first["asked"]) == [0] and first["groups"] == [("one", 0), ("one", 1), ("unless", 2, 3)]


def test_a_description_that_does_not_compile_makes_its_pair_asked_whole():
    two = 'check not all sentence where end_mark == "." and n_word > 3 | End every sentence with a period,'
    got = read(edit(5, two))
    assert "does not compile" in got["asked"][3] and "tied to a part that is asked" in got["asked"][4]
    assert got["parts"] == [SENTENCES, WORDS, NO_SO] and got["groups"] == [("if", 0, 1), ("one", 2)]
    assert got["checks"][3]["form"] is None and got["pairs"] == [("if", 0, 1), ("unless", 3, 4)]
    assert read(edit(5, two.replace("not all", "all")))["asked"] == {}                     # every sentence, both: it compiles


def test_a_tag_on_a_check_already_in_a_pair_asks_all_it_touches():
    got = read(edit(2, "check either next n word < 40 | use fewer than 40 words."))       # "if" then "either": three in a row
    assert set(got["asked"]) == {0, 1, 2} and "already in a pair" in got["asked"][1]
    assert got["parts"] == [PERIODS, ONE] and got["groups"] == [("unless", 0, 1)]
    assert read(LINES[:1] + ["none | " + TEXT[14:TEXT.index("\n")]] + ["none | " + TEXT[TEXT.index("Do not"):]])["program"] is None
