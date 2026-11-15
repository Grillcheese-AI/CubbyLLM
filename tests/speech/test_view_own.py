"""Order 0073, step 4 (cubbyllm/speech/view_own.py): what a line's own text lets its description say. A number is in
the text as digits or as a word, 0 and 1 always; a position is 1, an ordinal or a number of the text, or the last;
a string is its column's own or quoted in the text, free when there is neither; a view is one the text names, any
when it names none. `Own` gives the same values with the place where the text holds each, and what it holds by a
place is what `allows` allows on the text cut there."""
from __future__ import annotations

from cubbyllm.speech import view_own as O
from cubbyllm.speech import view_split as V

EN = 'Use fewer than 40 words and begin the third sentence with "So".'
FR = "Mets au moins trois phrases et termine la deuxième ligne par « fin »."


def n(view, op, v):
    return {"q": "n", "view": view, "op": op, "v": v}


def at(view, pos, col, val, op="=="):
    return {"q": "at", "view": view, "pos": pos, "col": col, "op": op, "val": val}


def test_numbers_are_the_texts_in_digits_or_in_words():
    assert O.allows(n("word", "<", 40), EN) and not O.allows(n("word", "<", 41), EN)
    assert O.allows(n("word", "<", 0), EN) and O.allows(n("word", "<", 1), EN)              # 0 and 1 always
    assert O.allows(n("sentence", ">=", 3), FR) and not O.allows(n("sentence", ">=", 2), "Mets au moins trois phrases.")
    assert not O.allows(n("word", "<", 4), "Use fewer than 40 words.")                      # a part of a number is not it


def test_positions_are_the_first_an_ordinal_a_number_or_the_last():
    assert O.allows(at("sentence", 3, "first_word", "So"), EN) and not O.allows(at("sentence", 2, "first_word", "So"), EN)
    assert O.allows(at("sentence", 1, "first_word", "So"), EN) and O.allows(at("sentence", -1, "first_word", "So"), EN)
    assert O.allows(at("line", 2, "first_word", "fin"), FR)                                  # "deuxième", accents folded
    assert O.allows(at("sentence", 40, "first_word", "So"), EN)                              # a number of the text


def test_strings_are_the_columns_own_or_quoted_and_free_when_neither():
    assert not O.allows(at("sentence", 3, "first_word", "so"), EN)                           # compared as written
    assert O.allows(at("sentence", 3, "end_mark", "?"), EN) and not O.allows(at("sentence", 3, "end_mark", "So!"), EN)
    word = {"q": "any", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}
    assert O.allows(word, EN) and O.allows(dict(word, where=[{"col": "lower", "op": "==", "v": "So"}]), EN)
    assert not O.allows(dict(word, where=[{"col": "lower", "op": "==", "v": "then"}]), EN)
    assert O.allows(dict(word, where=[{"col": "lower", "op": "==", "v": "then"}]), "Use the word then.")   # nothing quoted
    assert O.allows({"q": "any", "view": "char", "where": [{"col": "char", "op": "==", "v": '"'}]}, 'Use the character """.')


def test_a_view_is_one_the_text_names_and_any_when_it_names_none():
    assert not O.allows(n("paragraph", "<", 40), EN) and O.allows(n("sentence", "<", 40), EN)
    assert O.allows(n("paragraph", "<", 40), "Keep it under 40.")


def test_own_gives_each_value_with_the_place_the_text_holds_it():
    text = "Intro.\n" + EN + "  \nMore 7."
    own = O.Own(text, 7)
    assert text[own.start:own.stop] == EN                                                    # to the line break, no outer space
    after = lambda s: text.index(s) + len(s)                                                 # noqa: E731
    assert own.numbers == {0: 7, 1: 7, 40: after("40")} and 7 not in own.numbers             # the next line's is not its own
    assert own.positions == {1: 7, 3: after("third"), 40: after("40")}
    assert own.views == {"word": after("word"), "sentence": after("sentence")} and own.view_cap == after("word")
    assert own.quotes == {"So": after('"So"')} and own.quote_cap == after('"So"')
    assert own.strings("first_word") == {"So": after('"So"')}
    assert own.strings("lower") == {"So": after('"So"'), "so": after('"So"')}
    assert own.strings("end_mark") == {".": 7, "?": 7, "!": 7, "none": 7, "So": after('"So"')}
    assert own.inside == set(range(text.index('"So"') + 1, after('"So"')))
    fr = O.Own(FR, 0)
    assert fr.numbers[3] == FR.index("trois") + 5 and fr.positions[2] == FR.index("deuxième") + 8
    assert fr.quotes == {"fin": FR.index("»") + 1} and fr.views["line"] == FR.index("ligne") + 5


def test_what_own_holds_by_a_place_is_allowed_on_the_text_cut_there():
    for text in (EN, FR, 'End every line with "." and use at most twelve items, 3 of them long.'):
        own = O.Own(text, 0)
        for p in range(1, len(text) + 1):
            if p in own.inside or text[p - 1].isspace():
                continue
            cut = text[:p]
            view = next((v for v, e in own.views.items() if e <= p), "word")      # one the cut names, or any
            col = "lower" if view == "word" else "first_word"                     # a string column with no values of its own
            assert all(O.allows(n(view, ">", v), cut) for v, e in own.numbers.items() if e <= p), cut
            assert all(O.allows(at(view, k, col, s), cut) for k, ke in own.positions.items() if ke <= p
                       for s, e in own.quotes.items() if e <= p), cut
            assert O.allows(at(view, 1, col, "anything"), cut) == (own.quote_cap is None or own.quote_cap > p), cut


def test_parse_holds_a_description_to_its_text_when_asked():
    answer = '\ncheck n word < 40 | Use fewer than 40 words and begin the third sentence with "So".'
    assert "lines" in V.parse(answer, EN, "m", own_text=True)
    wrong = answer.replace("< 40", "< 14")
    assert "lines" in V.parse(wrong, EN, "m") and "says what its text does not" in V.parse(wrong, EN, "m", own_text=True)["error"]
