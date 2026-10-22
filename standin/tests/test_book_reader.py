"""The book reader's contract (standin/book_reader.py, order 0013) and its training labels
(standin/data/build_reader_sft.py): a line is kept only with a relation of the set and a value the sentence
writes; dates become the store's form; a fact reads as the wiki world's sentence whoever read it; a label is a
pair the sentence could state (the same date, close names, the relation's verb), held out by book.
Run: python -m pytest standin/tests/test_book_reader.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import book_reader as br  # noqa: E402
import build_reader_sft as rs  # noqa: E402
from cubbyllm.reasoning.planner import Triple  # noqa: E402

SENT = "Napoleon was born in Ajaccio on 15 August 1769."
SET = {"BORN_IN", "BORN_ON", "DIED_IN"}


def test_a_line_needs_a_relation_of_the_set_and_a_value_the_sentence_writes():
    out = br.parse("BORN_IN | Ajaccio\nFOUNDED_BY | Ajaccio\nBORN_ON | 15 August 1769\nDIED_IN | Paris\nBORN_IN Ajaccio\n"
                   "BORN_IN | Ajaccio", SENT, SET)
    assert out == [("BORN_IN", "Ajaccio"), ("BORN_ON", "15 August 1769")]   # unknown relation, unwritten value, no bar, twice
    assert br.parse("NONE", SENT, SET) == [] and br.parse("", SENT, SET) == []
    assert br.answer(out) == "BORN_IN | Ajaccio\nBORN_ON | 15 August 1769" and br.answer([]) == "NONE"
    assert br.prompt("  Napoleon was\n born. ", "Napoleon") == "Sentence: Napoleon was born.\nSubject: Napoleon\nFacts:"


def test_the_reader_gives_store_values_and_reads_a_sentence_once():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return "BORN_IN | Ajaccio\nBORN_ON | 15 August 1769"
    reader = br.BookReader(generate, SET)
    assert reader.read(SENT, "Napoleon") == [("BORN_IN", "Ajaccio"), ("BORN_ON", "1769-08-15")]
    assert reader.read(SENT, "Napoleon") and len(calls) == 1 and reader.calls == 1
    assert br.date_value("742") == "742" and br.date_value("480 BC") == "480 BC" and br.date_value("Ajaccio") is None


def test_a_fact_is_the_wiki_worlds_sentence_whoever_read_it():
    assert br.store_facts("Napoleon Bonaparte", "BORN_IN", "Ajaccio") == ["Ajaccio is the place of birth of Napoleon Bonaparte"]
    assert br.store_facts("Robinson Crusoe", "AUTHORED_BY", "Daniel Defoe") == ["Daniel Defoe is the author of Robinson Crusoe",
                                                                              "Robinson Crusoe is the work of Daniel Defoe"]
    assert br.store_facts("Kush", "capital", "Kerma") == ["Kerma is the capital of Kush"]     # a frame label without a code
    assert br.frame_fact(Triple(obj="Ajaccio", rel="place of birth", subj="Napoleon")) == ("BORN_IN", "Ajaccio")
    assert br.frame_fact(Triple(obj="Kerma", rel="capital", subj="Kush")) == ("capital", "Kerma")


def test_a_label_is_a_pair_the_sentence_could_state():
    m = rs.date_span("Desargues was born on 21 February 1591 in Lyon.", "1591-02-21")
    assert m is not None and m.group(0).strip() == "21 February 1591"
    assert rs.date_span("Desargues was born on 22 February 1591.", "1591-02-21") is None          # another day
    assert rs.date_span("Desargues was born in 1591.", "1591-02-21").group(0).strip() == "1591"     # the year alone agrees
    assert rs.date_span("He saw 480 BC.", "480") is None and rs.date_span("in 480 BC", "-480") is not None
    s = "Poseidon, son of Rhea and god of the sea, rose."
    a, b = rs.find(s, "Rhea"), rs.find(s, "poseidon")
    assert rs.surface(s, "poseidon") == "Poseidon" and rs.gap(s, a.span(), b.span()) == 2
    assert rs.JUNK.search("See Lakoff, Metaphors (1980), pp. 4-5.") and rs.JUNK.search("Burgundy, 25, 30, 31")
    assert not rs.JUNK.search("Rome was founded by Romulus in 753 BC.")
    assert {rs.split_of(h, 0.1) for h in ("00000000ab", "ffffffffab")} == {"test", "train"}
    assert rs.split_of("12345678", 0.1) == rs.split_of("12345678", 0.1)
