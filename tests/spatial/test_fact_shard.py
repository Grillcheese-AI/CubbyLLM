"""The fact shard (validation/make_fact_shard.py): a fact's address is the subject's cell plus
the relation's step, a name is a span up the z axis, the object's pieces and the question's
``here`` and answer pieces are exactly what is placed, and the control places nothing. Names in
readable words (``--names v2``, order 0011): relations, entities and event ids each render as
specified, no question doubles a word over all 2,902 relations, and without v2 every name renders
as it did."""
from __future__ import annotations

import pathlib as _pathlib  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[2]))      # the repo root, for:
from cubbyllm.local_paths import expand as _lp  # noqa: E402 -- local folders by name (order 0030)

import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "validation"))

from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from make_fact_shard import STEP, WIKI, address, cell, doubled, document, question, render, step, words  # noqa: E402

TOKENIZER = os.path.join(os.environ.get("CUBBY_MODELS", _lp(r"$CUBBY_MODELS")), "base450m", "grilly", "tokenizer.json")


@pytest.fixture(scope="module")
def tk():
    if not os.path.exists(TOKENIZER):
        pytest.skip("no bbpe128k tokenizer here")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


def test_names_become_words_and_addresses_are_fixed_and_separated():
    assert words("KeikoMatsuzaka") == "Keiko Matsuzaka" and words("WORKS_FOR") == "works for" and words("AB") == "ab"
    assert cell("Daiei") == cell("Daiei") and cell("Daiei") != cell("Daiei2")
    assert all(4096 <= v < 4096 + 1024 for v in cell("Daiei"))
    assert step(1)[0] - step(0)[0] == STEP and step(54)[1] - step(0)[1] == STEP   # a grid the kernel separates
    a0, a1 = address("Daiei", 7, 0), address("Daiei", 7, 1)
    assert a1 == (a0[0], a0[1], a0[2] + STEP)                                     # the next piece, up z


def test_a_document_places_the_objects_pieces_and_the_question(tk):
    rels = {"WORKS_FOR": 0, "HAS_PART": 1, "PERFORMED_BY": 2}
    facts = [("KeikoMatsuzaka", "WORKS_FOR", "Daiei"), ("Leadership", "HAS_PART", "PowerConcept"),
             ("InugamikeNoIchizoku", "PERFORMED_BY", "KeikoMatsuzaka")]
    qa = document(facts, rels, random.Random(0), 3)
    assert {q for q, _a, _at in qa} == {f"What is the {words(r)} of {words(s)}?" for s, r, _o in facts}
    by_q = {f"What is the {words(r)} of {words(s)}?": address(s, rels[r]) for s, r, _o in facts}
    assert all(at == by_q[q] for q, _a, at in qa)                                  # the question carries the fact's address
    eos = tk.token_to_id("</s>")
    ids, places, roles = render(tk, facts, rels, qa, "places", eos, 200)
    assert ((places[:, 0] != NO_PLACE) == (roles != 0)).all()
    assert ids.shape == (200,) and places.shape == (200, 3) and ids[-1] == eos
    placed = [i for i in range(200) if places[i, 0] != NO_PLACE]
    text = tk.decode(ids.tolist(), skip_special_tokens=True)
    assert text.startswith("Facts:\nKeiko Matsuzaka works for Daiei.\n") and "? here\nAnswer: " in text
    # the objects' pieces: piece i of "Daiei" at the fact's address + STEP * i in z
    o = tk.encode("Daiei", add_special_tokens=False).ids
    at = next(i for i in placed if ids[i] == o[0])
    a0 = address("KeikoMatsuzaka", 0)
    for i in range(len(o)):
        assert ids[at + i] == o[i] and tuple(int(v) for v in places[at + i]) == (a0[0], a0[1], a0[2] + STEP * i)
        assert roles[at + i] == 1                                       # a fact's object writes into the scene
    # every placed token is an object's piece, a question's here, or an answer's piece
    here = [i for i in placed if tk.decode([int(ids[i])]).strip() == "here"]
    assert len(here) == 3 and all(roles[i] == 2 for i in here)          # a question asks; it does not write
    for i in here:                                                      # here asks for piece 0; the answer's piece j asks for j + 1
        a = tuple(int(v) for v in places[i])
        j = i + 1
        while places[j, 0] == NO_PLACE:                                 # "\nAnswer: " is not placed
            j += 1
        ans = []
        while places[j, 0] != NO_PLACE:
            ans.append(tuple(int(v) for v in places[j]))
            j += 1
        assert ans == [(a[0], a[1], a[2] + STEP * (n + 1)) for n in range(len(ans))]
        assert all(roles[k] == 2 for k in range(i + 1, j) if places[k, 0] != NO_PLACE)   # the answer's pieces ask too
    ids_t, places_t, roles_t = render(tk, facts, rels, qa, "text", eos, 200)
    assert not roles_t.any()
    assert not (places_t[:, 0] != NO_PLACE).any() and "here" not in tk.decode(ids_t.tolist(), skip_special_tokens=True)
    with pytest.raises(ValueError):
        render(tk, facts * 20, rels, [], "places", eos, 64)


def test_far_fills_the_gap_with_text_and_lands_the_questions_at_the_end(tk):
    class Stub:                                                          # real text stands in: a counter
        def take(self, n):
            return [1000 + i % 50 for i in range(n)]
    rels = {"WORKS_FOR": 0, "HAS_PART": 1}
    facts = [("KeikoMatsuzaka", "WORKS_FOR", "Daiei"), ("Leadership", "HAS_PART", "PowerConcept")]
    qa = document(facts, rels, random.Random(1), 2)
    eos = tk.token_to_id("</s>")
    ids, places, _ = render(tk, facts, rels, qa, "text", eos, 1024, Stub())
    assert ids.shape == (1024,) and ids[-1] == eos and int((ids == eos).sum()) == 1   # no padding: the gap is text
    text = tk.decode(ids.tolist(), skip_special_tokens=True)
    assert text.startswith("Facts:\n") and text.rstrip().endswith(words(qa[-1][1]))     # the questions at the end
    first_q = next(i for i in range(1024) if tk.decode(ids[i:i + 3].tolist()).startswith("Question"))
    assert first_q > 1024 - 200 and not (places[:, 0] != NO_PLACE).any()              # ~800 tokens past the facts
    near, near_places, near_roles = render(tk, facts, rels, qa, "places", eos, 1024, Stub())
    assert near.shape == (1024,) and ((near_places[:, 0] != NO_PLACE) == (near_roles != 0)).all()   # places mode fills too


def test_lead_makes_the_token_before_the_answer_ask_for_its_first_piece(tk):
    rels = {"WORKS_FOR": 0, "HAS_PART": 1}
    facts = [("KeikoMatsuzaka", "WORKS_FOR", "Daiei"), ("Leadership", "HAS_PART", "PowerConcept")]
    qa = document(facts, rels, random.Random(1), 2)
    eos = tk.token_to_id("</s>")
    ids, places, roles = render(tk, facts, rels, qa, "places", eos, 200)
    ids_l, places_l, roles_l = render(tk, facts, rels, qa, "places", eos, 200, lead=True)
    assert np.array_equal(ids, ids_l)                                   # the same documents, token for token
    diff = [i for i in range(200) if roles[i] != roles_l[i]]
    assert len(diff) == len(qa)                                         # one more asker a question
    for i in diff:
        here = max(k for k in range(i) if roles[k] == 2 and tk.decode([int(ids[k])]).strip() == "here")
        assert roles_l[i] == 2 and tuple(places_l[i]) == tuple(places[here])     # it asks where here asks: piece 0
        assert tuple(places_l[i + 1]) == tuple(places[i + 1]) and roles_l[i + 1] == 2   # then the first piece


def test_v2_renders_relations_entities_and_event_ids_as_words():
    assert words("WORKS_FOR", True, True) == "works for" and words("VARIANT_OF", True, True) == "variant of"
    assert words("KeikoMatsuzaka", v2=True) == "Keiko Matsuzaka" and words("HexabranchusSanguineus", v2=True) == "Hexabranchus Sanguineus"
    assert [words(n, v2=True) for n in ("ABBA", "1 BCE", "FIDO2", "11.183N 4.283W")] == ["ABBA", "1 BCE", "FIDO2", "11.183N 4.283W"]
    for event, said in {"FortPeckDam_ConstructionBegan1933": "Fort Peck Dam Construction Began 1933",
                        "Rocket_UseInChina13thCentury": "Rocket Use In China 13th Century",
                        "FortPeckDam_HighRelease2011": "Fort Peck Dam High Release 2011",
                        "MatsWilander_1982FrenchOpenTitle": "Mats Wilander 1982 French Open Title",
                        "JangSongthaek_ViceChairmanNDC_End": "Jang Songthaek Vice Chairman NDC End",
                        "VietnameseBoatPeople_HoaExodus1978_1979": "Vietnamese Boat People Hoa Exodus 1978 1979",
                        "33.7714N_72.7518E": "33.7714N 72.7518E", "_": "_"}.items():    # nothing left: the name as it is
        assert words(event, v2=True) == said, event
    # without v2, every name as every shard before 0011 has it
    assert words("FortPeckDam_HighRelease2011") == "fortpeckdam highrelease2011" and words("ABBA") == "abba"
    assert words("KeikoMatsuzaka") == words("KeikoMatsuzaka", v2=True) and words("WORKS_FOR") == words("WORKS_FOR", True, True)


def test_no_question_doubles_a_word():
    assert question("SpanishDancer", "VARIANT_OF", True) == "What is Spanish Dancer variant of?"     # not the inverse
    assert question("Paris", "CAPITAL_OF", True) == "What is Paris capital of?"
    assert question("CampAlbum", "ALBUM_OF", True) == "What is Camp Album the album of?"            # a shared word parted
    assert doubled("What is Camp Album album of?", "Camp Album") == ["album"]
    assert question("SpanishDancer", "VARIANT_OF") == "What is the variant of of Spanish Dancer?"      # v1, as built
    assert question("KeikoMatsuzaka", "WORKS_FOR", True) == "What is the works for of Keiko Matsuzaka?"
    assert doubled("What is the variant of of X?") == ["of"] and not doubled("What is the member of Duran Duran?", "Duran Duran")
    assert not doubled("What is the author of Of Mice And Men?", "Of Mice And Men")      # the name's own word
    if not os.path.exists(WIKI):
        pytest.skip("no wiki world here")
    import pyarrow.parquet as pq
    rels = sorted(set(pq.read_table(WIKI, columns=["relation"]).column("relation").to_pylist()))
    assert len(rels) == 2902
    assert [r for r in rels if doubled(question("X", r, True))] == []
    assert len([r for r in rels if doubled(question("X", r))]) == 646                  # every relation ending in "of", before


def test_v2_places_the_readable_names_pieces(tk):
    rels = {"TIMELINE_EVENT": 0, "VARIANT_OF": 1}
    facts = [("FortPeckDam", "TIMELINE_EVENT", "FortPeckDam_ConstructionBegan1933"),
             ("SpanishDancer", "VARIANT_OF", "HexabranchusSanguineus")]
    qa = document(facts, rels, random.Random(0), 2, v2=True)
    assert {q for q, _a, _at in qa} == {"What is the timeline event of Fort Peck Dam?", "What is Spanish Dancer variant of?"}
    eos = tk.token_to_id("</s>")
    ids, places, roles = render(tk, facts, rels, qa, "places", eos, 200, lead=True, v2=True)
    text = tk.decode(ids.tolist(), skip_special_tokens=True)
    assert text.startswith("Facts:\nFort Peck Dam timeline event Fort Peck Dam Construction Began 1933.\n")
    o = tk.encode("Fort Peck Dam Construction Began 1933", add_special_tokens=False).ids
    w = [i for i in range(200) if roles[i] == 1]
    a0 = address("FortPeckDam", 0)
    assert [int(ids[i]) for i in w[:len(o)]] == o                                         # the object's pieces write, in order
    assert [tuple(int(v) for v in places[i]) for i in w[:len(o)]] == [(a0[0], a0[1], a0[2] + STEP * i) for i in range(len(o))]
    v1 = render(tk, facts, rels, document(facts, rels, random.Random(0), 2), "places", eos, 200, lead=True)
    assert "fortpeckdam constructionbegan1933" in tk.decode(v1[0].tolist(), skip_special_tokens=True)
