"""The fact shard (validation/make_fact_shard.py): a fact's address is the subject's cell plus
the relation's step, a name is a span up the z axis, the object's pieces and the question's
``here`` and answer pieces are exactly what is placed, and the control places nothing."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "validation"))

from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from make_fact_shard import STEP, address, cell, document, render, step, words  # noqa: E402

TOKENIZER = os.path.join(os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS"), "base450m", "grilly", "tokenizer.json")


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
    ids, places = render(tk, facts, rels, qa, "places", eos, 200)
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
    # every placed token is an object's piece, a question's here, or an answer's piece
    here = [i for i in placed if tk.decode([int(ids[i])]).strip() == "here"]
    assert len(here) == 3
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
    ids_t, places_t = render(tk, facts, rels, qa, "text", eos, 200)
    assert not (places_t[:, 0] != NO_PLACE).any() and "here" not in tk.decode(ids_t.tolist(), skip_special_tokens=True)
    with pytest.raises(ValueError):
        render(tk, facts * 20, rels, [], "places", eos, 64)
