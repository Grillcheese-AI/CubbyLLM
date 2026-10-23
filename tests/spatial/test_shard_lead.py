"""The village render in the far facts' shape (``make_spatial_shard.render(lead=True)``, order 0035 C).

Pinned: a listed name's piece i sits at its cell + 6 i z and writes there; after a placed question, the last token of
"Answer: " asks at the cell (for piece 0), the answer is the name in the token form it was written in (the listing's
own ids), and its piece j asks for piece j + 1's address; the probe's span finder reads every answer where it is; an
unplaced question and the text mode are as before.
"""
from __future__ import annotations

import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "validation") not in sys.path:
    sys.path.insert(0, str(ROOT / "validation"))

from cubbyllm.model.fact_layout import STEP  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from make_spatial_shard import ASK, WRITE, questions, render, village  # noqa: E402

TOK = ROOT / "data" / "grillcheese_bbpe128k.json"
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="the bbpe tokenizer is not in this checkout")


@pytest.fixture(scope="module")
def tk():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK))


def test_the_lead_shape_writes_pieces_up_z_and_asks_for_each_next_piece(tk):
    from exp_hypertoken_recall import answers
    rng = random.Random(7)
    sc = village(rng, 14)
    qa = questions(sc, rng, 12, "placed")
    eos = tk.token_to_id("</s>")
    ids, places, roles = render(tk, sc, qa, "places", eos, 1024, lead=True)
    written = {name: tk.encode(name, add_special_tokens=False).ids for name in sc.things}
    starts = [j for j in range(len(ids)) if roles[j] == WRITE and (j == 0 or roles[j - 1] != WRITE)]
    assert len(starts) == len(sc.things)
    for j in starts:                                          # every name's pieces: at the cell, then up z
        k = j
        while k < len(ids) and roles[k] == WRITE:
            k += 1
        base = tuple(int(v) for v in places[j])
        assert [tuple(int(v) for v in places[m]) for m in range(j, k)] == [(base[0], base[1], base[2] + STEP * n)
                                                                          for n in range(k - j)]
        assert sc.at(base) is not None and [int(x) for x in ids[j:k]] == written[sc.at(base)]
    spans = answers(tk, ids)
    assert len(spans) == len(qa)
    for (q, a, cell), (j, k) in zip(qa, spans):
        cell = tuple(cell)
        assert roles[j - 1] == ASK and tuple(int(v) for v in places[j - 1]) == cell          # the lead asks for piece 0
        assert [int(x) for x in ids[j:k + 1]] == tk.encode(a, add_special_tokens=False).ids    # the written form
        if a in written:
            assert [int(x) for x in ids[j:k + 1]] == written[a]
        for n, m in enumerate(range(j, k + 1)):                                                # piece n asks for n + 1
            assert roles[m] == ASK and tuple(int(v) for v in places[m]) == (cell[0], cell[1], cell[2] + STEP * (n + 1))
        assert tk.decode([int(x) for x in ids[j - 3:k + 1]]).endswith(f"Answer: {a}")
    assert all(places[m, 0] == NO_PLACE for m in range(len(ids)) if roles[m] == 0)


def test_the_space_form_writes_and_answers_every_name_as_it_is_spoken(tk):
    """``lead="space"``: " coal" in the listing and in the answer, and the ":" of "Answer:" asks for piece 0."""
    from exp_hypertoken_recall import answers
    rng = random.Random(9)
    sc = village(rng, 12)
    qa = questions(sc, rng, 10, "placed")
    ids, places, roles = render(tk, sc, qa, "places", tk.token_to_id("</s>"), 1024, lead="space")
    written = {name: tk.encode(" " + name, add_special_tokens=False).ids for name in sc.things}
    starts = [j for j in range(len(ids)) if roles[j] == WRITE and (j == 0 or roles[j - 1] != WRITE)]
    assert len(starts) == len(sc.things)
    for j in starts:
        base = tuple(int(v) for v in places[j])
        assert [int(x) for x in ids[j:j + len(written[sc.at(base)])]] == written[sc.at(base)]
    colon = tk.encode("Answer:", add_special_tokens=False).ids[-1]
    for (q, a, cell), (j, k) in zip(qa, answers(tk, ids)):
        assert int(ids[j - 1]) == colon and roles[j - 1] == ASK and tuple(int(v) for v in places[j - 1]) == tuple(cell)
        assert [int(x) for x in ids[j:k + 1]] == tk.encode(" " + a, add_special_tokens=False).ids
        if a in written:
            assert [int(x) for x in ids[j:k + 1]] == written[a]


def test_off_and_unplaced_and_text_are_as_before(tk):
    rng = random.Random(3)
    sc = village(rng, 10)
    eos = tk.token_to_id("</s>")
    learned = questions(sc, random.Random(4), 8)                           # unplaced: no address, as before
    for mode in ("places", "text"):
        a = render(tk, sc, learned, mode, eos, 512)
        b = render(tk, sc, learned, mode, eos, 512, lead=(mode == "text"))   # lead means nothing in text mode
        assert all((x == y).all() for x, y in zip(a, b))
    placed = questions(sc, random.Random(4), 8, "placed")
    a, b = render(tk, sc, placed, "places", eos, 512), render(tk, sc, placed, "places", eos, 512, lead=False)
    assert all((x == y).all() for x, y in zip(a, b))
