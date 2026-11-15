"""Order 0073, step 3 (validation/builders/view_draw.py): the draw of a long row's parts and groups. A said form's
bounds are read with their place and side; the draw takes the side that is behind, so its parts come out balanced;
a part whose folded form is of a kind kept for the test is not drawn, a negated position among them; the tied
pairs sit at each place in equal shares, and a row that must keep a lone part keeps one."""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import view_draw as D  # noqa: E402
import view_replies as VR  # noqa: E402
import view_said as S  # noqa: E402

from cubbyllm.speech import reply_views as RV  # noqa: E402

LOWER = {"q": "n", "view": "word", "op": ">=", "v": 40}
UPPER = {"q": "n", "view": "word", "op": "<", "v": 40}


def test_bounds_have_a_place_and_a_side():
    assert D.bounds(LOWER) == [("size", "lower")] and D.bounds(UPPER) == [("size", "upper")]
    assert D.bounds({"q": "n", "view": "word", "op": "==", "v": 0}) == []                    # a 0 is not counted
    count = {"q": "count", "view": "sentence", "op": "<", "v": 3, "where": [{"col": "n_word", "op": ">", "v": 5}]}
    assert D.bounds(count) == [("count", "upper"), ("filter", "lower")]
    assert D.bounds({"q": "at", "view": "sentence", "pos": 1, "col": "n_word", "op": "!=", "val": 7}) == [("position", "other")]
    assert D.bounds({"q": "at", "view": "sentence", "pos": 1, "col": "first_word", "op": "==", "val": "so"}) == []
    assert D.bounds({"q": "any", "view": "word", "where": [{"col": "lower", "op": "==", "v": "so"}]}) == []


def test_the_side_that_is_behind_is_wanted():
    tally = Counter({("size", "upper"): 3, ("size", "lower"): 1})
    assert D.wanted(LOWER, tally) == 1 and D.wanted(UPPER, tally) == -1
    assert D.wanted(LOWER, Counter()) == 0
    assert D.wanted(dict(LOWER, op="=="), tally) == 0                                         # "exactly" has no other side
    assert D.wanted(LOWER, Counter({("count", "upper"): 9})) == 0                             # another place's count
    assert D.lower_share(tally) == 0.25 and D.lower_share(tally, "count") is None


def test_a_draw_comes_out_balanced_and_without_the_kinds_kept_for_the_test():
    rng, tally, n = random.Random(5), Counter(), 0
    while n < 60:
        views = [RV.views(t) for t in VR.pool(rng, 80)]
        for _ in range(5):
            parts = D.draw_parts(rng, views, rng.randint(2, 4), tally)
            if parts is None:
                continue
            forms = [S.said(p, "en") for p in parts]
            assert all(f is not None for f in forms) and not any(D.held(p) for p in parts)
            D.tally_row(tally, forms)
            n += 1
    assert 0.45 <= D.lower_share(tally) <= 0.55, tally


def test_a_negated_position_of_a_kept_kind_is_held():
    first = {"q": "at", "view": "paragraph", "pos": 2, "col": "first_word", "op": "==", "val": "so"}
    assert D.held(first) and D.held({"not": first})          # 0067's draw let the second through
    word = {"q": "count", "view": "word", "op": ">=", "v": 1, "where": [{"col": "lower", "op": "==", "v": "so"}]}
    assert not D.held(word) and D.held({"not": word})        # "do not use the word": a word absent
    assert not D.held({"q": "n", "view": "sentence", "op": ">=", "v": 3})


def test_tied_pairs_sit_at_each_place_in_equal_shares():
    rng, n = random.Random(1), 40000
    for k, lone in ((3, False), (4, False), (4, True)):
        at = Counter(i for _ in range(n) for i in D.place(rng, k, lone))
        shares = [at[i] / n for i in range(k - 1)]
        assert max(shares) - min(shares) < 0.015, (k, lone, shares)
    assert all(D.place(rng, 2, True) == [] for _ in range(50))             # two parts: the flipped one stays lone
    assert all(len(D.place(rng, 4, True)) <= 1 for _ in range(300))
    for _ in range(300):                                                   # five to eight: no part in two pairs
        tied = D.place(rng, rng.randint(5, 8))
        assert all(b - a >= 2 for a, b in zip(tied, tied[1:]))


def test_the_rows_kept_have_the_shares_whatever_the_build_drops():
    rng, seen = random.Random(3), Counter()
    for _ in range(6000):
        k = rng.randint(2, 4)
        tied = D.place(rng, k, False, seen)
        if tied and rng.random() < 0.5:                                    # a row with a pair is dropped half the time
            continue
        seen[(k, D.placement([("either", i, i + 1) for i in tied] + [("one", 9)]))] += 1
    for k, share in ((2, 0.3), (3, 0.255), (4, 0.247)):
        rows = sum(n for (parts, _), n in seen.items() if parts == k)
        for i in range(k - 1):
            at = sum(n for (parts, p), n in seen.items() if parts == k and i in p)
            assert abs(at / rows - share) < 0.005, (k, i, at / rows)


def test_a_condition_is_said_of_a_plain_atom():
    rng, plain = random.Random(2), {"q": "n", "view": "sentence", "op": ">=", "v": 3}
    kinds = Counter()
    for _ in range(400):
        parts = [rng.choice([plain, {"not": plain}]) for _ in range(rng.randint(2, 8))]
        groups = D.groups_for(rng, parts)
        assert sorted(x for g in groups for x in g[1:]) == list(range(len(parts)))
        for g in groups:
            kinds[g[0]] += 1
            cond = g[1] if g[0] == "if" else g[2] if g[0] == "unless" else None
            assert cond is None or "not" not in parts[cond]
    assert all(kinds[k] for k in ("one", "if", "unless", "either"))
