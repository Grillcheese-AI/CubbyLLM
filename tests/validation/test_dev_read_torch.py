"""Order 0078, validation/dev_read_torch.py: the batched decode of the 16 tries is exact. Prompts of different lengths
sharing a prefix, decoded greedily by `rows_ids` (the prefix read once, its state copied, one batch per length), give
what each prompt gives decoded alone from a fresh state -- on a small random model whose attention ring wraps (window
smaller than the prompts), so the copied counters and rings are exercised."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import dev_read_torch as D  # noqa: E402

from cubbyllm.model.build import build_model  # noqa: E402

META = {"kind": "cubby_base", "D": 64, "L": 3, "heads": 2, "window": 8, "attn_every": 3, "ctx": 8, "slots": 2,
        "gen_basis": 4, "gen_rank": 2, "vocab": 128, "vocab_real": 128, "mem_every": 0, "tied": False,
        "causal_ctx": True}
EOS = 127


def alone(model, ids, max_new):
    with torch.no_grad():
        st = model.init_state(1, "cpu")
        for t in ids[:-1]:
            _, st = model.step(torch.tensor([t]), st)
        return D.decode(model, st, torch.tensor([ids[-1]]), EOS, max_new, 0.0, 0)[0]


def test_rows_equal_each_prompt_alone():
    torch.manual_seed(0)
    model = build_model(META, "cpu", seed=0)
    prefix = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    ids = [prefix + [50, 51], prefix + [60, 61, 62, 63], prefix + [70, 71], prefix + [80], prefix + [60, 61, 62, 64]]
    got = D.rows_ids(model, ids, EOS, 12, 0.0, 0, "cpu")
    assert got == [alone(model, i, 12) for i in ids]
    assert all(len(g) > 0 for g in got)


def test_clone_state_repeats_rows_and_copies_counters():
    model = build_model(META, "cpu", seed=0)
    with torch.no_grad():
        _, st = model.step(torch.tensor([3]), model.init_state(1, "cpu"))
    c = D.clone_state(st, 4)
    assert c["ctx_sum"].shape[0] == 4 and c["ctx_n"].shape == st["ctx_n"].shape
    c["ctx_n"].add_(1.0)
    assert float(st["ctx_n"]) == 1.0                                  # a copy, not the same buffer


def test_the_prefix_cache_changes_nothing():
    model = build_model(META, "cpu", seed=0)
    q = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
    calls = [[q + [90, 91]], [q + [95, 50, 51], q + [95, 60, 61, 62]], [q + [95, 50, 51, 52], q + [95, 70]]]
    cache: list = []
    for ids in calls:
        assert D.rows_ids(model, ids, EOS, 10, 0.0, 0, "cpu", cache) == D.rows_ids(model, ids, EOS, 10, 0.0, 0, "cpu")
    assert any(k == tuple(q) for k, _ in cache)                         # the question, kept between the steps
