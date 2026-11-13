"""The talk trainer's micro-batches (order 0067, round 6: long prompts ran a 22 GB card out of memory): a batch cut into
chunks gives the same gradient as the whole batch, every row lands in exactly one chunk, and no chunk pads past its
budget unless one row alone is longer."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import train_talk_torch as T  # noqa: E402
from talk_data import make_batch  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(50, 8)
        self.head = torch.nn.Linear(8, 50)

    def features(self, x):
        return torch.tanh(self.emb(x).cumsum(1))

    def logits_from(self, h):
        return self.head(h)


def rows(rng, n):
    return [{"p": [rng.randrange(1, 50) for _ in range(rng.randint(2, 30))],
             "t": [rng.randrange(1, 50) for _ in range(rng.randint(1, 6))]} for _ in range(n)]


def grads(model, parts):
    model.zero_grad()
    built = [make_batch(p) for p in parts]
    ans = sum(float(b[2].sum()) for b in built)
    for x, y, m, _ in built:
        (T.masked_loss(model, x, y, m, torch.device("cpu"), False) * (float(m.sum()) / ans)).backward()
    return [p.grad.clone() for p in model.parameters()]


def test_chunked_gradient_equals_the_whole_batch():
    torch.manual_seed(0)
    model, rs = Toy(), rows(random.Random(1), 16)
    whole = grads(model, [rs])
    parts = T.chunks(rs, 60)
    assert len(parts) > 1
    for a, b in zip(whole, grads(model, parts)):
        assert torch.allclose(a, b, atol=1e-6)


def test_every_row_in_one_chunk_within_its_budget():
    rs = rows(random.Random(2), 40)
    parts = T.chunks(rs, 80)
    assert [id(r) for p in parts for r in p] == [id(r) for r in rs]
    for p in parts:
        longest = max(len(r["p"]) + len(r["t"]) for r in p)
        assert longest * len(p) <= 80 or len(p) == 1
    assert T.chunks(rs, 0) == [rs]
