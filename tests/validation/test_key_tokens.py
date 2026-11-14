"""Order 0067 round 2: the key tokens' loss weights (validation/key_tokens.py, train_talk_torch.py --key-weight).
Pinned: the key characters are the program's operators, compared values, positions, view names and the join's
tests, never the split; one weight per answer token and the end token; off (weight 1), the batch's mask and the loss
are exactly as before; on, the loss is the weighted mean."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation")]
import key_tokens as K  # noqa: E402
import talk_data as TD  # noqa: E402

ANSWER = ("\ncheck 1 | Write fewer than 48 words.\n---\nprogram Instruction implements ISolve {\n"
          "    function part1(n_word: number): number {\n        create c1: number; assign c1 = 0;\n"
          "        if (n_word >= 48) { assign c1 = 1; }\n        return c1;\n    }\n}\n")


def test_the_key_characters_are_the_programs_meaning_only():
    keys = [ANSWER[s:e] for s, e in K.key_spans(ANSWER)]
    assert ">=" in keys and "48" in keys and "n_word" in keys
    assert all(s > ANSWER.index("---") for s, _ in K.key_spans(ANSWER))      # nothing in the split
    assert K.key_spans("check 1 | Write fewer than 48 words.") == []


class Tok:
    """A character tokenizer: one token per character, offsets exact."""
    class Enc:
        def __init__(self, text):
            self.ids, self.offsets = [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def encode(self, text):
        return self.Enc(text)


def test_one_weight_per_token_on_the_key_characters():
    w = K.weights(Tok(), ANSWER, 4.0)
    assert len(w) == len(ANSWER) + 1 + 1                                   # the leading space, then the end token
    marked = "".join((" " + ANSWER)[i] for i, x in enumerate(w[:-1]) if x == 4.0)
    assert ">=" in marked and "48" in marked and "n_word" in marked and "check" not in marked


def test_off_the_batch_is_unchanged_and_on_it_carries_the_weights():
    rows = [{"p": [1, 2, 3], "t": [4, 5, 6]}, {"p": [1, 2], "t": [7, 8, 9, 10]}]
    x, y, m, keep = TD.make_batch(rows)
    assert set(np.unique(m)) <= {0.0, 1.0}
    rows[0]["w"] = [1.0, 4.0, 1.0]
    m2 = TD.make_batch(rows)[2]
    assert m2[0, 2:5].tolist() == [1.0, 4.0, 1.0] and (m2[1] == m[1]).all()


def test_the_loss_is_the_weighted_mean():
    torch = pytest.importorskip("torch")
    import train_talk_torch as TT

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = torch.nn.Embedding(16, 4)
            self.o = torch.nn.Linear(4, 16)

        def features(self, x):
            return self.e(x)

        def logits_from(self, h):
            return self.o(h)

    torch.manual_seed(0)
    model = M()
    rows = [{"p": [1, 2, 3], "t": [4, 5, 6]}, {"p": [1, 2], "t": [7, 8, 9, 10]}]
    x, y, m, _ = TD.make_batch(rows)
    plain = TT.masked_loss(model, x, y, m, torch.device("cpu"), False)
    sel = m > 0
    logits = model.logits_from(model.features(torch.from_numpy(x))[torch.from_numpy(sel)])
    ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)[torch.from_numpy(sel)], reduction="none")
    assert torch.allclose(plain, ce.mean())
    rows[0]["w"] = [1.0, 4.0, 1.0]
    x, y, m, _ = TD.make_batch(rows)
    w = torch.from_numpy(m[m > 0])
    assert torch.allclose(TT.masked_loss(model, x, y, m, torch.device("cpu"), False), (ce * w).sum() / w.sum())
