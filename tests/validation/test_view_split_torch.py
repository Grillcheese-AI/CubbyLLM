"""Order 0073, step 6 (validation/exp_view_split_torch.py): the reader's decoder for arms m and w. With a model that
scores the gold next token highest, a batch of rows of unequal length decodes to the gold answers and each ends
``finished``; with a model that always wants the end token, every answer is still inside the grammar and whole (the
end token is taken only when the instruction is used up); an answer that outruns the limit ends ``limit``."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_split_torch as X  # noqa: E402

from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402

ROWS = [{"id": "a", "split": "dev", "instruction": 'Write a note. Use fewer than 40 words.\nDo not use the word "so".',
         "gold": {"m": '\ntask | Write a note.\ncheck n word < 40 | Use fewer than 40 words.\n'
                       'check none word where lower == "so" | Do not use the word "so".'}},
        {"id": "b", "split": "dev", "instruction": "Stay kind.", "gold": {"m": "\nasked | Stay kind."}}]
ROWS[0]["gold"]["w"] = ROWS[0]["gold"]["m"].replace("none word", "not any word")
ROWS[1]["gold"]["w"] = ROWS[1]["gold"]["m"]
LETTERS = sorted({ch for r in ROWS for t in (V.prompt(r["instruction"]), *r["gold"].values()) for ch in t})
PIECES = [""] + LETTERS + ["check ", " | ", "none ", "Use fewer than"]
IDS = {p: i for i, p in enumerate(PIECES)}
EOS = len(PIECES)
TK = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[IDS[ch] for ch in text]))


class Scripted:
    """Scores, for each row, the id its script gives for that step highest (the end token after the script)."""

    def __init__(self, scripts, prompts):
        self.scripts, self.starts = scripts, [len(p) - 1 for p in prompts]

    def init_state(self, n, dev):
        return 0

    def step(self, tokens, t):
        logits = torch.zeros(len(self.scripts), EOS + 1)
        for i, (script, start) in enumerate(zip(self.scripts, self.starts)):
            if t >= start:
                logits[i, script[t - start] if t - start < len(script) else EOS] = 5.0
        return logits, t + 1


def read(arm, scripts, max_new=4000, own=False):
    prompts = [TK.encode(V.prompt(r["instruction"])).ids for r in ROWS]
    return X.decode(Scripted(scripts, prompts), TK, T.Vocab(dict(enumerate(PIECES))), EOS, ROWS, arm, own, max_new, "cpu")


@pytest.mark.parametrize("arm,own", [("m", False), ("w", False), ("m", True), ("w", True)])
def test_a_model_that_scores_the_gold_token_highest_decodes_the_gold_answers(arm, own):
    got = read(arm, [[IDS[ch] for ch in r["gold"][arm]] for r in ROWS], own=own)
    assert [g["answer"] for g in got] == [r["gold"][arm] for r in ROWS]
    assert [(g["id"], g["arm"], g["end"]) for g in got] == [("a", arm, "finished"), ("b", arm, "finished")]
    assert [g["ids"] for g in got] == [len(r["gold"][arm]) for r in ROWS]


def test_a_model_that_always_wants_the_end_token_still_writes_a_whole_answer_inside_the_grammar():
    for arm in V.ARMS:
        got = read(arm, [[], []])
        for g, r in zip(got, ROWS):
            assert g["end"] == "finished" and "lines" in V.parse(g["answer"], r["instruction"], arm), g
            assert g["answer"] != r["gold"][arm]                               # not the gold answer: its own, but well formed
    assert X.best(torch.tensor([0.0, 3.0, 9.0, 1.0]), frozenset({0, 1, 3})) == 1  # the best-scored of the allowed, not of all


def test_an_answer_that_outruns_the_limit_ends_at_the_limit():
    got = read("m", [[IDS[ch] for ch in r["gold"]["m"]] for r in ROWS], max_new=30)
    assert [g["end"] for g in got] == ["limit", "finished"] and got[0]["ids"] == 30
    assert got[0]["answer"] == ROWS[0]["gold"]["m"][:30]
