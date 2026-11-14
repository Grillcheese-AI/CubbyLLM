"""The Colab read's grammar pool (validation/view_read_pool.py): checking each row's grammar in worker processes
changes no answer. A scripted model proposes, for three rows under the real typed grammar, the generator's own answer,
the same with a literal of the wrong type (refused: the runner-up is taken and the row goes its own way), and a
shorter one; the batch decoded with the pool gives the serial read's answers and errors, token for token."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT / "tests" / "speech")]
import exp_view_reader_torch as R  # noqa: E402
from test_view_types import GOLD  # noqa: E402
from view_read_pool import GrammarPool  # noqa: E402

EOS, V = 0, 256
INSTR = 'Do not include any sentence that has a number of words other than 2 and that begins with the word "-".'
SHORT = "Write at most 3 sentences."


class Tok:
    def encode(self, text):
        class E:
            ids = [min(ord(c), V - 1) for c in text]
        return E()

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class Model:
    """Each row's script scored highest after its prompt, then the end; ``"`` always the runner-up."""

    def __init__(self, prompts, answers):
        self.scripts = [p + a + [EOS] for p, a in zip(prompts, answers)]

    def init_state(self, batch, device):
        return {"seen": [[] for _ in range(batch)]}

    def step(self, token, state):
        logits = torch.full((len(self.scripts), V), -1.0)
        for i, tok in enumerate(token.tolist()):
            state["seen"][i].append(tok)
            s, n = self.scripts[i], len(state["seen"][i])
            logits[i, s[n] if n < len(s) else EOS] = 5.0
            logits[i, ord('"')] = 4.0
        return logits, state


def test_the_pool_reads_what_the_serial_read_reads():
    tk = Tok()
    good = f"\ncheck 1 | {INSTR}\n---\n" + GOLD
    rows = [{"id": "a", "split": "dev", "instruction": INSTR}, {"id": "b", "split": "dev", "instruction": INSTR},
            {"id": "c", "split": "dev", "instruction": SHORT}]
    answers = [good, good.replace('x4 == "-"', "x4 == 9"), f"\ncheck 1 | {SHORT}\n---\n" + GOLD[:200]]
    prompts = [tk.encode(R.VR.prompt(r["instruction"])).ids for r in rows]

    def read(pool):
        model = Model(prompts, [tk.encode(a).ids for a in answers])
        return R.decode(model, tk, EOS, rows, "typed", 1500, torch.device("cpu"), pool)
    serial = read(None)
    assert serial[0]["answer"] == good and "error" not in serial[0]          # the gold answer kept whole
    assert serial[1]["answer"] != answers[1]                                  # the wrong literal refused
    pool = GrammarPool(2)
    try:
        assert read(pool) == serial
    finally:
        pool.close()
