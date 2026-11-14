"""The view reader's Colab read (validation/exp_view_reader_torch.py): rows with prompts of different lengths decode in
lockstep, each row's first answer token taken from the logits after its own last prompt token, each held to its own
grammar, each stopping at its end token. A fake model scripts the answers; a fake grammar refuses one token."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation")]
import exp_view_reader_torch as R  # noqa: E402

EOS, V = 0, 40


class Tok:
    def encode(self, text):
        class E:
            ids = [ord(c) % 30 + 5 for c in text]
        return E()

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


class Model:
    """Scores the next token of each row's script highest: after the prompt, the answer's tokens, then the end."""

    def __init__(self, prompts, answers):
        self.scripts = [p + a + [EOS] for p, a in zip(prompts, answers)]

    def init_state(self, batch, device):
        return {"t": 0, "seen": [[] for _ in range(batch)]}

    def step(self, token, state):
        logits = torch.full((len(self.scripts), V), -1.0)
        for i, tok in enumerate(token.tolist()):
            state["seen"][i].append(tok)
            s, n = self.scripts[i], len(state["seen"][i])
            logits[i, s[n] if n < len(s) else EOS] = 5.0
            logits[i, 39] = 4.0                                          # a runner-up the grammar may need
        return logits, state


class Gram:
    """Viable unless the answer holds the letter of token 7 ('h'), which it then must not; complete once it has 3."""

    def viable(self, text):
        return "h" not in text

    def complete(self, text):
        return len(text) >= 3


def test_lockstep_rows_of_different_prompt_lengths(monkeypatch):
    rows = [{"id": "a", "split": "dev", "instruction": "xy"}, {"id": "b", "split": "dev", "instruction": "a much longer one"}]
    monkeypatch.setattr(R.VR, "prompt", lambda s: s)
    monkeypatch.setattr(R.VT, "TypedGrammar", lambda instr: Gram())
    tk = Tok()
    prompts = [tk.encode(r["instruction"]).ids for r in rows]
    answers = [[11, 12, 13], [21, 7, 23]]                 # row b's 7 is refused by its grammar: the runner-up, 39
    out = R.decode(Model(prompts, answers), tk, EOS, rows, "typed", 10, torch.device("cpu"))
    assert out[0]["answer"] == tk.decode([11, 12, 13]) and "error" not in out[0]
    assert out[1]["answer"].startswith(tk.decode([21, 39]))


def test_a_rows_answer_is_the_same_alone_and_inside_a_batch(monkeypatch):
    """Cowork 2026-11-13 22:20: padding and order change nothing. On a small hybrid trunk (MinGRU, a ring of window 4
    every 3rd layer, the real `CubbyModel.step`), three rows of different prompt lengths read alone, together and in
    another order give the same answers, each run past the attention window."""
    sys.path.insert(0, str(ROOT / "tests" / "model"))
    from test_grow import META, _build

    class Free:
        def viable(self, text):
            return True

        def complete(self, text):
            return False

    m = _build(META, seed=1)
    torch.manual_seed(7)
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.context_source.parameters()):
            p.copy_(torch.randn_like(p) * 0.3)
    monkeypatch.setattr(R.VR, "prompt", lambda s: s)
    monkeypatch.setattr(R.VT, "TypedGrammar", lambda instr: Free())
    rows = [{"id": k, "split": "dev", "instruction": s} for k, s in
            (("a", "ab"), ("b", "a much longer prompt"), ("c", "middling one"))]
    tk = Tok()

    def read(batch):
        return {o["id"]: o["answer"] for o in R.decode(m, tk, EOS, batch, "typed", 9, torch.device("cpu"))}
    alone = {k: v for r in rows for k, v in read([r]).items()}
    assert all(len(v) == 9 for v in alone.values()) and len(set(alone.values())) == 3, alone
    assert read(rows) == alone
    assert read(rows[::-1]) == alone
