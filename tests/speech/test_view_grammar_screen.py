"""The chooser's screen (`view_grammar.chooser`, ``SCREEN_AT``): on a long ranked list a piece whose first one or two
characters are refused is refused without being judged whole. Pinned: the same pick as judging every piece, at prefixes
of the generator's own answer across the split and the program, the piece that fits ranked last so the whole list is
scanned (the grammar's refusals are prefix-closed). Checked once on 20 prefixes of real answers over the whole 128k
vocabulary: the same allowed set at all 20, the scan 16 times faster."""
from __future__ import annotations

import random

import pytest

pytest.importorskip("regex")
from cubbyllm.speech import view_grammar as VG  # noqa: E402
from cubbyllm.speech import view_types as VT  # noqa: E402
from test_view_types import GOLD  # noqa: E402

INSTR = 'Do not include any sentence that has a number of words other than 2 and that begins with the word "-".'
EOS = 0


def _vocab(answer: str) -> list[str]:
    rng = random.Random(0)
    chars = sorted(set(answer))
    vocab = {c for c in chars} | {a + b for a in chars[:30] for b in chars[:30]}
    vocab |= {answer[i:i + k] for k in (3, 5, 8) for i in range(0, len(answer) - k, 7)}
    vocab |= {"".join(rng.choice(chars) for _ in range(4)) for _ in range(300)}
    return ["</s>"] + sorted(vocab)


def test_the_screen_picks_what_judging_every_piece_picks(monkeypatch):
    answer = f"\ncheck 1 | {INSTR}\n---\n" + GOLD
    vocab = _vocab(answer)
    index = {p: i for i, p in enumerate(vocab)}
    rng = random.Random(1)
    cuts = sorted(rng.sample(range(1, len(answer) - 8), 12))
    for cut in cuts:
        text = answer[:cut]
        ids = [index[c] for c in text]                                 # one character a token
        nxt = index[answer[cut]]
        ranked = [i for i in range(1, len(vocab)) if i != nxt]
        rng.shuffle(ranked)
        ranked.append(nxt)                                             # what surely fits, last
        picks = []
        for at in (10 ** 9, 64):
            monkeypatch.setattr(VG, "SCREEN_AT", at)
            choose = VG.chooser(VT.TypedGrammar(INSTR), vocab.__getitem__, EOS, 0)
            picks.append(choose(ids, ranked))
        assert picks[0] == picks[1] and picks[0], (cut, picks)
