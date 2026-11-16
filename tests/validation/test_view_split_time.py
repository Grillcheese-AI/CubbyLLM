"""Order 0073, line 8 (validation/exp_view_split_time.py): the read at home, one instruction at a time, held to the
machine through grilly2's ``prefix_allowed_tokens_fn``. With a trunk that writes a scripted answer: the hook offers
only what the machine allows and takes the whole gold answer to ``finished``; a read cut by the limit is not whole;
the sample is dev's and seeded; the line passes on the median over the rows of 2 to 4 parts."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation"), str(ROOT)]
import exp_view_split_time as X  # noqa: E402

from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402
from cubbyllm.speech.view_machine import Machine  # noqa: E402

TEXT = ('Write a note. If the reply has at least 3 sentences, use fewer than 40 words.\n'
        'Do not use the word "so". Stay kind.')
ANSWER = "\n" + "\n".join(["task | Write a note.",
                           "check if next n sentence >= 3 | If the reply has at least 3 sentences,",
                           "check n word < 40 | use fewer than 40 words.",
                           'check not any word where lower == "so" | Do not use the word "so".',
                           "asked | Stay kind."])
PIECES = [""] + sorted(set(TEXT + ANSWER + V.prompt("") + "abcdefghijklmnopqrstuvwxyz0123456789_<>=!|\"\\ \n"))
ID = {p: i for i, p in enumerate(PIECES)}
EOS = len(PIECES)


class Ids:
    """Stands for a row of ids on the device: all the hook asks of it is ``tolist``."""
    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


class Refused(Exception):
    """The machine does not allow the id the script writes next: (its piece, how many ids were written)."""


class Trunk:
    """Writes `script` an id at a step, asking the hook first, as grilly2's generate does."""
    eos, tokenizer = EOS, SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[ID[ch] for ch in text]))

    def __init__(self, script: str):
        self.script, self.most = [ID[ch] for ch in script] + [EOS], 0

    def generate(self, role, rows, max_new, prefix_allowed_tokens_fn=None):
        ids, out = list(rows[0]), []
        for want in self.script[:max_new]:
            allowed = prefix_allowed_tokens_fn(0, Ids(ids))
            if want not in allowed:
                raise Refused(PIECES[want] if want != EOS else "eos", len(out))
            self.most = max(self.most, len(allowed))
            ids.append(want)
            out.append(want)
        return [out]


VOCAB = T.Vocab(dict(enumerate(PIECES)))


def test_the_hook_holds_the_answer_to_the_machine_and_takes_the_gold_answer_whole():
    trunk = Trunk(" " + ANSWER)
    got = X.time_one(trunk, "split", VOCAB, TEXT, "w", False, 900)
    assert got["end"] == "finished" and got["whole"] and got["ids"] == len(ANSWER) + 1
    assert 0 <= got["grammar"] <= got["seconds"] and trunk.most < len(PIECES)
    reader = T.Reader(Machine(TEXT, "w"), VOCAB, EOS)
    allowed = X.holder(reader, 3, {})
    assert set(allowed(0, Ids([7, 8, 9]))) == {ID[" "], ID["\n"]}                           # only how an answer starts
    assert set(allowed(0, Ids([7, 8, 9, ID["\n"]]))) == {ID[k] for k in "ctapn"} and reader.n == 1   # the five kinds
    for k in range(len(ANSWER) + 1):
        allowed(0, Ids([7, 8, 9] + [ID[ch] for ch in ANSWER[:k]]))
    assert EOS in allowed(0, Ids([7, 8, 9] + [ID[ch] for ch in ANSWER]))                    # the instruction is used up
    assert allowed(0, Ids([7, 8, 9] + [ID[ch] for ch in ANSWER] + [EOS])) == [EOS] and reader.end == "finished"
    with pytest.raises(Refused) as refused:                                                 # 41 is not in its line's text
        X.time_one(Trunk(" " + ANSWER.replace("n word < 40", "n word < 41")), "split", VOCAB, TEXT, "w", True, 900)
    assert refused.value.args[0] == "1"
    assert X.time_one(Trunk(" " + ANSWER), "split", VOCAB, TEXT, "w", True, 900)["whole"]   # the gold answer, under the rule


def test_a_read_cut_by_the_limit_is_not_whole():
    got = X.time_one(Trunk(" " + ANSWER), "split", VOCAB, TEXT, "w", False, 30)
    assert got == dict(got, end="limit", whole=False, ids=30)
    with pytest.raises(Refused) as refused:                                                 # arm M's machine refuses the `not`
        X.time_one(Trunk(" " + ANSWER), "split", VOCAB, TEXT, "m", False, 900)
    assert refused.value.args[0] == "t"


def test_the_sample_is_devs_and_seeded():
    rows = [{"id": k, "split": s} for k, s in enumerate(["train", "dev", "test", "dev", "test_long", "dev", "dev"])]
    assert {r["split"] for r in X.pick(rows, 3, 0)} == {"dev"} and X.pick(rows, 3, 0) == X.pick(rows, 3, 0)
    assert len(X.pick(rows, 60, 0)) == 4 and X.pick(rows, 3, 0) != X.pick(rows, 3, 5)


def one(parts: int, seconds: float, end: str = "finished") -> dict:
    return {"n_parts": parts, "seconds": seconds, "grammar": seconds / 10, "ids": int(seconds * 40), "end": end, "whole": end == "finished"}


def test_the_line_passes_on_the_median_over_the_rows_of_two_to_four_parts():
    got = X.summary([one(2, 1.0), one(2, 2.0), one(2, 3.0), one(3, 9.0), one(3, 10.0, "limit")])
    assert got["by number of parts"]["2"] == {"rows": 3, "median seconds": 2.0, "95th percentile seconds": 3.0, "median ids": 80,
                                              "ended": {"finished": 3}, "answers whole": 3}
    assert got["by number of parts"]["3"]["median seconds"] == 9.5 and got["by number of parts"]["3"]["ended"] == {"finished": 1, "limit": 1}
    assert got["2 to 4 parts"]["median seconds"] == 3.0 and got["2 to 4 parts"]["95th percentile seconds"] == 10.0
    assert got["line 8"] == {"needs": "a median of at most 8 s for 2 to 4 parts", "median seconds": 3.0, "passes": True, "served": True}
    assert got["share of the time in the grammar"] == 0.1
    slow = X.summary([one(4, 8.0), one(4, 8.5), one(4, 26.0), one(6, 1.0)])
    assert slow["line 8"]["median seconds"] == 8.5 and not slow["line 8"]["passes"] and slow["line 8"]["served"]
    assert X.summary([one(3, 8.0)])["line 8"]["passes"] and not X.summary([one(3, 25.5)])["line 8"]["served"]
