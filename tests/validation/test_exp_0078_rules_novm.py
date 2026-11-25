"""Order 0078's addendum, parts 1 and 4 (validation/exp_0078_dev_score.py --rules, exp_0078_novm.py): the gold check
counts each rule's refusals on its own and drops a rule over the line; with the rules a breaking try does not vote and k,
m are chosen again without 80:200; off, the score is unchanged. The no-VM arm's shots stay outside dev and carry no
calculator marks; its answer is the number after the first ####, else the last number, cut at a next question; its rows
score as any arm's."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
import exp_0078_dev_score as S  # noqa: E402
import exp_0078_novm as N  # noqa: E402

APPLES = "Ann has 5 apples. How many apples are left?"


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def test_gold_check_counts_each_rule_and_drops_one_over_the_line(tmp_path):
    rows = [{"id": f"t{i}", "question": APPLES, "answer": "so\n#### -2", "dev": False} for i in range(8)]
    rows += [{"id": "t9", "question": APPLES, "answer": "#### 1,250"}]
    g = S.gold_check(_write(tmp_path / "train.jsonl", rows))
    assert g["golds"] == 9 and g["refused by"] == {"negative": 8, "whole": 0, "money": 0}
    assert g["dropped"] == ["negative"] and g["kept"] == ["whole", "money"]


def _rows():
    def row(i, answers):
        return {"id": f"q{i}", "i": i, "gold": "3", "greedy": {"answer": "3"}, "seconds": {"greedy": 1, "tries": 1},
                "tries": [{"answer": a} for a in answers]}
    rows = [row(i, ["3"] * 4 + ["2.5"] * 4 + [None] * 8) for i in range(80)]            # a tie: 2.5 breaks "whole"
    rows += [row(i, ["3"] * 4 + ["2.5"] * 4 + [None] * 8) for i in range(80, 200)]
    rows += [row(i, ["3"] * 4 + ["-1"] * 4 + [None] * 8) for i in range(200, 329)]
    return rows


def test_with_the_rules_a_breaking_try_does_not_vote_and_k_m_are_chosen_again(tmp_path):
    p = _write(tmp_path / "rows.jsonl", _rows())
    qs = {f"q{i}": APPLES for i in range(329)}
    off = S.score(p, "", S.HELD)
    assert off["held read"]["at m"]["right"] == 0                       # every question a tie without the rules
    on = S.score(p, "", S.HELD, ("negative", "whole", "money"), qs)
    assert on["held read"]["at m"]["right"] == 120 and on["held read"]["at m"]["wrong"] == 0
    assert on["tries out by rule"] == {"negative": {"tries": 129 * 4, "right": 0},
                                       "whole": {"tries": 200 * 4, "right": 0}, "money": {"tries": 0, "right": 0}}
    assert "rules" not in off and S.score(p, "", S.HELD) == off           # off: unchanged
    train = _write(tmp_path / "train.jsonl", [{"id": "t", "question": APPLES, "answer": "#### 3"}])
    rr = S.rules_read(p, train, qs)
    assert rr["verdict on the held read"]["kept"] and rr["with the rules"]["rules"] == ["negative", "whole", "money"]


def test_novm_shots_prompt_and_answer(tmp_path):
    rows = [{"id": f"gsm8k-train-{i}", "question": f"Q{i}?", "answer": f"2+{i} = <<2+{i}={2 + i}>>{2 + i}\n#### {2 + i}",
             "dev": i % 2 == 0} for i in range(20)]
    train = _write(tmp_path / "train.jsonl", rows)
    sh = N.shots(train, seed=0)
    assert len(sh) == 4 and all(int(s["id"].rsplit("-", 1)[1]) % 2 == 1 for s in sh)   # none from dev
    assert sh == N.shots(train, seed=0) and sh != N.shots(train, seed=1)
    assert all("<<" not in s["solution"] and s["solution"].endswith("#### " + s["solution"].split()[-1]) for s in sh)
    p = N.prompt(sh, " How many? ")
    assert p.count("Question: ") == 5 and p.endswith("Question: How many?\nAnswer:")
    assert N.answer_of(" 3 + 4 = 7\n#### 1,207\n\nQuestion: next 99") == "1207"
    assert N.answer_of(" she has 7 left, then 9.5 more.\n\nQuestion: #### 12") == "9.5"     # cut at the next question
    assert N.answer_of(" #### $-4.25") == "-4.25" and N.answer_of(" none at all") is None


def test_novm_reader_keeps_the_shots_state_and_it_changes_nothing():
    import pytest
    torch = pytest.importorskip("torch")
    import dev_read_torch as D
    from cubbyllm.model.build import build_model
    meta = {"kind": "cubby_base", "D": 64, "L": 3, "heads": 2, "window": 8, "attn_every": 3, "ctx": 8, "slots": 2,
            "gen_basis": 4, "gen_rank": 2, "vocab": 128, "vocab_real": 128, "mem_every": 0, "tied": False,
            "causal_ctx": True}

    class Tk:
        def encode(self, s):
            return SimpleNamespace(ids=[1 + ord(c) % 120 for c in s])

        def decode(self, ids):
            return ",".join(map(str, ids))

    def reader():
        r = object.__new__(N.NoVmReader)
        r.torch, r.dev, r.tk, r.eos = torch, "cpu", Tk(), 127
        r.model = build_model(meta, "cpu", seed=0)
        r.stepper, r.cache, r.pin = D.Eager(r.model), [], None
        return r
    sh = [{"question": f"q{i}", "solution": f"s{i} #### {i}"} for i in range(4)]
    a, b = reader(), reader()
    a.keep_shots(sh)
    assert a.pin is not None and len(a.pin[0]) > 100
    steps, real = [0], a.model.step

    def counted(*x, **kw):
        steps[0] += 1
        return real(*x, **kw)
    a.model.step = counted
    for qi in range(12):                                                # more questions than the cache's 8 entries
        p = N.prompt(sh, f"question {qi}?")
        steps[0] = 0
        ga, ta, _ = a.read(p, qi, k=2, new=6)
        assert steps[0] < len(a.pin[0])                                 # the shots are never read again
        b.cache = []
        gb, tb, _ = b.read(p, qi, k=2, new=6)
        assert (ga, ta) == (gb, tb)


def test_novm_rows_score_as_any_arm(tmp_path):
    class Reader:
        def read(self, text, qi, k):
            assert text.startswith("Question: ") and text.endswith("Answer:")
            return " so #### 3", [" #### 3"] * 9 + [" 4"] * 7, (0.1, 0.2)
    train = _write(tmp_path / "train.jsonl", [{"id": f"t{i}", "question": "q", "answer": "#### 1"} for i in range(6)])
    recs = [{"id": f"q{i}", "gold": "3", "question": f"dev {i}"} for i in range(329)]
    out = tmp_path / "novm.jsonl"
    a = SimpleNamespace(gsm8k_train=train, shots_seed=0, out=str(out), offset=0, k=16)
    N.run(a, recs[:5], Reader())
    N.run(a, recs, Reader())                                            # resumes: the first 5 are not read again
    rows = [json.loads(ln) for ln in open(out, encoding="utf-8")]
    assert [r["i"] for r in rows] == list(range(329)) and rows[0]["greedy"]["answer"] == "3"
    s = S.score(str(out), "", S.HELD)
    assert s["held read"]["at m"]["right"] == 120 and s["greedy"]["right"] == 209
