"""Order 0078 S1's set: the GSM8K id survives slotting and the re-cut; the cap keeps the long plans first."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.step_loop import recut  # noqa: E402
from emitter_data import slot_record  # noqa: E402
from validation.sft import build_s1 as B  # noqa: E402
from validation.sft import import_tinygsm as tg  # noqa: E402


def test_the_gsm8k_id_survives_slotting_and_the_recut():
    rec = tg.to_record("Tom has 12 apples and buys 9 more, then sells them at 3 dollars each. How much?",
                       "def f():\n    a = 12 + 9\n    return a * 3\n", 1)
    rec["gsm8k_id"] = "gsm8k-train-7"
    rows = recut(slot_record(rec))
    assert len(rows) == 3 and all(r["gsm8k_id"] == "gsm8k-train-7" for r in rows)
    del rec["gsm8k_id"]
    assert all("gsm8k_id" not in r for r in recut(slot_record(rec)))


def test_the_cap_keeps_every_long_plan_and_samples_the_short_ones():
    by = {1: 1000, 2: 1000, 5: 100}                       # rows: 2,000 + 3,000 short, 600 long
    assert B.keep_share(by, 0, 5) == 1.0
    assert B.keep_share(by, 600 + 2500, 5) == 0.5
    assert B.keep_share(by, 300, 5) == 0.0                # the long plans alone pass the cap: no short one kept
    ids = [f"t{i}" for i in range(20000)]
    assert 0.47 < sum(B.sampled(i, 0.5) for i in ids) / len(ids) < 0.53


def test_the_mix_draws_gsm8k_shares_to_the_rows_wanted():
    from collections import Counter
    mix, avail = Counter({1: 10, 2: 30, 3: 60}), Counter({1: 10**6, 2: 10**6, 3: 50})
    got = B.mix_counts(mix, 3000, Counter({1: 10**6, 2: 10**6, 3: 10**6}))
    assert got == {1: 86, 2: 257, 3: 514} and abs(sum((s + 1) * n for s, n in got.items()) - 3000) < 5
    assert B.mix_counts(mix, 3000, avail)[3] == 50          # never more than there is


def test_a_gold_s_annotations_become_a_program_reusing_its_results():
    from validation.sft import gsm8k_programs as GP
    row = {"id": "gsm8k-train-0", "question": "Natalia sold clips to 48 of her friends in April, and then she sold half "
           "as many clips in May. How many clips did Natalia sell altogether in April and May?",
           "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nIn all 48+24 = <<48+24=72>>72.\n#### 72"}
    assert "v1 = 48 + v0" in GP.to_code(row["answer"])
    rec, kind = GP.program(row, 0)
    assert kind == "kept" and rec["gold"] == "72" and rec["gsm8k_id"] == "gsm8k-train-0"
