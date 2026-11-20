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
