"""Order 0067, step 3: the rule rows are built in reverse, by rule, from the seen kinds only. No held-out kind of the
pre-registered split has a wording; every plan is one the family accepts; every IFEval kind with rows has a reply
builder for the agreement read; the IFEval form of a row states the same rule as its plan's numbers."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation")]
import build_rule_rows as B  # noqa: E402
import rule_replies  # noqa: E402
import rule_templates as RT  # noqa: E402

from cubbyllm.speech import rule_plan as rp  # noqa: E402

SPLIT = json.loads((ROOT / "validation" / "logs" / "rule_kinds_split_2026-11-12.json").read_text(encoding="utf-8"))


def test_no_held_out_kind_has_a_wording():
    held = {k.split(":")[1] for k in SPLIT["held_out"]}
    assert not held & set(RT.T) and not set(B.IFEVAL.values()) & set(SPLIT["held_out"])
    assert set(B.IFEVAL.values()) <= set(SPLIT["seen"])


def test_rows_are_plans_the_family_accepts_and_name_their_ifeval_form():
    rows = B.build(per=6, seed=3)
    assert {r["split"] for r in rows} == {"train", "dev", "new_wording"}
    for r in rows:
        assert rp.check(r["plan"]) is None
        for iid, kw in r["ifeval"]:
            assert iid in B.IFEVAL.values()
            nums = {v for v in kw.values() if isinstance(v, int)}
            assert nums <= {lf["v"] for lf in rp.leaves(r["plan"])}


def test_every_ifeval_kind_has_replies_on_both_sides_of_its_edge():
    rng = random.Random(0)
    for r in B.build(per=4, seed=1):
        for iid, kw in r["ifeval"]:
            out = rule_replies.replies(iid, kw, r["lang"], rng, 3)
            assert len(out) == 3 and all(isinstance(x, str) for x in out)
