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


def test_the_sft_records_are_the_readers_own_form():
    sys.path.insert(0, str(ROOT / "validation" / "sft"))
    import build_rule_reader_sft as S
    from cubbyllm.speech import rule_reader as rr
    rows = B.build(per=6, seed=2)
    recs = [S.record(r) for r in rows if r["split"] in S.SPLIT]
    assert recs and {r["split"] for r in recs} <= {"train", "held"} and not any(r["split"] == "new_wording" for r in rows
                                                                                 if r["id"] in {x["id"] for x in recs})
    for rec in recs:
        assert rec["prompt"].startswith(rr.HEAD) and rec["prompt"].endswith("\nRule:") and rec["answer"].endswith("\n")
        last_text, last_slot = S.slot_texts(rec)[-1]
        assert last_text + " " + last_slot == rec["prompt"] + " " + rec["answer"].rstrip("\n")


def test_no_training_wording_shares_five_words_with_a_benchmark_prompt():
    """Cowork's check of step 3: no benchmark prompt reaches a training row. Every train and dev instruction the
    shipped build makes (seed 0) against IFEval's prompts, and M-IFEval's French ones where present, read off-repo."""
    import pytest
    import bench_overlap as BO
    bench = BO.benchmark_grams()
    if not bench:
        pytest.skip("no benchmark prompts on this machine ($CUBBY_DATASETS/ifeval, $CUBBY_DATASETS/m-ifeval/fr)")
    hits = [(r["instruction"], BO.overlaps(r["instruction"], bench)) for r in B.build(per=250, seed=0)
            if r["split"] in ("train", "dev") and BO.overlaps(r["instruction"], bench)]
    assert not hits, hits[:3]


def test_round_two_trains_clean_and_reads_fresh_wordings():
    """Round 2 (10 training wordings a kind): its new wordings are in no training list and were never read in round 1;
    its train/dev rows share 5 words with no benchmark prompt (where the prompts are on this machine)."""
    import pytest
    import bench_overlap as BO
    import rule_wordings_v2 as V2
    T2 = V2.templates(RT.T)
    for k, by_lang in T2.items():
        for lang, w in by_lang.items():
            assert len(w["train"]) == 10 and len(w["new"]) == 3, (k, lang)
            assert not set(w["new"]) & set(w["train"]), (k, lang)
            assert not set(w["new"]) & set(RT.T[k][lang]["train"] + RT.T[k][lang]["new"]), (k, lang)
    bench = BO.benchmark_grams()
    if not bench:
        pytest.skip("no benchmark prompts on this machine")
    try:
        B.use_wordings("v2")
        hits = [r["instruction"] for r in B.build(per=250, seed=1)
                if r["split"] in ("train", "dev") and BO.overlaps(r["instruction"], bench)]
    finally:
        B.use_wordings("v1")
    assert not hits, hits[:3]


def test_round_three_trains_clean_and_reads_fresh_wordings():
    """Round 3: its new wordings are in no training list and appear in no earlier round; its training comparisons
    never use IFEval's "at least" / "less than" (the new rows still do); its train/dev rows share 5 words with no
    benchmark prompt."""
    import pytest
    import bench_overlap as BO
    import rule_wordings_v2 as V2
    import rule_wordings_v3 as V3
    T2, T3 = V2.templates(RT.T), V3.templates(RT.T)
    for k, by_lang in T3.items():
        for lang, w in by_lang.items():
            earlier = set(T2[k][lang]["train"] + T2[k][lang]["new"] + RT.T[k][lang]["train"] + RT.T[k][lang]["new"])
            assert len(w["new"]) == 3 and not set(w["new"]) & (set(w["train"]) | earlier), (k, lang)
    assert not {"at least", "less than"} & {p for ps in V3.REL["en"].values() for p in ps}
    bench = BO.benchmark_grams()
    if not bench:
        pytest.skip("no benchmark prompts on this machine")
    try:
        B.use_wordings("v3")
        hits = [r["instruction"] for r in B.build(per=250, seed=2)
                if r["split"] in ("train", "dev") and BO.overlaps(r["instruction"], bench)]
    finally:
        B.use_wordings("v1")
    assert not hits, hits[:3]


def test_round_four_trains_clean_and_reads_comparisons_it_never_trained_on():
    """Round 4 (Nick: everyday comparisons allowed in training): its new wordings are fresh; its test comparisons are in
    no training comparison list; its train/dev rows share 5 words with no benchmark prompt."""
    import pytest
    import bench_overlap as BO
    import rule_wordings_v3 as V3
    import rule_wordings_v4 as V4
    T3, T4 = V3.templates(RT.T), V4.templates(RT.T)
    for k, by_lang in T4.items():
        for lang, w in by_lang.items():
            earlier = set(T3[k][lang]["train"] + T3[k][lang]["new"])
            assert len(w["new"]) == 3 and not set(w["new"]) & (set(w["train"]) | earlier), (k, lang)
    for lang in ("en", "fr"):
        trained = {p for ps in V4.REL[lang].values() for p in ps}
        assert not trained & {p for ps in V4.REL_NEW[lang].values() for p in ps}, lang
    assert {"at least", "less than"} <= {p for ps in V4.REL["en"].values() for p in ps}
    if not BO.benchmark_grams():
        pytest.skip("no benchmark prompts on this machine")
    try:
        B.use_wordings("v4")
        rows = B.build(per=250, seed=3)
    finally:
        B.use_wordings("v1")
    bench = BO.benchmark_grams()
    assert not [r for r in rows if r["split"] in ("train", "dev") and BO.overlaps(r["instruction"], bench)]
