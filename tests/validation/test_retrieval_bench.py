"""Order 0046 -- the retrieval harness's own rules, offline: a shortlist takes a set or a run of equal scores whole or not
at all (no tie broken); a mention is located by its item's own names as whole words, or not at all; R0 labels each
refusal by the first stage that stopped it; the splits and samples are fixed by seed 0; a CronQuestions pickle builds
nothing but plain containers; an archive member that would land outside its folder is refused; a frozen lexicon is not
overwritten."""
from __future__ import annotations

import io
import json
import os
import pathlib
import pickle
import random
import sys
import tarfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_bench import r0_denominator as r0  # noqa: E402
from retrieval_bench.mentions import locate  # noqa: E402
from retrieval_bench.r1_finding import by_score, position, typo  # noqa: E402
from retrieval_sets import build_qa, download, lexicon  # noqa: E402
from retrieval_sets.common import fold, halves, sample  # noqa: E402


def test_a_shortlist_takes_each_set_or_run_of_equal_scores_whole():
    groups = [["Q1", "Q2", "Q3"], ["Q9"], ["Q4", "Q5"]]
    assert position(groups, "Q2") == 3 and position(groups, "Q9") == 4 and position(groups, "Q5") == 6
    assert position(groups, "Q7") is None
    assert position([["Q1", "Q2"], ["Q2", "Q3"]], "Q3") == 3                   # an item seen before counts once
    assert by_score([(0, "A"), (0, "B"), (3, "C"), (5, "D"), (5, "E")]) == [["A", "B"], ["C"], ["D", "E"]]
    # recall@K: hit iff the position is <= K; a name 300 items share fits no shortlist of 20, even with the gold in it
    assert (position([[f"Q{i}" for i in range(300)]], "Q7") or 10 ** 9) > 20


def test_a_mention_is_located_by_whole_words_of_its_items_own_names():
    q = "Quel est le quatrième livre de la série Twilight ?"
    assert locate(q, ["Twilight", "Fascination"]) == ("Twilight", [q.index("Twilight"), q.index("Twilight") + 8])
    text, span = locate("Où est né François Mitterrand ?", ["francois mitterrand"])
    assert text == "François Mitterrand"
    assert locate("what does the twilightzone show", ["Twilight"]) is None          # not a whole word: not located
    assert locate("Who wrote The Lord of the Rings?", ["Lord", "The Lord of the Rings"])[0] == "The Lord of the Rings"


def test_a_typo_changes_one_or_two_letters():
    rng = random.Random(0)
    for name in ("Salt Lake City", "Twilight", "Barack Obama"):
        t = typo(name, rng)
        assert t != name and abs(len(t) - len(name)) <= 2


def test_r0_labels_each_refusal_by_the_first_stage_that_stopped_it():
    g = {"Q1"}
    assert r0.label({"verified": False, "reason": "no_plan"}, [], g, set()) == ("reading", "no_plan")
    assert r0.label({"verified": False, "reason": "ambiguous_entity"}, [], g, set())[0] == "several"
    assert r0.label({"verified": False, "reason": "retrieval_exhausted"},
                    [{"qid": None, "ambiguous": ["Q1", "Q2"], "unresolved": False, "n": 0}], g, set())[0] == "several"
    assert r0.label({"verified": False, "reason": "retrieval_exhausted"},
                    [{"qid": "Q9", "ambiguous": [], "unresolved": False, "n": 40}], g, set())[0] == "wrong candidate"
    assert r0.label({"verified": False, "reason": "retrieval_exhausted"},
                    [{"qid": None, "ambiguous": [], "unresolved": True, "n": 0}], g, set())[0] == "no candidate"
    assert r0.label({"verified": False, "reason": "retrieval_exhausted"},
                    [{"qid": "Q1", "ambiguous": [], "unresolved": False, "n": 40}], g, set())[0] == "value missing"
    assert r0.label({"verified": False, "reason": "ambiguous_hop"}, [], g, set())[0] == "several values"
    assert r0.label({"verified": False, "reason": "unknown_relation"}, [], g, set())[0] == "relation"
    assert r0.label({"verified": True, "answer": "Paris"}, [], g, {"paris"}) == ("spoken", "right")
    assert r0.label({"verified": True, "answer": "Lyon"}, [], g, {"paris"}) == ("spoken", "wrong")
    assert r0.label({"verified": True, "answer": "Lyon"}, [], g, set()) == ("spoken", "unscored")


def test_splits_and_samples_are_fixed_by_seed_0():
    ids = [f"x:{i}" for i in range(101)]
    a, b = halves(ids), halves(list(reversed(ids)))
    assert a == b and sum(1 for v in a.values() if v == "dev") == 50
    rows = [{"id": f"x:{i}"} for i in range(50)]
    assert sample(rows, 10) == sample(list(reversed(rows)), 10) and len(sample(rows, 10)) == 10
    assert fold("  Québec-City!! ") == "quebec city"


def test_a_pickle_that_would_run_code_is_refused_and_plain_containers_load():
    class Boom:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    with pytest.raises(pickle.UnpicklingError):
        build_qa._Plain(io.BytesIO(pickle.dumps(Boom()))).load()
    data = [{"question": "q", "answers": {"Q1"}, "entities": {"Q2"}, "times": set(), "uniq_id": 3}]
    assert build_qa._Plain(io.BytesIO(pickle.dumps(data))).load() == data


def test_an_archive_member_that_would_land_outside_its_folder_is_refused(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"x"
        info.size = len(payload)
        t.addfile(info, io.BytesIO(payload))
    with pytest.raises(SystemExit):
        download.unpack(str(bad), str(tmp_path / "out"))
    assert not (tmp_path / "escape.txt").exists()


def test_a_frozen_lexicon_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(lexicon, "FROZEN", str(tmp_path / "frozen.json"))
    f = tmp_path / "lex.json"
    f.write_text(json.dumps({"a": 1}))
    h = lexicon.freeze("relation_lexicon", str(f), "test")
    assert lexicon.freeze("relation_lexicon", str(f), "test") == h                 # the same file: unchanged
    f.write_text(json.dumps({"a": 2}))
    with pytest.raises(SystemExit):
        lexicon.freeze("relation_lexicon", str(f), "test")
    table = tmp_path / "props.json"
    table.write_text(json.dumps({"fetched": "x", "n_properties": 1, "properties": {
        "P19": {"label": {"en": "place of birth", "fr": "lieu de naissance"}, "aliases": {"en": ["born in"], "fr": ["né à"]},
                "datatype": "wikibase-item"}}}))
    lx = lexicon.build(str(table))
    assert lx["by_wording"]["en"]["born in"] == ["P19"] and lx["by_wording"]["fr"]["ne a"] == ["P19"]


def test_a_quantity_is_near_after_unit_conversion_and_only_then():
    from retrieval_bench.units import near
    assert near("96 m", "315 feet") and near("180 cm", "180.5") and near("1,609 m", "1 mile")
    assert not near("96 m", "96 kg") and not near("96 m", "320 feet") and not near("Paris", "Paris") and not near("96", "96")
    from retrieval_bench.r0_denominator import label
    rec = {"verified": True, "answer": "96 m"}
    assert label(rec, [], set(), {"315 feet"}, ["315 feet"]) == ("spoken", "near")
    assert label(dict(rec, answer="Rome"), [], set(), {"paris"}, ["Paris"]) == ("spoken", "wrong")
