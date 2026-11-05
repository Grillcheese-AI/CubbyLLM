"""Order 0049's stream (``validation/selfimprove/stream.py``): train records only, the planted families marked, the
shuffled copy's golds a permutation of its own, held-out problems never in the stream, nights in order, deterministic."""
from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from selfimprove import stream as S  # noqa: E402


def _recs():
    out = []
    for fam, task, lang, sub in (("wd/en", "wd", "en", ""), ("wd/fr", "wd", "fr", ""),
                                 ("a1", "arithmetic", None, "step 0/1; steps=1"), ("a4", "arithmetic", None, "step 0/4; steps=4"),
                                 ("k", "kernel", None, "")):
        for i in range(40):
            for split in ("train", "val"):
                out.append({"id": f"{fam}-{i}-{split}", "task": task, "lang": lang, "subtype": sub, "split": split,
                            "gold": f"{fam}:{i}", "prompt": "q", "program": "p"})
    return out


def test_the_stream_plants_its_gaps_and_shuffles_one_family_only():
    h, st, held = S.build(_recs(), level=4, shuffled="arithmetic/steps=1", per_family=20, held=5, night=30, seed=0)
    assert h["planted"] == ["arithmetic/steps=4", "wd/en", "wd/fr"] and h["shuffled"] == "arithmetic/steps=1#shuffled"
    assert all(r["split"] == "train" for r in st + held)
    fams = collections.Counter(r["family"] for r in st)
    assert fams["wd/en"] == 20 and fams["arithmetic/steps=1#shuffled"] == 20
    assert fams["arithmetic/steps=1"] == 0 and any(r["family"] == "arithmetic/steps=1" for r in held)   # the copy alone
    sh = [r for r in st if r["shuffled"]]
    orig = {r["id"]: r["gold"] for r in _recs()}
    assert sorted(r["gold"] for r in sh) != sorted(orig[r["id"].split("#")[0]] for r in sh) or \
        any(r["gold"] != orig[r["id"].split("#")[0]] for r in sh)                       # golds moved
    assert {r["id"] for r in held}.isdisjoint({r["id"] for r in st})                     # held-out never served
    assert [r["night"] for r in st] == sorted(r["night"] for r in st) and st[-1]["night"] == (len(st) - 1) // 30 + 1
    assert S.build(_recs(), 4, "arithmetic/steps=1", 20, 5, 30, 0)[1] == st              # deterministic


def test_a_missing_planted_family_stops_the_build():
    import pytest
    with pytest.raises(SystemExit):
        S.build(_recs(), level=7, shuffled="arithmetic/steps=1", per_family=10, held=2, night=10, seed=0)
