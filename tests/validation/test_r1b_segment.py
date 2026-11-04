"""Order 0046, R1b: the names segment (``validation/retrieval_sets/names_segment.py``) carries every name of an item as a
key, merges one name's kinds into one fact, follows a redirect to its article's item and leaves a section redirect out;
the check (``names_check``) reports a field that differs; the VM arm (``validation/retrieval_bench/r1b_vm.py``) enters
each distance group whole and leaves out a group the typo lookup's limit may have cut."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from retrieval_bench import r1b_vm as V  # noqa: E402
from retrieval_sets import names_check as C  # noqa: E402
from retrieval_sets import names_segment as S  # noqa: E402


def test_the_fold_routes_a_name_whatever_its_case_and_punctuation():
    assert S.fold("L'Étranger (roman)") == "l étranger roman" and S.fold("São_Paulo!") == "são paulo"
    assert S.shard_of(S.fold("PARIS"), 32) == S.shard_of(S.fold("paris"), 32)


def test_a_name_s_kinds_merge_and_redirects_reach_the_item(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    try:
        exe = S.cubelang_exe()
    except SystemExit:
        pytest.skip("no CubeLang binary with sharded stores here")
    src = tmp_path / "src"
    (src / "names_x").mkdir(parents=True)
    rows = [("Q90", "Paris", "label", "en"), ("Q90", "Paris", "label", "fr"), ("Q90", "Paname", "alias", "fr"),
            ("Q90", "Paris", "title", "en"), ("P17", "country", "label", "en")]
    pq.write_table(pa.table({k: [r[i] for r in rows] for i, k in enumerate(("id", "name", "kind", "lang"))}),
                   src / "names_x" / "a.parquet")
    for lg, red in (("en", [("City of Light", "Paris", ""), ("Paris history", "Paris", "History")]), ("fr", [])):
        pq.write_table(pa.table({"from": pa.array([r[0] for r in red], pa.string()), "to": pa.array([r[1] for r in red], pa.string()),
                                 "fragment": pa.array([r[2] for r in red], pa.string())}), src / f"redirects_{lg}wiki.parquet")
    work = tmp_path / "work"
    work.mkdir()
    c = S.route(str(src), str(work), 2)
    assert c["redirects_kept"] == 1 and c["redirect_fragment"] == 1 and c["names"] == 4          # the property left out
    facts = []
    for i in range(2):
        S.build_shard(str(work / f"shard_{i:02d}.tsv"), str(work / f"s{i}.jsonl"), str(tmp_path / f"shard_{i:02d}.cubeseg"), exe)
        facts += [json.loads(x) for x in open(work / f"s{i}.jsonl", encoding="utf-8")]
    assert {"key": "paris", "text": "Q90", "source": "label:en,label:fr,title:en"} in facts
    assert {"key": "city of light", "text": "Q90", "source": "redirect:en"} in facts
    out = subprocess.run([exe, "knowledge", "lookup", str(tmp_path), "PANAME"], capture_output=True, text=True, encoding="utf-8")
    assert json.loads(out.stdout.splitlines()[0]) == {"text": "Q90", "source": "alias:fr"}          # the shards, as one store


def test_the_check_reports_what_differs():
    ext = {f: set() for f in C.FIELDS} | {"label:en": {"Paris"}, "alias:fr": {"Paname"}}
    live = {f: set() for f in C.FIELDS} | {"label:en": {"Paris"}, "alias:fr": {"Ville Lumière"}}
    assert C.compare(ext, live) == {"alias:fr": {"extract_only": ["Paname"], "live_only": ["Ville Lumière"]}}
    assert C.compare(ext, dict(ext)) == {} and C.compare(ext, None) == {"missing_live": True}


def test_each_distance_enters_whole_and_a_cut_group_is_left_out():
    hit = {"exact": [{"text": "Q1"}], "near": [{"edits": 1, "facts": [{"text": "Q2"}]}, {"edits": 2, "facts": [{"text": "Q3"}]}]}
    assert V.groups(hit) == {"vm_exact": [["Q1"]], "vm_edit1": [["Q1"], ["Q2"]], "vm_edit2": [["Q1"], ["Q2"], ["Q3"]]}
    assert V.groups(hit, limit=2)["vm_edit2"] == [["Q1"], ["Q2"], []]                            # at the limit: edits 2 may be cut
