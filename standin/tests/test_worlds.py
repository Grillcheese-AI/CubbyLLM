"""Order 0015's ingestion on synthetic fixtures (no source text in the repo): every manifest row gets a decision; each
route writes its world, a re-run adds nothing and a new file is the only one read; the release build leaves out what
the rights tracker does not clear; a book is classified fiction or reality; a fiction world holds its work's people
and answers only a question that names the work; the wiki world's debris is dropped at the mount.
Run: python -m pytest standin/tests/test_worlds.py -q
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

import worlds_adapters as wa  # noqa: E402
import worlds_fiction as wf  # noqa: E402
import worlds_ingest as wi  # noqa: E402
import worlds_manifest as wm  # noqa: E402
from books import Mentions  # noqa: E402

HEAD = "tree\tdataset\tfiles\tGB\tkind\troute\ttier\tworld\tnotes\n"
NOVEL = ("Arden's father, Brom, kept the mill at Telmar. Arden walked to the mill. Brom was old.\n\n"
         "Arden said nothing. Brom said nothing. \"Come,\" said Arden. Arden ran. Brom slept.\n\n") * 3
REFERENCE = ("Rome was founded by Romulus in 753 BC. The city grew on seven hills, 12 km from the sea.\n\n"
             "Table 3.1 lists the 14 regions of Augustus, with their areas in square km (1-14).\n\n") * 3


def _tree(tmp_path):
    src = tmp_path / "src"
    for rel, text in [("ref/Ann Lee - Rome.txt", REFERENCE), ("novels/Kim Dale - The Mill.txt", NOVEL),
                      ("mixed/Bo Ray - Rome Again.txt", REFERENCE), ("mixed/Cy Fox - Mill Days.txt", NOVEL.replace("mill at Telmar", "mill at Varn"))]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text, encoding="utf-8")
    (src / "quotes.json").write_text(json.dumps([{"quote": "Veni, vidi, vici.", "author": "Julius Caesar", "category": "[]"}]))
    (src / "uci").mkdir()
    (src / "uci" / "car.data").write_text("vhigh,vhigh,2,2,small,low,unacc\nlow,low,4,4,big,high,vgood\n")
    (src / "events.jsonl").write_text(json.dumps({"title": "Founding of Rome", "text": "Rome founded.", "year_start": -753}) + "\n")
    manifest, paths = tmp_path / "manifest.tsv", tmp_path / "paths.tsv"
    manifest.write_text(HEAD + "".join(f"E\t{d}\t1\t0\t{k}\t{r}\tattributed\treality\t\n" for d, k, r in [
        ("ref", "reference", "passages + frames"), ("novels", "FICTION", "fiction worlds, one per work"),
        ("mixed", "mixed", "classify per book first"), ("quotes", "quotes", "quote world"), ("uci", "tabular", "data world (rows)"),
        ("events", "events", "event world by date"), ("bench", "EVAL", "EXCLUDED"), ("web", "web text", "passages, lowest priority"),
        ("sft", "SFT", "order 0010 (human casual talk)")]), encoding="utf-8")
    paths.write_text("dataset\tpath\troute\tworld\ttier\tnote\n" + "".join(f"{d}\t{src / p}\t{r}\t{w}\tattributed\t\n" for d, p, r, w in [
        ("ref", "ref", "passages", "reality:reference"), ("novels", "novels", "fiction", "fiction"),
        ("mixed", "mixed", "classify", "reality:assorted"), ("quotes", "quotes.json", "quotes", "reality:quotes"),
        ("uci", "uci", "table", "data"), ("events", "events.jsonl", "events", "reality:events")]), encoding="utf-8")
    return src, manifest, paths


def _run(tmp_path, manifest, paths, **kw):
    return wi.run(tmp_path / "worlds", plan=wm.plan(manifest, paths), log=lambda m: None,
                  mentions=Mentions(["Rome", "Romulus", "Augustus"]), tracker=tmp_path / "no_tracker.csv", **kw)


def test_every_row_gets_a_decision_and_a_path_must_name_a_row(tmp_path):
    _src, manifest, paths = _tree(tmp_path)
    d = {x.dataset: x for x in wm.plan(manifest, paths)}
    assert {k: v.status for k, v in d.items()} == {"ref": "ingest", "novels": "ingest", "mixed": "ingest", "quotes": "ingest",
                                                   "uci": "ingest", "events": "ingest", "bench": "excluded", "web": "queued",
                                                   "sft": "excluded"}
    assert d["sft"].reason == "order 0010 (human casual talk)"                        # the manifest's own words
    with open(paths, "a", encoding="utf-8") as f:
        f.write("nowhere\tX:\\none\tpassages\treality\tattributed\t\n")
    with pytest.raises(ValueError):
        wm.plan(manifest, paths)


def test_each_route_writes_its_world_and_a_rerun_adds_nothing(tmp_path):
    src, manifest, paths = _tree(tmp_path)
    rows = {r["dataset"]: r for r in _run(tmp_path, manifest, paths)}
    out = tmp_path / "worlds"
    assert rows["ref"]["entries"][0]["files_added"] == 1 and rows["ref"]["entries"][0]["rows"] > 0
    assert sqlite3.connect(out / "quotes.sqlite").execute("SELECT speaker FROM quotes").fetchall() == [("Julius Caesar",)]
    assert sqlite3.connect(out / "tables.sqlite").execute("SELECT COUNT(*), MAX(safety) FROM t_car").fetchone() == (2, "low")
    assert sqlite3.connect(out / "events.sqlite").execute("SELECT world, era FROM events").fetchone() == ("reality:events:-753", "AncientClassical")
    mixed = rows["mixed"]["entries"][0]
    assert mixed["classified"]["fiction"] == 1 and mixed["classified"]["reality"] == 1
    again = {r["dataset"]: r for r in _run(tmp_path, manifest, paths)}
    assert all(e.get("files_added", 0) == 0 for r in again.values() for e in r["entries"] if "files_added" in e)
    (src / "ref" / "Dee Moss - Rome Later.txt").write_text(REFERENCE.replace("seven", "the seven"), encoding="utf-8")
    third = {r["dataset"]: r for r in _run(tmp_path, manifest, paths)}
    assert third["ref"]["entries"][0]["files_added"] == 1 and third["novels"]["entries"][0]["files_added"] == 0


def test_the_release_build_leaves_out_what_the_tracker_does_not_clear(tmp_path):
    _src, manifest, paths = _tree(tmp_path)
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("key,author,title,suggested,fiction,status\n|annlee|rome,Ann Lee,Rome,to_buy,,\n"
                       "kimdale|themill,Kim Dale,The Mill,public_domain (US; Project Gutenberg),yes,\n", encoding="utf-8")
    rel = wi.Release(tracker, only=True)
    assert not rel.cleared(pathlib.Path("Ann Lee - Rome.txt"), {"author": "Ann Lee", "title": "Rome"})
    assert rel.cleared(pathlib.Path("Kim Dale - The Mill.txt"), {"author": "Kim Dale", "title": "The Mill"})
    rows = {r["dataset"]: r for r in wi.run(tmp_path / "w", plan=wm.plan(manifest, paths), release=True, log=lambda m: None,
                                            mentions=Mentions(["Rome"]), tracker=tracker)}
    assert rows["ref"]["entries"][0]["left_out"] == 1 and rows["ref"]["entries"][0]["files_added"] == 0
    assert (tmp_path / "w" / "release" / "registry.sqlite").exists()


def test_a_fiction_world_answers_only_a_question_that_names_its_work(tmp_path):
    from ask import AskLoop
    from question_forms import TemplateEmitter
    from test_books import TimedStore, exact_vm
    from test_search_learn import STORE, DictSource
    assert "Arden" in wf.work_names(NOVEL, at_least=3) and ("Arden", "father", "Brom", ) == wf.read_facts(NOVEL, ["Arden", "Brom"])[0][:3]
    assert wa.looks_like_fiction(NOVEL)[0] and not wa.looks_like_fiction(REFERENCE)[0]
    _src, manifest, paths = _tree(tmp_path)
    _run(tmp_path, manifest, paths)
    fw = wf.FictionWorlds(tmp_path / "worlds" / "fiction" / "fiction.sqlite")
    assert set(fw.titles) == {"The Mill", "Mill Days"}
    loop = lambda world: AskLoop(TemplateEmitter(), world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm())  # noqa: E731
    reality = loop(TimedStore(STORE))
    rec = fw.ask("Who was the father of Arden?", reality, loop)
    assert rec["world"] == "reality" and not rec["verified"]                          # reality never hears of Arden
    rec = fw.ask("In The Mill, who was the father of Arden?", reality, loop)
    assert rec["world"] == "fiction:The Mill" and rec["verified"] and rec["answer"] == "Brom"
    assert fw.strip("Who was the father of Arden in \"The Mill\"?", "The Mill") == "Who was the father of Arden?"


def test_the_wiki_worlds_debris_is_dropped_at_the_mount():
    import wikikg as wk
    clean, counts = wk.hygiene([("A", "IS_A", "<class 'wikikg.entities.Event'>"), ("BryantPark", "LOCATION_OF", "_"),
                                ("<__main__.DynamicRelation object at 0x1>", "RELATED_TO", "B"), ("X", "SP0USE_OF", "Y"), ("Ada", "BORN_ON", "1815")])
    assert clean == [("X", "SPOUSE_OF", "Y"), ("Ada", "BORN_ON", "1815")]
    assert counts == {"dropped_not_a_name": 3, "relations_respelled": 1}
