"""Order 0054, Cowork's four proposals from the read of the "after" (Nick, 2026-11-11), each behind its own flag:
- CB_ORIGINAL_WORK=1: a work made from another item of the same name steps aside; the original joins from the alias
  tier ("The Gospel According to John"); a score based on an anthem no longer answers "who wrote" the anthem;
- CB_ALIAS_SEARCH=1: "Emperor Constantine" (an alias of one item) also gathers the emperors labelled "Constantine";
- CB_OWN_DATE=1: a value with its own point in time is said as of that year, not the dump's;
- CB_ANSWER_FITS=1: where -> a place, who -> a person or an organisation, when -> a date, else refused.
Off, each is as before (pinned beside each).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.reasoning import ask_word_fits as awf  # noqa: E402
from cubbyllm.reasoning.ask_memory import AskMemory  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan, Triple  # noqa: E402
from cubbyllm.worlds import wd_store_world as wsw  # noqa: E402

_spec = importlib.util.spec_from_file_location("_wd_fake", pathlib.Path(__file__).with_name("test_wd_store_world.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
st = T.st

T.PROPS.update({"P50": ("author", "wikibase:WikibaseItem"), "P921": ("main subject", "wikibase:WikibaseItem"),
                "P144": ("based on", "wikibase:WikibaseItem"), "P20": ("place of death", "wikibase:WikibaseItem"),
                "P17": ("country", "wikibase:WikibaseItem")})
YEAR11 = {"kind": "wikibase:TimeValue", "wikibase:timePrecision": "11", "wikibase:timeCalendarModel": "wd:Q1985727"}
T.ITEMS.update({
    "Q8001": ([("en", 0, "religious text")], [], {}, []), "Q8002": ([("en", 0, "literary work")], [], {}, []),
    "Q8003": ([("en", 0, "national anthem")], [], {}, []), "Q8004": ([("en", 0, "musical score")], [], {}, []),
    "Q8010": ([("en", 0, "Gospel of John"), ("en", 1, "The Gospel According to John")],
              [st(31, {"entity": "Q8001"}), st(50, {"entity": "Q8020"})], {}, ["Q8001"]),
    "Q8011": ([("en", 0, "The Gospel According to John")],
              [st(31, {"entity": "Q8002"}), st(50, {"entity": "Q8021"}), st(921, {"entity": "Q8010"})], {}, ["Q8002"]),
    "Q8020": ([("en", 0, "John the Apostle")], [st(31, {"entity": "Q5"})], {}, ["Q5"]),
    "Q8021": ([("en", 0, "D. A. Carson")], [st(31, {"entity": "Q5"})], {}, ["Q5"]),
    "Q8030": ([("en", 0, "Hymn of the Land")], [st(31, {"entity": "Q8003"})], {}, ["Q8003"]),
    "Q8031": ([("en", 0, "Hymn of the Land")],
              [st(31, {"entity": "Q8004"}), st(50, {"entity": "Q8032"}), st(144, {"entity": "Q8030"})], {}, ["Q8004"]),
    "Q8032": ([("en", 0, "Herbert Arranger")], [st(31, {"entity": "Q5"})], {}, ["Q5"]),
    "Q8040": ([("en", 0, "Constantine II"), ("en", 1, "Emperor Constantine")],
              [st(31, {"entity": "Q5"}), st(20, {"entity": "Q8050"})], {}, ["Q5"]),
    "Q8041": ([("en", 0, "Constantine")], [st(31, {"entity": "Q5"}), st(20, {"entity": "Q8051"})], {}, ["Q5"]),
    "Q8042": ([("en", 0, "Constantine")], [st(17, {"entity": "Q8051"})], {}, []),
    "Q8050": ([("en", 0, "Aquileia")], [st(17, {"entity": "Q8051"})], {}, []),
    "Q8051": ([("en", 0, "Byzantium")], [], {}, []),
    "Q8060": ([("en", 0, "Fayette")], [{**st(1082, {"typed": ["+208501", "xsd:decimal"]}, "n9"),
                                        "hot_qualifiers": [[585, {"typed": ["2020-04-01T00:00:00Z", "xsd:dateTime"]}]],
                                        "hot_qualifier_nodes": [[585, "n10"]]}],
              {"n9": {"kind": "wikibase:QuantityValue"}, "n10": YEAR11}, []),
})
DESC = {"Q8040": "Roman emperor (316-340)", "Q8041": "9th-century Byzantine emperor", "Q8042": "city in Algeria"}
T.NAMES.update({
    "the gospel according to john": [("Q8011", 0, "exact"), ("Q8010", 1, "exact")],
    "gospel of john": [("Q8010", 0, "exact")], "hymn of the land": [("Q8030", 0, "exact"), ("Q8031", 0, "exact")],
    "emperor constantine": [("Q8040", 1, "exact")], "constantine": [("Q8041", 0, "exact"), ("Q8042", 0, "exact")],
    "fayette": [("Q8060", 0, "exact")], "constantine ii": [("Q8040", 0, "exact")],
})


class Store(T.FakeStore):
    def __call__(self, r):
        got = super().__call__(r)
        if r["op"] == "brief":
            for q, b in got.items():
                b["description"] = DESC.get(q)
        return got


@pytest.fixture()
def world(tmp_path):
    from cubbyllm.worlds.wd_store_kinds import build
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kind_rels.tsv").write_text("Q8001\tP31\t10\nQ8001\tP50\t10\nQ8002\tP31\t10\nQ8002\tP50\t10\nQ8003\tP31\t10\nQ8004\tP31\t10\nQ8004\tP50\t10\nQ5\tP20\t50\n"
                                       "Q5\tP31\t100\n", encoding="utf-8")
    (raw / "labels.tsv").write_text("Q8001\treligious text\nQ8002\tliterary work\nQ8003\tnational anthem\nQ8004\tmusical score\n"
                                    "Q5\thuman\nP50\tauthor\nP31\tinstance of\nP20\tplace of death\n", encoding="utf-8")
    (raw / "meta.tsv").write_text("snapshot\tabc123\nitems\t200\ntyped\t110\n", encoding="utf-8")
    out = str(tmp_path / "store.kinds.sqlite")
    build(str(raw), out)
    return wsw.WdStoreWorld(Store(), "2025-12-29T23:00:38Z", kinds=out)


def hop(w, rel, name):
    w.clear_withheld()
    return {t.obj for _f, t in w.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)}


def test_the_original_work_first(world, monkeypatch):
    monkeypatch.delenv("CB_ORIGINAL_WORK", raising=False)
    assert hop(world, "author", "The Gospel According to John") == {"D. A. Carson"}          # off: the commentary
    assert hop(world, "author", "Hymn of the Land") == {"Herbert Arranger"}                  # off: the score's author
    monkeypatch.setenv("CB_ORIGINAL_WORK", "1")
    world._tiers.cache_clear(), world._named.cache_clear(), world._rows.cache_clear()
    assert hop(world, "author", "The Gospel According to John") == {"John the Apostle"}      # the original joins, first
    assert world.decision("the gospel according to john")["derived"] == ["Q8011"]
    assert hop(world, "author", "Hymn of the Land") == set()                                 # the score steps aside: ask
    assert set(world.withheld("hymn of the land")) <= {"Q8030", "Q8031"}


def test_an_alias_match_does_not_end_the_name_search(world, monkeypatch):
    monkeypatch.delenv("CB_ALIAS_SEARCH", raising=False)
    assert hop(world, "place of death", "Emperor Constantine") == {"Aquileia"}                # off: the alias's one item
    monkeypatch.setenv("CB_ALIAS_SEARCH", "1")
    world._tiers.cache_clear(), world._named.cache_clear(), world._rows.cache_clear()
    assert set(world.namesakes("emperor constantine")) == {"Q8040", "Q8041"}                 # not the city
    assert hop(world, "place of death", "Emperor Constantine") == set()                       # two emperors: asked


def test_a_value_is_said_with_its_own_date(world, monkeypatch):
    t = next(t for _f, t in world._by_subj.get("fayette"))
    monkeypatch.delenv("CB_OWN_DATE", raising=False)
    assert t.obj == "208501" and world.fact_year(t) is None
    monkeypatch.setenv("CB_OWN_DATE", "1")
    assert world.fact_year(t) == "2020"
    me = SimpleNamespace(world=SimpleNamespace(held_by=lambda f: world))
    lr = SimpleNamespace(result=SimpleNamespace(trace=[SimpleNamespace(fact=_f, triple=t) for _f, t in world._by_subj.get("fayette")]))
    assert AskMemory._snapshot(me, "What is the population of Fayette?", lr) == ("as_of", "2020")
    assert AskMemory._snapshot(me, "What was the population of Fayette in 2020?", lr) is None
    assert AskMemory._snapshot(me, "What was the population of Fayette in 2013?", lr)[0] == "refuse"
    monkeypatch.delenv("CB_OWN_DATE")
    assert AskMemory._snapshot(me, "What is the population of Fayette?", lr) == ("as_of", "2025")


def test_the_value_kinds_come_from_the_values_own_item(world):
    death = {t.obj: t for _f, t in world._by_subj.get("constantine ii")}
    assert world.value_kinds(death["Aquileia"]) == {"place"}
    author = {t.obj: t for _f, t in world._by_subj.get("gospel of john")}
    assert world.value_kinds(author["John the Apostle"]) == {"person"}
    assert world.value_kinds(Triple("x", "y", "z")) is None                                   # no item handed out


def test_the_answer_fits_the_question_word(world, monkeypatch):
    assert awf.asked("Where was it filmed?") == "where" and awf.asked("Whom did she marry?") == "who"
    assert awf.asked("What is it?") is None
    assert awf.fits("when", "16 February 2015", None) and awf.fits("when", "500 BC", None) and not awf.fits("when", "Paris", None)
    assert awf.fits("where", "1987", None) is False and awf.fits("where", "Havana Nights", None) is None
    assert awf.fits("who", "Acme", {"organisation"}) and not awf.fits("where", "A Film", set())
    death = {t.obj: t for _f, t in world._by_subj.get("constantine ii")}
    author = {t.obj: t for _f, t in world._by_subj.get("gospel of john")}

    def run(q, t, on=True):
        monkeypatch.setenv("CB_ANSWER_FITS", "1" if on else "0")
        rec = {"answer": t.obj, "verified": True}
        lr = SimpleNamespace(result=SimpleNamespace(trace=[SimpleNamespace(fact="f", triple=t)]))
        return awf.check(SimpleNamespace(world=SimpleNamespace(others=[world])), q, lr, rec), rec

    assert run("Where did he die?", death["Aquileia"])[0]
    ok, rec = run("Where was it written?", author["John the Apostle"])
    assert not ok and rec["reason"] == "word_mismatch" and rec["answer"] is None
    assert run("Where was it written?", author["John the Apostle"], on=False)[0]                # off: as before
    assert run("Who wrote it?", author["John the Apostle"])[0] and not run("Who died there?", death["Aquileia"])[0]
