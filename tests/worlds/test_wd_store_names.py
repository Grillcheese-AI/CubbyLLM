"""Order 0066, the four checks (Cowork, approved by Nick 2026-11-11), over a fake store with its own class-roots table:
- check 1 (CB_QWORD_KIND): "where" refuses a film and a book and passes a hospital, a park, a neighbourhood, a village
  and a city; "who" refuses a place and passes a person, a band and a company; "when" refuses a place; "what culture
  region" (no class) is read as "region": a company refused, a region passed; "which city" is untouched;
- check 2 (CB_ORIGINAL_FIRST): a commentary and its Gospel under one name -> the Gospel; an anthem and its
  arrangement -> the anthem; two works with no statement between them -> asked, both named;
- check 3 (CB_ALIAS_OPEN): "Emperor Constantine" (one alias item, several emperors) -> asked; a year only one's term
  covers -> that one; an alias with no role word -> as today;
- check 4 (CB_VALUE_DATE): a 2020 point in time and a 2025 dump -> "as of 2020"; no date of its own -> the dump's
  year; a question naming 2023 against a 2020 value -> refused;
- each flag off: as before.
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

from cubbyllm.reasoning import ask_class, ask_namesakes, ask_qword  # noqa: E402
from cubbyllm.reasoning.ask_memory import AskMemory  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from cubbyllm.worlds import wd_store_roots as R  # noqa: E402
from cubbyllm.worlds import wd_store_world as wsw  # noqa: E402

_spec = importlib.util.spec_from_file_location("_wd_fake", pathlib.Path(__file__).with_name("test_wd_store_world.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
st = T.st
FLAGS = ("CB_QWORD_KIND", "CB_ORIGINAL_FIRST", "CB_ALIAS_OPEN", "CB_VALUE_DATE")


def sq(pid, value, quals=()):
    """A statement with hot qualifiers [(pid, year)]."""
    return {**st(pid, value), "hot_qualifiers": [[p, {"typed": [f"{y:04d}-01-01T00:00:00Z", "xsd:dateTime"]}] for p, y in quals]}


def item(label, *sts, aliases=()):
    return ([("en", 0, label)] + [("en", 1, a) for a in aliases], list(sts), {}, [])


def of(*classes):
    return [st(31, {"entity": c}) for c in classes]


T.PROPS.update({"P50": ("author", "wikibase:WikibaseItem"), "P921": ("main subject", "wikibase:WikibaseItem"),
                "P144": ("based on", "wikibase:WikibaseItem"), "P20": ("place of death", "wikibase:WikibaseItem"),
                "P39": ("position held", "wikibase:WikibaseItem")})
CLASSES = {   # class: (label, superclass)
    "Q9001": ("geographic location", None), "Q9003": ("work", None), "Q9004": ("group of humans", None),
    "Q9005": ("organization", None), "Q9006": ("position", None), "Q9015": ("human settlement", "Q9001"),
    "Q9010": ("hospital", "Q9001"), "Q9012": ("park", "Q9001"), "Q9013": ("neighbourhood", "Q9001"),
    "Q9014": ("village", "Q9015"), "Q9016": ("city", "Q9015"), "Q9017": ("region", "Q9001"),
    "Q9020": ("film", "Q9003"), "Q9021": ("book", "Q9003"), "Q8001": ("religious text", "Q9003"),
    "Q8002": ("literary work", "Q9003"), "Q8003": ("national anthem", "Q9003"), "Q8004": ("musical score", "Q9003"),
    "Q9030": ("band", "Q9004"), "Q9031": ("company", "Q9005"),
    "Q9040": ("emperor", "Q9006"), "Q9041": ("Roman emperor", "Q9040"),
}
for q, (lab, sup) in CLASSES.items():
    T.ITEMS[q] = item(lab, *([st(279, {"entity": sup})] if sup else []))
T.ITEMS.update({
    "Q9100": item("Saint Mary's Hospital", *of("Q9010")), "Q9101": item("Hyde Park", *of("Q9012")),
    "Q9102": item("Upper East Side", *of("Q9013")), "Q9103": item("Nine Mile", *of("Q9014")),
    "Q9104": item("Lyon", *of("Q9016")), "Q9105": item("Havana Nights", *of("Q9020")), "Q9106": item("The Beers", *of("Q9021")),
    "Q9107": item("Ada", *of("Q5")), "Q9108": item("The Quiet Ones", *of("Q9030")), "Q9109": item("Acme", *of("Q9031")),
    "Q9110": item("Central Europe", *of("Q9017")),
    "Q9199": ([("en", 0, "Probe")], [st(20, {"entity": q}) for q in ("Q9100", "Q9101", "Q9102", "Q9103", "Q9104", "Q9105",
                                                                        "Q9106", "Q9107", "Q9108", "Q9109", "Q9110")]
              + [st(569, {"typed": ["1815-12-10T00:00:00Z", "xsd:dateTime"]}, "n1")], {"n1": T.DAY}, []),
    # check 2
    "Q8010": item("Gospel of John", *of("Q8001"), st(50, {"entity": "Q8020"}), aliases=("The Gospel According to John",)),
    "Q8011": item("The Gospel According to John", *of("Q8002"), st(50, {"entity": "Q8021"}), st(921, {"entity": "Q8010"})),
    "Q8020": item("John the Apostle", *of("Q5")), "Q8021": item("D. A. Carson", *of("Q5")),
    "Q8030": item("Hymn of the Land", *of("Q8003")),
    "Q8031": item("Hymn of the Land", *of("Q8004"), st(50, {"entity": "Q8032"}), st(144, {"entity": "Q8030"})),
    "Q8032": item("Herbert Arranger", *of("Q5")),
    "Q8070": item("Twin Tale", *of("Q8002"), st(50, {"entity": "Q8020"})),
    "Q8071": item("Twin Tale", *of("Q8002"), st(50, {"entity": "Q8021"})),
    # check 3
    "Q8040": item("Constantine II", *of("Q5"), st(20, {"entity": "Q9104"}), sq(39, {"entity": "Q9041"}, [(580, 337), (582, 340)]),
                  aliases=("Emperor Constantine",)),
    "Q8041": item("Constantine", *of("Q5"), st(20, {"entity": "Q9103"}), sq(39, {"entity": "Q9041"}, [(580, 913), (582, 959)])),
    "Q8043": item("Constantine", *of("Q5"), st(20, {"entity": "Q9101"}), sq(39, {"entity": "Q9041"}, [(580, 868), (582, 879)])),
    "Q8042": item("Constantine", *of("Q9016")),
    "Q8080": item("New York City", *of("Q9016"), aliases=("Big Apple",)),
    # check 4
    "Q8060": ([("en", 0, "Fayette")], [{**st(1082, {"typed": ["+208501", "xsd:decimal"]}, "n9"),
                                        "hot_qualifiers": [[585, {"typed": ["2020-04-01T00:00:00Z", "xsd:dateTime"]}]],
                                        "hot_qualifier_nodes": [[585, "n10"]]}],
              {"n9": {"kind": "wikibase:QuantityValue"}, "n10": T.DAY}, []),
    "Q8061": ([("en", 0, "Dale")], [st(1082, {"typed": ["+1200", "xsd:decimal"]}, "n9")], {"n9": {"kind": "wikibase:QuantityValue"}}, []),
})
for k, ids in {"probe": ["Q9199"], "gospel of john": ["Q8010"], "constantine ii": ["Q8040"], "fayette": ["Q8060"],
               "dale": ["Q8061"], "emperor": ["Q9040"], "new york city": ["Q8080"], "human": ["Q5"]}.items():
    T.NAMES[k] = [(q, 0, "exact") for q in ids]
T.NAMES.update({"the gospel according to john": [("Q8011", 0, "exact"), ("Q8010", 1, "exact")],
                "hymn of the land": [("Q8030", 0, "exact"), ("Q8031", 0, "exact")],
                "twin tale": [("Q8070", 0, "exact"), ("Q8071", 0, "exact")],
                "emperor constantine": [("Q8040", 1, "exact")],
                "constantine": [("Q8041", 0, "exact"), ("Q8042", 0, "exact"), ("Q8043", 0, "exact")],
                "big apple": [("Q8080", 1, "exact")]})


@pytest.fixture()
def world(tmp_path, monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    from cubbyllm.worlds.wd_store_kinds import build
    raw = tmp_path / "raw"
    raw.mkdir()
    rels = ["Q8001\tP31\t10", "Q8001\tP50\t10", "Q8002\tP31\t10", "Q8002\tP50\t10", "Q8003\tP31\t10", "Q8004\tP31\t10",
            "Q8004\tP50\t10", "Q5\tP31\t100", "Q5\tP20\t50", "Q5\tP39\t20"]
    (raw / "kind_rels.tsv").write_text("\n".join(rels) + "\n", encoding="utf-8")
    (raw / "labels.tsv").write_text("".join(f"{q}\t{CLASSES[q][0]}\n" for q in ("Q8001", "Q8002", "Q8003", "Q8004"))
                                    + "Q5\thuman\nP50\tauthor\nP31\tinstance of\nP20\tplace of death\nP39\tposition held\n",
                                    encoding="utf-8")
    (raw / "meta.tsv").write_text("snapshot\tabc123\nitems\t200\ntyped\t110\n", encoding="utf-8")
    build(str(raw), str(tmp_path / "kinds.sqlite"))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "subclass.tsv").write_text("".join(f"{q}\t{s}\n" for q, (_l, s) in CLASSES.items() if s), encoding="utf-8")
    (sub / "meta.tsv").write_text("snapshot\tabc123\nedges\t20\n", encoding="utf-8")
    roots = {"place": ["Q9001"], "person": ["Q5"], "group": ["Q9004"], "organisation": ["Q9005"], "character": [],
             "work": ["Q9003"], "role": ["Q9006"]}
    R.build(str(sub), roots, str(tmp_path / "roots.sqlite"))
    return wsw.WdStoreWorld(T.FakeStore(), "2025-12-29T23:00:38Z", kinds=str(tmp_path / "kinds.sqlite"),
                            roots=str(tmp_path / "roots.sqlite"))


def fresh(w):
    for c in (w._tiers, w._named, w._rows):
        c.cache_clear()
    w.clear_withheld()


def hop(w, rel, name):
    fresh(w)
    return {t.obj for _f, t in w.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)}


def test_the_roots_table_reads_the_closure_and_the_resolution_rule(world, tmp_path):
    assert world._roots.of("Q9014") == {"place"} and world._roots.of("Q9041") == {"role"} and world._roots.of("Q9999") == set()
    assert R.resolve(T.FakeStore(), {"Q5", "Q41"}, {"person": (("Q5", "human"),)}) == {"person": ["Q5"]}
    with pytest.raises(SystemExit):                                   # an id that does not carry its label: stop
        R.resolve(T.FakeStore(), {"Q5", "Q41"}, {"person": (("Q41", "human"),)})
    with pytest.raises(SystemExit):                                   # an id that is no class of the store: stop
        R.resolve(T.FakeStore(), {"Q41"}, {"person": (("Q5", "human"),)})
    assert set(R.ROOTS) == set(R.BIT) and all(q.startswith("Q") for ps in R.ROOTS.values() for q, _l in ps)
    assert R.for_store(str(tmp_path / "roots.sqlite"), "another") is None


def check1(world, monkeypatch, question, value, on=True):
    monkeypatch.setenv("CB_QWORD_KIND", "1" if on else "0")
    f, t = {t.obj: (f, t) for f, t in world._by_subj.get("probe")}[value]
    rec = {"answer": value, "verified": True}
    lr = SimpleNamespace(result=SimpleNamespace(trace=[SimpleNamespace(fact=f, triple=t)]))
    return ask_class.check(SimpleNamespace(world=SimpleNamespace(others=[world]), source=None, _everyday=None), question, lr, rec), rec


def test_check_1_the_question_word(world, monkeypatch):
    for v in ("Saint Mary's Hospital", "Hyde Park", "Upper East Side", "Nine Mile", "Lyon"):
        assert check1(world, monkeypatch, "Where did he die?", v)[0], v
    for v in ("Havana Nights", "The Beers"):
        ok, rec = check1(world, monkeypatch, "Where was it filmed?", v)
        assert not ok and rec["reason"] == "class_mismatch" and rec["answer"] is None, v
        assert check1(world, monkeypatch, "Where was it filmed?", v, on=False)[0]                # off: as before
    assert not check1(world, monkeypatch, "Who won it?", "Lyon")[0]
    for v in ("Ada", "The Quiet Ones", "Acme"):
        assert check1(world, monkeypatch, "Who won it?", v)[0], v
    assert not check1(world, monkeypatch, "When was it?", "Lyon")[0]
    assert check1(world, monkeypatch, "Quand est-il né ?", "10 December 1815")[0]
    assert ask_qword.asked("Où est-il mort ?") == "where" and ask_qword.asked("To whom was it given?") == "who"
    assert ask_qword.asked("Which city?") is None


class Src:
    """The live source as ask_class reads it: search by phrase, an item's claims."""
    CL = {"Q9017": {"P279": ["Q9001"]}, "Q9016": {"P279": ["Q9015"]}, "Q9110": {"P31": ["Q9017"]},
          "Q9109": {"P31": ["Q9031"]}, "Q9104": {"P31": ["Q9016"]}}
    links: dict = {}

    def _get(self, params):
        hit = {"region": "Q9017", "city": "Q9016", "Central Europe": "Q9110", "Acme": "Q9109", "Lyon": "Q9104"}.get(params.get("search"))
        return {"search": [{"id": hit, "label": params["search"]}] if hit else []}

    def _claims(self, q):
        return {p: [{"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": v}}}} for v in vs]
                for p, vs in self.CL.get(q, {}).items()}

    def _labels_for(self, ids):
        return {q: q for q in ids}


def test_check_1b_the_head_noun(world, monkeypatch):
    def run(q, value, on):
        monkeypatch.setenv("CB_QWORD_KIND", "1" if on else "0")
        rec = {"answer": value, "verified": True}
        h = SimpleNamespace(fact=f"{value} is the part of of Germany", triple=SimpleNamespace(obj=value))
        return ask_class.check(SimpleNamespace(world=None, source=Src(), _everyday=None), q,
                               SimpleNamespace(result=SimpleNamespace(trace=[h])), rec)
    assert run("What culture region is Germany a part of?", "Acme", on=False)                    # off: unchecked
    assert not run("What culture region is Germany a part of?", "Acme", on=True)
    assert run("What culture region is Germany a part of?", "Central Europe", on=True)
    assert run("Which city is it in?", "Lyon", on=True) and run("Which city is it in?", "Lyon", on=False)


def test_check_2_the_original_first(world, monkeypatch):
    assert hop(world, "author", "The Gospel According to John") == {"D. A. Carson"}            # off: the commentary
    assert hop(world, "author", "Hymn of the Land") == {"Herbert Arranger"}                    # off: the score's
    monkeypatch.setenv("CB_ORIGINAL_FIRST", "1")
    assert hop(world, "author", "The Gospel According to John") == {"John the Apostle"}        # the Gospel
    fresh(world)
    assert world._tiers("hymn of the land") == (["Q8030"], [])                                   # the anthem
    assert hop(world, "author", "Hymn of the Land") == set()                                     # (it holds no author)
    assert hop(world, "author", "Twin Tale") == set() and set(world.withheld("twin tale")) == {"Q8070", "Q8071"}


def test_check_3_an_alias_does_not_end_the_search(world, monkeypatch):
    assert hop(world, "place of death", "Emperor Constantine") == {"Lyon"}                      # off: the alias's item
    monkeypatch.setenv("CB_ALIAS_OPEN", "1")
    assert hop(world, "place of death", "Emperor Constantine") == set()                          # three emperors: asked
    assert set(world.namesakes("emperor constantine")) == {"Q8040", "Q8041", "Q8043"}            # not the city
    assert world.pick_year("emperor constantine", 870) == "Q8043" and world.pick_year("emperor constantine", 2000) is None
    got = ask_namesakes._by_year(SimpleNamespace(others=[world]), "emperor constantine",
                                 "Where did Emperor Constantine, emperor in 870, die?")
    assert got[0] == "Q8043"
    fresh(world)
    assert world._tiers("big apple") == ([], ["Q8080"])                                          # no role word: as today


def test_check_4_a_value_keeps_its_own_date(world, monkeypatch):
    me = SimpleNamespace(world=SimpleNamespace(held_by=lambda f: world))

    def lr(name):
        return SimpleNamespace(result=SimpleNamespace(trace=[SimpleNamespace(fact=f, triple=t) for f, t in world._by_subj.get(name)]))
    q = "What is the population of Fayette?"
    assert AskMemory._snapshot(me, q, lr("fayette")) == ("as_of", "2025")                        # off: the dump's year
    monkeypatch.setenv("CB_VALUE_DATE", "1")
    assert AskMemory._snapshot(me, q, lr("fayette")) == ("as_of", "2020")
    assert AskMemory._snapshot(me, "What is the population of Dale?", lr("dale")) == ("as_of", "2025")   # none of its own
    assert AskMemory._snapshot(me, "What was the population of Fayette in 2023?", lr("fayette"))[0] == "refuse"
    rec = {}
    AskMemory._dates(me, lr("fayette"), rec)
    assert rec["dates"] == {"value": "2020", "dump": "2025"}


def test_all_four_on_together(world, monkeypatch):
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    assert hop(world, "author", "The Gospel According to John") == {"John the Apostle"}
    assert hop(world, "place of death", "Emperor Constantine") == set()
    assert not check1(world, monkeypatch, "Where was it filmed?", "Havana Nights")[0]
    assert check1(world, monkeypatch, "Where did he die?", "Hyde Park")[0]
    me = SimpleNamespace(world=SimpleNamespace(held_by=lambda f: world))
    lr = SimpleNamespace(result=SimpleNamespace(trace=[SimpleNamespace(fact=f, triple=t) for f, t in world._by_subj.get("fayette")]))
    assert AskMemory._snapshot(me, "What is the population of Fayette?", lr) == ("as_of", "2020")


def test_an_unknown_value_given_as_a_plain_string_is_read_without_error(world, monkeypatch):
    T.ITEMS["Q8090"] = item("Mystery Man", *of("Q5"), st(569, "somevalue"), st(570, "novalue"), st(20, {"entity": "Q9104"}))
    T.NAMES["mystery man"] = [("Q8090", 0, "exact")]
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)                   # off: the bookkeeping still runs, and must not fail
    assert hop(world, "place of death", "Mystery Man") == {"Lyon"}
    assert world._rec()["life"]["Q8090"] == (None, None)
