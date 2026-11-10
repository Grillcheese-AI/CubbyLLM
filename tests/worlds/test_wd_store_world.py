"""Order 0055 step 3: the hot Wikidata store as a world (`cubbyllm/worlds/wd_store_world.py`), over a fake that answers
as `wdstore serve` does (snapshot, brief, labels, name, get with nodes, pointing_at). Pinned:
- values said exactly: a date at its precision (a year-precision date is a year, a Julian one its year only, a decade
  not said), a quantity with its unit's label, never an external id, never a deprecated statement; an ended statement
  dropped only on a relation whose value changes;
- names as the Wikidata5M world matches them: by label, then alias; "the" + a name; an accent or punctuation form
  kept only when the item's own name normalizes to the key; never a typo hit; a Wikimedia page is not an item;
- a name several items share gives none of their facts and notes the holders; the asker's choice reads one; a name
  of more than NAMESAKE_CAP items reads and lists none;
- an object's facts are its subjects' own truthy statements; a time-sensitive relation is the world's, dated 2025;
- the server's builder refuses to run with the flag on and no store (never a silent off).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from cubbyllm.worlds import wd_store_world as wsw  # noqa: E402
from cubbyllm.worlds.wd_store_values import say_time  # noqa: E402

PROPS = {"P31": ("instance of", "wikibase:WikibaseItem"), "P19": ("place of birth", "wikibase:WikibaseItem"),
         "P569": ("date of birth", "wikibase:Time"), "P2048": ("height", "wikibase:Quantity"),
         "P214": ("VIAF cluster ID", "wikibase:ExternalId"), "P69": ("educated at", "wikibase:WikibaseItem"),
         "P26": ("spouse", "wikibase:WikibaseItem"), "P1082": ("population", "wikibase:Quantity")}
DAY = {"kind": "wikibase:TimeValue", "wikibase:timePrecision": "11", "wikibase:timeCalendarModel": "wd:Q1985727"}


def st(pid, value, node=None, best=True, rank=1, ended=False):
    return {"pid": pid, "value": value, "value_node": node, "best": best, "rank": rank,
            "hot_qualifiers": [[582, {"typed": ["2001-01-01T00:00:00Z", "xsd:dateTime"]}]] if ended else []}


ITEMS = {   # id: (names [(lang, kind, text)], statements, nodes, classes)
    "Q1": ([("en", 0, "Ada Example"), ("en", 1, "Countess Ada")], [
        st(31, {"entity": "Q5"}), st(19, {"entity": "Q60"}),
        st(569, {"typed": ["1815-12-10T00:00:00Z", "xsd:dateTime"]}, "n1"),
        st(2048, {"typed": ["+1.65", "xsd:decimal"]}, "n2"), st(214, {"str": "12345"}),
        st(19, {"entity": "Q61"}, best=False, rank=0),                                    # deprecated
        st(69, {"entity": "Q70"}, ended=True), st(26, {"entity": "Q71"}, ended=True)],   # ended: kept / dropped
        {"n1": DAY, "n2": {"kind": "wikibase:QuantityValue", "wikibase:quantityUnit": "wd:Q11573"}}, ["Q5"]),
    "Q2": ([("en", 0, "Globe")], [st(19, {"entity": "Q60"})], {}, ["Q5"]),
    "Q3": ([("en", 0, "Globe")], [st(19, {"entity": "Q61"})], {}, ["Q5"]),
    "Q4": ([("en", 0, "Globe")], [st(31, {"entity": "Q4167410"})], {}, ["Q4167410"]),        # a disambiguation page
    "Q5": ([("en", 0, "human")], [], {}, []),
    "Q6": ([("en", 0, "The Quiet Band")], [st(31, {"entity": "Q5"})], {}, ["Q5"]),
    "Q7": ([("en", 0, "St. Elmo")], [st(19, {"entity": "Q60"})], {}, ["Q5"]),
    "Q60": ([("en", 0, "London")], [], {}, []), "Q61": ([("en", 0, "Paris")], [], {}, []),
    "Q70": ([("en", 0, "Oxford")], [], {}, []), "Q71": ([("en", 0, "Bob")], [], {}, []),
    "Q11573": ([("en", 0, "metre")], [], {}, []),
}
NAMES = {   # the store's own lookups: (text) -> hits, as its exact / folded / typo tiers give them
    "ada example": [("Q1", 0, "exact")], "countess ada": [("Q1", 1, "exact")], "globe": [(q, 0, "exact") for q in ("Q2", "Q3", "Q4")],
    "the quiet band": [("Q6", 0, "exact")], "st elmo": [("Q7", 0, "folded")], "london": [("Q60", 0, "exact")],
    "ada exampel": [("Q1", 0, "typo1")], "crowd": [(f"Q{100 + i}", 0, "exact") for i in range(30)],
}
for i in range(30):
    ITEMS[f"Q{100 + i}"] = ([("en", 0, "Crowd")], [st(19, {"entity": "Q60"})], {}, ["Q5"])


class FakeStore:
    def __init__(self):
        self.asked = []

    def __call__(self, r):
        self.asked.append(r["op"])
        op = r["op"]
        if op == "snapshot":
            return {"snapshot": "abc123"}
        if op == "brief":
            out = {}
            for i in r["ids"]:
                if i in PROPS:
                    out[i] = {"id": i, "label": PROPS[i][0], "datatype": PROPS[i][1], "statements": 1, "types": []}
                elif i in ITEMS:
                    out[i] = {"id": i, "label": ITEMS[i][0][0][2], "types": ITEMS[i][3], "statements": len(ITEMS[i][1]), "datatype": None}
            return out
        if op == "labels":
            return {i: ITEMS[i][0][0][2] for i in r["ids"] if i in ITEMS}
        if op == "name":
            return {"hits": [{"id": q, "kind": k, "how": how, "lang": "en", "key": r["text"]} for q, k, how in NAMES.get(r["text"], [])]}
        if op == "get":
            if r["id"] not in ITEMS:
                return {"found": False}
            names, sts, nodes, _ = ITEMS[r["id"]]
            return {"found": True, "id": r["id"], "names": [list(n) for n in names], "statements": sts,
                    **({"nodes": nodes} if r.get("nodes") else {})}
        if op == "pointing_at":
            hits = [[s["pid"], q, s["rank"], s["best"]] for q, (_n, sts, _d, _c) in ITEMS.items() for s in sts
                    if s["value"] == {"entity": r["value"]}]
            return {"total": len(hits), "hits": hits[:r["limit"]]}
        return {"error": f"unknown op {op}"}


@pytest.fixture()
def world():
    return wsw.WdStoreWorld(FakeStore(), "2025-12-29T23:00:38Z", statements=42)


def rels(world, name):
    return {(t.rel, t.obj) for _f, t in world._by_subj.get(name)}


def test_values_are_said_exactly_and_only_what_can_be(world):
    got = rels(world, "ada example")
    assert ("date of birth", "10 December 1815") in got and ("height", "1.65 metre") in got
    assert ("place of birth", "London") in got and ("place of birth", "Paris") not in got        # deprecated: never
    assert not any(r == "VIAF cluster ID" for r, _o in got)                                       # an external id: never
    assert ("educated at", "Oxford") in got and ("spouse", "Bob") not in got   # ended: kept, unless the value changes
    assert say_time("1815-00-00T00:00:00Z", {"wikibase:timePrecision": "9"}) == "1815"
    assert say_time("1700-03-01T00:00:00Z", {"wikibase:timePrecision": "11", "wikibase:timeCalendarModel": "wd:Q1985786"}) == "1700"
    assert say_time("-0500-01-01T00:00:00Z", {"wikibase:timePrecision": "9"}) == "500 BC"
    assert say_time("1810-01-01T00:00:00Z", {"wikibase:timePrecision": "8"}) is None and say_time("1810-01-01T00:00:00Z", None) is None


def test_names_by_label_then_alias_with_the_article_and_never_a_typo(world):
    assert rels(world, "ada example") == rels(world, "countess ada") != set()        # the alias tier, the same item
    assert world.label("countess ada") == "Ada Example" and world.label("q1") == "Ada Example"
    assert rels(world, "quiet band") == {("instance of", "human")}                 # "the" + the name
    assert rels(world, "st elmo") == {("place of birth", "London")}                 # punctuation: its own name normalizes
    assert rels(world, "ada exampel") == set()                                       # a typo hit is never read


def test_a_shared_name_reads_none_until_one_is_chosen_and_a_page_is_no_item(world):
    assert rels(world, "globe") == set()
    assert set(world.namesakes("globe")) == {"Q2", "Q3"}                             # the disambiguation page is not one
    plan = QuestionPlan(relations=[None], tail="place of birth of Globe", n_hop=1)
    assert world.hop(plan, 0, None) == [] and set(world.withheld("globe")) == {"Q2", "Q3"}
    assert {t.obj for _f, t in world.item_facts("globe", "Q3")} == {"Paris"}
    with world.pinned("globe", "globe", "Q2"):
        assert {t.obj for _f, t in world.hop(plan, 0, None)} == {"London"}


def test_a_crowded_name_reads_and_lists_none(world):
    assert rels(world, "crowd") == set() and world.namesakes("crowd") == {}
    assert world._entity.cache_info().currsize < 20                                  # none of its 30 items read


def test_an_objects_facts_are_its_subjects_own(world):
    got = {(t.subj, t.rel) for _f, t in world._by_obj.get("london")}
    assert ("Ada Example", "place of birth") in got and ("St. Elmo", "place of birth") in got


def test_the_world_says_its_date_and_its_time_sensitive_relations(world):
    assert world.snapshot == "2025" and "spouse" in world.time_sensitive and "place of birth" not in world.time_sensitive
    assert "VIAF cluster ID".lower() not in {r.lower() for r in world.relations()} and len(world) == 42
    assert "10 December 1815 is the date of birth of Ada Example" in world


def test_the_flag_on_with_no_store_stops_the_server(monkeypatch):
    from cubby_serve import wd_world
    monkeypatch.setattr(wd_world, "value", lambda k: None)
    with pytest.raises(SystemExit):
        wd_world.build(lambda m: None)


# ---- option (a): the store's kinds table, order 0037's rule on its shared names ----

PROPS["P36"] = ("capital", "wikibase:WikibaseItem")
ITEMS.update({
    "Q41": ([("en", 0, "country")], [], {}, []), "Q44": ([("en", 0, "painting")], [], {}, []),
    "Q40": ([("en", 0, "Gaul")], [st(31, {"entity": "Q41"}), st(36, {"entity": "Q60"})], {}, ["Q41"]),
    "Q43": ([("en", 0, "Gaul")], [st(31, {"entity": "Q44"}), st(19, {"entity": "Q61"})], {}, ["Q44"]),
    "Q300": ([("en", 0, "Rome")], [st(31, {"entity": "Q41"}), st(36, {"entity": "Q61"})], {}, ["Q41"]),
})
NAMES["gaul"] = [("Q40", 0, "exact"), ("Q43", 0, "exact")]
NAMES["rome"] = [("Q300", 0, "exact")] + [(f"Q{400 + i}", 0, "exact") for i in range(30)]
for i in range(30):
    ITEMS[f"Q{400 + i}"] = ([("en", 0, "Rome")], [st(31, {"entity": "Q44"}), st(19, {"entity": "Q60"})], {}, ["Q44"])


@pytest.fixture()
def kinds_world(tmp_path):
    from cubbyllm.worlds.wd_store_kinds import build
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kind_rels.tsv").write_text("Q41\tP31\t10\nQ41\tP36\t9\nQ44\tP31\t100\nQ44\tP19\t60\n", encoding="utf-8")
    (raw / "labels.tsv").write_text("Q41\tcountry\nQ44\tpainting\nP31\tinstance of\nP36\tcapital\nP19\tplace of birth\n", encoding="utf-8")
    (raw / "meta.tsv").write_text("snapshot\tabc123\nitems\t200\ntyped\t110\n", encoding="utf-8")
    out = str(tmp_path / "store.kinds.sqlite")
    build(str(raw), out)
    return wsw.WdStoreWorld(FakeStore(), "2025-12-29T23:00:38Z", kinds=out), out


def capital(world, name):
    world.clear_withheld()
    return {t.obj for _f, t in world.hop(QuestionPlan(relations=[None], tail=f"capital of {name}", n_hop=1), 0, None)}


def test_the_kinds_table_decides_a_shared_name_as_in_wikidata5m(world, kinds_world):
    w, _path = kinds_world
    assert w._kinds.carries("country", "capital") and not w._kinds.carries("painting", "capital")
    assert capital(w, "Gaul") == {"London"}                      # the country: the only kind that carries a capital
    assert capital(world, "Gaul") == set() and set(world.withheld("gaul")) == {"Q40"}   # no table: held, as before


def test_a_crowded_name_is_narrowed_to_what_the_rule_could_pick(kinds_world):
    w, _path = kinds_world
    assert capital(w, "Rome") == {"Paris"}                       # 31 items: the country read, no painting read
    assert w._entity.cache_info().currsize < 10                  # (of 31 items: the country and its values)
    assert {t.obj for _f, t in w.item_facts("rome", "Q405")} == {"painting", "London"}    # the asker's choice reads it


def test_a_fact_the_world_handed_out_is_held_whatever_its_names_count(kinds_world):
    w, _path = kinds_world                                       # Rome: 31 items, past the cap, so its name reads none
    assert "Paris is the capital of Rome" not in w               # not yet handed out: the name's rule, as before
    assert capital(w, "Rome") == {"Paris"}
    assert "Paris is the capital of Rome" in w                   # the loop names this world as its holder ("as of 2025")
    assert "Lyon is the capital of Rome" not in w


def test_a_table_from_another_state_of_the_store_is_not_read(kinds_world):
    _w, path = kinds_world
    from cubbyllm.worlds.wd_store_kinds import for_store
    assert for_store(path, "abc123") is not None and for_store(path, "another") is None


# ---- the cue table and the fictional classes (wd_store_cues) ----

PROPS["P166"] = ("award received", "wikibase:WikibaseItem")
ITEMS.update({
    "Q520": ([("en", 0, "ship")], [], {}, []), "Q610": ([("en", 0, "Star Trek location")], [], {}, []),
    "Q510": ([("en", 0, "Nobel Prize in Physics")], [], {}, []), "Q511": ([("en", 0, "Blue Riband")], [], {}, []),
    "Q500": ([("en", 0, "Curie")], [st(31, {"entity": "Q5"}), st(166, {"entity": "Q510"})], {}, ["Q5"]),
    "Q501": ([("en", 0, "Curie")], [st(31, {"entity": "Q520"}), st(166, {"entity": "Q511"})], {}, ["Q520"]),
    "Q600": ([("en", 0, "Albion")], [st(31, {"entity": "Q41"}), st(36, {"entity": "Q60"})], {}, ["Q41"]),
    "Q601": ([("en", 0, "Albion")], [st(31, {"entity": "Q610"}), st(36, {"entity": "Q61"})], {}, ["Q610"]),
    "Q602": ([("en", 0, "Holmes")], [st(31, {"entity": "Q610"}), st(36, {"entity": "Q60"})], {}, ["Q610"]),
})
NAMES.update({"curie": [("Q500", 0, "exact"), ("Q501", 0, "exact")], "albion": [("Q600", 0, "exact"), ("Q601", 0, "exact")],
              "holmes": [("Q602", 0, "exact")]})
# the refinements (wd_store_refine): a property that fits anything, items that agree, a fictional item alone fitting
PROPS["P1889"] = ("different from", "wikibase:WikibaseItem")
ITEMS.update({
    "Q700": ([("en", 0, "opera")], [], {}, []),
    "Q710": ([("en", 0, "Thespis")], [st(31, {"entity": "Q700"}), st(1889, {"entity": "Q60"})], {}, ["Q700"]),
    "Q711": ([("en", 0, "Thespis")], [st(31, {"entity": "Q5"}), st(19, {"entity": "Q60"})], {}, ["Q5"]),
    "Q720": ([("en", 0, "Twinland")], [st(31, {"entity": "Q41"}), st(36, {"entity": "Q60"})], {}, ["Q41"]),
    "Q721": ([("en", 0, "Twinland")], [st(31, {"entity": "Q41"}), st(36, {"entity": "Q60"})], {}, ["Q41"]),
    "Q730": ([("en", 0, "Vulcan")], [st(31, {"entity": "Q44"}), st(19, {"entity": "Q60"})], {}, ["Q44"]),
    "Q731": ([("en", 0, "Vulcan")], [st(31, {"entity": "Q610"}), st(36, {"entity": "Q61"})], {}, ["Q610"]),
})
NAMES.update({"thespis": [("Q710", 0, "exact"), ("Q711", 0, "exact")], "twinland": [("Q720", 0, "exact"), ("Q721", 0, "exact")],
              "vulcan": [("Q730", 0, "exact"), ("Q731", 0, "exact")]})


@pytest.fixture()
def full_world(tmp_path):
    from cubbyllm.worlds import wd_store_cues, wd_store_kinds
    raw = tmp_path / "raw_full"
    raw.mkdir()
    (raw / "kind_rels.tsv").write_text("Q41\tP31\t10\nQ41\tP36\t9\nQ44\tP31\t100\nQ5\tP31\t50\nQ5\tP166\t40\n"
                                       "Q520\tP31\t20\nQ520\tP166\t15\nQ610\tP31\t5\nQ610\tP36\t4\n"
                                       "Q700\tP31\t10\nQ700\tP1889\t8\n", encoding="utf-8")
    (raw / "labels.tsv").write_text("Q41\tcountry\nQ44\tpainting\nQ5\thuman\nQ520\tship\nQ610\tStar Trek location\n"
                                    "Q700\topera\nP31\tinstance of\nP36\tcapital\nP166\taward received\nP1889\tdifferent from\n",
                                    encoding="utf-8")
    (raw / "meta.tsv").write_text("snapshot\tabc123\nitems\t200\ntyped\t185\n", encoding="utf-8")
    wd_store_kinds.build(str(raw), str(tmp_path / "s.kinds.sqlite"))
    cues = tmp_path / "cues"
    cues.mkdir()
    (cues / "word_kinds.tsv").write_text("nobel\tQ5\t150\nriband\tQ520\t120\n", encoding="utf-8")
    (cues / "fictional.tsv").write_text("Q14897293\nQ610\n", encoding="utf-8")
    (cues / "meta.tsv").write_text("snapshot\tabc123\n", encoding="utf-8")
    wd_store_cues.build(str(cues), str(raw / "labels.tsv"), str(tmp_path / "s.cues.sqlite"))
    return wsw.WdStoreWorld(FakeStore(), "2025-12-29T23:00:38Z", kinds=str(tmp_path / "s.kinds.sqlite"),
                            cues=str(tmp_path / "s.cues.sqlite"))


def awards(world, name, cues=()):
    world.clear_withheld()
    with world.cued(list(cues)):
        return {t.obj for _f, t in world.hop(QuestionPlan(relations=[None], tail=f"award received of {name}", n_hop=1), 0, None)}


def test_a_cue_word_rules_a_kind_out(full_world):
    assert awards(full_world, "Curie") == set()                   # a person and a ship both hold awards: no pick alone
    assert full_world.decision("curie")["fit"] == ["Q500", "Q501"]
    assert awards(full_world, "Curie", ["nobel"]) == {"Nobel Prize in Physics"}   # "Nobel" never goes with a ship


def test_a_fictional_item_steps_aside_only_when_a_real_one_fits(full_world):
    assert capital(full_world, "Albion") == {"London"}            # the real country fits: the Star Trek location steps aside
    d = full_world.decision("albion")
    assert (d["fit"], d["stepped_aside"]) == (["Q600"], ["Q601"])
    assert set(full_world.namesakes("albion")) == {"Q600", "Q601"}  # still among the name's items (the closing line)
    assert capital(full_world, "Vulcan") == {"Paris"}            # no real item fits (a painting): the fictional one does
    assert capital(full_world, "Holmes") == {"London"}            # only a fictional item carries the name: it stands


def rel_of(world, rel, name):
    world.clear_withheld()
    return {t.obj for _f, t in world.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)}


def test_a_property_that_fits_anything_never_picks(full_world):
    assert rel_of(full_world, "different from", "Thespis") == set()      # hop 42: the opera alone holds it, no pick
    assert full_world.decision("thespis")["neutral"] and set(full_world.withheld("thespis")) == {"Q710", "Q711"}


def test_items_that_agree_are_said_once(full_world):
    assert capital(full_world, "Twinland") == {"London"}
    assert full_world.decision("twinland")["agree"] == ["Q720", "Q721"]
