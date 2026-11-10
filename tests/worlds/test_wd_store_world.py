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
