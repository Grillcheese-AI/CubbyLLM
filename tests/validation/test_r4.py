"""Order 0046, R4 (``validation/retrieval_bench/r4_*.py``): the source's own order (preferred rank, single-value, no
stated end), the interval filter for an explicit year, "present tense means now", and the statements read with their
rank and time. Nothing ranked; several values a set or a refusal."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_bench import r4_statements as S  # noqa: E402
from retrieval_bench import r4_values as V  # noqa: E402


def t(y):
    return {"y": y, "m": 0, "d": 0, "prec": 9}


MADRID = {"v": "Q2807", "rank": "normal", "start": t(1561)}
TOLEDO = {"v": "Q5836", "rank": "normal", "start": t(1085), "end": t(1561)}
VALLADOLID = {"v": "Q8356", "rank": "normal", "start": t(1601), "end": t(1606)}


def test_the_source_order_never_speaks_an_ended_capital_beside_madrid():
    caps = [TOLEDO, MADRID, VALLADOLID]
    assert V.order(caps, single=True) == {"spoken": ["Q2807"], "why": "single-value: the one with no end"}
    assert V.order([dict(TOLEDO, rank="preferred"), MADRID], single=False)["spoken"] == ["Q5836"]   # the source's own word
    assert V.order([TOLEDO, VALLADOLID], single=True)["spoken"] is None                           # every value ended
    two_open = [{"v": "Q1", "rank": "normal"}, {"v": "Q2", "rank": "normal"}]
    assert V.order(two_open, single=True)["spoken"] is None                                        # never a pick
    assert V.order(two_open, single=False)["spoken"] == ["Q1", "Q2"]                               # a set, whole
    assert V.order([{"v": None, "rank": "normal"}, dict(MADRID, rank="deprecated")], single=False)["spoken"] is None


def test_an_explicit_year_keeps_only_the_values_placed_at_it():
    caps = [TOLEDO, MADRID, VALLADOLID]
    assert V.interval(caps, True, 1400)["spoken"] == ["Q5836"] and V.interval(caps, True, 2010)["spoken"] == ["Q2807"]
    assert V.interval(caps, True, 1603)["spoken"] is None    # Valladolid and Madrid's open interval: single-value refuses
    assert V.interval(caps, False, 1603)["spoken"] == ["Q2807", "Q8356"]     # a property that may hold several: the set
    assert V.interval([{"v": "Q1", "rank": "normal"}], False, 2010)["spoken"] is None   # an undated value is never placed
    award = {"v": "Q9", "rank": "normal", "point": t(2001)}
    assert V.holds_at(award, 2001) is True and V.holds_at(award, 2002) is False
    assert V.holds_at({"v": "Q1", "end": t(1990)}, 2000) is False and V.holds_at({"v": "Q1", "end": t(1990)}, 1980) is None
    assert V.interval(caps, True, None) == V.order(caps, True)


def test_present_tense_means_now_only_in_its_own_arm():
    vals = [{"v": "Q1", "rank": "normal", "start": t(1990), "end": t(2050)}, {"v": "Q2", "rank": "normal", "start": t(1900)}]
    today = dt.date(2026, 11, 4)
    assert V.order(vals, single=True)["spoken"] == ["Q2"]                                         # no stated end
    assert V.now(vals, True, None, "What is the currency of X?", today)["spoken"] is None          # both hold now
    assert V.now(vals, False, None, "What is the currency of X?", today)["spoken"] == ["Q1", "Q2"]
    assert V.now(vals, True, None, "What was the currency of X?", today) == V.order(vals, True)    # past tense: no "now"
    assert V.asked_year("what is the capital of spain in 2010?") == 2010 and V.asked_year("x", [1993]) == 1993
    assert V.asked_year("from 1990 to 2000?") is None


def test_judging_counts_a_value_outside_the_gold_against_the_answer():
    assert V.judge(["Q1"], {"Q1", "Q2"}) == "right" and V.judge(["Q1", "Q3"], {"Q1"}) == "partly"
    assert V.judge(["Q3"], {"Q1"}) == "wrong" and V.judge(None, {"Q1"}) == "refused"


def test_statements_keep_their_rank_their_time_and_the_era():
    claims = {"P36": [
        {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q2807"}}}, "rank": "preferred",
         "qualifiers": {"P580": [{"datavalue": {"value": {"time": "+1561-00-00T00:00:00Z", "precision": 9}}}]}},
        {"mainsnak": {"snaktype": "novalue"}, "rank": "normal"},
        {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"time": "-0753-04-21T00:00:00Z"}}}, "rank": "normal",
         "qualifiers": {"P585": [{"datavalue": {"value": {"time": "-0753-00-00T00:00:00Z", "precision": 9}}}]}}]}
    v = S.values_of(claims, "P36")
    assert v[0] == {"v": "Q2807", "rank": "preferred", "start": {"y": 1561, "m": 0, "d": 0, "prec": 9}}
    assert v[1]["v"] is None and v[2]["point"]["y"] == -753


# Cowork's four rules after dev (``r4_rules``), read once on CronQuestions' test.
from retrieval_bench import r4_rules as R  # noqa: E402


SECRETARY = {"v": "Q100", "rank": "normal", "start": t(1782)}            # began 1782, no end stated


def test_rule_1_an_open_end_is_not_placed_once_the_record_says_it_ended():
    ends = {"Q1": None, "Q100": 1823}                                       # the holder died in 1823
    assert R.why_ended(SECRETARY, [SECRETARY], 1800, "Q1", ends, False) is None
    assert R.why_ended(SECRETARY, [SECRETARY], 2005, "Q1", ends, False) == "the value ended in 1823"
    assert R.why_ended(SECRETARY, [SECRETARY], 2005, "Q1", {"Q1": 1900}, False) == "the asked item ended in 1900"
    nxt = {"v": "Q101", "rank": "normal", "start": t(1830)}
    vals = [SECRETARY, nxt]
    assert R.why_ended(SECRETARY, vals, 1840, "Q1", {}, False) is None                      # several may hold: no
    assert R.why_ended(SECRETARY, vals, 1840, "Q1", {}, True) == "a later value started in 1830"   # one at a time
    assert R.why_ended(dict(SECRETARY, replaced=True), vals, 1840, "Q1", {}, False) == "a later value started in 1830"
    assert R.why_ended(SECRETARY, vals, 1830, "Q1", {}, True) is None       # the successor's own year: both stand
    assert R.why_ended(SECRETARY, vals, 1840, "Q1", {"Q100": "unread"}, False) == "the value's end could not be read"
    closed = {"v": "Q5", "start": t(1900), "end": t(1950)}
    assert R.why_ended(closed, [closed], 1920, "Q1", {"Q5": 1910}, True) is None            # a stated end decides
    keep, dropped = R.drop_ended(vals, 1840, "Q1", {}, True)
    assert keep == [nxt] and dropped == [("Q100", "a later value started in 1830")]
    assert R.drop_ended(vals, None, "Q1", {}, True) == (vals, [])
    assert R.ended_ids(vals + [closed, {"v": "1990"}], "Q1") == ["Q1", "Q100", "Q101"]


def test_rule_2_more_than_ten_values_is_a_count_and_ask():
    eleven = {"spoken": [f"Q{i}" for i in range(11)], "why": "x"}
    assert R.bound(eleven) == {"spoken": None, "ask": 11, "why": "11 values: the count and ASK"}
    ten = {"spoken": [f"Q{i}" for i in range(10)], "why": "x"}
    assert R.bound(ten) is ten and R.bound({"spoken": None}) == {"spoken": None}


class _Src:
    """A source with Wikidata's search and claims: Gibraltar is a territory, Spain a country."""
    CLAIMS = {"Q6256": {"P279": ["Q1048835"]}, "Q29": {"P31": ["Q6256"]}, "Q1410": {"P31": ["Q46395"]},
              "Q46395": {"P279": ["Q161243"]}}

    def _get(self, params):
        return {"search": [{"id": "Q6256", "label": "country"}]} if params.get("search") == "country" else {}

    def _claims(self, q):
        return {p: [{"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": x}}}} for x in xs]
                for p, xs in self.CLAIMS.get(q, {}).items()}


def test_rule_3_the_class_asked_drops_a_value_of_another_class():
    q, res = "what country is the iberian peninsula located in?", {"spoken": ["Q1410", "Q29"], "why": "set"}
    out = R.class_filter(_Src(), q, res, {"country"}, exempt=False)
    assert out["spoken"] == ["Q29"] and out["class_dropped"] == ["Q1410"]
    assert R.class_filter(_Src(), q, {"spoken": ["Q1410"]}, None, exempt=False)["spoken"] is None
    kept = R.class_filter(_Src(), q, res, {"country"}, exempt=True)       # the serving path's exemption: the word is
    assert kept["spoken"] == ["Q1410", "Q29"]                              # the relation asked, so no check
    assert R.class_filter(_Src(), "who was it?", res, None) is res         # no class named


def test_rule_4_gold_ids_follow_merges_and_a_namesake_stays_wrong():
    gold = R.gold_through_merges({"Q7"}, {"Q7": "Q8"})
    assert gold == {"Q7", "Q8"} and V.judge(["Q8"], gold) == "right" and V.judge(["Q9"], gold) == "wrong"


def test_statements_read_replaced_by_and_an_items_own_end(monkeypatch, tmp_path):
    claims = {"P39": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q5"}}}, "rank": "normal",
                       "qualifiers": {"P1366": [{"datavalue": {"value": {"id": "Q6"}}}]}}]}
    assert S.values_of(claims, "P39")[0]["replaced"] is True
    import retrieval_sets.common as C
    died = {"P570": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"time": "+1823-05-01T00:00:00Z"}}},
                      "rank": "normal"}]}
    monkeypatch.setattr(C, "entities", lambda src, ids, **k: {q: {"claims": died} for q in ids if q == "Q100"})
    st = S.Statements(None, path=str(tmp_path / "s.jsonl"))
    assert st.ends(["Q100", "Q1"]) == {"Q100": 1823, "Q1": "unread"}


class _VSrc:
    """P17's value type: country or dependent territory, instance of; Gibraltar a British overseas territory."""
    def __init__(self, classes):
        st = {"mainsnak": {"datavalue": {"value": {"id": "Q21510865"}}}, "rank": "normal",
              "qualifiers": {"P2308": [{"datavalue": {"value": {"id": c}}} for c in classes],
                             "P2309": [{"datavalue": {"value": {"id": "Q21503252"}}}]}}
        snak = lambda x: {"mainsnak": {"datavalue": {"value": {"id": x}}}, "rank": "normal"}   # noqa: E731
        self.claims = {"P17": {"P2302": [st]}, "Q29": {"P31": [snak("Q3624078")]}, "Q3624078": {"P279": [snak("Q6256")]},
                       "Q1410": {"P31": [snak("Q46395")]}, "Q46395": {"P279": [snak("Q161243")]}, "P99": {}}

    def _claims(self, q):
        return self.claims.get(q, {})


def test_rule_3_valuetype_checks_each_value_against_the_propertys_type():
    from retrieval_bench import r4_valuetype as VT
    res = {"spoken": ["Q1410", "Q29"], "why": "set"}
    strict = VT.check(_VSrc(["Q6256"]), "P17", res, {})               # countries only: Gibraltar drops
    assert strict["spoken"] == ["Q29"] and strict["type_dropped"] == ["Q1410"]
    wide = VT.check(_VSrc(["Q6256", "Q161243"]), "P17", res, {})      # dependent territories too: it stands
    assert wide["spoken"] == ["Q1410", "Q29"] and wide["value_type"] == "every value of the type"
    assert VT.check(_VSrc(["Q6256"]), "P99", res, {})["value_type"] == "no constraint"
    assert VT.check(_VSrc(["Q6256"]), "P17", {"spoken": ["Q1410"]}, {})["spoken"] is None
