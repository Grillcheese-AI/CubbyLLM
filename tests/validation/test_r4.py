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
