"""The possibility oracle: what a world says about a fact before the host keeps it (H-F2 M4, 2026-09-29)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import cubbyllm.bridges.possibility as mod
from cubbyllm.bridges.possibility import (LATENT, Possibility, PossibilityOracle, PossibleRelations,
                                          StorePossibility, Verdict, same_subject_clash)
from cubbyllm.core.protocols import Wiring

sys.path.append(str(Path(__file__).resolve().parents[2] / "standin"))   # the stand-in is not installed (appended: its
                                                                          # hypothesis.py must not shadow pytest's plugin)
from worlds import FactStore, route_world          # noqa: E402
from capsules import CapsuleStore                  # noqa: E402

GEO = ["Paris is the capital of France", "Berlin is the capital of Germany", "Rome is the capital of Italy",
       "Seine is the river of Paris", "Spree is the river of Berlin"]
MUSIC = ["Thriller is the album of Michael Jackson", "Abbey Road is the album of The Beatles",
         "Liverpool is the hometown of The Beatles"]


def _worlds():
    return {"geo": FactStore(GEO, name="geo"), "music": FactStore(MUSIC, name="music")}


def test_wiring_and_protocol():
    assert mod.__wiring__ is Wiring.STANDALONE
    assert isinstance(StorePossibility(_worlds()), PossibilityOracle)


def test_clash_rule_is_same_subject_same_relation_different_object():
    assert same_subject_clash("Lyon is the capital of France", GEO) == "Paris is the capital of France"
    assert same_subject_clash("Paris is the capital of France", GEO) is None            # the same fact
    assert same_subject_clash("Lyon is the largest city of France", GEO) is None        # another relation
    assert same_subject_clash("not a fact", GEO) is None


def test_member_is_possible_with_full_support():
    o = StorePossibility(_worlds())
    p = o.possible("Paris is the capital of France")
    assert p.verdict is Verdict.POSSIBLE and p.world_id == "geo" and p.why == "member"
    assert p.support == (("Paris is the capital of France", 1.0),) and p.score == 1.0
    assert p.latent is None and p.predicted == ()


def test_contradiction_is_impossible_and_names_the_member():
    o = StorePossibility(_worlds())
    p = o.possible("Lyon is the capital of France")
    assert p.impossible and p.world_id == "geo"
    assert p.support[0][0] == "Paris is the capital of France" and p.why == "contradiction"


def test_contradiction_in_another_world_is_still_impossible():
    """The planted fact's new object pulls its best hit into the music world; the geo world holds the subject."""
    w = _worlds()
    f = "Michael Jackson Thriller Beatles is the capital of France"   # obj | capital | France, worded like music
    o = StorePossibility(w)
    assert o.route(f)[0] == "music"
    p = o.possible(f)
    assert p.impossible and p.world_id == "geo" and p.why == "contradiction_elsewhere"
    assert p.support[0][0] == "Paris is the capital of France"
    assert not StorePossibility(w, cross_world=False).possible(f).impossible   # time worlds: off


def test_uncovered_is_unknown_and_goes_latent():
    o = StorePossibility(_worlds())
    p = o.possible("Quuxium is the element of Fnordovia")
    assert p.unknown and p.latent == LATENT and p.world_id == LATENT and p.why == "below_tau_match"


def test_covered_but_unsupported_is_unknown_in_the_routed_world():
    o = StorePossibility(_worlds(), tau_answer=0.95)                # only an exact member supports
    p = o.possible("Lyon is the largest city of France")
    assert p.unknown and p.world_id == "geo" and p.latent == LATENT and p.why == "covered_unsupported"


def test_resemblance_supports_at_tau_answer():
    o = StorePossibility(_worlds(), tau_answer=0.3)
    p = o.possible("Lyon is the largest city of France")
    assert p.possible and p.why == "resembles" and p.world_id == "geo"
    assert all(s >= 0.3 for _m, s in p.support) and p.score == p.support[0][1]


def test_route_matches_route_world_and_reports_margin():
    w = _worlds()
    o = StorePossibility(w)
    for q in ("Which river runs through Berlin?", "Whose album is Thriller?", "capital of Italy"):
        name, score, margin = o.route(q)
        assert (name, score) == route_world(w, q)                   # parity with today's best-top-score rule
        assert margin is not None and margin >= 0
    assert route_world(w, "Whose album is Thriller?", oracle=o) == ("music", o.route("Whose album is Thriller?")[1])
    assert o.worlds() == ["geo", "music"]


def test_latent_world_is_never_routed_to():
    w = _worlds()
    w[LATENT] = FactStore(["Quuxium is the element of Fnordovia"], name=LATENT)
    o = StorePossibility(w)
    assert o.worlds() == ["geo", "music"]
    assert o.possible("Quuxium is the element of Fnordovia").unknown


def test_ambiguous_margin_is_unknown_when_gated():
    w = {"a": FactStore(["Paris is the capital of France"], name="a"),
         "b": FactStore(["Paris is the capital of France"], name="b")}      # a tie: margin 0
    assert StorePossibility(w, tau_margin=0.02).possible("Paris is the capital of France").why == "ambiguous_margin"
    assert StorePossibility(w).possible("Paris is the capital of France").possible   # no gate: ties break by name
    assert StorePossibility(w).route("Paris")[0] == "a"


def test_relation_kind_reused_paraphrase_unknown():
    o = StorePossibility(_worlds())
    assert o.possible("capital", kind="relation").possible                    # reused (3 facts)
    assert o.possible("hometown", kind="relation").unknown                    # one fact: not reused
    assert o.possible("the capital", kind="relation").possible                # normalized
    known = PossibleRelations(o)
    assert "capital" in known and "hometown" not in known
    assert known.match("capital city") is None or known.match("capital city") == "capital"


def test_question_kind_routes_and_supports_by_hits():
    o = StorePossibility(_worlds(), tau_answer=0.3)
    p = o.possible("What is the capital of Germany?", kind="question")
    assert p.world_id == "geo" and p.possible and p.support
    assert o.asked[-1]["kind"] == "question"


def test_route_world_falls_back_when_the_oracle_fails_or_names_a_stranger():
    class Broken:
        def possible(self, *a, **k): raise RuntimeError
        def route(self, text): raise RuntimeError
        def worlds(self): return []

    class Stranger:
        def possible(self, *a, **k): raise RuntimeError
        def route(self, text): return "mars", 0.9, None
        def worlds(self): return ["mars"]
    w = _worlds()
    assert route_world(w, "Whose album is Thriller?", oracle=Broken()) == route_world(w, "Whose album is Thriller?")
    assert route_world(w, "Whose album is Thriller?", oracle=Stranger()) == route_world(w, "Whose album is Thriller?")


# -- how the capsule stores read it ------------------------------------------------------------------------

def test_capsule_store_refuses_impossible_and_ledgers_it():
    store = CapsuleStore(GEO, name="geo", gpu=False)
    store.oracle = StorePossibility({"geo": store})
    assert store.add("Lyon is the capital of France", source="nick") is False
    assert "Lyon is the capital of France" not in store
    assert store.last_refusal["world"] == "geo" and store.last_refusal["verdict"] == "impossible"
    assert store.ledger[-1]["support"][0][0] == "Paris is the capital of France"
    assert store.ledger[-1]["source"] == "nick"


def test_capsule_store_keeps_unknown_as_latent_and_attests():
    store = CapsuleStore(GEO, name="geo", gpu=False)
    store.oracle = StorePossibility({"geo": store})
    f = "Quuxium is the element of Fnordovia"
    assert store.add(f, source="lfm") is True
    assert store.meta[f]["latent"] == LATENT and store.meta[f]["possibility"] == "unknown"
    assert store.provenance[f] == "lfm (latent)"                    # learn.py's tier: not spoken until attested
    assert store.latent_facts() == [f]
    assert store.attest(f, source="wiki") is True
    assert store.meta[f]["latent"] is None and store.meta[f]["was_latent"] == LATENT
    assert store.provenance[f] == "lfm, wiki" and store.latent_facts() == []
    assert store.attest(f) is False and store.attest("never stored") is False


def test_capsule_store_keeps_possible_with_support_and_a_passed_in_verdict():
    store = CapsuleStore(GEO, name="geo", gpu=False)
    store.oracle = StorePossibility({"geo": store}, tau_answer=0.3)
    f = "Lyon is the largest city of France"
    assert store.add(f) is True
    assert store.meta[f]["possibility"] == "possible" and store.meta[f]["support"] and store.meta[f]["latent"] is None
    given = Possibility("elsewhere", Verdict.UNKNOWN, 0.1, latent="elsewhere", why="test")
    assert store.add("Nice is the port of France", possibility=given) is True
    assert store.meta["Nice is the port of France"]["latent"] == "elsewhere"     # judged once, by the caller
    assert store.add("Nice is the port of France", possibility=given) is False    # a duplicate is still a duplicate


def test_capsule_store_without_oracle_is_unchanged():
    store = CapsuleStore(GEO, name="geo", gpu=False)
    assert store.add("Lyon is the capital of France") is True                 # serve.contradiction is the caller's gate
    assert store.meta["Lyon is the capital of France"]["possibility"] is None and store.ledger == []


def test_capsule_store_round_trips_verdicts_and_ledger(tmp_path):
    store = CapsuleStore(GEO, name="geo", gpu=False)
    store.oracle = StorePossibility({"geo": store})
    store.add("Lyon is the capital of France", source="nick")
    store.add("Quuxium is the element of Fnordovia", source="lfm")
    store.save(tmp_path / "geo")
    back = CapsuleStore.load(tmp_path / "geo", gpu=False)
    assert back.texts == store.texts and back.ledger == store.ledger
    assert back.meta["Quuxium is the element of Fnordovia"]["latent"] == LATENT
    assert back.provenance["Quuxium is the element of Fnordovia"] == "lfm (latent)"
    assert back.meta["Paris is the capital of France"]["possibility"] is None      # a seed, added before the oracle


@pytest.mark.parametrize("n", [200])
def test_possible_is_cheap(n):
    import time
    store = FactStore([f"Town{i} is the capital of Land{i}" for i in range(2000)], name="big")
    o = StorePossibility({"big": store, "geo": FactStore(GEO, name="geo")})
    t0 = time.perf_counter()
    for i in range(n):
        o.possible(f"Town{i} is the capital of Land{i}")
    per = (time.perf_counter() - t0) / n
    # the lexical fallback walks python postings; the hypothesis gate (1 ms) is measured on the real stores
    # with the encoder (one matmul per world)
    assert per < 0.02, f"{per * 1e3:.2f} ms per possible()"
