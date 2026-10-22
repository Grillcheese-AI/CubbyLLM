"""Pins for standin/data/import_science.py: the host keeps a proposed science fact only when both ends are
copied from the row, the relation is in the vocabulary and the template round-trips; the worlds hold a
first statement as latent, attest it on a second row, refuse a clash on a functional relation and not on a
many-valued one; only attested facts are mounted. Fakes only (no network, no LLM). Run:
python -m pytest standin/tests/test_import_science.py -q -p no:hypothesispytest"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.append(p)

import import_science as s  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402

ROW = {"id": "MegaScience/TextbookReasoning#1000", "dataset": "MegaScience/TextbookReasoning", "subject": "physics",
       "question": "What is the definition of moment of momentum, and how does it differ from momentum?",
       "answer": "Moment of momentum (also called angular momentum) is given by \\( \\mathbf{L} = \\mathbf{r} \\times "
                 "\\mathbf{p} \\). Momentum is \\( \\mathbf{p} = m\\mathbf{v} \\). Bacteria are 0.5-5 µm long."}
TN = normalize(s.plain(ROW["question"] + " " + ROW["answer"]))


def test_latex_and_symbols_become_plain_text_the_grounding_can_match():
    assert "l r x p" in TN and "p mv" in TN and "0 5 5 um" in TN
    assert s.plain("\\frac{a}{b} \\cdot \\alpha") == "(a)/(b) * alpha"


def test_the_host_keeps_only_copied_vocabulary_facts_that_round_trip():
    ok = {"subject": "moment of momentum", "relation": "other name", "object": "angular momentum"}
    assert s.check_fact(ok, TN) == (None, "angular momentum is the other name of moment of momentum")
    assert s.check_fact({"subject": "angular momentum", "relation": "formula", "object": "L = r × p"}, TN)[0] is None
    assert s.check_fact({**ok, "object": "spin"}, TN)[0] == "ungrounded"            # the row never says it
    assert s.check_fact({**ok, "relation": "cousin"}, TN)[0] == "relation"
    assert s.check_fact({"subject": "p", "relation": "formula", "object": "m v"}, TN)[0] == "variable"
    assert s.check_fact({"subject": "dip angle", "relation": "value", "object": "50 deg"}, TN)[0] == "relation"
    assert s.plain("H₂O and Ca²⁺") == "H_2O and Ca^2^+"
    assert s.check_fact({**ok, "object": "角动量"}, TN)[0] == "non_ascii"
    assert s.check_fact({**ok, "object": "moment of momentum"}, TN)[0] == "trivial"
    assert s.check_fact("not a dict", TN)[0] == "shape"


def test_extract_records_every_candidate_with_the_hosts_verdict():
    class Fake:
        def chat(self, system, user, max_tokens=600, temperature=None):
            assert "relation is exactly one of" in system and "moment of momentum" in user.lower()
            return ('Sure: [{"subject": "moment of momentum", "relation": "other name", "object": "angular momentum"},'
                    ' {"subject": "angular momentum", "relation": "inventor", "object": "Newton"}]')
    out, stats = s.extract([ROW], Fake(), log=lambda *a: None)
    assert stats["rows"] == 1 and stats["candidates"] == 2 and stats["kept"] == 1 and stats["rejected:relation"] == 1
    assert out[0]["fact"] and out[0]["area"] == "physics" and out[0]["row"] == ROW["id"] and out[1]["fact"] is None
    assert s.parse_candidates("no json here") == [] and s.parse_candidates("[1, 2") == []

    class Uncached:
        def chat(self, *a, **k):
            return ""
    out, stats = s.extract([ROW, ROW], Uncached(), log=lambda *a: None, skip_empty=True)
    assert out == [] and stats["rows_skipped_uncached"] == 2 and "rows" not in stats


def _f(fact, row, area="physics"):
    return {"row": row, "dataset": "MegaScience/TextbookReasoning", "area": area, "fact": fact, "why": None}


def test_worlds_hold_latent_attest_on_a_second_row_and_refuse_functional_clashes_only(tmp_path):
    facts = [_f("kg m^2/s is the unit of angular momentum", "r1"),
             _f("kg m^2/s is the unit of angular momentum", "r1"),              # the same row again: nothing new
             _f("kg m^2/s is the unit of angular momentum", "r2"),              # a second row: attested
             _f("N m is the unit of angular momentum", "r3"),                   # functional clash: refused
             _f("spinning top is the example of angular momentum", "r4"),
             _f("gyroscope is the example of angular momentum", "r5"),          # many-valued: a new fact
             _f("L is the symbol of angular momentum", "r6", area="chemistry"),
             _f("Gyroscope is the example of Angular Momentum.", "r7")]      # the same fact up to case: attests
    _w, default = s.build_worlds(facts)                     # the default: no relation is functional in textbook science
    assert "refused" not in default and default["second_object:unit"] == 1
    worlds, stats = s.build_worlds(facts, functional={"unit"})
    phys = worlds["science_physics"]
    assert stats["attested"] == 2 and stats["refused"] == 1 and stats["second_object_multivalued"] == 1
    assert "kg m^2/s is the unit of angular momentum" not in phys.latent_facts()
    assert "gyroscope is the example of angular momentum" not in phys.latent_facts()
    assert "spinning top is the example of angular momentum" in phys.latent_facts()
    assert phys.ledger and phys.ledger[0]["fact"] == "N m is the unit of angular momentum"
    assert "science_chemistry" in worlds and stats["attested_facts"] == 2 and stats["latent_facts"] == 2
    s.save_worlds(worlds, str(tmp_path), ["MegaScience/TextbookReasoning"], stats)
    meta = json.loads((tmp_path / "science_physics" / "world.json").read_text(encoding="utf-8"))
    assert meta["license"] == ["CC-BY-NC-SA-4.0"] and meta["refused"] == 1

    class Brain:
        worlds = {}
    mounted = s.mount_science(Brain(), str(tmp_path))
    assert mounted == {"science_physics": 2}, "only attested facts are mounted; a world with none is not"
    assert Brain.worlds["science_physics"].texts == ["kg m^2/s is the unit of angular momentum",
                                                     "gyroscope is the example of angular momentum"]


def test_fetch_pages_the_rows_api_at_spread_offsets():
    calls = []

    def get(url):
        calls.append(url)
        if url.endswith("length=1"):
            return {"num_rows_total": 1000}
        off = int(url.split("offset=")[1].split("&")[0])
        return {"rows": [{"row_idx": off + i, "row": {"question": f"q{off + i}", "answer": "a",
                                                      "subject": "math" if i % 2 else "physics"}} for i in range(100)]}
    rows = s.fetch_rows("MegaScience/TextbookReasoning", 250, seed=1, get=get, drop_areas=("math",), threads=3)
    assert len(rows) == 250 and rows[0]["id"].startswith("MegaScience/TextbookReasoning#")
    assert {r["subject"] for r in rows} == {"physics"}, "a dropped area never reaches the rows"
    assert len({r["id"] for r in rows}) == 250 and all("datasets-server.huggingface.co/rows" in c for c in calls)
