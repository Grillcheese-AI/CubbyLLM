"""cubbyllm/reasoning/sleep.py -- the nightly cycle.

Pinned: every refusal reason the loop can produce is routed (or is an answered kind), and a new one
cannot appear without a decision; a certified chain becomes ONE episode however often the night is run;
only facts the gate accepted from a trusted, non-latent source become durable; re-running a night
changes nothing (ledger, queues, striatum); queue items merge across nights; a spoken answer the asker
calls wrong is a defect and a -5 for its shape; a profile answer is not a refusal; an episode whose
chain lost a fact is retired, never deleted.
"""
from __future__ import annotations

import json
import pathlib
import re

from cubbyllm.reasoning import sleep as S
from cubbyllm.reasoning.hippocampus import Hippocampus
from cubbyllm.reasoning.striatum import Striatum, question_shape

REPO = pathlib.Path(S.__file__).resolve().parents[2]

GOOD = {"question": "What is the capital of the country of citizenship of Marie?", "verified": True,
        "answer": "ottawa", "reason": None, "plan": ["country of citizenship", "capital"], "seed": "marie",
        "trace": ["canada is the country of citizenship of marie", "ottawa is the capital of canada"],
        "learned": [
            {"fact": "canada is the country of citizenship of marie", "status": "accepted", "source": "wikidata",
             "entity": "marie", "rel": "country of citizenship", "time": {"point": "2001"}},
            {"fact": "ottawa is the capital of canada", "status": "accepted", "source": "wikidata (latent)",
             "entity": "canada"},
            {"fact": "paris is the capital of france", "status": "duplicate", "source": "wikidata", "entity": "x"},
            {"fact": "rome is the capital of italy", "status": "accepted", "source": "somewhere", "entity": "x"},
        ]}


def _rec(q, reason, **kw):
    return {"question": q, "verified": False, "answer": None, "reason": reason, "plan": kw.pop("plan", None),
            "seed": kw.pop("seed", None), "trace": [], "learned": [], **kw}


DAY1 = [GOOD,
        _rec("Which lake in Kashmir is the jewel?", "unknown_relation", plan=["lake known as the jewel"],
             unknown=["lake known as the jewel"]),
        _rec("Tell me something about Marie", "no_plan"),
        _rec("Who founded the Acme club?", "retrieval_exhausted", plan=["founder"], seed="acme club"),
        _rec("Who is Jean?", "profile", answer="jean — occupation: baker"),
        dict(GOOD, question="What is the capital of the country of citizenship of Hans?", answer="berlin",
             feedback="wrong", trace=["germany is the country of citizenship of hans"], learned=[])]


def _write(tmp, name, rows):
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_every_reason_the_loop_writes_is_routed_or_an_answered_kind():
    """A reason literal in the loop's code that the table does not know would land in 'unrouted' in
    silence. Scanned: every snake_case literal on or right after a line that assigns a reason."""
    found = set()
    files = list((REPO / "cubbyllm" / "reasoning").glob("*.py")) + [REPO / "standin" / "ask.py"]
    for f in files:
        if f.name == "sleep.py":
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if re.search(r"reason\"?\]?\s*=|reason=", line):
                for s in (line, lines[i + 1] if i + 1 < len(lines) else ""):
                    found |= set(re.findall(r"[\"']([a-z]+(?:_[a-z]+)+)[\"']", s))
    # keys of the record and of its `refused` dict that share a line with a reason; not reasons
    not_reasons = {"tail_relation", "latent_facts", "unknown_relations", "tau_vm", "wall_s"}
    missing = sorted(found - set(S.ROUTES) - S.ANSWERED - not_reasons)
    assert not missing, f"reasons with no route: {missing}. Route them in sleep.ROUTES (a decision), don't widen the scan."
    assert {"plan_does_not_cover_question", "unknown_relation", "retrieval_exhausted"} <= found


def test_a_certified_chain_becomes_one_episode_however_often_the_night_runs(tmp_path):
    day = _write(tmp_path, "day1.jsonl", DAY1)
    for _ in range(2):
        S.sleep([day], tmp_path / "sleep", night="2026-09-24", trusted={"wikidata"})
    hip = Hippocampus.load(tmp_path / "sleep" / "hippocampus.jsonl")
    assert len(hip.episodes) == 2                      # Marie, and Hans (spoken, later called wrong)
    ep = next(e for e in hip.episodes if e.seed == "marie")
    assert ep.chain == GOOD["trace"] and ep.provenance["night"] == "2026-09-24"


def test_only_accepted_facts_from_a_trusted_non_latent_source_become_durable(tmp_path):
    day = _write(tmp_path, "day1.jsonl", DAY1)
    rep = S.sleep([day], tmp_path / "sleep", night="2026-09-24", trusted={"wikidata"})
    live = S.load_facts(tmp_path / "sleep" / "learned_facts.jsonl")
    assert list(live) == ["canada is the country of citizenship of marie"]
    kept = live["canada is the country of citizenship of marie"]
    assert kept["rel"] == "country of citizenship" and kept["time"] == {"point": "2001"}
    assert rep["consolidate"]["facts_untrusted_skipped"] == 2   # the latent one and 'somewhere'


def test_rerunning_a_night_changes_nothing(tmp_path):
    day = _write(tmp_path, "day1.jsonl", DAY1)
    out = tmp_path / "sleep"
    S.sleep([day], out, night="2026-09-24")
    snap = {p.name: p.read_text(encoding="utf-8") for p in [out / "ledger.jsonl", out / "striatum.json",
                                                             out / "learned_facts.jsonl", *sorted((out / "queues").glob("*.jsonl"))]}
    S.sleep([day], out, night="2026-09-24")
    after = {p.name: p.read_text(encoding="utf-8") for p in [out / "ledger.jsonl", out / "striatum.json",
                                                              out / "learned_facts.jsonl", *sorted((out / "queues").glob("*.jsonl"))]}
    assert after == snap


def test_queue_items_merge_across_nights_and_a_recurring_item_says_so(tmp_path):
    out = tmp_path / "sleep"
    S.sleep([_write(tmp_path, "d1.jsonl", DAY1)], out, night="2026-09-24")
    S.sleep([_write(tmp_path, "d2.jsonl", DAY1[1:2])], out, night="2026-09-25")
    items = [json.loads(l) for l in (out / "queues" / "relation_wordings.jsonl").read_text().splitlines()]
    it = next(i for i in items if i["key"] == "lake known as the jewel")
    assert it["count"] == 2 and it["nights"] == 2 and it["first_seen"] == "2026-09-24" and it["status"] == "open"
    report = (out / "nights" / "2026-09-25" / "report.md").read_text(encoding="utf-8")
    assert "recurring" in report and "previous night" in report


def test_each_refusal_lands_in_the_queue_for_the_level_that_must_change(tmp_path):
    rep = S.sleep([_write(tmp_path, "d1.jsonl", DAY1)], tmp_path / "sleep", night="2026-09-24")
    t = rep["triage"]
    assert t["relation_wordings"]["tonight"] == 1 and t["emitter_curriculum"]["tonight"] == 1
    assert t["source_gaps"]["top"][0][0] == "acme club | founder"
    assert t["unrouted"]["tonight"] == 0


def test_a_spoken_answer_the_asker_calls_wrong_is_a_defect_and_costs_its_shape(tmp_path):
    rep = S.sleep([_write(tmp_path, "d1.jsonl", DAY1)], tmp_path / "sleep", night="2026-09-24")
    assert rep["audit"]["wrong"] == 1 and rep["triage"]["defects"]["tonight"] == 1
    st = Striatum.load(tmp_path / "sleep" / "striatum.json")
    q = DAY1[-1]["question"]
    assert st.value("emitter", question_shape(q)) < 0            # -5 then +1 on the same shape, alpha 0.2
    assert "Kill line crossed" in (tmp_path / "sleep" / "nights" / "2026-09-24" / "report.md").read_text()


def test_a_profile_answer_is_not_a_refusal(tmp_path):
    rep = S.sleep([_write(tmp_path, "d1.jsonl", DAY1)], tmp_path / "sleep", night="2026-09-24")
    assert "profile" not in rep["refusals"] and rep["verified"] == 3   # Marie, Hans (spoken), Jean's profile


def test_an_episode_whose_chain_lost_a_fact_is_retired_never_deleted(tmp_path):
    out = tmp_path / "sleep"
    S.sleep([_write(tmp_path, "d1.jsonl", DAY1[:1])], out, night="2026-09-24")
    S.sleep([_write(tmp_path, "d2.jsonl", [])], out, night="2026-09-25",
            contains=lambda f: f != "ottawa is the capital of canada")
    hip = Hippocampus.load(out / "hippocampus.jsonl")
    assert len(hip.episodes) == 1 and len(hip) == 0
    assert hip.episodes[0].provenance["retired"]["facts"] == ["ottawa is the capital of canada"]


def test_a_retired_fact_leaves_the_durable_log_without_being_deleted(tmp_path):
    p = tmp_path / "learned_facts.jsonl"
    p.write_text(json.dumps({"fact": "a is the b of c"}) + "\n" + json.dumps({"fact": "d is the e of f"}) + "\n"
                 + json.dumps({"retire": "a is the b of c", "reason": "audit"}) + "\n", encoding="utf-8")
    assert list(S.load_facts(p)) == ["d is the e of f"]
    assert "a is the b of c" in p.read_text()
