"""The board's Knowledge section (order 0055's store), over a fake store client (no binary): the public audience gets a
notice and nothing else; an unconfigured store says so; status, search, entity (labelled, as of a date), graph (both
directions), who has, path (both ends, at most 3 steps) and resolve answer from the store's own answers."""
from __future__ import annotations

import json

from cubby_serve.board.store import board_store


def st(pid, value, rank=1, quals=()):
    return {"guid": f"g{pid}{value}", "pid": pid, "rank": rank, "best": rank != 0, "value": value, "value_node": None,
            "hot_qualifiers": list(quals), "hot_qualifier_nodes": [], "cold_qualifiers": [], "cold_qualifier_nodes": [], "references": ["r1"]}


E = {"Q1": [st(31, {"entity": "Q5"}), st(26, {"entity": "Q2"}, quals=[[580, {"typed": ["2001-01-01T00:00:00Z", "xsd:dateTime"]}]])],
     "Q2": [st(31, {"entity": "Q5"}), st(1082, {"typed": ["+12", "xsd:decimal"]})],
     "Q3": [st(40, {"entity": "Q1"}), st(31, {"entity": "Q9"}, rank=0)],
     "Q5": [], "Q9": []}
LABELS = {"en": {"Q1": "Ada", "Q2": "Bea", "Q3": "Cy", "P31": "instance of", "P26": "spouse", "P40": "child", "P580": "start time"},
          "mul": {"Q5": "human"}, "fr": {"Q9": "chose"}}


class Fake:
    def __init__(self, store):
        self.store, self.asked = store, []

    def ask(self, r):
        self.asked.append(r["op"])
        op = r["op"]
        if op == "snapshot":
            return {"snapshot": "abc123"}
        if op == "labels":
            return {i: LABELS[r["lang"]][i] for i in r["ids"] if i in LABELS[r["lang"]]}
        if op == "resolve":
            return {"id": "Q1" if r["id"] == "Q7" else r["id"]}
        if op == "get":
            i = "Q1" if r["id"] == "Q7" else r["id"]
            if i not in E:
                return {"found": False, "id": i}
            out = {"id": i, "found": True, "cold": False, "revision": 5, "modified": "2025-12-01T00:00:00Z", "datatype": None,
                   "names": [["en", 0, LABELS["en"].get(i, i)], ["en", 2, f"about {i}"]], "sitelinks": [], "statements": json.loads(json.dumps(E[i]))}
            if r.get("as_of"):
                out["as_of"] = r["as_of"]
                for s in out["statements"]:
                    s["holds"] = "holds" if s["hot_qualifiers"] else "undated"
            if i != r["id"]:
                out["redirected_from"] = r["id"]
            return out
        if op == "pointing_at":
            hits = [[s["pid"], src, s["rank"], s["best"]] for src, ss in sorted(E.items()) for s in ss if s["value"] == {"entity": r["value"]}]
            return {"total": len(hits), "hits": hits[:r.get("limit", 1000)]}
        if op == "who":
            hits = [[src, s["rank"], s["best"]] for src, ss in sorted(E.items()) for s in ss
                    if s["pid"] == r["pid"] and s["value"] == {"entity": r["value"]}]
            return {"total": len(hits), "hits": hits[:r.get("limit", 1000)]}
        if op == "name":
            return {"hits": [{"id": "Q1", "lang": "en", "kind": 0, "how": "exact", "key": "ada"}]} if r["text"].lower() == "ada" else {"too_many": 65}
        return {"error": "unknown op"}


def test_the_public_audience_gets_a_notice_and_an_unconfigured_store_says_so(monkeypatch):
    f = Fake("x")
    out = board_store("/board/store/entity/Q1", "", "user", f)
    assert out["audit_only"] and "Q1" not in json.dumps(out) and f.asked == []
    monkeypatch.setattr("cubby_serve.board.store.from_env", lambda: None)
    assert board_store("/board/store", "", "audit")["configured"] is False


def test_status_reads_the_manifest_and_the_snapshot(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dump_date": "2025-12-29T23:00:38Z", "counts": {"entities_hot": 3},
        "files": [{}, {}], "inputs": [{"parts": 1, "release": {"name": "d.ttl.bz2", "md5": "m", "licence": "cc-zero"}}]}))
    out = board_store("/board/store", "", "audit", Fake(str(tmp_path)))
    assert (out["snapshot"], out["counts"]["entities_hot"], out["files"], out["release"]["licence"]) == ("abc123", 3, 2, "cc-zero")


def test_search_entity_graph_who_path_resolve():
    f = Fake("x")
    s = board_store("/board/store/search", "q=Ada", "audit", f)
    assert s["hits"] == [{"id": "Q1", "how": "exact", "lang": "en", "matched": "ada", "description": "about Q1", "statements": 2, "label": "Ada"}]
    assert board_store("/board/store/search", "q=xyz", "audit", f)["too_many"] == 65
    e = board_store("/board/store/entity/Q7", "as_of=2019", "audit", f)
    assert (e["id"], e["redirected_from"], e["as_of"], e["label"]) == ("Q1", "Q7", "2019", "Ada")
    g26 = next(g for g in e["groups"] if g["pid"] == "P26")
    assert (g26["label"], g26["statements"][0]["value"], g26["statements"][0]["holds"]) == ("spouse", "Bea (Q2)", "holds")
    assert g26["statements"][0]["qualifiers"][0] == {"pid": "P580", "label": "start time", "value": "2001-01-01T00:00:00Z", "time": True}
    assert next(g for g in e["groups"] if g["pid"] == "P31")["statements"][0]["value"] == "human (Q5)"     # a mul label
    gr = board_store("/board/store/graph/Q1", "", "audit", f)
    assert {(x["from"], x["pid"], x["to"]) for x in gr["edges"]} == {("Q1", "P31", "Q5"), ("Q1", "P26", "Q2"), ("Q3", "P40", "Q1")}
    w = board_store("/board/store/who", "p=P31&v=Q5", "audit", f)
    assert ([h["id"] for h in w["hits"]], w["p_label"], w["v_label"]) == (["Q1", "Q2"], "instance of", "human")
    w1 = board_store("/board/store/who", "p=P31&v=Q5&limit=1", "audit", f)
    assert ([h["id"] for h in w1["hits"]], w1["total"]) == (["Q1"], 2)                                 # the first, and the total
    g1 = board_store("/board/store/graph/Q5", "limit=1", "audit", f)
    assert (len(g1["edges"]), g1["more"]["in"]) == (1, 1)
    p = board_store("/board/store/path", "from=Q3&to=Q5", "audit", f)
    assert [(x["from"], x["pid"], x["to"]) for x in p["steps"]] == [("Q3", "P40", "Q1"), ("Q1", "P31", "Q5")]
    assert board_store("/board/store/path", "from=Q5&to=Q3", "audit", f)["steps"] is None          # no chain that way
    assert board_store("/board/store/path", "from=Q3&to=Q9", "audit", f)["steps"] is None          # deprecated: not a step
    assert board_store("/board/store/resolve", "id=Q7", "audit", f) == {"id": "Q1"}
    assert "error" in board_store("/board/store/entity/nope", "", "audit", f)


def test_the_board_is_on_by_default_and_no_board_turns_it_off():
    from cubby_serve.cli import parser
    ap = parser()
    assert ap.parse_args([]).board is True                      # on by default (Nick, 2026-11-10)
    assert ap.parse_args(["--no-board"]).board is False
    assert ap.parse_args(["--board"]).board is True             # the old switch still parses
