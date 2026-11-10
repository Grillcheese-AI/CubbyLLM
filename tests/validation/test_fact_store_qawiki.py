"""QAWiki's ids by role (order 0055's addendum on merged items): from the query, the mentions and the gold answers
(the training file's list of result sets and the test file's one), then counted as the store resolves them."""
import json

from validation.fact_store import qawiki_golds as G


def _files(d):
    train = {"dataset": {}, "questions": [
        {"id": 0, "query": {"sparql": "SELECT ?x WHERE { wd:Q19798734 p:P161 ?s . ?s pq:P453 wd:Q27955792 }"},
         "mentions": [{"entity": "http://www.wikidata.org/entity/Q19798734"}],
         "answers": [{"results": {"bindings": [{"x": {"type": "uri", "value": "http://www.wikidata.org/entity/Q7"}}]}}]}]}
    test = {"dataset": {}, "questions": [
        {"id": 1, "query": {"sparql": "ASK { wd:Q42 wdt:P31 wd:Q5 }"}, "mentions": [],
         "answers": {"head": {}, "results": {"bindings": [{"y": {"type": "literal", "value": "Q99"}}]}}}]}
    (d / "wikikgqa.json").write_text(json.dumps(train), encoding="utf-8")
    (d / "x_solutions.json").write_text(json.dumps(test), encoding="utf-8")


def test_ids_by_role_then_counted_as_the_store_resolves_them(tmp_path):
    _files(tmp_path)
    ids = tmp_path / "ids.txt"
    G.main(["--qawiki", str(tmp_path), "--out", str(ids)])
    assert ids.read_text().split() == ["Q5", "Q7", "Q42", "Q19798734", "Q27955792"]   # a literal "Q99" is no id
    res = tmp_path / "resolved.tsv"
    res.write_text("Q5\thot\nQ7\t-> Q8\thot\nQ42\tcold\nQ19798734\tmissing\nQ27955792\t-> Q1\tmissing\n")
    out = tmp_path / "out.json"
    G.main(["--qawiki", str(tmp_path), "--resolved", str(res), "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["by_role"]["answer"] == {"ids": 1, "redirect": 1}
    assert r["by_role"]["query"] == {"ids": 4, "cold": 1, "hot": 1, "missing": 1, "redirect_missing": 1}
    assert r["through_a_redirect"] == ["Q7", "Q27955792"] and r["missing"] == ["Q19798734", "Q27955792"]
