"""validation/system_eval/bench.py -- order 0054's five benchmarks, full test splits, as one row shape.

  simpleqa  SimpleQA's test set (4,326; MIT)                       the data root's ``simpleqa/simple_qa_test_set.csv``
  nq        Natural Questions, open: its test is nq_open's validation split (3,610; CC BY-SA 3.0)
                                                                   ``$CUBBY_DATASETS/webquestions/nq_validation.parquet``
  webq      WebQuestions' test split (2,032; CC BY 4.0)            ``$CUBBY_DATASETS/webquestions/test.parquet``
  clutrr    CLUTRR ``gen_train23_test2to10``, test (1,146; CC BY-NC 4.0), each story's relations given as facts
                                                                   (H-A13's setting) ``$CUBBY_DATASETS/benchmarks``
  aa        AA-Omniscience's public split, v1 (600; Apache 2.0)    ``$CUBBY_DATASETS/benchmarks/aa_omniscience/``

A row: {id, bench, question, golds, seen, meta}. ``seen``: the question was read in development before this run --
WebQuestions questions on 0038's scoreboard (``webq_na_eu.csv``, both splits) or among 0046's WebQSP rows (WebQSP's
questions are WebQuestions'); reported apart as well as in the total. ``check`` compares each count with the
order's and returns the differences (reported, never fixed by dropping rows).

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

BENCHES = ("simpleqa", "nq", "webq", "clutrr", "aa")
EXPECTED = {"simpleqa": 4326, "nq": 3610, "webq": 2032, "clutrr": 1146, "aa": 600}
LICENCES = {"simpleqa": "MIT", "nq": "CC BY-SA 3.0", "webq": "CC BY 4.0", "clutrr": "CC BY-NC 4.0", "aa": "Apache 2.0"}
CLUTRR_TASK = "gen_train23_test2to10"


def _path(kind: str) -> str:
    from cubbyllm.local_paths import data_dir, expand
    return {"simpleqa": str(data_dir("simpleqa", "simple_qa_test_set.csv")),
            "nq": expand("$CUBBY_DATASETS/webquestions/nq_validation.parquet"),
            "webq": expand("$CUBBY_DATASETS/webquestions/test.parquet"),
            "webq_na_eu": expand("$CUBBY_DATASETS/webquestions/webq_na_eu.csv"),
            "benchmarks": expand("$CUBBY_DATASETS/benchmarks"),
            "aa": expand("$CUBBY_DATASETS/benchmarks/aa_omniscience/aa_omniscience_public_train.parquet")}[kind]


def qkey(q: str) -> str:
    """A question as compared across files: lowercase, punctuation and spaces folded."""
    return " ".join(re.findall(r"[a-z0-9]+", str(q).lower()))


def _parquet(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def seen_webq() -> set[str]:
    """Question keys development already read: 0038's scoreboard and 0046's WebQSP rows (any split)."""
    out = set()
    p = _path("webq_na_eu")
    if os.path.exists(p):
        out |= {qkey(r["problem"]) for r in csv.DictReader(open(p, encoding="utf-8"))}
    try:
        from retrieval_sets.common import read_rows
        out |= {qkey(r["question"]) for r in read_rows("webqsp_el")}
    except (OSError, KeyError):
        pass
    return out


def clutrr_items(recs: list[dict]) -> list[dict]:
    """CLUTRR records as rows: the question, the relations as facts (``meta['facts']``), A and B; a record whose
    story cannot be read or repeats a name is kept, with no facts (every arm then refuses it, and the count holds)."""
    from exp_r34_clutrr_relations import graph
    import clutrr as C
    from cubbyllm.reasoning.planner import normalize
    rows = []
    for i, rec in enumerate(recs):
        g = graph(rec)
        meta = {"facts": None, "a": None, "b": None, "depth": None, "why": None}
        q = None
        if g is None:
            meta["why"] = "unparseable"
        else:
            names, edges, types, q0, q1 = g
            q = f"How is {names[q1]} related to {names[q0]}?"
            if len(set(n.lower() for n in names)) != len(names):
                meta["why"] = "repeated_name"
            else:
                meta.update(facts=[C.fact(names[a], r, names[b]) for (a, b), r in zip(edges, types)],
                            a=names[q0], b=names[q1], depth=len(types))
        rows.append({"id": f"clutrr:{rec.get('id') or i}", "bench": "clutrr", "question": q or "",
                     "golds": [normalize(rec["target_text"])], "seen": False, "meta": meta})
    return rows


def load(bench: str) -> list[dict]:
    """The benchmark's rows, in file order."""
    if bench == "simpleqa":
        rs = list(csv.DictReader(open(_path("simpleqa"), encoding="utf-8")))
        return [{"id": f"simpleqa:{i}", "bench": bench, "question": r["problem"], "golds": [r["answer"]], "seen": False,
                 "meta": {}} for i, r in enumerate(rs)]
    if bench == "nq":
        return [{"id": f"nq:{i}", "bench": bench, "question": r["question"], "golds": list(r["answer"]), "seen": False,
                 "meta": {}} for i, r in enumerate(_parquet(_path("nq")))]
    if bench == "webq":
        seen = seen_webq()
        return [{"id": f"webq:{i}", "bench": bench, "question": r["question"], "golds": list(r["answers"]),
                 "seen": qkey(r["question"]) in seen, "meta": {"url": r.get("url")}}
                for i, r in enumerate(_parquet(_path("webq")))]
    if bench == "clutrr":
        import clutrr as C
        return clutrr_items(C.load_raw("test", CLUTRR_TASK, _path("benchmarks")))
    if bench == "aa":
        return [{"id": f"aa:{r['question_id']}", "bench": bench, "question": r["question"], "golds": [r["answer"]],
                 "seen": False, "meta": {k: r[k] for k in ("domain", "topic", "subtopic")}}
                for r in _parquet(_path("aa"))]
    raise KeyError(bench)


def check(rows_by_bench: dict[str, list[dict]]) -> dict:
    """{bench: {"rows": n, "expected": m}} for every count that is not the order's."""
    return {b: {"rows": len(rs), "expected": EXPECTED[b]} for b, rs in rows_by_bench.items() if len(rs) != EXPECTED[b]}


def export(path: str, benches=BENCHES) -> dict:
    """Every benchmark's rows into one jsonl (what Colab's M arm reads); returns the counts and their check."""
    rb = {b: load(b) for b in benches}
    with open(path, "w", encoding="utf-8") as f:
        for rs in rb.values():
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": {b: len(rs) for b, rs in rb.items()}, "differs": check(rb),
            "seen": {b: sum(r["seen"] for r in rs) for b, rs in rb.items()}}


__all__ = ["BENCHES", "EXPECTED", "LICENCES", "check", "clutrr_items", "export", "load", "qkey", "seen_webq"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="questions.jsonl")
    print(json.dumps(export(ap.parse_args().export), indent=1))

