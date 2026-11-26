"""replay_arith_cert - order 0085, step 3: an arithmetic record replayed from the record and the question text alone.

- the question's SHA-256 against the record's;
- each ``$N`` re-read from its span: the question's text there is the recorded text, and what the host fills in for
  it (`arith_trace.read_span`) is the recorded value; one ``$N`` is one value wherever it is written;
- each ``$K`` found in the arithmetic world's closed list (`arith_world.CONVERSIONS`), fact and value;
- the vote recounted from the recorded tries' values (with the record's answer-kind rules) at its k and m;
- a certificate's steps re-run in order in a fresh VM: each step's program slot-checked and filled from the
  re-read values, its SHA-256 compared, run, and its outcome and value compared; each value becomes the next ``$S``;
  then the stop's slot is the answer, the answer is the vote's, and the answer-kind checks come out as recorded.

``replay(rec, question, vm)`` -> "same", or the first field that differs (``slots.$N2.span``,
``steps.1.value``, ``vote``, ...). **Kill:** one replay that is not "same" stops the writer until the cause is fixed
and recorded.

    python validation/replay_arith_cert.py --records <records.jsonl> --data <emitter_sft_v12e_w_slots.jsonl>
        [--out validation/logs/arith_cert_replay_<date>.json]

Wiring: STANDALONE (a measurement; never imported by cubbyllm/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.reasoning.arith_world import CONVERSIONS  # noqa: E402
from cubbyllm.reasoning.slots import SlotTable, Span, canon, num_value  # noqa: E402
from cubbyllm.records.arith_trace import _check_rows, read_span, sha256_text  # noqa: E402

__wiring__ = "STANDALONE"

FACTS = {f.text: canon(f.value) for f in CONVERSIONS}


def _slots(rec: dict, question: str) -> "tuple[list, str | None]":
    spans, seen = [], {}
    for s in rec["slots"]:
        sid = s["id"]
        if s["kind"] == "N":
            if not (0 <= s["start"] < s["end"] <= len(question)) or question[s["start"]:s["end"]] != s["text"]:
                return [], f"slots.{sid}.span"
            if read_span(s["text"]) != s["value"]:
                return [], f"slots.{sid}.value"
            if seen.setdefault(sid, s["value"]) != s["value"]:
                return [], f"slots.{sid}.value"
            spans.append(Span(sid, s["text"].replace(",", ""), s["start"], s["end"], "N",
                              None if s["value"] == s["text"].replace(",", "") else s["value"]))
        elif s["kind"] == "K":
            if FACTS.get(s["fact"]) != s["value"]:
                return [], f"slots.{sid}.fact"
            spans.append(Span(sid, s["fact"], -1, -1, "K", s["value"]))
    return spans, None


def _vote(rec: dict, question: str) -> dict:
    from cubbyllm.reasoning.answer_kind import kept
    from cubbyllm.reasoning.speak_vote import vote
    values = [num_value(t["value"]) for t in rec["tries"]][:rec["k"]]
    if rec.get("rules"):
        values = kept(values, question, tuple(rec["rules"]))
    v = vote(values, rec["m"])
    return {"answer": canon(v["answer"]) if v["answer"] is not None else None, "votes": v["votes"], "why": v["why"]}


def replay(rec: dict, question: str, vm) -> str:
    from cubbyllm.reasoning.step_loop import wrap_step
    if sha256_text(question) != rec["question_sha256"]:
        return "question_sha256"
    spans, bad = _slots(rec, question)
    if bad:
        return bad
    if len(rec["tries"]) != rec["k"] or _vote(rec, question) != rec["vote"]:
        return "vote"
    if rec["kind"] == "refusal":
        return "same"
    table = SlotTable(question, spans)
    for j, st in enumerate(rec["steps"]):
        if st.get("k") != j or "refused" in st:
            return f"steps.{j}"
        if not table.check(st["program"]).ok:
            return f"steps.{j}.slot check"
        ran = wrap_step(table.fill(st["program"]))
        if sha256_text(ran) != st["ran_sha256"]:
            return f"steps.{j}.program"
        ok, result = vm(ran)
        v = num_value(result) if ok else None
        if ("return" if v is not None else "error") != st["outcome"]:
            return f"steps.{j}.outcome"
        if v is None or canon(v) != st["value"]:
            return f"steps.{j}.value"
        table.spans.append(Span(f"$S{j + 1}", "", -1, -1, "S", canon(v)))
    for s in rec["slots"]:
        if s["kind"] == "S" and (table.get(s["id"]) is None or table.get(s["id"]).filled != s["value"]):
            return f"slots.{s['id']}"
    s = table.get(rec["stop"] or "")
    if s is None or s.filled != rec["answer"]:
        return "answer"
    if rec["vote"]["answer"] is None or num_value(rec["answer"]) is None or \
            abs(num_value(rec["answer"]) - num_value(rec["vote"]["answer"])) > 1e-6 * max(1.0, abs(num_value(rec["answer"]))):
        return "answer"
    if rec.get("rules") and _check_rows(num_value(rec["answer"]), question, tuple(rec["rules"])) != rec["checks"]:
        return "checks"
    return "same"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--data", required=True, help="the questions, by id (the read's --data)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    from exp_0078_dev_read import load
    from cubbyllm.emit.format import vm_run
    qs = {r["id"]: r["question"] for r in load(a.data)}
    out, t0 = {"same": 0, "kinds": {}, "not same": []}, time.perf_counter()
    for ln in open(a.records, encoding="utf-8"):
        rec = json.loads(ln)
        got = replay(rec, qs[rec["id"]], vm_run)
        out["kinds"][rec["kind"]] = out["kinds"].get(rec["kind"], 0) + 1
        if got == "same":
            out["same"] += 1
        else:
            out["not same"].append({"id": rec["id"], "kind": rec["kind"], "first difference": got})
    out["records"] = os.path.basename(a.records)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["line 2 (every replay same)"] = not out["not same"]
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


__all__ = ["FACTS", "replay"]

if __name__ == "__main__":
    main()
