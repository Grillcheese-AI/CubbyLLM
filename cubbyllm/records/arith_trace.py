"""cubbyllm/records/arith_trace.py -- what a spoken arithmetic answer rests on, or why one was refused (order 0085,
step 1), written by the host from the step loop's state after the fact: never from what the model says it did. The
only model text in it is each emitted step, as the host re-rendered it (`step_loop.render_step`): what the VM ran.

``trace(question, chains, k, m, rules, freeze)`` -> one record (a dict, JSON lines off the repository):
- ``question_sha256``;
- ``slots``: each ``$N`` with its span (``start``, ``end``, the ``text`` exactly as the question has it, number words
  included) and the ``value`` the host fills in; each ``$K`` by its ``fact`` in the arithmetic world's closed list
  and its value; for a certificate, each ``$S`` by the step that made it;
- ``tries``: every try's final value (None for a try that reached none) and its refusal reason;
- ``vote``: k, m, the value said (or None), how many tries reached it, and why not (``speak_vote.vote``), counted
  over the values the answer-kind rules keep when ``rules`` is given;
- a **certificate** (``kind`` "certificate"): ``chosen`` (the first try that reached the answer), its ``steps``
  (each: the step program as the VM got it with its slots, ``ran_sha256`` of the whole filled program the VM ran,
  the ``outcome`` "return" | "error", the ``value``), its ``stop`` and the ``answer``; ``checks``: the answer-kind
  rules on the answer, each "satisfied", "violated" or "does not apply";
- a **refusal** (``kind`` "refusal"): ``why`` -- "no try reached a value" (with ``try reasons`` counted), "rule:
  <name>" (the rules took out the value the vote would have said), or the vote's own "too few agree" / "tie";
- ``freeze``: the hashes taken at the read's start (`validation/arith_freeze.py`), the same on every record.

A chain hands the writer ``table`` (its `SlotTable`), ``log`` (one entry a step, kept by the dev read when it
records), ``stop``, ``answer`` and ``refused``.

Wiring: STANDALONE (`validation/exp_0078_dev_read.py --records`; `validation/replay_arith_cert.py` replays it).
"""
from __future__ import annotations

import hashlib
from collections import Counter

from ..reasoning.slots import NUM_IN_TEXT_RX, canon, num_value, number_words

__wiring__ = "STANDALONE"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_span(text: str) -> "str | None":
    """What the host fills in for a span's text: digits as written without commas ("1,200" -> "1200", "3.50" stays
    "3.50"), a number word by its value ("four" -> "4", "a dozen" -> "12"); None for anything else."""
    if NUM_IN_TEXT_RX.fullmatch(text):
        return text.replace(",", "")
    w = number_words(text)
    if len(w) == 1 and w[0][2] == 0 and w[0][3] == len(text):
        return canon(w[0][1])
    return None


def slot_rows(table) -> list[dict]:
    out = []
    for s in table.spans:
        if s.kind == "N":
            out.append({"id": s.id, "kind": "N", "start": s.start, "end": s.end,
                        "text": table.question[s.start:s.end], "value": s.filled})
        elif s.kind == "K":
            out.append({"id": s.id, "kind": "K", "fact": s.text, "value": s.filled})
    return out


def _check_rows(answer: float, question: str, rules) -> list[dict]:
    from ..reasoning.answer_kind import asked, broken
    a = asked(question)
    applies = {"negative": True, "whole": bool(a and a[0] == "many"), "money": bool(a and a[0] == "much" and a[1])}
    return [{"rule": r, "outcome": "violated" if broken(answer, question, (r,)) else
             "satisfied" if applies[r] else "does not apply"} for r in rules]


def refusal_why(chains, k: int, m: int, values: list, question: str, rules) -> dict:
    from ..reasoning.answer_kind import broken
    from ..reasoning.speak_vote import vote
    reasons = Counter(str(c.refused).split(":")[0] for c in chains[:k] if c.answer is None)
    if all(v is None for v in (num_value(c.answer) for c in chains[:k])):
        return {"why": "no try reached a value", "try reasons": dict(reasons)}
    if rules:
        off = vote([num_value(c.answer) for c in chains[:k]], m)
        hit = broken(off["answer"], question, rules) if off["answer"] is not None else None
        if hit:
            return {"why": f"rule: {hit}", "try reasons": dict(reasons)}
    return {"why": vote(values, m)["why"], "try reasons": dict(reasons)}


def trace(question: str, chains: list, k: int, m: int, rules=(), freeze: "dict | None" = None) -> dict:
    from ..reasoning.answer_kind import kept
    from ..reasoning.speak_vote import same, vote
    raw = [num_value(c.answer) for c in chains[:k]]
    values = kept(raw, question, rules) if rules else raw
    v = vote(values, m)
    rec = {"question_sha256": sha256_text(question), "slots": slot_rows(chains[0].table), "k": k, "m": m,
           "rules": list(rules), "tries": [{"value": c.answer, "refused": c.refused} for c in chains[:k]],
           "vote": {"answer": canon(v["answer"]) if v["answer"] is not None else None, "votes": v["votes"],
                    "why": v["why"]}, "freeze": freeze}
    if v["answer"] is None:
        return {**rec, "kind": "refusal", **refusal_why(chains, k, m, values, question, rules)}
    i = next(j for j, x in enumerate(values) if x is not None and same(x, v["answer"]))
    c = chains[i]
    rec["slots"] += [{"id": f"$S{s['k'] + 1}", "kind": "S", "step": s["k"], "value": s["value"]}
                     for s in c.log if "value" in s]
    return {**rec, "kind": "certificate", "chosen": i, "steps": list(c.log), "stop": c.stop, "answer": c.answer,
            "checks": _check_rows(num_value(c.answer), question, rules) if rules else []}


__all__ = ["read_span", "refusal_why", "sha256_text", "slot_rows", "trace"]
