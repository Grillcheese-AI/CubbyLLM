"""validation/selfimprove/mark.py -- order 0049: the VM marks an attempt against the request's gold, and says why.

  arithmetic (step rows)  the step school's judge (``train_school_torch.StepJudge``): the block runs in the VM and its
                          value is the step's gold; another step's value of the same plan, a copied literal, the gold
                          written in, or an unknown slot is refused
  wd                      the program runs in the VM, and its HOP1 word names the gold property among the Wikidata
                          properties it names (labels and aliases, EN and FR: the local property table). In serving the
                          host keeps only the named properties the item carries; this harness has no item store, so a
                          word naming several is right when the gold is one of them (13 of 30 gold FR programs name
                          several: ``selfimprove_marker_on_golds_2026-11-05.json``). Several without the gold: refused;
                          one other: wrong
  the other families      the program runs; right if its value matches the gold, or, without a gold, if it runs.
                          The "answer written in" guard is arithmetic's: a kernel program keeps its constants as
                          literals by design (8 of 30 gold kernel programs tripped it), so kernel skips it
The ledger's verdicts: "right" (ran and would be spoken; ``gold_right`` says whether it matched), "refused" (the host
would not speak it: a slot error, a hack, no or several properties), "caught" (the VM could not run it).

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL), os.path.join(os.path.dirname(VAL), "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

_HOP1 = re.compile(r'bind\s+\w+\s*,\s*HOP1\s*,\s*"([^"]*)"')


def plan_values(records: list[dict]) -> dict:
    """{problem id: its gold step values}: the values a step may compute without being wrong (the step school's)."""
    from cubbyllm.reasoning.slots import num_value
    rows = collections.defaultdict(list)
    for r in records:
        if r.get("task") == "arithmetic" and "#" in r.get("id", ""):
            rows[r["id"].split("#")[0]].append(r)
    return {k: [num_value(str(r["gold"])) for r in sorted(v, key=lambda r: r["stats"]["k"])
                if not r["program"].lstrip().startswith("return")] for k, v in rows.items()}


class Marker:
    """``Marker(records, props)(record, slotted program)`` -> {"verdict", "gold_right", "why"}."""

    def __init__(self, records: list[dict], props: str | None = None, judge=None, aliases=None) -> None:
        if judge is None:
            from train_school_torch import StepJudge
            judge = StepJudge()
        self.judge, self.plans = judge, plan_values(records)
        if aliases is None and props:
            from cubbyllm.sources.wikidata_aliases import PropertyAliases
            aliases = PropertyAliases(props)
        self.pa = aliases

    def selfcheck(self, records: list[dict]) -> None:
        """Fail fast when the VM cannot be reached (the step school's check: a few gold programs run to their gold)."""
        steps = [r for r in records if r.get("task") == "arithmetic"][:8]
        self.judge.selfcheck(steps, step=True)

    def __call__(self, rec: dict, slotted: str) -> dict:
        if rec["task"] == "arithmetic":
            return self._step(rec, slotted)
        return self._program(rec, slotted)

    def _step(self, rec: dict, slotted: str) -> dict:
        from cubbyllm.reasoning.school import CORRECT, WRONG
        row = {**rec, "plan_values": [] if rec.get("shuffled") else self.plans.get(rec["id"].split("#")[0], [])}
        a = self.judge.attempt_step(row, slotted)
        if a["outcome"] == CORRECT:
            return {"verdict": "right", "gold_right": True, "why": ""}
        if a["outcome"] == WRONG:
            return {"verdict": "right", "gold_right": False, "why": a["why"] or "the wrong value"}
        caught = a["why"] in ("does not run", "malformed")
        return {"verdict": "caught" if caught else "refused", "gold_right": None, "why": a["why"]}

    def _program(self, rec: dict, slotted: str) -> dict:
        from cubbyllm.reasoning.school import answer_written_in
        from cubbyllm.reasoning.slots import _norm, num_value
        from emitter_data import table_of
        table = table_of(rec["spans"], rec.get("question", ""))
        v = table.check(slotted)
        if not v.ok:
            return {"verdict": "refused", "gold_right": None, "why": f"slots: {v.reason}"}
        gold = rec.get("gold")
        if rec["task"] not in ("wd", "kernel") and num_value(str(gold)) is not None:
            hack = answer_written_in(slotted, gold, [num_value(s.filled) for s in table.spans if s.kind == "N"])
            if hack:
                return {"verdict": "refused", "gold_right": None, "why": hack}
        filled = table.fill(slotted)
        ok, res = self.judge._run(filled)
        if not ok:
            return {"verdict": "caught", "gold_right": None, "why": "does not run"}
        if rec["task"] == "wd":
            return self._property(rec, filled)
        if gold is None:
            return {"verdict": "right", "gold_right": None, "why": "runs (no gold)"}
        g, r = num_value(str(gold)), num_value(str(res))
        same = (abs(r - g) <= 1e-6 * max(1.0, abs(g))) if (g is not None and r is not None) else \
            _norm(str(res)) == _norm(str(gold))
        return {"verdict": "right", "gold_right": bool(same), "why": "" if same else f"value {str(res)[:40]}"}

    def _property(self, rec: dict, filled: str) -> dict:
        from cubbyllm.sources.wikidata_aliases import _normalize
        words = _HOP1.findall(filled)
        if not words:
            return {"verdict": "refused", "gold_right": None, "why": "no relation word"}
        pids = {self.pa._pid_of.get(_normalize(lab)) for lab in self.pa.relations(words[0])} - {None}
        if not pids:
            return {"verdict": "refused", "gold_right": None, "why": f"no property is named {words[0][:40]!r}"}
        if rec["gold"] in pids:
            return {"verdict": "right", "gold_right": True, "why": "" if len(pids) == 1 else f"among {len(pids)} named"}
        if len(pids) > 1:
            return {"verdict": "refused", "gold_right": None, "why": f"{len(pids)} properties are named so, not the gold"}
        pid = pids.pop()
        return {"verdict": "right", "gold_right": pid == rec["gold"], "why": "" if pid == rec["gold"] else f"names {pid}"}


__all__ = ["Marker", "plan_values"]
