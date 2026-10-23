"""s12_act -- order 0034's act line, counted: a world's own act log (one JSON line per plan: how it was drawn, the cell
the host computed and the one the test expected, whether an order was sent, whether the marker stands there, how many
stand elsewhere, why a plan was refused) summed to counts that name no cell and no thing. The world's own words (why
it refused an order, why a village could take none) are counted, never copied.

The line: every plan drawn placeable is placed on the computed cell, and that cell is the expected one; none is
placed anywhere else; every plan drawn unplaceable is refused before an order is sent.

Wired: STANDALONE (``exp_s12_scene_files.py`` adds it to its log when ``$CUBBY_SCENES/act_log.jsonl`` is there).
"""
from __future__ import annotations

import collections
import json

__wiring__ = "STANDALONE"


def counts(path: str) -> dict:
    every = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in every if "skipped" not in r]                     # a village that could take no order: no plan
    ok = [r for r in rows if r.get("drawn") == "placeable"]
    bad = [r for r in rows if r.get("drawn") == "unplaceable"]
    placed = [r for r in ok if r.get("sent") and r.get("ok") and r.get("stands") and r.get("host_cell_is_expected")]
    return {"plans": len(rows), "villages": len({r.get("seed") for r in rows}),
            "placeable": len(ok), "placed_on_the_expected_cell": len(placed),
            "host_resolved": sum(bool(r.get("verified")) for r in rows),
            "host_cell_not_expected": sum(1 for r in rows if r.get("cell") and not r.get("host_cell_is_expected")),
            "placed_elsewhere": sum(int(r.get("elsewhere") or 0) for r in rows),
            "sent_and_failed": sum(1 for r in rows if r.get("sent") and not (r.get("ok") and r.get("stands"))),
            "unplaceable": len(bad), "unplaceable_refused_before_the_order": sum(1 for r in bad if not r.get("sent")),
            "refused_why": dict(collections.Counter(r.get("refused") for r in rows if not r.get("sent"))),
            # the world's own refusals and why a village was skipped stay in the world's log: their words are its rules
            "order_refused_by_the_world": sum(1 for r in rows if r.get("sent") and not r.get("ok")),
            "villages_skipped": sum(1 for r in every if "skipped" in r)}
