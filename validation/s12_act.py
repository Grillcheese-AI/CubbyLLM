"""s12_act -- order 0034's act line, counted: a world's own act log (one JSON line per plan: how it was drawn, the cell
the host computed and the one the test expected, whether an order was sent, whether the marker stands there, how many
stand elsewhere, why a plan was refused) summed to counts that name no cell and no thing.

The line: every plan drawn placeable is placed on the computed cell, and that cell is the expected one; none is
placed anywhere else; every plan drawn unplaceable is refused before an order is sent.

Wired: STANDALONE (``exp_s12_scene_files.py`` adds it to its log when ``$CUBBY_SCENES/act_log.jsonl`` is there).
"""
from __future__ import annotations

import collections
import json

__wiring__ = "STANDALONE"


def counts(path: str) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
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
            "refused_why": dict(collections.Counter(r.get("refused") for r in rows if not r.get("sent")))}
