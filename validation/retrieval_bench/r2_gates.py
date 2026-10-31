"""r2_gates -- order 0046, R2's pre-registered lines read off a run's summary (``r2_choosing``), per arm.

  bare          fame set, bare names: single picks (line: 0; kill: any)
  qualified     fame set, qualified variants: wrong (line: 0) and resolved = picks / rows (line: >= 0.70)
  shadowlink    accuracy among spoken picks, Top and Shadow, and the gap in points (line: <= 5; kill: > 10)
  mintaka, webqsp_el
                wrong and decided (picks), wrong per row and per pick (line, on test: no more wrong than tiers, at
                most 1 in 50, more decided)

    python validation/retrieval_bench/r2_gates.py validation/logs/retrieval_R2_dev_<date>.json [...]

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json
import sys

__wiring__ = "STANDALONE"


def _acc(c: dict) -> float | None:
    n = c.get("right", 0) + c.get("wrong", 0)
    return round(c.get("right", 0) / n, 4) if n else None


def gates(summary: dict) -> dict:
    arms = sorted({a for g in summary.values() for a in g if a not in ("rows", "gold_in_candidates")})
    out = {}
    for a in arms:
        row = {}
        if "fame/bare" in summary and a in summary["fame/bare"]:
            row["bare_single"] = summary["fame/bare"][a].get("single", 0)
        if "fame/qualified" in summary and a in summary["fame/qualified"]:
            c, n = summary["fame/qualified"][a], summary["fame/qualified"]["rows"]
            row["qualified"] = {"wrong": c.get("wrong", 0), "right": c.get("right", 0),
                                "resolved": round((c.get("right", 0) + c.get("wrong", 0)) / n, 4)}
        top, sh = (summary.get(f"shadowlink/{k}", {}).get(a) for k in ("Top", "Shadow"))
        if top and sh:
            t, s = _acc(top), _acc(sh)
            row["shadowlink"] = {"top": t, "shadow": s,
                                 "gap_points": round(abs(t - s) * 100, 1) if t is not None and s is not None else None}
        for s in ("mintaka", "webqsp_el"):
            if s in summary and a in summary[s]:
                c, n = summary[s][a], summary[s]["rows"]
                picks = c.get("right", 0) + c.get("wrong", 0)
                row[s] = {"wrong": c.get("wrong", 0), "decided": picks, "rows": n,
                          "wrong_per_row": round(c.get("wrong", 0) / n, 4),
                          "wrong_per_pick": round(c.get("wrong", 0) / picks, 4) if picks else None}
        out[a] = row
    return out


def budget(rows: list[dict], arm: str = "b1") -> dict:
    """B1's wall per question by its number of branches k (the order's line: p50 and p95 for k in 1, 2, 3, 5)."""
    by = {}
    for r in rows:
        o = (r.get("arms") or {}).get(arm) or {}
        if "k" in o and "wall_ms" in o:
            by.setdefault(str(o["k"]) if o["k"] <= 5 else "6+", []).append(o["wall_ms"])
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]                       # noqa: E731
    return {k: {"n": len(v), "p50_ms": q(v, 0.5), "p95_ms": q(v, 0.95)} for k, v in sorted(by.items())}


def main() -> None:
    for path in sys.argv[1:]:
        rep = json.load(open(path, encoding="utf-8"))
        print(path)
        print(json.dumps(gates(rep["summary"]), indent=1))


if __name__ == "__main__":
    main()
