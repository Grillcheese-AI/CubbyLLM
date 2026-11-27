"""tools/queue.py -- live work-order queue view from docs/orders/0*.md.

Replaces hand-maintained queue snapshots in CLAUDE.md and ad-hoc `grep -H "^status:"` scans.
Supports both the current line-2 `status: ...` format and optional YAML frontmatter
(`status`, `assignee`, `depends_on`, `updated`).

    python tools/queue.py [--for code-a|code-b|cowork|all] [--all] [--json]

Wired: STANDALONE (maintenance / session-start CLI).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

__wiring__ = "STANDALONE"

REPO = Path(__file__).resolve().parents[1]
ORDERS_DIR = REPO / "docs" / "orders"
_STATUS_RE = re.compile(r"^status:\s*(.+)$", re.M)
_WRITTEN_RE = re.compile(r"^written:\s*(\d{4}-\d{2}-\d{2})", re.M)
_HEAD_RE = re.compile(r"^#\s*(\d{4})\s*[-\u2014]\s*(.+)$", re.M)
_DEPS_RE = re.compile(r"\b(\d{4})\b")

STATES = ("in_progress", "review", "waiting_gpu", "waiting_nick", "open", "blocked", "proposed", "parked", "done")


def _norm_state(raw: str) -> str:
    head = raw.split("(", 1)[0].strip().lower().replace(" ", "_").replace("-", "_")
    return head if head in STATES else "open"


def _infer_assignee(status_line: str) -> str:
    low = status_line.lower()
    if "code b" in low or "code-b" in low or "agent-b" in low:
        return "code-b"
    if "cowork builds" in low or "cowork:" in low:
        return "cowork"
    if "code a" in low or "code-a" in low:
        return "code-a"
    return "unassigned"


def _parse_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    meta: dict = {}
    for line in text[4:end].splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v.strip("'\"")
    return meta


def parse_order(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = _parse_frontmatter(text)
    m_head = _HEAD_RE.search(text)
    order_id = m_head.group(1) if m_head else path.name[:4]
    title = m_head.group(2).strip() if m_head else path.stem
    m_stat = _STATUS_RE.search(text[:800])
    raw_status = fm.get("status") or (m_stat.group(1).strip() if m_stat else "unknown")
    state = _norm_state(raw_status)
    note = raw_status.split("(", 1)[1].rstrip(")").strip() if "(" in raw_status else ""
    assignee = fm.get("assignee") or _infer_assignee(raw_status)
    deps = fm.get("depends_on")
    if deps is None:
        m_after = re.search(r"(?:after|waits for)\s+([^;)]+)", raw_status, re.I)
        deps = [d for d in _DEPS_RE.findall(m_after.group(1)) if d != order_id] if m_after else []
    m_date = _WRITTEN_RE.search(text[:1000])
    return {
        "id": order_id,
        "file": path.relative_to(REPO).as_posix(),
        "title": title,
        "state": state,
        "assignee": assignee,
        "depends_on": [str(d).zfill(4) for d in deps],
        "note": note,
        "written": fm.get("updated") or (m_date.group(1) if m_date else ""),
    }


def load_orders(orders_dir: Path = ORDERS_DIR) -> list[dict]:
    return [parse_order(p) for p in sorted(orders_dir.glob("0*.md"))]


def _matches_worker(order: dict, worker: str) -> bool:
    if worker == "all":
        return True
    if worker == "code-a":
        return order["assignee"] in ("code-a", "unassigned")
    return order["assignee"] == worker


def summarize(orders: list[dict], worker: str = "all", include_done: bool = False) -> dict:
    done_ids = {o["id"] for o in orders if o["state"] == "done"}
    grouped: dict[str, list[dict]] = {s: [] for s in STATES}
    for o in orders:
        o = dict(o)
        o["unmet_deps"] = [d for d in o["depends_on"] if d not in done_ids]
        grouped.setdefault(o["state"], []).append(o)
    counts = {s: len(v) for s, v in grouped.items() if v}
    shown = {}
    for s in STATES:
        if s == "done" and not include_done:
            continue
        items = [o for o in grouped[s] if _matches_worker(o, worker)]
        if items:
            shown[s] = items
    warnings = []
    if len(grouped["in_progress"]) > 3:
        warnings.append(f"{len(grouped['in_progress'])} orders marked in_progress (recommended WIP <= 1 per worker)")
    if grouped["review"]:
        warnings.append(f"{len(grouped['review'])} order(s) waiting in review: {', '.join(o['id'] for o in grouped['review'])}")
    return {"total": len(orders), "counts": counts, "warnings": warnings, "orders": shown}


def _print_text(data: dict) -> None:
    counts = ", ".join(f"{k}={v}" for k, v in data["counts"].items())
    print(f"Orders ({data['total']} total): {counts}")
    for w in data["warnings"]:
        print(f"  [!] {w}")
    for state, items in data["orders"].items():
        print(f"\n== {state} ({len(items)}) ==")
        for o in items:
            tag = f" [{o['assignee']}]" if o["assignee"] != "unassigned" else ""
            deps = f" (waits: {','.join(o['unmet_deps'])})" if o["unmet_deps"] else ""
            title = o["title"] if len(o["title"]) <= 78 else o["title"][:75] + "..."
            print(f"  {o['id']}{tag}  {title}{deps}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show live CubbyLLM work-order queue.")
    ap.add_argument("--for", dest="worker", choices=("all", "code-a", "code-b", "cowork"), default="all")
    ap.add_argument("--all", dest="include_done", action="store_true", help="Include done orders.")
    ap.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON.")
    args = ap.parse_args(argv)
    data = summarize(load_orders(), worker=args.worker, include_done=args.include_done)
    if args.as_json:
        print(json.dumps(data, indent=2))
    else:
        _print_text(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
