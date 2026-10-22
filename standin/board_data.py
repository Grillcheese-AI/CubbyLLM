"""standin/board_data.py -- what the control board shows, read from what the server already has (order 0027, P1).

Each function returns a plain dict for one audience: ``who`` is ``"user"`` (0016's public audience: no source, site or
dataset named) or ``"audit"``. Nothing here changes state, and nothing invents a number: a section whose order is not
done says so (``SECTIONS``: the order it waits for) and carries no numbers.

- ``home(brain, who)``: per section, built or waiting, and for a built one three cards (a number, a caption, a trend);
- ``journey(rec, times, forms, who)``: one answer's steps (sense, route, recall, check, say) from its thought graph, with
  their timings where the server's event ring still holds them, and the question's spans (entity, relation, time);
- ``system(brain, who)``: the order queue, the news poller, the latest copy-or-refuse rates, training runs.

Wired: WIRED (``board_api``).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import time
from pathlib import Path

__wiring__ = "WIRED"

ROOT = Path(__file__).resolve().parents[1]
# (id, name, second word, the order the section waits for: None = built in P1)
SECTIONS = [("watch", "World Watch", "live", "0024"), ("chat", "Chat", "& prompt", None),
            ("worlds", "Worlds", "& relations", "0015"), ("forecast", "Forecast", "tree", "0026"),
            ("mood", "Mood", "& arcs", "0020"), ("system", "System", "health", None),
            ("agents", "Agents", "soon", "a later order")]
STEPS = [("sense", "Sense", ("question",)), ("route", "Route", ("plan", "verdict", "alias")),
         ("recall", "Recall", ("hop", "fact", "fetch", "gate", "gates", "latent")),
         ("check", "Check", ("walk", "branch", "vm")), ("say", "Say", ("answer",))]
SOURCE_KEYS = ("provenance", "source", "clash", "via")        # node fields that name where a fact came from


def empty(order: str) -> dict:
    return {"built": False, "order": order, "message": f"not built yet (order {order})"}


def _today(rec) -> bool:
    return str(rec.get("ts", ""))[:10] == dt.date.today().isoformat()


def _hours(recs, n: int = 12) -> list[int]:
    now = time.time()
    out = [0] * n
    for r in recs:
        try:
            age = now - time.mktime(time.strptime(str(r.get("ts"))[:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            continue
        if 0 <= age < n * 3600:
            out[n - 1 - int(age // 3600)] += 1
    return out


def refuse_rates() -> list[dict]:
    """The latest copy-or-refuse measurements (orders 0021-0022), newest first, from their logs."""
    out = []
    for pat, what in (("exp_refuse_450m_2*.json", "450M, run 5"), ("k10_refuse_*_2*.json", "toy")):
        for f in sorted(glob.glob(str(ROOT / "validation" / "logs" / pat)), key=os.path.getmtime, reverse=True)[:6]:
            d = json.load(open(f, encoding="utf-8"))
            t = d.get("test_model") or {}
            if t:
                tag = re.sub(r"^k10_refuse_|_\d{4}-\d{2}-\d{2}\.json$", "", os.path.basename(f)).replace("_", " ")
                out.append({"what": what if what != "toy" else f"toy {tag}",
                            "log": os.path.basename(f), **{k: round(t[k], 4) for k in
                            ("exact", "present_wrong", "present_refused", "absent_answered") if k in t}})
    return out


def home(brain, who: str) -> dict:
    loop = getattr(brain, "ask_loop", None)
    hist = list(getattr(loop, "history", []) or [])
    out = []
    for sid, name, dim, order in SECTIONS:
        s = {"id": sid, "name": name, "dim": dim}
        if order is not None:
            out.append({**s, **empty(order)})
            continue
        if sid == "chat":
            today = [r for r in hist if _today(r)]
            refused = sum(1 for r in today if not r.get("verified") and not r.get("answer"))
            rates = refuse_rates()
            last = journey(hist[-1], {}, None, who) if hist else None
            cards = [{"t": "Today", "s": "Questions asked", "big": len(today), "unit": "asked",
                      "cap": f"{refused} refused", "trend": _hours(hist)},
                     {"t": "Copy or refuse", "s": rates[0]["what"] if rates else "not measured yet",
                      "big": round(100 * rates[0]["present_wrong"], 1) if rates else None, "unit": "% wrong",
                      "cap": (f"{round(100 * rates[0]['present_refused'])}% refused, "
                              f"{round(100 * rates[0]['absent_answered'], 1)}% of absent answered") if rates else "no log"},
                     {"t": "Last question", "s": "Its journey", "big": len(last["steps"]) if last else None,
                      "unit": "steps", "cap": " → ".join(x["title"].lower() for x in last["steps"]) if last else "none yet"}]
        else:
            sysd = system(brain, who)
            q, p = sysd["orders"]["counts"], sysd["poller"]
            cards = [{"t": "Orders", "s": "The queue", "big": q.get("open", 0), "unit": "open",
                      "cap": f"{q.get('in progress', 0)} in progress · {q.get('review', 0)} in review · "
                             f"{q.get('proposed', 0)} proposed"},
                     {"t": "News poller", "s": "Feeds read last poll", "big": p.get("up"), "unit": f"of {p.get('active')}",
                      "cap": f"last poll {p.get('last_poll_ago') or 'never'} · {p.get('items_today', 0)} items today"},
                     {"t": "Training", "s": "Runs with logs", "big": len(sysd["training"]), "unit": "runs",
                      "cap": (sysd["training"][0]["run"] + f" at step {sysd['training'][0]['step']}") if sysd["training"]
                      else "no run logs here"}]
        out.append({**s, "built": True, "cards": cards})
    return {"sections": out}


def spans(question: str, forms) -> list[dict]:
    """The question's words, each with its role: entity, relation, time, or none."""
    words = re.findall(r"\S+", question or "")
    roles = ["" for _ in words]
    parsed = forms.parse(question) if forms is not None else None
    wording = forms.wording(question) if forms is not None else None

    def mark(phrase, role):
        toks = [t.lower() for t in re.findall(r"[\w'-]+", phrase or "")]
        if not toks:
            return
        clean = [re.sub(r"[^\w'-]", "", w).lower() for w in words]
        for i in range(len(clean) - len(toks) + 1):
            if clean[i:i + len(toks)] == toks:
                for j in range(i, i + len(toks)):
                    roles[j] = roles[j] or role
    if parsed:
        mark(parsed[0], "entity")
    mark(wording, "relation")
    for i, w in enumerate(words):
        if re.fullmatch(r"\(?\d{3,4}\)?[?.,]?|today|now|currently|ago", w.lower()):
            roles[i] = roles[i] or "time"
    return [{"text": w, "role": r} for w, r in zip(words, roles)]


def journey(rec: dict, times: dict, forms, who: str) -> dict:
    """``rec``: a kept record; ``times``: event id -> its time (the server's ring)."""
    g = rec.get("graph") or {}
    nodes = [n for n in g.get("nodes", []) if isinstance(n, dict)]
    steps, prev = [], None
    for sid, title, kinds in STEPS:
        mine = [n for n in nodes if n.get("kind") in kinds]
        if not mine:
            continue
        ts = [times[n["id"]] for n in mine if n.get("id") in times]
        end = max(ts) if ts else None
        ms = round(1000 * (end - prev)) if end is not None and prev is not None else None
        prev = end if end is not None else prev
        detail = {}
        for n in mine:
            detail[n["kind"]] = detail.get(n["kind"], 0) + 1
        facts = [f"{n.get('subj')} · {n.get('rel')} · {n.get('obj')}" for n in mine if n.get("kind") == "fact" and n.get("subj")][:4]
        step = {"step": sid, "title": title, "ms": ms, "counts": detail, "facts": facts,
                "stop": any(n.get("verdict") == "refused" for n in mine)}
        if who == "audit":
            step["sources"] = sorted({str(n[k]) for n in mine for k in SOURCE_KEYS if n.get(k)})
        steps.append(step)
    said = rec.get("reply") or rec.get("answer")
    return {"id": rec.get("id"), "question": rec.get("question"), "tokens": spans(rec.get("question"), forms),
            "steps": steps, "wall_s": rec.get("wall_s"), "verified": bool(rec.get("verified")),
            "said": said if rec.get("verified") or rec.get("reply") else None, "tier": rec.get("tier"),
            "reason": rec.get("reason")}


def orders() -> dict:
    out, counts = [], {}
    for f in sorted(glob.glob(str(ROOT / "docs" / "orders" / "0*.md"))):
        lines = open(f, encoding="utf-8").read().splitlines()
        title = re.sub(r"^#\s*\d+\s*[—-]\s*", "", lines[0]) if lines else ""
        status = next((l.split(":", 1)[1].strip() for l in lines[:6] if l.startswith("status:")), "?")
        status = status.split()[0] if status else "?"
        status = "in progress" if status == "in" else status
        out.append({"no": os.path.basename(f)[:4], "title": title.split(":")[0][:80], "status": status})
        counts[status] = counts.get(status, 0) + 1
    return {"list": out, "counts": counts}


def poller(who: str) -> dict:
    root = os.environ.get("CUBBY_NEWS_ARCHIVE")
    from news_archive import feeds
    fs = feeds()
    active = [f for f in fs if f.get("status") == "active"]
    out = {"active": len(active), "held": len(fs) - len(active), "up": None, "last_poll_ago": None, "items_today": 0}
    if not root or not Path(root, "_polls.jsonl").exists():
        return out
    last = {}
    with open(Path(root, "_polls.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                last[r.get("feed")] = r
    up = [k for k, r in last.items() if "items" in r and not r.get("error")]
    out["up"] = len([f for f in active if f["id"] in up])
    newest = max((r.get("at", "") for r in last.values()), default="")
    if newest:
        mins = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(newest)).total_seconds() / 60
        out["last_poll_ago"] = f"{int(mins)} min ago" if mins < 120 else f"{mins / 60:.0f} h ago"
    day = Path(root, f"{dt.date.today().isoformat()}.jsonl")
    out["items_today"] = sum(1 for _ in open(day, encoding="utf-8")) if day.exists() else 0
    if who == "audit":
        out["feeds"] = [{"id": f["id"], "kind": f.get("kind"), "every_h": f.get("every_h", 1), "status": f.get("status"),
                         "internal": f.get("licence") == "publisher", "last": (last.get(f["id"]) or {}).get("at"),
                         "error": (last.get(f["id"]) or {}).get("error") or (last.get(f["id"]) or {}).get("skipped")}
                        for f in fs]
    return out


def training() -> list[dict]:
    """Runs with a metrics log under ``CUBBY_RUNS`` (a folder of run folders): name, last step, last loss."""
    root = os.environ.get("CUBBY_RUNS")
    out = []
    for m in sorted(glob.glob(os.path.join(root, "*", "*_metrics.jsonl")) if root else [], key=os.path.getmtime, reverse=True)[:6]:
        try:
            with open(m, "rb") as f:
                f.seek(max(0, os.path.getsize(m) - 65536))     # probe and eval lines follow the last loss
                tail = [json.loads(l) for l in f.read().decode("utf-8", "replace").splitlines()[1:] if l.strip().startswith("{")]
        except (OSError, ValueError):
            continue
        r = tail[-1] if tail else {}
        lossy = next((x for x in reversed(tail) if isinstance(x.get("loss"), (int, float))), {})
        out.append({"run": os.path.basename(os.path.dirname(m)), "step": r.get("step"), "loss": lossy.get("loss"),
                    "updated": dt.datetime.fromtimestamp(os.path.getmtime(m)).isoformat(timespec="minutes")})
    return out


def system(brain, who: str) -> dict:
    return {"orders": orders(), "poller": poller(who), "copy_or_refuse": refuse_rates(), "training": training(),
            "gpu": None}
