"""standin/news_trigger.py -- news as the trigger that a fact may have changed (order 0024).

A reported role change about a person or an organisation a world holds marks that entity's time-sensitive facts as
possibly stale: "X named CEO" today leaves the worlds saying the old CEO, as of 2019, until the structured source
catches up. The marks are **reported**, never a new value: they make the ask loop say the fact with its date and
add that later reports say it may have changed (``ask.py``'s "as of" branch, with ``news_changes``).

Where the cues come from (no model reads the news):
- **news items** (kinds ``news`` and ``statement``: an office announcing an appointment is the strongest report of
  one): a change word in the headline or summary (``CUES``, English and French) and, in the same headline or summary,
  a run of capitalised words the world holds as a person or an organisation (``known``);
- **GDELT events** (``news_gdelt``): an actor the world holds and a change word in the event's source URL (its words
  only: the URL is never fetched).

``scan(root, known)`` appends new marks to ``<root>/_changes.jsonl`` (entity, cue, day, source, evidence) and returns
them; ``Changes(root).since(entity, year)`` -- the marks about an entity dated after a year (a world's "as of").

Wired: WIRED (``ask.AskLoop(news_changes=...)``; off when absent).
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

__wiring__ = "WIRED"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from news_events import fold  # noqa: E402

CUES = {
    "appointed": r"appoint(?:s|ed|ment)|named (?:as |the )?(?:new )?(?:ceo|chief|head|director|minister|president|chair)"
                 r"|nomm[ée]e?s?\b|nomination|d[ée]sign[ée]e?",
    "resigned": r"resign(?:s|ed|ation)?|steps? down|stepping down|d[ée]missionn?e?|d[ée]mission|quitte ses fonctions",
    "elected": r"\b(?:re-?)?elected\b|wins? (?:the )?(?:election|presidency|leadership)|[ée]lue?s?\b|r[ée][ée]lue?",
    "sworn in": r"sworn in|assermentée?s?|prête serment",
    "died": r"\bdies\b|\bdied\b|dead at|passed away|\bmeurt\b|\bmort\b|d[ée]c[èe]de|d[ée]c[ée]d[ée]e?|d[ée]c[èe]s",
    "acquired": r"acquir(?:es|ed)|acquisition of|takes? over|\brach[èe]te\b|\brachat\b|\bacquiert\b",
    "removed": r"\bousted\b|\bfired\b|\bsacked\b|\bdismissed\b|limog[ée]e?|destitu[ée]e?|\bremplac[ée]e?\b",
}
_CUE = {k: re.compile(v, re.I) for k, v in CUES.items()}
_NAME = re.compile(r"(?:[A-ZÀ-Ý][\w'’.-]+)(?:\s+(?:de |du |da |van |von |al-|el-)?[A-ZÀ-Ý][\w'’.-]+){1,3}")


def cues(text: str) -> list[str]:
    return [k for k, rx in _CUE.items() if rx.search(text or "")]


def names(text: str) -> list[str]:
    """Runs of two to four capitalised words (a person, an organisation)."""
    return [m.group(0).rstrip(".") for m in _NAME.finditer(text or "")]


def from_items(rows, known, kinds=None) -> list[dict]:
    out = []
    for r in rows:
        if (kinds or {}).get(r["feed"], r.get("kind", "news")) not in ("news", "statement"):
            continue
        for part in (r.get("title", ""), r.get("summary", "")):
            cs = cues(part)
            if not cs:
                continue
            for n in dict.fromkeys(names(part)):
                if known(n):
                    out += [{"entity": n, "cue": c, "day": r.get("published") or r["first_seen"][:10],
                             "source": f"item:{r['key']}", "lang": r.get("lang")} for c in cs]
    return out


def from_gdelt(events, known) -> list[dict]:
    out = []
    for e in events:
        words = " ".join(re.split(r"[/_.\-?=&]+", (e.get("source_url") or "").lower()))
        cs = cues(words)
        if not cs:
            continue
        for n in {e.get("a1_name", ""), e.get("a2_name", "")}:
            if n and len(n.split()) >= 2 and known(n):
                day = e.get("day") or ""
                out += [{"entity": n.title(), "cue": c, "day": f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else day,
                         "source": f"gdelt:{e.get('id')}", "lang": None} for c in cs]
    return out


def scan(root, known, kinds=None, start: str | None = None) -> list[dict]:
    """New marks from the archive's items and GDELT's kept events (from ``start``), appended once to the marks file."""
    from news_events import load
    root = pathlib.Path(root)
    marks = from_items(load(root, start), known, kinds)
    for f in sorted((root / "gdelt").glob("*.events.jsonl")):
        if start and f.name[:10] < start:
            continue
        marks += from_gdelt((json.loads(l) for l in f.open(encoding="utf-8") if l.strip()), known)
    path = root / "_changes.jsonl"
    seen = set()
    if path.exists():
        seen = {(m["entity"], m["cue"], m["day"], m["source"]) for m in map(json.loads, path.open(encoding="utf-8"))}
    new = [m for m in marks if (m["entity"], m["cue"], m["day"], m["source"]) not in seen]
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for m in new:
            seen.add((m["entity"], m["cue"], m["day"], m["source"]))
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return new


class Changes:
    """The marks file, read once: ``since(entity, year)`` -> the marks about it dated after ``year``, oldest first."""

    def __init__(self, root) -> None:
        p = pathlib.Path(root) / "_changes.jsonl"
        self.by: dict[str, list[dict]] = {}
        for m in (map(json.loads, p.open(encoding="utf-8")) if p.exists() else ()):
            self.by.setdefault(fold(m["entity"]), []).append(m)

    def since(self, entity: str, year) -> list[dict]:
        y = str(year or "")[:4]
        return sorted((m for m in self.by.get(fold(entity or ""), []) if m["day"][:4] > y), key=lambda m: m["day"])
