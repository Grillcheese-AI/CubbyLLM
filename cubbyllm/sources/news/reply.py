"""standin/news_reply.py -- what the news archive says about someone or something on a day, in Cubby's words
(order 0024). No model reads the news: the reply is built from what the reports have in common.

``answer(question, root, today)`` for the archive's own questions, in English and French:

    What happened with <X> on <date>? / What was reported about <X> this week?
    Que s'est-il passé avec <X> le <date> ? / Qu'a-t-on rapporté sur <X> cette semaine ?

finds the news items (kind ``news``: an office's statement or an organisation's analysis is not an event) dated in
the window that name X, groups them (``news_events``: one event across outlets), and says the biggest groups as
**reported**, dated by their first day, in Cubby's words: "On 21 October, reports about Mark Carney also named
Turkey (visit, alliance)." Nothing else is claimed: who did what is the reports', not Cubby's. A reply never names
an outlet (0016) and never carries a run of 8 words from an item (checked against every item it rests on); nothing
found is a refusal ("I have no reports about X on 21 October."). The record keeps the items (the audit view).

Wired: STANDALONE (``validation``; the ask loop's news branch takes it after 0024's bench).
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from collections import Counter
from pathlib import Path

__wiring__ = "STANDALONE"

from .events import fold, group, load  # noqa: E402
from .events import features  # noqa: E402

MONTHS_EN = "january february march april may june july august september october november december".split()
MONTHS_FR = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
_Q = [("en", re.compile(r"^what happened (?:with|to|in) (?P<x>.+?) (?P<when>on .+|this week|today|yesterday)\s*\??$", re.I)),
      ("en", re.compile(r"^what (?:was|has been) reported (?:about|on) (?P<x>.+?)(?: (?P<when>on .+|this week|today|yesterday))?\s*\??$", re.I)),
      ("fr", re.compile(r"^(?:que s'est-il passé|qu'est-ce qui s'est passé) (?:avec|à|au|en) (?P<x>.+?) "
                        r"(?P<when>le .+|cette semaine|aujourd'hui|hier)\s*\??$", re.I)),
      ("fr", re.compile(r"^qu'?a-t-on rapporté (?:sur|à propos de) (?P<x>.+?)(?: (?P<when>le .+|cette semaine|aujourd'hui|hier))?\s*\??$", re.I))]
MAX_RUN = 7
# capitals that are not names of anyone or anything in the event: days, months, titles
NOT_NAMES = set(fold(w) for w in """monday tuesday wednesday thursday friday saturday sunday lundi mardi mercredi jeudi
    vendredi samedi dimanche minister ministre president président prime premier""".split()) | set(map(fold, MONTHS_EN + MONTHS_FR))


def parse(question: str):
    """(lang, subject, first day, last day) or None for a question this source does not answer."""
    q = " ".join(question.replace("’", "'").split())
    for lang, rx in _Q:
        m = rx.match(q)
        if m:
            return lang, m.group("x").strip(" ,"), m.group("when")
    return None


def window(when: str | None, today: dt.date) -> tuple[dt.date, dt.date] | None:
    w = (when or "this week").lower().strip()
    if w in ("this week", "cette semaine"):
        return today - dt.timedelta(days=6), today
    if w in ("today", "aujourd'hui"):
        return today, today
    if w in ("yesterday", "hier"):
        return today - dt.timedelta(days=1), today - dt.timedelta(days=1)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", w)
    if m:
        d = dt.date(*map(int, m.groups()))
        return d, d
    for months in (MONTHS_EN, MONTHS_FR):
        for i, name in enumerate(months):
            m = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th|er)?\s+{name}|{name}\s+(\d{{1,2}})", fold(w))
            if m and fold(name) in fold(w):
                day, year = int(m.group(1) or m.group(2)), today.year
                y = re.search(r"\b(19|20)\d{2}\b", w)
                d = dt.date(int(y.group(0)) if y else year, i + 1, day)
                return d, d
    return None


def _say_date(d: dt.date, lang: str) -> str:
    return f"{d.day} {MONTHS_EN[d.month - 1].capitalize()}" if lang == "en" else f"{'1er' if d.day == 1 else d.day} {MONTHS_FR[d.month - 1]}"


def _day(r: dict) -> dt.date:
    return dt.date.fromisoformat(r.get("published") or r["first_seen"][:10])


def outlets(feeds: list[dict]) -> set[str]:
    """Every outlet's names as the archive knows them, folded: its feed id's words and its hosts' labels."""
    out = set()
    for f in feeds:
        out |= {w for w in f["id"].split("-") if len(w) > 2}
        for h in [f.get("url", "")] + list(f.get("publisher_hosts") or []):
            host = re.sub(r"^https?://", "", h).split("/")[0]
            out |= {p for p in host.split(".") if len(p) > 2 and p not in ("www", "com", "org", "net", "feeds", "rss")}
    return {fold(o) for o in out} | {"reuters", "associated press", "afp", "presse canadienne", "canadian press"}


def outlet_words(text: str, names_out: set[str]) -> set[str]:
    """The folded words of ``text`` that spell an outlet's name together: adjacent capitalised words whose run, spaces
    taken out, is one of ``names_out`` ("the Daily Planet" for ``dailyplanet``) -- each word alone would pass."""
    caps = [m.group(0) for m in re.finditer(r"[^\W\d_][\w'-]*", text)]
    out = set()
    for i in range(len(caps)):
        if not caps[i][:1].isupper():
            continue
        for j in range(i + 1, min(i + 4, len(caps)) + 1):
            if not caps[j - 1][:1].isupper():
                break
            if fold("".join(caps[i:j])) in names_out:
                out |= {fold(w) for w in caps[i:j]}
    return out


def answer(question: str, root, today: dt.date, feeds: list[dict] | None = None) -> dict:
    p = parse(question)
    if p is None:
        return {"answered": False, "reason": "not_a_news_question"}
    lang, subject, when = p
    span = window(when, today)
    if span is None:
        return {"answered": False, "reason": "date_unread", "subject": subject, "lang": lang}
    if feeds is None:
        from .archive import feeds as manifest
        feeds = manifest()
    kinds, names_out = {f["id"]: f.get("kind", "news") for f in feeds}, outlets(feeds)
    want = [t for t in re.findall(r"[^\W_][\w'-]*", fold(subject)) if t not in ("the", "le", "la", "les")]
    rows = [r for r in load(root, span[0].isoformat(), span[1].isoformat()) if kinds.get(r["feed"], "news") == "news"]
    hit = [r for r in rows if all(t in set(re.findall(r"[\w'-]+", fold(r.get("title", "") + " " + r.get("summary", "")))) for t in want)]
    rec = {"question": question, "lang": lang, "subject": subject, "from": span[0].isoformat(), "to": span[1].isoformat()}
    if not hit:
        none = (f"I have no reports about {subject} " + (f"on {_say_date(span[0], lang)}." if span[0] == span[1] else "for those days.")
                if lang == "en" else f"Je n'ai aucun article sur {subject} " +
                (f"le {_say_date(span[0], lang)}." if span[0] == span[1] else "pour ces jours-là."))
        return {**rec, "answered": False, "reason": "no_reports", "reply": none, "items": []}
    lines, used = [], []
    for g in group(hit, kinds=kinds)[:2]:
        first = min(_day(r) for r in g)
        named, words, shown = Counter(), Counter(), {}
        for r in g:
            n, w = features(r.get("title", ""), r.get("summary", ""))
            spelled = outlet_words(r.get("title", "") + " " + r.get("summary", ""), names_out)
            named.update(n - set(want) - names_out - spelled)
            words.update(w - set(want) - spelled)
            for tok in re.findall(r"[^\W\d_][\w'-]*", r.get("title", "") + " " + r.get("summary", "")):
                f = fold(tok)
                shown.setdefault(f, tok)
                shown.setdefault(f[:-1] if f.endswith("s") else f, tok)      # a key word is kept stemmed: show a word
        others = [shown.get(n, n) for n, _ in named.most_common(6) if n not in NOT_NAMES and "'" not in n and "’" not in n][:3]
        keys = [shown.get(w, w).lower() for w, c in words.most_common(6) if fold(w) not in names_out][:3]
        if lang == "en":
            s = f"On {_say_date(first, lang)}, reports about {subject}" + (f" also named {', '.join(others)}" if others else "")
            s += f" ({', '.join(keys)})." if keys else "."
        else:
            s = f"Le {_say_date(first, lang)}, des articles sur {subject}" + (f" citaient aussi {', '.join(others)}" if others else "")
            s += f" ({', '.join(keys)})." if keys else "."
        lines.append(s)
        used += g
    reply = " ".join(lines)
    from ...speech.grounding import longest_run
    run = max(longest_run(reply, r.get("title", "") + " " + r.get("summary", "")) for r in used)
    if run > MAX_RUN:
        return {**rec, "answered": False, "reason": "quote_guard", "items": [r["key"] for r in used]}
    return {**rec, "answered": True, "tier": "reported", "reply": reply, "longest_run": run,
            "items": [r["key"] for r in used], "groups": len(lines)}
