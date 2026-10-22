"""cubbyllm/reasoning/ask_dated.py -- "what happened in <date>?": the date source's facts, the dated answer, the
question too broad to answer and the date with no events. Split out of ask.py (order 0031) as a mixin of AskLoop,
verbatim.

Wired: WIRED (AskLoop).
"""
from __future__ import annotations

import time
import re as _re
from .ask_claims import _learned
from .ask_forms import PROFILE_MAX_LINES
from .ask_words import say

__wiring__ = "WIRED"


class AskDated:
    """AskLoop's date asks (a mixin; order 0031)."""

    def _fetch_news(self, qid: int, entity: str, rec: dict) -> int:
        """A date's headlines into the store, through the SAME gate every other fact passes, each
        keeping its own publisher as provenance and its own day in `times`. Nothing downstream
        changes: `dated_facts` reads `times`, so the news arrives by the front door."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import gate
        from cubbyllm.reasoning.planner import normalize
        if self.news is None:
            return 0
        try:
            items = list(self.news.facts(entity))
        except Exception as e:                               # noqa: BLE001 -- a feed that is down is not an answer
            ev.emit("fetch", qid, source="news-rss", entity=entity, n=0, latent=False,
                    how=f"news unavailable: {str(e)[:80]}")
            return 0
        prov = getattr(self.world, "provenance", None)
        times = getattr(self.world, "times", None)
        fid = ev.emit("fetch", qid, source="news-rss", entity=entity, n=len(items), latent=False,
                      how=(getattr(self.news, "last", None) or {}).get("how"),
                      feeds=(getattr(self.news, "last", None) or {}).get("feeds"))
        n = 0
        for t in items:
            fact = f"{t.obj} is the {t.rel} of {t.subj}"
            key = normalize(fact)
            publisher = self.news.provenance.get(key, "news-rss")
            if hasattr(self.world, "index") and hasattr(self.world.index, "declare_relation"):
                self.world.index.declare_relation(t.rel)
            if hasattr(self.known, "declare"):
                self.known.declare(t.rel)
            g = gate(fact, self.world, publisher, entity)
            rec["learned"].append(_learned(g))
            if g.status == "accepted":
                self.world.add(fact)
                if hasattr(self.known, "add"):
                    self.known.add(fact)
                if prov is not None:
                    prov[" ".join(fact.split())] = publisher       # the PUBLISHER, not "news-rss"
                w = self.news.times.get(key)
                if times is not None and w:
                    times[" ".join(fact.split())] = w
                n += 1
            ev.emit("gate", fid, fact=fact, status=g.status, clash=g.clash, lifted=False, provenance=publisher)
        return n

    # -- the date index: "what happened in <year>" over what the store knows the date of -------
    def dated_facts(self, when: str, limit: int | None = PROFILE_MAX_LINES) -> list[tuple[str, dict]]:
        """The stored facts whose recorded time falls in the asked year or date -- the `times` map the
        source fills (`start`/`end`/`point`), never the fact text. A span counts for every year it
        covers ('position held' 2011-2019 answers 'what happened in 2015'), a point for its own.

        `limit=None` returns them ALL. The caller that has to decide whether a date is too broad
        to answer needs the true count, and a truncated list always looked answerable -- the
        clarify threshold could never fire behind a cap of nine."""
        m = _re.search(r"\b(1[0-9]{3}|2[0-9]{3})\b", when)
        if not m:
            return []
        year = int(m.group(1))
        day = None
        d = _re.search(r"\b(1[0-9]{3}|2[0-9]{3})-(\d{2})-(\d{2})\b", when)
        if d:
            day = d.group(0)
        out = []
        for fact, t in (getattr(self.world, "times", None) or {}).items():
            pt, st, en = t.get("point"), t.get("start"), t.get("end")
            hit = None
            if pt and (pt == day or (not day and pt[:4] == str(year))):
                hit = "on"
            elif st or en:
                # a CLOSED span is known to have covered its years, so it counts for each of them; an OPEN
                # one (a start with no end) is a state whose end the source never stated, and counting it
                # forward claims something nobody said. 2026-09-14, live and WRONG: "what happened in 2026?"
                # answered "Quebec City was the capital of Quebec" -- capital SINCE 1867, open -- and listed
                # seven twin cities as 2026 events. An open span counts for its start year only: that is the
                # part that happened.
                y0 = int(st[:4]) if st else None
                y1 = int(en[:4]) if en else None
                if y0 is not None and y0 == year:
                    hit = "began"
                elif y1 is not None and y1 == year:
                    hit = "ended"
                elif y0 is not None and y1 is not None and y0 < year < y1:
                    hit = "ongoing"
            if hit:
                out.append((fact, dict(t, hit=hit)))
        out.sort(key=lambda p: (p[1].get("point") or p[1].get("start") or "", p[0]))
        return out if limit is None else out[:limit]

    def dated(self, rec: dict, qid: int, pid: int, entity: str, dated: list, t0: float) -> dict:
        """The dated facts, each recovered by the VM like any profile line, as the answer."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import Triple, normalize, parse_fact
        from cubbyllm.reasoning.programs import build_chain_program
        prov = getattr(self.world, "provenance", None)
        wid = ev.emit("walk", pid, verified=None, reason="profile", answer=None, refused=None)
        kept = []
        for fact, t in dated:
            tr = parse_fact(fact)
            if tr is None:
                continue
            source, fns = build_chain_program([Triple(obj=tr.obj, rel=tr.rel, subj=tr.subj)], [tr.rel])
            out = self._run_fn(source, fns[0])
            sim = out.get("similarity")
            ok = sim is not None and sim >= self.tau_profile and normalize(out.get("result") or "") == normalize(tr.obj)
            span = t.get("point") or "–".join(x for x in (t.get("start"), t.get("end")) if x)
            src_name = (prov or {}).get(fact) or ("store" if fact in self.world else None)
            hid = ev.emit("hop", wid, hop=0, query=f"what happened in {entity}", fact=fact, how="dated · VM recover",
                          similarity=sim, provenance=src_name, verified=ok, when=span)
            ev.emit("fact", hid, text=fact, subj=tr.subj, rel=tr.rel, obj=tr.obj, provenance=src_name, when=span)
            if ok:
                kept.append({"relation": tr.rel, "value": tr.obj, "subject": tr.subj, "fact": fact,
                             "provenance": src_name, "similarity": sim, "when": span, "hit": t.get("hit")})
        ev.emit("vm", wid, verified=bool(kept) and len(kept) == len(dated), program_lines=len(dated) + 2,
                recovered=len(kept), bound=len(dated))
        rec["profile"] = kept; rec["trace"] = [k["fact"] for k in kept]
        if kept:
            rec["answer"] = f"{entity} — " + "; ".join(f"{k['when']}: {k['value']} is the {say(k['relation'])} of {k['subject']}" for k in kept)
            rec["prose"] = self.paraphrase(pid, entity, "when", kept)
            rec["scope"] = "the facts the store knows a date for -- not a record of the year"
        else:
            rec["reason"] = "profile_unrecovered"
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"], facts=rec["answer"],
                profile=rec["profile"] or None, prose=rec["prose"], scope=rec.get("scope"),
                entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def too_broad(self, rec: dict, qid: int, entity: str, dated: list, t0: float) -> dict:
        """Too much is dated in that range to answer in one breath, so the loop asks instead of
        picking. Nick, 2026-09-14: "when ambiguous it should either select A based on context or
        B ask the user: do you want to know more about: choices."

        The choices are a COUNT over the facts actually fetched -- the recurring names in them,
        commonest first. That is deliberately not a ranking of importance: this loop has no way
        to know what mattered on a day and no business pretending it does. It offers what was
        written about, says how much there is, and lets the asker narrow it."""
        from cubbyllm.reasoning import events as ev
        choices: list[str] = []
        if self.news is not None:
            try:
                choices = [w for w, _n in self.news.topics(entity)[:8]]
                if not choices:
                    # One day's headlines rarely repeat a name, so the recurrence bar that makes
                    # a YEAR's menu meaningful empties a DAY's (2026-09-14, live: 51 facts and
                    # nothing to offer). "51 things happened and I have no suggestions" is a
                    # worse answer than a list of single mentions, and both are honest.
                    choices = [w for w, _n in self.news.topics(entity, min_n=1)[:8]]
            except Exception:                                # noqa: BLE001
                choices = []
        if not choices:                                      # no news source: the relations themselves
            choices = list(dict.fromkeys(t.get("relation") or "" for _f, t in dated if isinstance(t, dict)))[:8]
        rec["reason"] = "date_too_broad"
        rec["clarify"] = {"entity": entity, "n": len(dated), "choices": [c for c in choices if c],
                          "how": "a count over the headlines fetched -- what was written about, not what mattered"}
        rec["needs"] = (f"{len(dated)} facts are dated in {entity}; ask about one of them, "
                        f"or a narrower date")
        ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"],
                clarify=rec["clarify"], entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def no_events(self, rec: dict, qid: int, t0: float) -> dict:
        """Nothing in the store is dated in that range, and a `when` ask does not fetch a date's own item."""
        from cubbyllm.reasoning import events as ev
        rec["reason"] = "no_events_for_date"
        rec["needs"] = ("no source indexes events by date: the event facts are keyed by entity, not by "
                        "year, and a date's own item carries only its trivia")
        ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"],
                entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec
