"""cubbyllm/reasoning/ask_memory.py -- what the ask loop keeps between questions: the facts held and learned (from the
sleep cycle's log), the store's snapshot and changes, a subject's life span, and the time checks on undated values.
Split out of ask.py (order 0031) as a mixin of AskLoop, verbatim.

Wired: WIRED (AskLoop).
"""
from __future__ import annotations

import pathlib
import time
from .ask_words import hedged, say, strip_as_of

__wiring__ = "WIRED"


class AskMemory:
    """AskLoop's memory between questions (a mixin; order 0031)."""

    def _keep(self, rec: dict) -> None:
        """Keep the record in memory and, with a history path, append it as one JSON line: an id, the
        time, the store's snapshot hash, and for each learned fact its relation (as the index split it
        now, so a reload splits it the same way) and WHEN the source said it was true. Inside an ask,
        the record gets its thought graph first, so the history keeps how the answer came about."""
        from ..speech.audience import mouth, record_hash
        mouth(rec, self.guard)                           # the last check before it is spoken (order 0016)
        g = getattr(self, "_graph", None)
        if g is not None and "graph" not in rec:
            try:
                rec["graph"] = g[0].graph(g[1])
            except Exception as e:                       # noqa: BLE001 -- a record is kept even if its graph is not
                rec["graph"] = {"error": str(e)[:200]}
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec.setdefault("ts", ts)
        rec.setdefault("id", f"{ts}-{len(self.history) + 1}")   # the handle an asker's feedback names
        rec["hash"] = record_hash(rec)                   # the audit chain's seal (audience.audit_chain)
        self.history.append(rec)
        if self.history_path is None:
            return
        try:
            import json as _json

            from cubbyllm.reasoning.learn import snapshot
            from cubbyllm.reasoning.planner import parse_fact
            known = getattr(getattr(self.world, "index", None), "_reused", None)
            times = getattr(self.world, "times", None) or {}
            for l in rec.get("learned") or []:
                key = " ".join(str(l.get("fact", "")).split())
                if l.get("status") == "accepted" and "rel" not in l:
                    t = parse_fact(key, known=known)
                    if t is not None:
                        l["rel"] = t.rel
                if key in times:
                    l.setdefault("time", times[key])
            line = dict(rec, store_snapshot=snapshot(self.world))
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception as e:                           # noqa: BLE001 -- a history that cannot be written never costs an answer
            print(f"[ask] history not written: {type(e).__name__}: {e}", flush=True)

    def load_held(self, path) -> int:
        """The facts a bulk pass held (two values of a relation that has one: order 0013's `_held.jsonl`), as a veto:
        an answer whose last fact is about a held (subject, relation) is refused `contested`, whoever states it --
        a live read that sees one value only does not make the other go away."""
        from cubbyllm.reasoning.planner import normalize, parse_fact
        from cubbyllm.reasoning.sleep import load_facts
        n = 0
        for key, r in load_facts(pathlib.Path(path)).items():
            t = parse_fact(key, known={r["rel"]} if r.get("rel") else None) if r.get("rel") else parse_fact(key)
            subj = r.get("subject") or (t.subj if t else None)
            if subj and (r.get("rel") or t):
                self.contested.add((normalize(subj), normalize(r.get("rel") or t.rel))); n += 1
        return n

    def load_learned(self, path) -> int:
        """Facts the sleep cycle made durable (its ``learned_facts.jsonl``), back into the store at boot,
        with their relation declared, their source as provenance and their time. What the loop learned
        yesterday is looked up today like any other fact -- and verified by the VM like any other."""
        from cubbyllm.reasoning.sleep import load_facts
        p = pathlib.Path(path)
        if not p.exists():
            return 0
        idx = getattr(self.world, "index", None)
        prov = getattr(self.world, "provenance", None)
        times = getattr(self.world, "times", None)
        n = 0
        for key, r in load_facts(p).items():
            rel = r.get("rel")
            if rel:
                if idx is not None and hasattr(idx, "declare_relation"):
                    idx.declare_relation(rel)
                if hasattr(self.known, "declare"):
                    self.known.declare(rel)
            if self.world.add(key):
                n += 1
                if hasattr(self.known, "add"):
                    self.known.add(key)
            if prov is not None:
                prov.setdefault(key, r.get("source") or "learned")
            if times is not None and r.get("time"):
                times.setdefault(key, r["time"])
        return n

    def _changed(self, h, as_of, rec) -> str:
        """Order 0024: " Reports on <day> say this may have changed." when the news marks the fact's subject after the
        world's date (``news_trigger``); the marks go to the record (internal), the reply names no outlet."""
        if self.news_changes is None or h is None or h.triple is None:
            return ""
        marks = self.news_changes.since(h.triple.subj, as_of)
        if not marks:
            return ""
        rec["news_changes"] = marks
        return f" Reports on {marks[-1]['day']} say this may have changed."

    def _checked(self, question: str, lr, rec: dict, again=None) -> None:
        """The checks a walked answer passes before it is said, on its record: the bulk pass's veto (order 0013), the
        question's tense and year (0012), a snapshot's "as of" (0014), the time rule on an undated value that changes
        (0015, Addendum 2), one witness hedged. Only for a walk that verified and a record that still says so (a
        namesake's refusal stands: order 0037). Out of `AskLoop._ask` so each namesake walked apart passes them too.
        `again` (the same walk once more): before a snapshot's changing value is said "as of", the source is asked
        for today's (`ask_source_first`, Nick 2026-10-27), and its value goes through these checks in its turn."""
        from cubbyllm.reasoning.planner import normalize
        from .ask_source_first import source_wins
        if not lr.result.verified or not rec.get("verified"):
            return
        from .ask_class import check as class_check       # an answer of the class asked (order 0047; CB_HELDOUT_FIXES=0 off)
        if not class_check(self, question, lr, rec):
            return
        why = self._time_check(question, lr) if self.tense else None
        snap = self._snapshot(question, lr) if not why else None
        if snap and snap[0] != "as_of":                   # a snapshot's value asked about another time (order 0014)
            why = snap[1]
        undated = self._undated(question, lr) if not why and not snap else None
        h = lr.result.trace[-1] if lr.result.trace else None
        contested = h is not None and h.triple is not None and (normalize(h.triple.subj), normalize(h.triple.rel)) in self.contested
        as_of = snap[1] if snap else undated[1] if undated and undated[0] == "as_of" else None
        if as_of and not why and not contested and source_wins(self, question, lr, rec, again, str(as_of)):
            return                                       # the source's value, checked in its turn
        if contested:
            rec.update(answer=None, verified=False, reason="contested",    # the bulk pass held two values: never said
                       refused={"answer": lr.result.answer, "why": "the sources disagree on it"})
        elif why:                                           # the question's tense or year is not the fact's: not said
            rec.update(answer=None, verified=False, reason="time_mismatch", refused={"answer": lr.result.answer, "why": why})
        elif snap:                                        # a value that changes, from a dated snapshot: said "as of"
            rec.update(tier="as_of", reply=f"As of {snap[1]}, {rec['answer']}." + self._changed(h, snap[1], rec))
        elif undated and undated[0] == "refuse":          # the time rule (0012): a year asked, the books date nothing
            rec.update(answer=None, verified=False, reason="time_undated", refused={"answer": lr.result.answer, "why": undated[1]})
        elif undated and undated[0] == "as_of":           # a world's undated value that changes, with the world's date
            rec.update(tier="as_of", reply=f"As of {undated[1]}, {rec['answer']}." + self._changed(h, undated[1], rec))
        elif undated:                                     # a value that changes, undated: true at one point, said so
            rec.update(tier="at_one_point", reply=hedged(undated[1].rstrip(".")) if lr.attributed else undated[1])
        elif lr.attributed:                               # one witness only: said hedged, the source never named
            rec.update(tier="attributed", reply=hedged(rec["answer"]))

    def _snapshot(self, question: str, lr):
        """For an answer whose last fact a snapshot world holds on a relation whose value changes (heads of state,
        teams, employers: `hdc_store`): ('as_of', year) to say it as of the snapshot; ('refuse', why) for another
        year asked; None otherwise -- the year asked IS the snapshot's, or the fact is not a snapshot's."""
        from cubbyllm.reasoning.planner import normalize
        h = lr.result.trace[-1] if lr.result.trace else None
        w = self.world.held_by(h.fact) if h is not None and h.triple is not None and hasattr(self.world, "held_by") else None
        if w is None or normalize(h.triple.rel) not in getattr(w, "time_sensitive", ()):
            return None
        from .question_forms import tense_of
        year = strip_as_of(question)[1]
        if year is not None:
            return None if str(year) == w.snapshot else ("refuse", f"asked about {year}, the snapshot holds {w.snapshot}")
        if tense_of(question) == "future":
            return ("refuse", f"asked in the future, the snapshot holds {w.snapshot}")
        return ("as_of", w.snapshot)

    def _life(self, entity: str):
        """(born, died) years the world holds for `entity` (either None): the books' same-name guard."""
        from ..worlds.book_source import year
        from cubbyllm.reasoning.planner import normalize
        out = [None, None]
        for _f, t in getattr(getattr(self.world, "index", None), "_by_subj", {}).get(normalize(entity or ""), []):
            k = {"birth date": 0, "date of birth": 0, "death date": 1, "date of death": 1}.get(normalize(t.rel))
            if k is not None and out[k] is None:
                out[k] = year(t.obj)
        return tuple(out)

    def _undated(self, question: str, lr):
        """The time rule (`time_rule`): for an answer whose last fact the books witness on a relation that changes
        over its subject's life, and not every witness dates it -- ('refuse', why) when the question names a year,
        ('hedge', the answer at one point) otherwise; None when the rule does not apply."""
        from cubbyllm.reasoning.planner import normalize
        from ..worlds.time_rule import at_one_point, undated
        h = lr.result.trace[-1] if lr.result.trace else None
        if h is None or h.triple is None:
            return None
        ws = getattr(self.books, "citations", {}).get(normalize(h.fact))
        key = " ".join(h.fact.split())
        prov = (getattr(self.world, "provenance", None) or {}).get(key, "")
        if ws is None and prov and all(x.startswith("book:") for x in prov.replace(" (attributed)", "").split("+")):
            ws = [{"when": (getattr(self.world, "times", None) or {}).get(key) or {}}]   # a book fact from a store (0013)
        i = question.lower().find(h.triple.subj.lower())                # the subject as the asker wrote it
        subj = question[i:i + len(h.triple.subj)] if i >= 0 else h.triple.subj
        year = strip_as_of(question)[1]
        if ws is None:                                   # a world's own fact (0015, Addendum 2): its "as of", or "at one point"
            from ..worlds.time_rule import changes, dated
            when = (getattr(self.world, "times", None) or {}).get(key) or {}
            if not self.time_rule or not changes(h.triple.rel) or dated(when):
                return None
            holder = self.world.held_by(h.fact) if hasattr(self.world, "held_by") else None
            as_of = getattr(holder, "snapshot", None) or getattr(holder or self.world, "as_of", None)
            if as_of:
                return ("as_of", str(as_of)) if year is None or str(year) == str(as_of) else \
                       ("refuse", f"asked about {year}, the world holds it as of {as_of}")
            if year is not None:
                return ("refuse", f"the world does not date when {h.triple.obj} was the {say(h.triple.rel)} of {subj}")
            return ("hedge", at_one_point(lr.result.answer, say(h.triple.rel), subj))
        if not undated(h.triple.rel, ws):
            return None
        if year is not None:
            return ("refuse", f"no witness dates when {h.triple.obj} was the {say(h.triple.rel)} of {subj}")
        return ("hedge", at_one_point(lr.result.answer, say(h.triple.rel), subj))

    def _time_check(self, question: str, lr) -> str | None:
        """Why the answer's last fact cannot be what the question asks about in time, or None: its tense (the
        form it parses to, else its auxiliary) and its year against the dates the fact carries
        (`question_forms.time_check`; an undated fact is not checked)."""
        from cubbyllm.reasoning.planner import normalize
        from .question_forms import Forms, noun_tense, tense_of, time_check
        h = lr.result.trace[-1] if lr.result.trace else None
        if h is None or h.triple is None:
            return None
        self._forms = self._forms or Forms()
        p = self._forms.parse(question)
        year = strip_as_of(question)[1]
        times = getattr(self.world, "times", None) or {}
        when = times.get(" ".join(h.fact.split())) or times.get(normalize(h.fact))
        kind = self._forms.aspect(h.triple.rel)
        tense = p[2] if p else tense_of(question)
        if p is None and kind == "event":                # "What is the date of birth of X?": the noun's "is", no tense
            tense = noun_tense(question, h.triple.rel, tense)
        return time_check(kind, tense, int(year) if year else None, when, h.triple.obj, time.localtime().tm_year)
