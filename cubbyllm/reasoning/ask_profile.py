"""cubbyllm/reasoning/ask_profile.py -- "who / what / where is X?": the profile program the host runs, the self
profile (what the loop knows of itself), and the paraphrase a talk adapter may give it, kept only when grounded.
Split out of ask.py (order 0031) as a mixin of AskLoop, verbatim.

Wired: WIRED (AskLoop).
"""
from __future__ import annotations

import time
from ..worlds.self_source import LOOP, SelfSource   # noqa: E402
from .ask_forms import (FALLBACK_KINDS, NEWS_CLARIFY_MIN, PARAPHRASE_SYSTEM, PROFILE, PROFILE_KINDS,
                        PROFILE_MAX_LINES, PROFILE_MAX_PER_RELATION, RESOLVE_HINT, cot_profile, cot_self)
from .ask_words import grounded_prose, say, when_text
from .ask_namesakes import pick as namesake_pick, refuse as namesake_refuse

__wiring__ = "WIRED"


class AskProfile:
    """AskLoop's profile asks (a mixin; order 0031)."""

    def introspect(self, question: str, kind: str, entity: str, t0: float, how: str) -> dict:
        """The loop asked about ITSELF. Same shape as a profile ask and a different store.

        The loop is the one subject it can witness directly -- it does not take anybody's word
        for its own history -- so these facts need no source and carry `self` as provenance.
        That privilege is fenced: a self fact never enters the world store, the answer says
        plainly that it is about the loop, and there is no relation here that carries a
        judgement. "refused 3 times" is a fact; "is unreliable" is an opinion, and this path
        cannot express one.

        The facts are DERIVED on every ask and never stored, so they cannot go stale and cannot
        be edited into something the loop did not do."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        qid = ev.emit("question", text=question, source="self")
        pid = ev.emit("plan", qid, seed=normalize(entity), relations=[], tail=entity, n_hop=0, ask=kind,
                      program=cot_self(normalize(entity), kind), how=how)
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": "self",
                     "plan": [], "seed": normalize(entity), "ask": kind, "world": "self",
                     "program": cot_self(normalize(entity), kind), "learned": [], "entities": [entity],
                     "aliased": [], "snapped": None, "trace": [], "profile": [], "prose": None,
                     "candidates": None, "clarify": None, "wall_s": 0.0}
        src = SelfSource(self)
        items = src.facts(entity)
        ev.emit("fetch", qid, source="self", entity=entity, n=len(items), latent=False,
                how=(src.last or {}).get("how"))
        if not items:
            rec["reason"] = "no_self_record"
            rec["needs"] = (src.last or {}).get("how") or "the loop has no record of that subject"
            ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"])
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            return rec
        lines = []
        for t in items:
            fact = f"{t.obj} is the {t.rel} of {t.subj}"
            lines.append({"relation": t.rel, "value": t.obj, "fact": fact, "provenance": "self",
                          "similarity": 1.0, "when": None})
        rec["profile"] = lines
        rec["trace"] = [l["fact"] for l in lines]
        by_rel: dict[str, list[str]] = {}
        for l in lines:
            by_rel.setdefault(say(l["relation"]), []).append(l["value"])
        rec["answer"] = entity + " \u2014 " + "; ".join(f"{r}: {', '.join(v)}" for r, v in by_rel.items())
        # NO PARAPHRASE ON THIS PATH, deliberately. 2026-09-14, live and wrong: given
        # `fact learned: 152; source: wikidata (152)` the talk adapter wrote "referenced in 152
        # sources" (it is 152 facts from ONE source), and given `last answer: 574482` it wrote
        # "the last answer was given 574482 units ago". Every name and number in both sentences
        # occurs in the facts, so the grounding guard passed them: the guard checks VOCABULARY,
        # not semantics, and it holds up on world facts only because the relations there are
        # ordinary English the adapter has seen a million times. "fact learned", "call",
        # "provenance" are not, and it confabulates what they mean.
        #
        # A loop reporting on itself has no business being fluent about it. The counts ARE the
        # answer, they are already readable, and prose can only add an interpretation of numbers
        # whose meaning the speaker does not know. Fluency here buys nothing and costs the kill
        # line.
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"],
                facts=rec["answer"], profile=rec["profile"], prose=rec["prose"], entities=[entity], fetched=0)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def profile(self, question: str, kind: str, entity: str, t0: float, how: str, item: str | None = None, asker: str | None = None) -> dict:
        """'who / what / where is X' -- the retrieval program. The plan is SEED + ASK (what the emitter
        learns to write); the host fills the store from the source when it holds nothing about X; the
        facts under the ask kind's relations are bound into a program and the VM recovers each one --
        only what the VM returned at the 1-hop threshold is spoken. The reply is those facts; the talk
        adapter, when one is mounted, paraphrases them, and the paraphrase is kept only when every name
        and number in it is in the facts. `reason: profile` -- a retrieval, not a walked chain."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        from cubbyllm.reasoning.programs import build_chain_program
        qid = ev.emit("question", text=question, source=getattr(self.source, "name", None))
        pid = ev.emit("plan", qid, seed=normalize(entity), relations=[], tail=entity, n_hop=0, ask=kind,
                      program=cot_profile(normalize(entity), kind), how=how)
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": "profile", "plan": [], "seed": normalize(entity),
                     "ask": kind, "program": cot_profile(normalize(entity), kind), "learned": [], "entities": [], "aliased": [],
                     "snapped": None, "trace": [], "profile": [], "prose": None, "candidates": None, "clarify": None, "wall_s": 0.0}
        prov = getattr(self.world, "provenance", None)
        rels = PROFILE_KINDS[kind]
        if kind == "when":
            # the date index: the facts the store knows the DATE of. A `when` ask never FETCHES the date
            # itself -- a year's own item is trivia (2026-09-14, live: asking "what happened in 2026?"
            # gated `2026 is the point in time of 2026` into the store and then read it back as the
            # answer). Events come from facts about entities, dated by their source; nothing here
            # invents them, and an empty index is an honest refusal.
            dated = self.dated_facts(entity, limit=None)         # the TRUE count: see too_broad
            if not dated and self.news is not None:
                # ... and when the store knows nothing about that date, ask something that might.
                # This is the fix for the refusal Nick hit head-on: "if I ask what happened in
                # 2026? it refuses it" -- correctly, because the store held nothing dated. It
                # holds something now, and it came through the gate like everything else.
                self._fetch_news(qid, entity, rec)
                dated = self.dated_facts(entity, limit=None)
            if len(dated) > NEWS_CLARIFY_MIN:
                return self.too_broad(rec, qid, entity, dated, t0)
            if dated:
                return self.dated(rec, qid, pid, entity, dated[:PROFILE_MAX_LINES], t0)
            have = [t for t in self.facts_about(entity) if normalize(t.rel) in PROFILE_KINDS["when"]]
            if not have:
                return self.no_events(rec, qid, t0)
        store_key = item or entity                       # an item the asker chose is read as itself, never as the bare
        have = self.facts_about(store_key)               # label, which may be another namesake's (order 0036)
        last: dict = {}
        known_rels = set(rels) | (set(PROFILE) if kind in FALLBACK_KINDS else set())
        # a name the stores hold only as several items' (order 0033): the asker or the question's own qualifier decide
        # which item, and that item's facts alone are read (never beside another world's facts about the bare name,
        # which may be another namesake's); when nothing decides, the asker is asked
        ns = namesake_pick(self, entity, item) if item or not any(normalize(t.rel) in known_rels for t in have) else None
        picked = (ns or {}).get("rows") or []
        if picked:
            entity = ns["label"]; rec["namesake"] = {"item": ns["item"], "how": ns["how"]}
            have = [t for _f, t in picked]
        elif not any(normalize(t.rel) in known_rels for t in have):
            last = self._fetch(qid, entity, list(RESOLVE_HINT[kind]), rec, item, asker)
            have = self.facts_about(store_key)
        # the lines: the ask kind's relations first, then any other profile relation the store holds
        chosen: list[tuple[str, str, str]] = []                       # (rel, obj, the stored fact text)
        by_key = {(normalize(t.rel), t.obj): " ".join(f.split())
                  for f, t in (picked or self.world.index._by_subj.get(normalize(store_key), [])) if t is not None}
        order = tuple(rels) + (tuple(r for r in PROFILE if r not in rels) if kind in FALLBACK_KINDS else ())
        times = getattr(self.world, "times", None) or {}
        for rel in order:
            objs = list(dict.fromkeys(t.obj for t in have if normalize(t.rel) == rel))
            # a value whose truth depends on WHEN: Quebec City's population is 547 and 516,622 and
            # 546,958, each true at its own census (2026-09-14, Nick: `547 is the population of Quebec
            # City` gated in, clashing with 516622 -- the source dated every one of them and the store
            # dropped the date). A present-tense ask wants the LATEST, said with its date; the older
            # values stay stored, and a question that binds a year reads them through the date index.
            dated = [(o, (times.get(by_key.get((rel, o), "")) or {})) for o in objs]
            n_dated = sum(1 for _o, w in dated if w)
            if n_dated > 1:
                dated.sort(key=lambda q: (q[1].get("point") or q[1].get("start") or ""), reverse=True)
                objs = [o for o, _w in dated]
            for obj in objs[:1 if n_dated > 1 else PROFILE_MAX_PER_RELATION]:
                chosen.append((rel, obj, by_key.get((rel, obj), f"{obj} is the {rel} of {entity}")))
            if len(chosen) >= PROFILE_MAX_LINES:
                break
        chosen = chosen[:PROFILE_MAX_LINES]
        if not chosen:
            amb = last.get("ambiguous") or (ns or {}).get("ambiguous") or []
            if amb:
                # the asker decides, the host never picks (Nick, 2026-09-14: "when ambiguous it should
                # either select A based on context or B ask the user: do you want to know more about:
                # choices"). A is the resolution tiers above -- the hop the question needs, label over
                # alias, the meta-pages dropped; when they leave more than one, this is B: the choices go
                # back with the item ids, and the answer re-asks with the one the asker names.
                namesake_refuse(rec, amb, None if last.get("ambiguous") else (ns or {}).get("of"))
            else:
                rec["reason"] = ("no_events_for_date" if kind == "when" else
                                 ("entity_unresolved" if rec["entities"] and not rec["learned"] else "profile_empty"))
                if kind == "when":
                    # the honest diagnosis (2026-09-14): nothing here indexes events BY DATE. The wiki world
                    # holds 76,897 `timeline event` facts, every one keyed by its ENTITY ('Girard Desargues'
                    # -> 'Girard Desargues Birth'), and 0 facts whose subject is a year; a year's item carries
                    # the year's own trivia, not its events. `unknown_relation` blamed the word 'happened'
                    # for a missing index. A dated-news world is what would answer this shape.
                    rec["needs"] = ("no source indexes events by date: the event facts are keyed by entity, not by "
                                    "year, and a year's own item carries only its trivia")
            ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], candidates=rec["candidates"], clarify=rec.get("clarify"), needs=rec.get("needs"),
                    entities=list(rec["entities"]), fetched=len(rec["learned"]))
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            return rec
        # the retrieval program: every chosen fact bound; the VM recovers each role -- what it returns is spoken
        from cubbyllm.reasoning.planner import Triple
        wid = ev.emit("walk", pid, verified=None, reason="profile", answer=None, refused=None)
        kept: list[dict] = []; n_calls = 0
        for rel, obj, fact in chosen:
            triple = Triple(obj=obj, rel=rel, subj=entity)
            source, fns = build_chain_program([triple], [rel])
            out = self._run_fn(source, fns[0]); n_calls += 1
            sim = out.get("similarity"); got = out.get("result")
            ok = sim is not None and sim >= self.tau_profile and normalize(got or "") == normalize(obj)
            src_name = (prov or {}).get(fact) or ("store" if fact in self.world else None)
            hid = ev.emit("hop", wid, hop=0, query=f"what is the {rel} of {entity}", fact=fact, how="profile · VM recover",
                          similarity=sim, provenance=src_name, verified=ok)
            ev.emit("fact", hid, text=fact, subj=entity, rel=rel, obj=obj, provenance=src_name)
            if ok:
                w = (getattr(self.world, "times", None) or {}).get(fact) or {}
                kept.append({"relation": rel, "value": obj, "fact": fact, "provenance": src_name, "similarity": sim,
                             "when": (w.get("point") or "–".join(x for x in (w.get("start"), w.get("end")) if x)) or None})
        ev.emit("vm", wid, verified=bool(kept) and len(kept) == len(chosen), program_lines=len(chosen) + 2,
                recovered=len(kept), bound=len(chosen))
        rec["profile"] = kept; rec["trace"] = [k["fact"] for k in kept]
        if kept:
            by_rel: dict[str, list[str]] = {}
            for l in kept:
                by_rel.setdefault(say(l["relation"]), []).append(l["value"] + (f" ({when_text(l['when'])})" if l.get("when") else ""))
            rec["answer"] = entity + " — " + "; ".join(f"{r}: {', '.join(v)}" for r, v in by_rel.items())
            rec["prose"] = self.paraphrase(pid, entity, kind, kept)
        else:
            rec["reason"] = "profile_unrecovered"
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"], facts=rec["answer"],
                profile=rec["profile"] or None, prose=rec["prose"], entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def paraphrase(self, pid: int, entity: str, kind: str, lines: list[dict]) -> str | None:
        """The talk adapter says the facts in a sentence or two; kept only when grounded -- every name and
        number in it occurs in the facts. No talk adapter mounted (a programs-only emitter): None, and
        the facts are the reply."""
        from cubbyllm.reasoning import events as ev
        adapters = getattr(self.emitter, "adapters", None)
        if not isinstance(adapters, dict) or "talk" not in adapters or not getattr(self.emitter, "is_split", False):
            return None
        facts = [l["fact"] for l in lines]
        # the adapter reads the SAID form (say() has already dropped the schema words) -- asking a 4B model
        # to translate "the administrative territorial entity" itself only spends its attention on the
        # database's vocabulary instead of on the sentence. 2026-09-14, Nick, live.
        said = [f"{say(l['relation'])}: {l['value']}" + (f" ({when_text(l['when'])})" if l.get("when") else "")
                for l in lines]
        ask = {"who": "who", "what": "what", "where": "where", "when": "what happened in"}.get(kind, "what")
        prompt = (f"Facts about {entity}:\n" + "\n".join(f"- {s}" for s in said) +
                  f"\n\nIn one or two plain sentences, say {ask} {entity} is, using ONLY these facts. "
                  "Every NAME, DATE and NUMBER must appear exactly as given, and you may add none that is "
                  "not in the facts.")
        from ..emit.base import clean_reply
        # the structured form rides beside the prompt (2026-10-17): the 450M talk adapter was trained on the
        # facts block ("Facts: / - entity -- relation: value / Question: / Answer:") and renders it itself;
        # the stand-in reads the instruction above and ignores these keys
        talk_ctx = {"role": "talk",
                    "facts": [(entity, say(l["relation"]), l["value"] + (f" ({when_text(l['when'])})" if l.get("when") else ""))
                              for l in lines],
                    "question": f"{ {'who': 'Who', 'where': 'Where', 'when': 'What happened in'}.get(kind, 'What')} is {entity}?"}
        try:
            raw = self.emitter.emit(prompt, max_new_tokens=160, context=talk_ctx, temperature=0.0,
                                    system=PARAPHRASE_SYSTEM)     # not the identity persona: under it the talk adapter says nothing here
            text = " ".join(clean_reply(raw).split())
        except Exception as e:                                   # noqa: BLE001 -- no paraphrase, the facts stand
            ev.emit("paraphrase", pid, ok=False, text=None, rejected=[f"talk adapter failed: {str(e)[:80]}"])
            return None
        ok, bad = grounded_prose(text, facts + said, entity)
        ev.emit("paraphrase", pid, ok=ok, text=text if ok else None, rejected=bad or None, draft=text, attempt=1)
        if ok:
            return text
        # A refusal is a result, but it is a poor one when the facts are right there and only one word
        # was invented (2026-09-14: the Bank of England lost its whole paraphrase over "institution").
        # The adapter is told which words it made up and gets ONE more turn. The guard is unchanged --
        # the second draft passes the same check or is refused the same way.
        try:
            retry = (prompt + "\n\nYour last attempt used " + ", ".join(f"'{b}'" for b in bad[:6]) +
                     " -- no fact above says that. Write it again without those words, and do not say "
                     "what KIND of thing it is unless a fact names the kind.")
            raw = self.emitter.emit(retry, max_new_tokens=160, context=talk_ctx, temperature=0.0,
                                    system=PARAPHRASE_SYSTEM)
            text2 = " ".join(clean_reply(raw).split())
        except Exception as e:                                   # noqa: BLE001
            ev.emit("paraphrase", pid, ok=False, text=None, rejected=[f"retry failed: {str(e)[:60]}"], attempt=2)
            return None
        ok2, bad2 = grounded_prose(text2, facts + said, entity)
        ev.emit("paraphrase", pid, ok=ok2, text=text2 if ok2 else None, rejected=bad2 or None, draft=text2, attempt=2)
        return text2 if ok2 else None
