"""The verified-program loop on ONE natural question, live -- what exp_r11 runs per row, as a service.

Wired: WIRED (the server mounts it behind `POST /ask` when started with `--ask`); every step it takes
is an event on `/loop/stream`,
so the control panel (dashboard/control_panel.html, served at /panel) draws the tree as it grows:
question -> plan (the emitter's proposal, the host's aliases) -> walk -> hops -> facts; the fetch
from the source and the gate's verdict on each fact; the answer or the refusal.

The seven invariants hold here exactly as in the benches: the emitter proposes a plan, the host
disposes (covers, lever 4/6/7), the VM verifies, the source's facts pass the gate with provenance,
and the answer is spoken only when a VM-verified chain reaches it -- never from the model.
"""
from __future__ import annotations

import collections
import inspect as _inspect
import os
import pathlib
import sys
import threading
import time

from ..worlds.self_source import LOOP, SelfSource  # noqa: F401 -- ask.LOOP, ask.SelfSource as before
# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .ask_forms import (FALLBACK_KINDS, NEWS_CLARIFY_MIN, PARAPHRASE_SYSTEM, PROFILE, PROFILE_KINDS,  # noqa: F401
                        PROFILE_MAX_LINES, PROFILE_MAX_PER_RELATION, RESOLVE_HINT, TAU_VM, _ASK_ROLE, _HOP_ROLE,
                        _PARTICIPLES, _PROFILE_ASK, _RELATION_ASKS, _SEED_ROLE, _SELF_ABOUT, _SELF_ABOUT2, _SELF_LOOP,
                        _WHEN_ASK, _WORLD_ROLE, _verb_tail, cot_profile, cot_self, profile_ask, profile_program,
                        program_world, relation_ask, self_ask)
from .ask_words import (ABSENT_REPLY, SAY, _AS_OF_PHRASE, _CATEGORY, _GROUP, _ISO, _MONTHS, _SMALL,  # noqa: F401
                        _TOKEN, _num_key, _vnorm, category_claims, date_words, grounded_prose, has_value, hedged, say,
                        strip_as_of, value_check, when_text)
from .ask_claims import (BACKWARD_CUES, FORWARD_CUES, _BEGIN, _CAUSAL_REL, _END, _PRONOUN, _SENT,  # noqa: F401
                         _aliases, _cue, _learned, _mentions, branch_clarify, claim_check, parse_choose, talk_reply)
from .ask_namesakes import after_walk, also, seed_for
from ..spatial.scene_ask import scene_ask
from .ask_memory import AskMemory
from .ask_relation import AskRelation
from .ask_profile import AskProfile
from .ask_dated import AskDated

ROOT = pathlib.Path(__file__).resolve().parents[2]

__wiring__ = "WIRED"


class AskLoop(AskDated, AskProfile, AskRelation, AskMemory):
    """`ask(question)` -> the loop's record. `emitter` is any object with `.emit(prompt, max_new_tokens=)`
    (the serving emitter, shared); `world` a FactStore with a TripleIndex (the wiki world when None);
    `source` 'wikidata' | 'wikidata-offline' | a Source object (the server builds the LFM source for `--ask lfm` and
    passes it, order 0031); `lexicon` adds lever 5; `scene` a
    `cubbyllm.spatial.Scene`, the source for a question whose relation or seed is the scene's (`spatial.route`)."""

    def __init__(self, emitter, world=None, source="wikidata", lexicon: bool = True, exe: str | None = None,
                 max_new: int = 300, run_fn=None, tau_profile: float = TAU_VM[1],
                 news=None, history_path: str | None = None, web=None, skills=None, scene=None, books=None,
                 tense: bool | None = None, loci=None, time_rule: bool | None = None, news_changes=None,
                 everyday: bool | None = None):
        from cubbyllm.bridges import cubelang_client as cc
        from cubbyllm.reasoning.plan_verify import StoreRelations
        self.emitter = emitter
        if world is None:
            from ..worlds import wikikg as wk
            wk.ensure_data(("triplets",))
            world = wk.wiki_world()
        self.world = world
        self.known = StoreRelations(world.index._seen)
        if hasattr(world, "known_relations"):           # several worlds (hdc_store.Stores): theirs too, by count
            world.known_relations(self.known)
        if isinstance(source, str):
            if source.startswith("wikidata"):
                from ..sources.wikidata import WikidataSource
                source = WikidataSource(offline=source.endswith("offline"), use_choices=True)   # serving learns what askers mean
            else:
                raise ValueError(f"unknown source {source!r}")
        self.source = source
        # The date source. A `when` ask is the one question the wiki world provably cannot answer
        # (measured: 0 facts whose subject is a year), so it gets its own source -- and only a
        # `when` ask ever consults it. `news="rss"` builds the default feed set.
        if news == "rss":
            from ..sources.news.source import NewsSource
            news = NewsSource()
        self.news = news
        # the web (2026-09-24): asked only after the source above has nothing more for the entity the walk
        # stalled on, and what it finds is held until two independent sites agree (standin/web_source.py)
        self.web = web
        self.fallbacks = [web] if web is not None else []
        # the books (order 0012): asked before the source -- local, free -- through the same gate; a fact one
        # author alone states is spoken hedged until Wikidata or another author states it too (an index path,
        # or a BookSource; None = off, the loop unchanged)
        if isinstance(books, (str, pathlib.PurePath)):
            from ..worlds.book_source import BookSource
            books = BookSource(books)
        self.books = books
        self.contested: set[tuple[str, str]] = set(getattr(world, "contested", None) or ())   # held: `load_held`, and
                                                         # the pairs two mounted worlds disagree on (0015, addendum 3)
        if books is not None and hasattr(books, "life") and books.life is None:
            books.life = self._life                      # the same-name guard reads the subject's life from this world
        # the question's tense and year against the answer's dates (order 0012: "Who rules England?" on a reign
        # that ended is not answered); on unless `tense` says otherwise, the books or not (Nick, 2026-10-24: on always)
        self.tense = True if tense is None else bool(tense)
        # the time rule on a world's own facts (order 0015, Addendum 2): on with the tense check unless told otherwise
        self.time_rule = self.tense if time_rule is None else bool(time_rule)
        # the 450M's loci path as the reader (order 0007): a checkpoint path, or a LociReader; None = today's loop
        if isinstance(loci, (str, pathlib.PurePath)):
            from .loci_reader import LociReader
            loci = LociReader(str(loci))
        self.loci = loci
        # order 0024: the news archive's change marks (news_trigger.Changes); a value said "as of" a date that reports
        # since say changed carries that too. None = off, the reply as before.
        self.news_changes = news_changes
        from ..speech.audience import LeakGuard                   # sources are internal (order 0016): the guard at the mouth,
        self.guard = LeakGuard(index=getattr(books, "index", None))   # the books' passages for its 8-word scan
        self._forms = None
        if books is not None:                            # natural questions (order 0012): a parsed form is a relation
            from .question_forms import Forms             # ask, never a profile ("Who was the founder of Rome?"), and
            self._forms = Forms()                        # its words name the relation (levers 4 and 6: "ruled")
        self._everyday = None                            # everyday wording (2026-10-24): the host restates "Who wrote X?"
        if os.environ.get("CB_EVERYDAY") != "0" if everyday is None else everyday:    # as the author of X; on unless
            from .everyday import EverydayForms          # CB_EVERYDAY=0 (Nick, 2026-10-24); the model gets every
            self._everyday = EverydayForms()             # question the table does not read, as before
        # the scene (2026-10-17): the host's route to space. A question the scene can state -- a spatial
        # relation at any hop, or a seed that is a thing in it -- is sourced from the scene's exact reading
        # instead of the world's source; the walk and the VM verify what it states like any other fact.
        self.scene = scene
        if web is not None and getattr(web, "wikidata", None) is None and hasattr(source, "resolve"):
            web.wikidata = source                        # a Wikipedia hit hands over to Wikidata's own claims
        if web is not None and getattr(web, "containment", None) is None and hasattr(source, "_claims"):
            web.containment = source                     # which places contain which: Warsaw vouches for Poland
        if web is not None and getattr(web, "_aliases", None) is None and getattr(source, "aliases", None) is not None:
            web._aliases = source.aliases                # one property table for both
        self.resolvers = []
        if lexicon:
            from cubbyllm.reasoning.lexicon import Lexicon
            self.resolvers.append(Lexicon())
        if self._forms is not None:                      # first: the first resolver to name a wording decides, and the
            self.resolvers.insert(0, self._forms)        # lexicon's synonyms named 'die' 'dice' and 'born' 'max born'
                                                         # (2026-10-22, the books bench: 67 of 76 unknown_relation)
        self._vm = run_fn                                # tests: a stand-in VM; otherwise the resident CubeLang session
        self.tau_profile = float(tau_profile)            # a profile line is spoken only when the VM recovers it at this similarity
        self.session = None if run_fn else cc.CubelangSession(exe=exe)
        self.max_new = int(max_new)
        self.calls = collections.Counter()
        self.history: list[dict] = []
        # every record, one JSON line each, for the night's sleep cycle (cubbyllm/reasoning/sleep.py):
        # the day's refusals are its curriculum and the day's accepted facts what it makes durable
        self.history_path = pathlib.Path(history_path) if history_path else None
        # the skill library (cubbyllm/reasoning/skills.py): composition rules the sleep cycle adopted, read
        # from its ledger; empty without one, and then a relation ask is answered only by a direct fact
        from cubbyllm.reasoning.skills import Library
        self.skills = skills if isinstance(skills, Library) else Library(skills)
        self._lock = threading.Lock()                    # one question at a time: the emitter and the VM session are not re-entrant


    def _run_fn(self, source: str, fn: str) -> dict:
        self.calls["vm"] += 1
        return self._vm(source, fn) if self._vm else self.session.run(source, fn=fn)


    def ask(self, question: str, item: str | None = None, asker: str | None = None, choose=None) -> dict:
        """One question through the loop; the record it returns (and keeps) carries the run's thought
        graph (`cubbyllm.reasoning.graph`), built from the events the ask emitted."""
        from cubbyllm.reasoning.graph import Collector
        with Collector() as col:
            self._graph = (col, " ".join(question.split()))
            try:
                return self._ask(question, item=item, asker=asker, choose=choose)
            finally:
                self._graph = None

    def _ask(self, question: str, item: str | None = None, asker: str | None = None, choose=None) -> dict:
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import learn_and_answer
        from cubbyllm.reasoning.planner import QuestionPlan, normalize
        from ..emit.base import strip_fences
        from cubbyllm.reasoning.emitted import emitted_plan
        t0 = time.perf_counter()
        question = " ".join(question.split())
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": None, "plan": None, "seed": None,
                     "learned": [], "entities": [], "aliased": [], "snapped": None, "trace": [], "wall_s": 0.0}
        with self._lock:
            self.calls["asked"] += 1
            sa = self_ask(question)                      # about the LOOP, not about the world
            if sa:
                rec = self.introspect(question, sa[0], sa[1], t0, how="the question's shape (about you)")
                self._keep(rec)
                return rec
            ra = relation_ask(question)                  # "how is B related to A?": composed, not looked up
            if ra:
                rec = self.relation(question, ra[0], ra[1], t0)
                self._keep(rec)
                return rec
            # a mounted scene's own shapes, read off the question and restated in its grammar (order 0035)
            sa = scene_ask(question, self.scene) if self.scene is not None else None
            ed = None if sa or self._everyday is None else self._everyday.restate(question)   # (restated, relation, seed)
            pa = None if sa or ed or (self._forms is not None and self._forms.parse(question)) else \
                profile_ask(question, self._is_relation)   # the host reads the shape off the question (gen 2 cannot emit it yet)
            if pa:
                rec = self.profile(question, pa[0], pa[1], t0, how="the question's shape (who / what / where / when)", item=item, asker=asker)
                self._keep(rec)
                return rec
            asked, _year = strip_as_of(question)         # the year is a constraint, not a relation
            try:
                raw = "no plan\n" if sa or ed else self.emitter.emit(asked, max_new_tokens=self.max_new)
                if program_world(strip_fences(raw)) == "self":
                    pp = profile_program(strip_fences(raw))
                    if pp:
                        rec = self.introspect(question, pp[0], pp[1], t0, how='the emitter\'s program (WORLD "self")')
                        self._keep(rec)
                        return rec
                program = strip_fences(raw)
                pp = profile_program(program)            # the emitter wrote the retrieval program: SEED + ASK, no hop
                if pp:
                    rec = self.profile(question, pp[0], pp[1], t0, how="the emitter's program (ASK role)", item=item, asker=asker)
                    self._keep(rec)
                    return rec
                ep = emitted_plan(program, normalize)
            except Exception as e:                       # noqa: BLE001 -- the proposer failed; the loop has nothing to dispose
                ep = None; rec["reason"] = f"emit_error: {str(e)[:120]}"
            if sa:
                ep, rec["scene_shape"] = ([sa[1]], sa[2]), sa[0]
            if ed:
                ep, rec["restated"] = ([ed[1]], ed[2]), ed[0]
            if not ep or not ep[0] or not ep[1]:
                rec["reason"] = rec["reason"] or "no_plan"
                qid = ev.emit("question", text=question, source=getattr(self.source, "name", None))
                ev.emit("answer", qid, answer=None, verified=False, reason=rec["reason"])
                rec["wall_s"] = round(time.perf_counter() - t0, 3); self._keep(rec); return rec
            rels, seed = ep
            one_item = seed_for(self, seed, item, question=question, rels=rels)  # a shared name (orders 0033, 0037)
            plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))
            rec["plan"], rec["seed"] = rels, seed
            source = self.source
            if self.scene is not None:
                from cubbyllm.spatial.route import route
                if route(plan, self.scene):
                    source = self.scene
                    rec["world"] = "scene"
            def learn(src=None):                         # `src`: another source for this walk (the reader's: none)
                return learn_and_answer(sa[0] if sa else ed[0] if ed else question, lambda q, k: [], self._run_fn, store=self.world, known=self.known,
                                        source=src or source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                        plan=plan, resolvers=self.resolvers, choose=parse_choose(choose) or None,
                                        fallbacks=self.fallbacks if src is None else [],
                                        before=[self.books] if self.books is not None and src is None else None)
            if self.loci is not None:                    # the trunk reads first; a miss is looked up once (order 0007)
                from .loci_reader import through_loci
                if through_loci(self, rec, rels, seed, learn) is not None:
                    rec["wall_s"] = round(time.perf_counter() - t0, 3)
                    self._keep(rec)
                    return rec
            with one_item:
                lr = learn()

        def again():                                     # the same walk once more, under the same pin or cue: a snapshot's
            with self._lock, seed_for(self, seed, item, question=question, rels=rels):    # changing value, the source
                return learn()                           # asked for today's (ask_source_first; Nick, 2026-10-27)
        rec.update(answer=lr.result.answer if lr.result.verified else None, verified=bool(lr.result.verified),
                   reason=None if lr.result.verified else lr.result.reason,
                   learned=[_learned(p) for p in lr.learned],
                   entities=list(lr.entities), aliased=[list(a) for a in lr.aliased], snapped=list(lr.snapped) if lr.snapped else None,
                   trace=[h.fact for h in lr.result.trace if h.fact], refused=lr.result.refused,
                   branches=lr.result.branches, chose=choose or None,
                   clarify=branch_clarify(lr.plan or plan, lr.result, parse_choose(choose)),
                   wall_s=round(time.perf_counter() - t0, 3))
        after_walk(self, rec, rec["seed"] or "", learn=learn)  # which item a shared name is: decided, or the asker is asked
        self._checked(question, lr, rec, again=again)    # the veto, the tense, "as of" (the source first), the time rule, the hedge
        also(rec)                                        # then the other namesakes, after the reply is what it will be
        if self.books is not None:                        # the record's evidence: internal, the audit view's (0016)
            rec["evidence"] = {f: self.books.citations[normalize(f)] for f in rec["trace"] if normalize(f) in self.books.citations}
        self._keep(rec)
        return rec


    FAMILY_FETCH_MAX = 40                                # source calls one relation ask may spend


