"""The verified-program loop on ONE natural question, live -- what exp_r11 runs per row, as a service.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this). serve_api mounts it behind
`POST /ask` when started with `--ask`, and every step it takes is an event on `/loop/stream`,
so the control panel (dashboard/control_panel.html, served at /panel) draws the tree as it grows:
question -> plan (the emitter's proposal, the host's aliases) -> walk -> hops -> facts; the fetch
from the source and the gate's verdict on each fact; the answer or the refusal.

The seven invariants hold here exactly as in the benches: the emitter proposes a plan, the host
disposes (covers, lever 4/6/7), the VM verifies, the source's facts pass the gate with provenance,
and the answer is spoken only when a VM-verified chain reaches it -- never from the model.
"""
from __future__ import annotations

import collections
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}       # exp_r11's thresholds, by hop count


class AskLoop:
    """`ask(question)` -> the loop's record. `emitter` is any object with `.emit(prompt, max_new_tokens=)`
    (the serving emitter, shared); `world` a FactStore with a TripleIndex (the wiki world when None);
    `source` 'wikidata' | 'wikidata-offline' | 'lfm' | a Source object; `lexicon` adds lever 5."""

    def __init__(self, emitter, world=None, source="wikidata", lexicon: bool = True, exe: str | None = None,
                 max_new: int = 300, lfm_gguf: str | None = None, run_fn=None):
        from cubbyllm.bridges import cubelang_client as cc
        from cubbyllm.reasoning.plan_verify import StoreRelations
        self.emitter = emitter
        if world is None:
            import wikikg as wk
            wk.ensure_data(("triplets",))
            world = wk.wiki_world()
        self.world = world
        self.known = StoreRelations(world.index._seen)
        if isinstance(source, str):
            if source.startswith("wikidata"):
                from sources import WikidataSource
                source = WikidataSource(offline=source.endswith("offline"))
            elif source == "lfm":
                from lfm_source import LfmSource
                source = LfmSource(lfm_gguf or str(ROOT / "standin" / "models" / "LFM2.5-2.6B.Q4_K_M.gguf"))
            else:
                raise ValueError(f"unknown source {source!r}")
        self.source = source
        self.resolvers = []
        if lexicon:
            from cubbyllm.reasoning.lexicon import Lexicon
            self.resolvers.append(Lexicon())
        self._vm = run_fn                                # tests: a stand-in VM; otherwise the resident CubeLang session
        self.session = None if run_fn else cc.CubelangSession(exe=exe)
        self.max_new = int(max_new)
        self.calls = collections.Counter()
        self.history: list[dict] = []
        self._lock = threading.Lock()                    # one question at a time: the emitter and the VM session are not re-entrant

    def _run_fn(self, source: str, fn: str) -> dict:
        self.calls["vm"] += 1
        return self._vm(source, fn) if self._vm else self.session.run(source, fn=fn)

    def ask(self, question: str) -> dict:
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import learn_and_answer
        from cubbyllm.reasoning.planner import QuestionPlan, normalize
        from eval_emitter_vm import strip_fences
        from exp_r9_matched_pairs import emitted_plan
        t0 = time.perf_counter()
        question = " ".join(question.split())
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": None, "plan": None, "seed": None,
                     "learned": [], "entities": [], "aliased": [], "snapped": None, "trace": [], "wall_s": 0.0}
        with self._lock:
            self.calls["asked"] += 1
            try:
                raw = self.emitter.emit(question, max_new_tokens=self.max_new)
                ep = emitted_plan(strip_fences(raw), normalize)
            except Exception as e:                       # noqa: BLE001 -- the proposer failed; the loop has nothing to dispose
                ep = None; rec["reason"] = f"emit_error: {str(e)[:120]}"
            if not ep or not ep[0] or not ep[1]:
                rec["reason"] = rec["reason"] or "no_plan"
                qid = ev.emit("question", text=question, source=getattr(self.source, "name", None))
                ev.emit("answer", qid, answer=None, verified=False, reason=rec["reason"])
                rec["wall_s"] = round(time.perf_counter() - t0, 3); self.history.append(rec); return rec
            rels, seed = ep
            plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))
            rec["plan"], rec["seed"] = rels, seed
            lr = learn_and_answer(question, lambda q, k: [], self._run_fn, store=self.world, known=self.known,
                                  source=self.source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                  plan=plan, resolvers=self.resolvers)
        rec.update(answer=lr.result.answer if lr.result.verified else None, verified=bool(lr.result.verified),
                   reason=None if lr.result.verified else lr.result.reason,
                   learned=[{"fact": p.fact, "status": p.status, "source": p.source, "entity": p.entity} for p in lr.learned],
                   entities=list(lr.entities), aliased=[list(a) for a in lr.aliased], snapped=list(lr.snapped) if lr.snapped else None,
                   trace=[h.fact for h in lr.result.trace if h.fact], refused=lr.result.refused,
                   wall_s=round(time.perf_counter() - t0, 3))
        self.history.append(rec)
        return rec
