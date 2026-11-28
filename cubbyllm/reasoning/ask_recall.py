"""cubbyllm/reasoning/ask_recall.py -- order 0083: a checked plan remembered before the 450M writes one.

With ``CB_ASK_MEMORY=1`` (off by default; off, the ask loop is as before), the ask loop asks the hippocampus
(``hippocampus.py``: the episodes ``sleep.replay`` writes) for plan candidates after the host's own shapes (self,
relation, scene, everyday, profile) and before the emitter, when the striatum (``striatum.py``) orders the hippocampus
first for the question's shape. A candidate takes the walk an emitted plan takes (``_walk_setup``, split out of
``_ask`` verbatim): covers(), levers 4 and 6, the walk, the VM. It walks over the store as it stands and fetches
nothing (``StoreOnly``): a remembered plan answers from what the store holds, and a miss costs a recall, not a source
call. The first certified candidate is spoken, with ``how = "a remembered plan"`` and the episode's id (the sleep
ledger's); if none is, the emitter runs as before. Each proposer's try goes to the striatum under its reward rule (+1
certified, 0 refused). ``CB_ASK_MEMORY_WRITE=1`` (off by default) also writes a certified answer as an episode at
once, as ``sleep.replay`` would write it at night; ``sleep`` still retires an episode whose chain lost a fact.

Memory proposes; the gate disposes. Nothing here changes plan_verify, the walk or the VM.

Wired: WIRED (AskLoop, a mixin; ``cubby_serve.cli`` mounts the memory, ``--memory <folder>``).
"""
from __future__ import annotations

import os
import pathlib
import threading
import time

__wiring__ = "WIRED"

PROPOSERS = ("hippocampus", "emitter")
COST_S = {"hippocampus": 0.05, "emitter": 1.0}       # seconds a proposal, exp_r15_striatum's: the effort-aware order
K = 3                                                # candidates the hippocampus proposes (recalled, then rebound)
_LOCK = threading.Lock()                             # the striatum and the live write, after the ask's own lock
_CALL = threading.local()                            # this thread's ask: the memory step's record, for _remember


def on() -> bool:
    """``CB_ASK_MEMORY=1``: the hippocampus is asked before the emitter. Off by default."""
    return os.environ.get("CB_ASK_MEMORY") == "1"


def write_on() -> bool:
    """``CB_ASK_MEMORY_WRITE=1``: a certified answer becomes an episode at once. Off by default."""
    return os.environ.get("CB_ASK_MEMORY_WRITE") == "1"


def episode_id(e) -> str:
    """The sleep ledger's id for an episode (``sleep.replay``): its question and chain, hashed."""
    from .planner import normalize
    from .sleep import _sha
    return _sha(normalize(e.question), *e.chain)


def mount(loop, folder: str | None = None) -> str:
    """The hippocampus and the striatum on `loop`, from `folder` (``hippocampus.jsonl`` and ``striatum.json``, the files
    ``sleep.py`` writes) when it holds them, else empty. Returns the boot line."""
    from .hippocampus import Hippocampus
    from .striatum import Striatum
    p = pathlib.Path(folder) if folder else None
    hp, sp = (p / "hippocampus.jsonl", p / "striatum.json") if p else (None, None)
    loop.hippocampus = Hippocampus.load(hp) if hp and hp.exists() else Hippocampus()
    loop.striatum = Striatum.load(sp) if sp and sp.exists() else Striatum()
    where = f"from {p.name}" if p else "empty (no --memory folder)"
    return (f"memory first (CB_ASK_MEMORY=1): {len(loop.hippocampus)} live episodes, {len(loop.striatum.expected)} "
            f"striatum values, {where}; live write {'on' if write_on() else 'off'}")


class StoreOnly:
    """A source that fetches nothing: a remembered plan walks over the store as it stands. Everything else (the property
    table levers 4 and 6 read, the names) is the wrapped source's own."""

    def __init__(self, source) -> None:
        self._source = source
        self.last: dict = {}

    def __getattr__(self, name):
        return getattr(self._source, name)

    def facts(self, entity, relations=None, via=None):
        return []


def answered(rec: dict) -> bool:
    """Spoken (sleep's rule): a VM-verified chain, or a profile / introspection answer."""
    reason = (rec.get("reason") or "").split(":", 1)[0].strip()
    return bool(rec.get("verified")) or (reason in ("profile", "self") and bool(rec.get("answer")))


class AskRecall:
    """AskLoop's memory-first step (a mixin; order 0083). ``mount`` sets both; None, and memory is off whatever the flag."""

    hippocampus = None
    striatum = None

    def _walk_setup(self, question: str, walked: str, rels: list, seed: str, item, choose, rec: dict):
        """A plan (relations, seed) -> (plan, source, learn, one_item): the walk ``_ask`` runs for an emitted plan, split
        out of it verbatim (order 0083) so a remembered plan takes the same one. `walked`: the question the walk is
        asked (the scene's or the everyday table's restatement, else the question)."""
        from cubbyllm.reasoning.learn import learn_and_answer
        from cubbyllm.reasoning.planner import QuestionPlan
        from .ask_claims import parse_choose
        from .ask_forms import TAU_VM
        from .ask_namesakes import seed_for
        one_item = seed_for(self, seed, item, question=question, rels=rels)  # a shared name (orders 0033, 0037)
        plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))
        rec["plan"], rec["seed"] = rels, seed
        source = self.source
        if hasattr(source, "resolve"):                 # the name with the article the asker wrote (order 0047)
            from .heldout import asked_span, on as heldout_on
            from ..sources.wikidata_resolve import name_key
            asked = asked_span(question, seed) if heldout_on() else None
            source.asked_names = {name_key(seed): asked} if asked else {}
        if self.scene is not None:
            from cubbyllm.spatial.route import route
            if route(plan, self.scene):
                source = self.scene
                rec["world"] = "scene"
        from . import cover_names
        names = cover_names.slots(self, question) if cover_names.on() else None   # order 0087's slot table
        def learn(src=None, store=None):             # `src`: another source for this walk (the reader's: none);
            return learn_and_answer(walked, lambda q, k: [], self._run_fn,   # `store`: a scratch
                                    store=self.world if store is None else store, known=self.known,   # first world (a namesake apart, 0043;
                                    # an empty store is falsy)
                                    source=src or source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                    plan=plan, resolvers=self.resolvers, choose=parse_choose(choose) or None,
                                    fallbacks=self.fallbacks if src is None else [],
                                    before=[self.books] if self.books is not None and src is None else None,
                                    names=names)
        return plan, source, learn, one_item

    def _recall(self, question: str, asked: str, item, choose, rec: dict) -> dict | None:
        """The remembered plans, before the emitter, when the striatum orders the hippocampus first for this shape: the
        first that certifies -> {rels, seed, plan, source, learn, lr}; None (the emitter runs) when memory is off, the
        striatum orders the emitter first, or no candidate certifies. Off: None, and `rec` untouched."""
        if not on() or self.hippocampus is None or self.striatum is None:
            return None
        t = time.perf_counter()
        with _LOCK:
            order = self.striatum.order(question, list(PROPOSERS), costs=COST_S)
            cands = self.hippocampus.propose(asked, k=K) if order[0] == "hippocampus" else []
        mem = rec["memory"] = _CALL.memory = {"order": order, "tried": []}
        if order[0] != "hippocampus":
            return None
        before = {k: rec.get(k) for k in ("plan", "seed")}
        got = None
        for plan_c, ep, kind in cands:
            rels, seed = list(ep.relations), plan_c.tail[len(ep.relations[0]) + 4:]
            plan, source, learn, one_item = self._walk_setup(question, question, rels, seed, item, choose, rec)
            with one_item:
                lr = learn(src=StoreOnly(source))
            ok = bool(lr.result.verified)
            mem["tried"].append({"episode": episode_id(ep), "kind": kind, "plan": rels, "seed": seed,
                                 "reason": None if ok else lr.result.reason})
            if ok:
                got = {"rels": rels, "seed": seed, "plan": plan, "source": source, "learn": learn, "lr": lr}
                rec.update(how="a remembered plan", episode=episode_id(ep))
                with _LOCK:
                    self.hippocampus.reinforce(ep)
                break
            rec.pop("world", None)
        if got is None:
            rec.update(before)                           # the emitter's walk writes its own
        with _LOCK:
            self.striatum.reward(question, "hippocampus", "certified" if got else "refused")
        mem["s"] = round(time.perf_counter() - t, 3)
        return got

    def _remember(self, question: str, rec: dict | None) -> None:
        """After an ask, memory on: when the ask reached the memory step and the emitter ran, the emitter's try to the
        striatum; with ``CB_ASK_MEMORY_WRITE=1``, any certified chain (the host's shapes' too, as at night) written as the
        episode ``sleep.replay`` would write for this record. `rec`: the ask's record (None when it raised). Off: nothing."""
        mem, _CALL.memory = getattr(_CALL, "memory", None), None
        if rec is None or not on() or self.hippocampus is None:
            return
        with _LOCK:
            if mem is not None and rec.get("how") != "a remembered plan":
                self.striatum.reward(question, "emitter", "certified" if answered(rec) else "refused")
            if write_on():
                write_episode(self.hippocampus, rec)


def write_episode(hip, rec: dict):
    """The episode ``sleep.replay`` writes for an ask record (a certified chain with its plan, seed and answer; not a
    relation ask), unless the hippocampus already holds this question with this chain. Returns it, or None."""
    from .planner import normalize
    chain = list(rec.get("trace") or [])
    if not (rec.get("verified") and chain and rec.get("plan") and rec.get("seed") and rec.get("answer")):
        return None
    if rec.get("kind") == "relation":
        return None
    keys = getattr(hip, "_keys", None)               # (question, chain) of every episode, built once
    if keys is None:
        keys = hip._keys = {(normalize(e.question), tuple(e.chain)) for e in hip.episodes}
    key = (normalize(rec["question"]), tuple(chain))
    if key in keys:
        return None
    keys.add(key)
    return hip.write(rec["question"], rec["plan"], rec["seed"], chain, rec["answer"],
                     provenance={"live": True, "record": rec.get("id"), "run": "live", "ts": rec.get("ts")})
