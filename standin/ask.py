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

# "who is X?" / "what is X?" / "where is X?" -- no relation asked: a PROFILE of the entity
# (2026-09-14, Nick: "seems like it does not get who is" -- the emitter had planned a relation named
# 'who is'; then: "the model needs to learn to retrieve: if it's a question about who is an entity,
# it needs to call a program that will retrieve the details about the entity and paraphrase it";
# "who, what, where"). The shape is a PROGRAM the emitter can learn to emit -- `bind frame, SEED,
# "x"; bind frame, ASK, "who";` -- and the host executes it: the facts the store holds about X under
# the ask kind's relations, fetched through the gate first when the store has none, each line with its
# provenance; then the talk adapter paraphrases those lines and the host keeps the paraphrase only
# when every name and number in it is in the facts. Until gen 3 emits the program, the host reads the
# shape off the question itself (the same regex builds gen 3's records). Not a walk, not VM-verified --
# a readout of the store, and the record says so (`reason: profile`).
import re as _re
_PROFILE_ASK = _re.compile(r"^\s*(?P<kind>who|what|where)\s+(?:is|was|are|were)\s+(?:the\s+)?(?P<ent>[^?]*?)\s*\??\s*$", _re.I)
_ASK_ROLE = _re.compile(r'bind\s+frame\s*,\s*ASK\s*,\s*"(?P<kind>who|what|where)"\s*;', _re.I)
_SEED_ROLE = _re.compile(r'bind\s+frame\s*,\s*SEED\s*,\s*"(?P<seed>(?:[^"\\]|\\.)*)"\s*;')
_HOP_ROLE = _re.compile(r'bind\s+frame\s*,\s*(?:HOP\d+|H\d+_\w+)\s*,')
PROFILE_KINDS = {
    "who": ("occupation", "country of citizenship", "date of birth", "place of birth", "date of death", "place of death",
            "position held", "notable work", "field of work", "award received", "educated at", "employer", "member of",
            "member of political party", "spouse", "father", "mother", "sex or gender", "instance of"),
    "what": ("instance of", "subclass of", "part of", "developer", "creator", "author", "manufacturer", "genre",
             "inception", "publication date", "country", "country of origin", "founded by", "headquarters location",
             "official language", "currency", "capital", "population", "occupation", "date of birth"),
    "where": ("instance of", "country", "located in the administrative territorial entity", "capital of", "continent",
              "part of", "capital", "population", "located in or next to body of water", "headquarters location",
              "located on terrain feature", "official language"),
}
PROFILE = tuple(dict.fromkeys(r for k in ("who", "what", "where") for r in PROFILE_KINDS[k]))
RESOLVE_HINT = {k: tuple(r for r in v if r not in ("instance of", "subclass of", "part of", "sex or gender"))[:8] for k, v in PROFILE_KINDS.items()}
PROFILE_MAX_PER_RELATION = 3
PROFILE_MAX_LINES = 9
PARAPHRASE_SYSTEM = "You are a careful writer. You restate given facts in plain sentences and never add anything."


def profile_ask(question: str) -> tuple[str, str] | None:
    """(kind, entity) of a 'who / what / where is X' question, or None: a question with a relation in
    it ('the capital of France', "Canada's capital") is a plan for the emitter, not a profile."""
    m = _PROFILE_ASK.match(question)
    if not m:
        return None
    ent = m.group("ent").strip()
    if not ent or " of " in f" {ent} " or "'s " in ent or ent.endswith("'s") or len(ent.split()) > 6:
        return None
    return m.group("kind").lower(), ent


def profile_program(program: str) -> tuple[str, str] | None:
    """(kind, seed) when an emitted program is the retrieval shape -- SEED and ASK bound, no hop."""
    a, s = _ASK_ROLE.search(program or ""), _SEED_ROLE.search(program or "")
    if not a or not s or _HOP_ROLE.search(program or ""):
        return None
    return a.group("kind").lower(), s.group("seed").replace('\\"', '"')


def cot_profile(seed: str, kind: str) -> str:
    """The retrieval program, in the emitter's CotPlan form: what gen 3 learns to emit for this shape."""
    esc = seed.replace("\\", "\\\\").replace('"', '\\"')
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n"
            "        create frame: number;\n"
            f'        bind frame, SEED, "{esc}";\n        bind frame, ASK, "{kind}";\n'
            "        return recover(frame, SEED);\n    }\n}\n")


_TOKEN = _re.compile(r"[A-Za-zÀ-ÿ][\w'’\-]*|\d[\d.,:/\-]*")
_SMALL = frozenset("the a an and or of in on at to for by with from as is was were are be been being it its this that "
                   "he she they his her their who whom which what where when also known born died one two three four five".split())


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")
_ISO = _re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def date_words(facts: list[str]) -> list[str]:
    """The month name, day and year of every ISO date in the facts: '2015-12-11' is also 'December 11,
    2015' (2026-09-14, live: the grounding rule refused an exact paraphrase of OpenAI's inception over
    the word 'December'). A date the store holds, spelled the way people spell it, is not an addition."""
    out: list[str] = []
    for f in facts:
        for y, m, d in _ISO.findall(f):
            i = int(m)
            if 1 <= i <= 12:
                out.append(_MONTHS[i - 1])
            out.append(y)
            if d:
                out.append(str(int(d)))
    return out


def grounded_prose(text: str, facts: list[str], entity: str) -> tuple[bool, list[str]]:
    """Every capitalised word and every number in the paraphrase must occur in the facts (or the entity's
    name, or a date the facts hold written out): the talk adapter says what the store says, in its words,
    and nothing it adds is spoken. -> (ok, the offending tokens)."""
    hay = " ".join(facts + [entity] + date_words(facts)).lower()
    bad: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower().strip("'’-.,:")
        if not low or low in _SMALL:
            continue
        is_name = tok[0].isupper() and low not in _SMALL
        is_num = tok[0].isdigit()
        if (is_name or is_num) and low not in hay:
            bad.append(tok.strip("'’-.,:"))
    return (not bad), bad


class AskLoop:
    """`ask(question)` -> the loop's record. `emitter` is any object with `.emit(prompt, max_new_tokens=)`
    (the serving emitter, shared); `world` a FactStore with a TripleIndex (the wiki world when None);
    `source` 'wikidata' | 'wikidata-offline' | 'lfm' | a Source object; `lexicon` adds lever 5."""

    def __init__(self, emitter, world=None, source="wikidata", lexicon: bool = True, exe: str | None = None,
                 max_new: int = 300, lfm_gguf: str | None = None, run_fn=None, tau_profile: float = TAU_VM[1]):
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
        self.tau_profile = float(tau_profile)            # a profile line is spoken only when the VM recovers it at this similarity
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
            pa = profile_ask(question)                   # the host reads the shape off the question (gen 2 cannot emit it yet)
            if pa:
                rec = self.profile(question, pa[0], pa[1], t0, how="the question's shape (who / what / where is X)")
                self.history.append(rec)
                return rec
            try:
                raw = self.emitter.emit(question, max_new_tokens=self.max_new)
                program = strip_fences(raw)
                pp = profile_program(program)            # the emitter wrote the retrieval program: SEED + ASK, no hop
                if pp:
                    rec = self.profile(question, pp[0], pp[1], t0, how="the emitter's program (ASK role)")
                    self.history.append(rec)
                    return rec
                ep = emitted_plan(program, normalize)
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

    # -- the profile ask ---------------------------------------------------------------------
    def facts_about(self, entity: str) -> list:
        """The Triples the store holds with `entity` as subject (the index's subject lookup)."""
        from cubbyllm.reasoning.planner import normalize
        index = getattr(self.world, "index", None)
        by = getattr(index, "_by_subj", None)
        if by is None:
            return []
        return [t for _f, t in by.get(normalize(entity), []) if t is not None]


    def _fetch(self, qid: int, entity: str, hint: list[str], rec: dict) -> dict:
        """The source asked about the entity (the ask kind's relations as the resolution hint), every
        fact through the gate, admitted ones into the store with provenance; the fetch and gate events."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import gate, _takes_relations
        from cubbyllm.reasoning.planner import Triple
        prov = getattr(self.world, "provenance", None)
        rec["entities"].append(entity)
        items = list(self.source.facts(entity, relations=hint) if _takes_relations(self.source) else self.source.facts(entity))
        last = getattr(self.source, "last", None) if isinstance(getattr(self.source, "last", None), dict) else {}
        fid = ev.emit("fetch", qid, source=getattr(self.source, "name", None), entity=entity, n=len(items), latent=False,
                      needs=hint, how=last.get("how"), item=last.get("qid"),
                      ambiguous=[list(a) for a in last.get("ambiguous", [])] or None)
        for item in items:
            fact = f"{item.obj} is the {item.rel} of {item.subj}" if isinstance(item, Triple) else item
            if isinstance(item, Triple):
                if hasattr(self.world, "index") and hasattr(self.world.index, "declare_relation"):
                    self.world.index.declare_relation(item.rel)
                if hasattr(self.known, "declare"):
                    self.known.declare(item.rel)
            p = gate(fact, self.world, self.source.name, entity)
            rec["learned"].append({"fact": p.fact, "status": p.status, "source": p.source, "entity": p.entity})
            if p.status == "accepted":
                self.world.add(fact)
                if hasattr(self.known, "add"):
                    self.known.add(fact)
                if prov is not None:
                    prov[" ".join(fact.split())] = self.source.name
            ev.emit("gate", fid, fact=fact, status=p.status, clash=p.clash, lifted=False,
                    provenance=(prov or {}).get(" ".join(fact.split())))
        return last

    def profile(self, question: str, kind: str, entity: str, t0: float, how: str) -> dict:
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
                     "snapped": None, "trace": [], "profile": [], "prose": None, "candidates": None, "wall_s": 0.0}
        prov = getattr(self.world, "provenance", None)
        rels = PROFILE_KINDS[kind]
        have = self.facts_about(entity)
        last: dict = {}
        if not any(normalize(t.rel) in PROFILE for t in have):
            last = self._fetch(qid, entity, list(RESOLVE_HINT[kind]), rec)
            have = self.facts_about(entity)
        # the lines: the ask kind's relations first, then any other profile relation the store holds
        chosen: list[tuple[str, str, str]] = []                       # (rel, obj, the stored fact text)
        by_key = {(normalize(t.rel), t.obj): " ".join(f.split()) for f, t in self.world.index._by_subj.get(normalize(entity), []) if t is not None}
        for rel in tuple(rels) + tuple(r for r in PROFILE if r not in rels):
            objs = list(dict.fromkeys(t.obj for t in have if normalize(t.rel) == rel))
            for obj in objs[:PROFILE_MAX_PER_RELATION]:
                chosen.append((rel, obj, by_key.get((rel, obj), f"{obj} is the {rel} of {entity}")))
            if len(chosen) >= PROFILE_MAX_LINES:
                break
        chosen = chosen[:PROFILE_MAX_LINES]
        if not chosen:
            amb = last.get("ambiguous") or []
            if amb:
                rec["reason"] = "ambiguous_entity"; rec["candidates"] = [[a[1], a[2]] for a in amb]
            else:
                rec["reason"] = "entity_unresolved" if rec["entities"] and not rec["learned"] else "profile_empty"
            ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], candidates=rec["candidates"],
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
                kept.append({"relation": rel, "value": obj, "fact": fact, "provenance": src_name, "similarity": sim})
        ev.emit("vm", wid, verified=bool(kept) and len(kept) == len(chosen), program_lines=len(chosen) + 2,
                recovered=len(kept), bound=len(chosen))
        rec["profile"] = kept; rec["trace"] = [k["fact"] for k in kept]
        if kept:
            by_rel: dict[str, list[str]] = {}
            for l in kept:
                by_rel.setdefault(l["relation"], []).append(l["value"])
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
        ask = {"who": "who", "what": "what", "where": "where"}[kind]
        prompt = (f"Facts about {entity}:\n" + "\n".join(f"- {f}" for f in facts) +
                  f"\n\nIn one or two plain sentences, say {ask} {entity} is, using ONLY these facts. "
                  "Do not add any name, date, number or detail that is not in the facts.")
        try:
            from emitter import clean_reply
            raw = self.emitter.emit(prompt, max_new_tokens=160, context={"role": "talk"}, temperature=0.0,
                                    system=PARAPHRASE_SYSTEM)     # not the identity persona: under it the talk adapter says nothing here
            text = " ".join(clean_reply(raw).split())
        except Exception as e:                                   # noqa: BLE001 -- no paraphrase, the facts stand
            ev.emit("paraphrase", pid, ok=False, text=None, rejected=[f"talk adapter failed: {str(e)[:80]}"])
            return None
        ok, bad = grounded_prose(text, facts, entity)
        ev.emit("paraphrase", pid, ok=ok, text=text if ok else None, rejected=bad or None, draft=text)
        return text if ok else None
