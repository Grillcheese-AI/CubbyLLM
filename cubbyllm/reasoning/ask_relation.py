"""cubbyllm/reasoning/ask_relation.py -- "how is B related to A?": the relation between two entities through the
composition rules, the family facts it fetches, the feedback that teaches a rule; and the fetch through the gate.
Split out of ask.py (order 0031) as a mixin of AskLoop, verbatim.

Wired: WIRED (AskLoop).
"""
from __future__ import annotations

import inspect as _inspect
import time
from .ask_claims import _learned
from .ask_forms import TAU_VM

__wiring__ = "WIRED"


class AskRelation:
    """AskLoop's relation questions and its fetch (a mixin; order 0031)."""

    def _is_relation(self, head: str) -> bool:
        """Does the store know `head` as a relation? `self.known` is StoreRelations -- a set over parsed
        `Triple.rel`, which is the question being asked. (The first cut of this read `index._seen`, which
        holds whole FACT SENTENCES, not relations: every membership test was False, so every entity whose
        name contains 'of' stayed a profile AND every real plan became one too. Live, 2026-09-14.)"""
        if not head:
            return False
        try:
            if head in self.known:
                return True
        except Exception:                                # noqa: BLE001
            return True                                  # a store that cannot answer keeps the old, safe guard
        for res in self.resolvers:                       # the lexicon's own names for a relation count too
            fn = getattr(res, "relations", None)
            if not callable(fn):
                continue
            try:
                if any(c in self.known for c in (fn(head) or ())):
                    return True
            except Exception:                            # noqa: BLE001 -- a resolver that cannot answer is not a yes
                continue
        # a mounted scene's own relations (order 0034): "What is the east neighbour of the well?" is a plan for the
        # scene, not a profile of 'east neighbour of the well' -- the route (spatial.route) never saw it otherwise
        scene = getattr(self, "scene", None)
        if scene is not None and not head[:1].isupper():
            try:
                if scene.relations(head):
                    return True
            except Exception:                            # noqa: BLE001 -- a scene that cannot answer is not a yes
                pass
        # the source's own property table (2026-10-16, live: "Who is the mother of Justin Trudeau?" read as
        # a profile of 'mother of justin trudeau' because the wiki world holds no 'mother'). A wording the
        # table names is a relation whether or not the store holds it yet -- unheld, the loop fetches it
        # (the first-resolver rule, learn.resolve_relations); a name's head ('bank', 'university',
        # 'great pyramid') the table does not name stays a profile, and so does a capitalised head
        # ('the Duke of Edinburgh', 'the City of London'): the asker wrote a name. The LOCAL table only.
        table = getattr(getattr(self.source, "aliases", None), "relations", None)
        if callable(table) and not head[:1].isupper():
            try:
                return bool(table(head))
            except Exception:                            # noqa: BLE001
                return False
        return False

    def feedback(self, record_id: str, verdict: str | None = None, relation: str | None = None,
                 asker: str | None = None) -> dict:
        """The asker on one of the loop's records: `verdict` right / wrong, and for a relation ask the relation
        they say it is. Written to the history as its own line, which the sleep cycle folds into the record:
        a spoken answer called wrong is a defect, and a relation the asker states makes the ask an episode
        for the skill library -- the one way a live refusal teaches a rule. The loop's own answers never do.
        Several askers are needed: the library counts distinct askers as support, so one asker alone cannot
        put a rule in."""
        from cubbyllm.reasoning.planner import normalize
        rec = next((r for r in reversed(self.history) if r.get("id") == record_id), None)
        if verdict not in (None, "right", "wrong"):
            raise ValueError("verdict is 'right' or 'wrong'")
        if relation is not None and rec is not None and rec.get("kind") != "relation":
            raise ValueError("a relation corrects a relation ask ('how is B related to A?') only")
        stated = normalize(relation) if relation else None
        if stated and rec is not None and rec.get("verified"):
            verdict = "right" if normalize(rec.get("answer") or "") == stated else "wrong"
        if verdict is None and stated is None:
            raise ValueError("say right / wrong, or the relation it is")
        line = {"feedback_for": record_id, "verdict": verdict, "relation": stated, "asker": asker,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if rec is not None:
            if verdict:
                rec["feedback"] = verdict
            if stated:
                rec["gold"] = stated
        if self.history_path is not None:
            import json as _json
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(line, ensure_ascii=False) + "\n")
        return {"ok": True, "record": record_id, "known": rec is not None, "verdict": verdict, "relation": stated,
                "episode": bool(stated and (rec is None or rec.get("kind") == "relation"))}

    # -- the relation ask --------------------------------------------------------------------
    def relation(self, question: str, a: str, b: str, t0: float) -> dict:
        """How is B related to A: every chain of held facts from A to B, each composed by the skill
        library over every bracketing and certified in the VM with its facts; one relation is spoken only
        if every path that composes agrees (`skills.relate`). The loop's own answers never teach the
        library -- only gold or an asker does, at night."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        from cubbyllm.reasoning.skills import MAX_HOPS, MAX_PATHS, relate, relate_back
        qid = ev.emit("question", text=question, source="skills")
        learned: list = []
        asked: list = []
        paths, overflow = self.world.index.paths(a, b, max_hops=MAX_HOPS, limit=MAX_PATHS)
        if not paths and not overflow and hasattr(self.source, "facts"):
            # nothing joins them in the store: ask the source for the two families, through the gate
            learned, asked = self._fetch_family(a, b)
            paths, overflow = self.world.index.paths(a, b, max_hops=MAX_HOPS, limit=MAX_PATHS)
        res = relate(paths, self.skills, self._run_fn, tau_vm=TAU_VM, chunk=2, overflow=overflow)
        if not res["verified"] and res["reason"] in ("no_path", "no_rule") and self.skills.inverses:
            # the store may only lead the other way, from B to A: compose that, and turn it round with the
            # inverse rule for B's sex (skills.relate_back) -- certified with B's sex fact in the VM
            back, back_over = self.world.index.paths(b, a, max_hops=MAX_HOPS, limit=MAX_PATHS)
            if back and not back_over:
                turned = relate_back(back, self._sex_of(b), self.skills, self._run_fn, tau_vm=TAU_VM, chunk=2)
                if turned["verified"]:
                    res = turned
        first = res["paths"][0]["relations"] if res["paths"] else []
        ev.emit("plan", qid, seed=normalize(a), relations=first, tail=normalize(b), n_hop=len(first),
                how=f"relation ask: {len(paths)} path(s) from {a} to {b}, composed by {len(self.skills)} rule(s)")
        rec: dict = {"question": question, "kind": "relation", "answer": res["answer"], "verified": res["verified"],
                     "reason": res["reason"], "plan": first, "seed": normalize(a), "target": normalize(b),
                     "statement": f"{b} is the {res['answer']} of {a}" if res["verified"] else None,
                     "paths": [p["relations"] for p in res["paths"]], "relation_paths": res["paths"],
                     "direction": res.get("direction", "forward"),
                     "trace": [f for p in res["paths"] if p["status"] == "derived" for f in p["facts"]],
                     "learned": [_learned(p) for p in learned], "entities": asked,
                     "aliased": [], "snapped": None, "wall_s": round(time.perf_counter() - t0, 3)}
        ev.emit("answer", qid, answer=rec["answer"], verified=rec["verified"], reason=rec["reason"])
        return rec

    def _sex_of(self, person: str):
        """(sex, the stored fact that says it) for a person the store knows the sex of, else None."""
        from cubbyllm.reasoning.planner import normalize
        for _f, t in self.world.index._by_subj.get(normalize(person), []):
            if t is not None and normalize(t.rel) == "sex or gender" and normalize(t.obj) in ("male", "female"):
                return normalize(t.obj), t
        return None

    def _fetch_family(self, a: str, b: str) -> list:
        """The two families from the source: each person's father, mother, children, siblings and spouses,
        then each of THOSE people once -- for their sex (a male child is a son) and their own family, which
        is what a path of up to four hops needs. Every fact passes the gate with its provenance; the edge back
        is the one Wikidata itself declares (father/mother inverse of child, sibling and spouse symmetric,
        `skills.family_facts`). Returns (the gate's records of what was accepted, the people asked about)."""
        from cubbyllm.reasoning.learn import gate
        from cubbyllm.reasoning.planner import normalize, parse_fact
        from cubbyllm.reasoning.skills import FAMILY, family_facts
        sex: dict[str, str] = {}
        edges: list[tuple[str, str, str]] = []
        budget = [self.FAMILY_FETCH_MAX]
        asked: list[str] = []

        def family(entity: str, via: str | None) -> list[tuple[str, str, str]]:
            if budget[0] <= 0:
                return []
            budget[0] -= 1
            asked.append(entity)
            try:
                try:
                    got = self.source.facts(entity, via=via) if via else self.source.facts(entity)
                except TypeError:                        # a source that takes the entity alone
                    got = self.source.facts(entity)
            except Exception:                            # noqa: BLE001 -- a source that fails states nothing
                return []
            out = []
            for t in got or []:
                t = parse_fact(t) if isinstance(t, str) else t
                if t is None:
                    continue
                r = normalize(t.rel)
                if r == "sex or gender" and normalize(t.obj) in ("male", "female"):
                    sex[normalize(t.subj)] = normalize(t.obj)
                elif r in FAMILY:
                    out.append((t.subj, r, t.obj))
            return out

        for person in (a, b):
            ring = family(person, None)
            edges += ring
            for x, r, y in ring:
                edges += family(y, f"{y} is the {r} of {x}")
        name = getattr(self.source, "name", "source")
        prov = getattr(self.world, "provenance", None)
        learned = []
        for x, r, y in edges:
            for f in family_facts(x, r, y, {x: sex.get(normalize(x)), y: sex.get(normalize(y))}):
                p = gate(f, self.world, name, x)
                if p.status != "accepted":
                    continue
                self.world.add(f)
                if hasattr(self.known, "add"):
                    self.known.add(f)
                if prov is not None:
                    prov.setdefault(f, name)
                learned.append(p)
        return learned, asked

    # -- the profile ask ---------------------------------------------------------------------
    def facts_about(self, entity: str) -> list:
        """The Triples the store holds with `entity` as subject (the index's subject lookup)."""
        from cubbyllm.reasoning.planner import normalize
        index = getattr(self.world, "index", None)
        by = getattr(index, "_by_subj", None)
        if by is None:
            return []
        return [t for _f, t in by.get(normalize(entity), []) if t is not None]

    def _fetch(self, qid: int, entity: str, hint: list[str], rec: dict, item: str | None = None, asker: str | None = None) -> dict:
        """The source asked about the entity (the ask kind's relations as the resolution hint), every
        fact through the gate, admitted ones into the store with provenance; the fetch and gate events."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import gate, _takes_item, _takes_relations
        from cubbyllm.reasoning.planner import Triple, normalize
        prov = getattr(self.world, "provenance", None)
        rec["entities"].append(entity)
        kw = {}
        if _takes_relations(self.source):
            kw["relations"] = hint
        if _takes_item(self.source):
            kw.update(qid=item, asker=asker)
        if "classes" in _inspect.signature(self.source.facts).parameters:
            kw["classes"] = True                         # a profile ask wants the KIND: see WikidataSource.facts
        items = list(self.source.facts(entity, **kw))
        last = getattr(self.source, "last", None) if isinstance(getattr(self.source, "last", None), dict) else {}
        if item and last.get("qid") == item:             # the asker's choice: kept as itself, never as the bare label
            items = [Triple(obj=t.obj, rel=t.rel, subj=item) if isinstance(t, Triple) else t for t in items]
            rec["kept_as"] = item                        # (order 0036: a bare question later still asks which)
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
            rec["learned"].append(_learned(p))
            if p.status == "accepted":
                self.world.add(fact)
                if hasattr(self.known, "add"):
                    self.known.add(fact)
                if prov is not None:
                    prov[" ".join(fact.split())] = self.source.name
            # the time is recorded for a DUPLICATE too. The preloaded world carries no qualifiers,
            # so every fact it already holds is undated -- and when the source restates one, the
            # gate calls it a duplicate and the old code dropped the date with it. That is why
            # "what was the population of Quebec City in 2011" stayed ambiguous through a refetch
            # that had the answer in hand (2026-09-14, live). Learning WHEN a fact was true adds
            # no claim: it is the same fact, said with the qualifier the source always had.
            if p.status in ("accepted", "duplicate"):
                times = getattr(self.world, "times", None)        # WHEN the source said the fact was true
                src_times = getattr(self.source, "times", None)
                if times is not None and src_times:
                    w = src_times.get(normalize(fact))
                    if w:
                        times[" ".join(fact.split())] = w
            ev.emit("gate", fid, fact=fact, status=p.status, clash=p.clash, lifted=False,
                    provenance=(prov or {}).get(" ".join(fact.split())))
        return last
