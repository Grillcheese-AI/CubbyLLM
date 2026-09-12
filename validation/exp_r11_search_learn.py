"""exp_r11_search_learn — the store lever: a refusal becomes a fetch, a gate, a write and a verified answer.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

exp_r10 put 4,326 free-text questions through the gate and every one was refused or failed, because
the store holds 3 of the answers. That is the baseline search-and-learn has to move: coverage on
free text is a STORE problem before it is a planner problem. `cubbyllm.reasoning.learn` is the host
loop (refusal -> entity the walk stalled on -> source -> gate -> store with provenance -> walk
again -> VM). Two arms, two questions:

  --heldout   Does the MECHANISM work, and what does the gate protect against? The eval store with
              the hop-0 fact of N verified chains withheld; a source that serves exactly the
              withheld facts by subject. Every question must go refused -> learned -> verified, at
              the harvest's precision. Then the same source POISONED (a contradicting object for the
              same subject and relation), served after the true fact (the gate must refuse it) and
              before it (the gate cannot: a novel falsehood contradicts nothing -- the true fact is
              then the one refused, and the VM verifies the poison). Both orders are reported,
              because that asymmetry IS the gate's contract: it keeps the store consistent, not true;
              truth is the source's policy plus retire-never-delete on the provenance record.

  --wikidata  Does it move SimpleQA? Gen-2 emitter plans, the wiki world, and a Wikidata-backed
              source (standin/sources.py, cached) asked about the entity the walk stalled on. Only
              the learnable refusals fetch (unknown_relation / retrieval_exhausted; a plan that
              does not cover its question stays refused). Reported: fetched / admitted / refused by
              the gate / verified / correct / near / WRONG, and every verified answer in full.

  python validation/exp_r11_search_learn.py --heldout --n 200 --resident
  python validation/exp_r11_search_learn.py --wikidata --n 600 --resident
"""
from __future__ import annotations

import argparse, collections, csv, io, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
SIMPLEQA = ROOT / "standin" / "data" / "out" / "simpleqa" / "simple_qa_test_set.csv"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}

_MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october november december".split(), 1)}


def as_date(s: str):
    """'1932-03-23' / 'March 23, 1932' / '23 March 1932' -> (1932, 3, 23); None if not a date.
    A deterministic normalizer, not a judge: two spellings of one date are one fact."""
    import re
    s = s.strip().lower().rstrip(".")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m: return tuple(int(x) for x in m.groups())
    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1) in _MONTHS: return (int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(r"(\d{1,2})\s+([a-z]+),?\s+(\d{4})", s)
    if m and m.group(2) in _MONTHS: return (int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    return None


def match(answer: str, gold: str, normalize) -> str:
    x, g = normalize(answer), normalize(gold)
    if x == g: return "correct"
    if as_date(answer) is not None and as_date(answer) == as_date(gold): return "correct"
    if g and (g in x or x in g): return "near"
    return "WRONG"


class LookupStore:
    """texts + membership + add + TripleIndex + lookup: what the loop needs, nothing else."""
    def __init__(self, texts):
        from cubbyllm.reasoning import TripleIndex
        self.texts: list[str] = []; self._seen: set[str] = set(); self.index = TripleIndex(texts)
        for t in texts: self.add(t)
    def add(self, fact):
        key = " ".join(fact.split())
        if not key or key in self._seen: return False
        self._seen.add(key); self.texts.append(key); self.index.add(key); return True
    def __contains__(self, fact): return " ".join(fact.split()) in self._seen
    def lookup(self, plan, hop, entity): return self.index.hop(plan, hop, entity)


class HeldOutSource:
    """Serves the withheld facts whose SUBJECT is the entity asked; `poison` adds a contradicting
    object for each, `poison_first` serves it before the true fact."""
    name = "heldout"
    def __init__(self, withheld, normalize, poison=False, poison_first=False):
        from cubbyllm.reasoning.planner import parse_fact
        self.by = collections.defaultdict(list); self.calls = 0
        for f in withheld:
            t = parse_fact(f)
            if t is None: continue
            true = f; bad = f"NOT-{t.obj} is the {t.rel} of {t.subj}"
            self.by[normalize(t.subj)].extend(([bad, true] if poison_first else [true, bad]) if poison else [true])
        self.n = normalize
    def facts(self, entity):
        self.calls += 1; return list(self.by.get(self.n(entity), []))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--heldout", action="store_true"); g.add_argument("--wikidata", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--harvest", default=str(LOGS / "cot_harvest_split.jsonl"))
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--offline", action="store_true", help="wikidata: cache only, no network")
    ap.add_argument("--lexicon", action="store_true", help="add the WordNet+WOLF synonym oracle as a second relation resolver (lever 5)")
    ap.add_argument("--proposer", default=None, help="wikidata arm: 'openrouter:<model id>' puts a frontier model in the emitter's seat "
                    "(the ceiling probe; same disposer, walk, VM and kill line; never the serving model)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize, parse_question
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    def no_search(q, k): return []
    out: dict = {"arm": "heldout" if a.heldout else "wikidata", "n": a.n, "seed": a.seed}

    if a.heldout:
        import exp_m3_cot_pipeline as m
        _q, _a, _h, _c, store_texts = m.load_sample(800, seed=0)
        recs = [json.loads(l) for l in open(a.harvest, encoding="utf-8")]
        ver = [r for r in recs if r.get("verified") and r.get("correct") and r.get("trace")]
        rng = random.Random(a.seed); rng.shuffle(ver); ver = ver[:a.n]
        withheld = {r["trace"][0]["fact"] for r in ver}          # the hop-0 fact of each chain
        base = [f for f in store_texts if f not in withheld]
        log(f"eval store {len(store_texts)} facts; {len(ver)} verified-correct chains sampled (seed {a.seed}); "
            f"{len(withheld)} hop-0 facts withheld -> store of {len(base)}")
        arms = [("clean", False, False), ("poison_after_truth", True, False), ("poison_before_truth", True, True)]
        out["arms"] = {}
        for name, poison, first in arms:
            store, known = LookupStore(base), StoreRelations(base)
            src = HeldOutSource(withheld, normalize, poison=poison, poison_first=first)
            c = collections.Counter(); ex = []
            for r in ver:
                q = r["question"]; plan = parse_question(q)
                if plan is None: c["unparseable"] += 1; continue
                lr = learn_and_answer(q, no_search, run_fn, store=store, known=known, source=src,
                                      tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1)
                c[f"first:{lr.first.reason or ('verified' if lr.first.verified else 'other')}"] += 1
                for p in lr.learned: c[f"gate:{p.status}"] += 1
                if lr.result.verified:
                    c["verified"] += 1
                    ok = normalize(lr.result.answer) == normalize(r["gold_answer"])
                    c["correct" if ok else "WRONG"] += 1
                    if not ok and len(ex) < 4: ex.append((q, lr.result.answer, r["gold_answer"], [p.fact for p in lr.accepted]))
                else:
                    c[f"final:{lr.result.reason}"] += 1
            log(f"\n[{name}] {dict(sorted(c.items()))} | source calls {src.calls}")
            for q, ans, gold, learned in ex:
                log(f"    WRONG {q[:70]!r} -> {ans!r} (gold {gold!r}) learned {learned}")
            out["arms"][name] = dict(c)
        log("\nthe mechanism: clean -> every withheld chain should read refused -> learned -> verified at the harvest's precision;")
        log("poison after truth -> refused by the gate as a contradiction; poison before truth -> the gate keeps the store CONSISTENT, not true.")

    else:
        import wikikg as wk
        from exp_r9_matched_pairs import emitted_plan
        from emitter import LlamaCppEmitter
        from eval_emitter_vm import strip_fences
        from sources import WikidataSource
        wk.ensure_data(("triplets",))
        world = wk.wiki_world()
        known = StoreRelations(world.index._seen)
        rows = list(csv.DictReader(io.StringIO(SIMPLEQA.read_text(encoding="utf-8"))))
        qcol = next(c for c in rows[0] if c.lower() in ("problem", "question"))
        acol = next(c for c in rows[0] if c.lower() in ("answer", "gold", "target"))
        rows = random.Random(a.seed).sample(rows, a.n) if a.n else rows
        if a.proposer and a.proposer.startswith("openrouter:"):
            from openrouter import OpenRouterProposer
            em = OpenRouterProposer(a.proposer.split(":", 1)[1], offline=a.offline)
            proposer_name = a.proposer
        else:
            em = LlamaCppEmitter(a.gguf); proposer_name = pathlib.Path(a.gguf).name
        src = WikidataSource(offline=a.offline)
        resolvers = []
        if a.lexicon:
            from cubbyllm.reasoning.lexicon import Lexicon
            resolvers.append(Lexicon())
        log(f"wiki world {len(world)} facts, {len(known)} relations | SimpleQA sample {len(rows)} (seed {a.seed}) | proposer {proposer_name} | "
            f"source wikidata{' (offline cache)' if a.offline else ''}" + (f" | lexicon {len(resolvers[0])} synsets" if resolvers else ""))
        out["proposer"] = proposer_name
        c = collections.Counter(); verified_ex = []; learned_ex = []; rows_out = []; out["world0"] = len(world)
        def build_plan(rels, seed):
            return QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels)) if rels and seed else None
        for i, r in enumerate(rows):
            q, gold = r[qcol], r[acol]
            try:
                ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
            except Exception:                                # noqa: BLE001
                c["emit_error"] += 1; continue
            plan = build_plan(*ep) if ep else None
            if plan is None: c["no_plan"] += 1; continue
            c["plan"] += 1
            lr = learn_and_answer(q, no_search, run_fn, store=world, known=known, source=src,
                                  tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan,
                                  resolvers=resolvers)
            c[f"first:{lr.first.reason or ('verified' if lr.first.verified else 'other')}"] += 1
            if lr.entities:
                c["fetched_questions"] += 1; c["facts_fetched"] += lr.fetched
                for p in lr.learned: c[f"gate:{p.status}"] += 1
                if lr.accepted and len(learned_ex) < 6:
                    learned_ex.append((q, lr.entities, len(lr.accepted), [p.fact for p in lr.accepted[:3]]))
            if lr.aliased:
                c["aliased_questions"] += 1
            rows_out.append({"q": q, "gold": gold, "plan": ep[0], "seed": ep[1], "first": lr.first.reason,
                             "unknown": (lr.first.refused or {}).get("unknown_relations"), "entities": lr.entities,
                             "admitted": len(lr.accepted), "aliased": lr.aliased, "final": lr.result.reason,
                             "verified": lr.result.verified, "answer": lr.result.answer})
            if lr.result.verified:
                c["verified"] += 1
                m = match(lr.result.answer, gold, normalize)
                c[m] += 1
                verified_ex.append({"q": q, "plan": ep[0], "seed": ep[1], "answer": lr.result.answer, "gold": gold, "match": m,
                                    "learned": [p.fact for p in lr.accepted], "trace": [h.fact for h in lr.result.trace]})
            else:
                c[f"final:{lr.result.reason}"] += 1
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s, api calls {src.calls}) plan {c['plan']} fetched {c['fetched_questions']} "
                    f"verified {c['verified']} correct {c['correct']} near {c['near']} WRONG {c['WRONG']}")
        log(f"\nSIMPLEQA + SEARCH-AND-LEARN: {dict(sorted(c.items()))}")
        log(f"api calls {src.calls} | VM calls {calls['vm']} | store grew {len(world) - out.get('world0', len(world))}")
        if hasattr(em, "usage"):
            log(f"proposer {proposer_name}: {em.calls} live calls, usage {em.usage}")
            out["proposer_usage"] = dict(em.usage, live_calls=em.calls)
        log("\nevery verified answer:")
        for v in verified_ex:
            log(f"  [{v['match']:7s}] {v['q'][:90]!r}\n           plan {v['plan']} seed {v['seed']!r} -> {v['answer']!r} (gold {v['gold']!r})\n           via {v['trace']}")
        log("\nlearned (first 6):")
        for q, ents, n, fs in learned_ex:
            log(f"  {q[:70]!r} asked {ents} admitted {n} e.g. {fs}")
        out.update({"counts": dict(c), "verified": verified_ex, "api_calls": src.calls, "rows": rows_out})
        al = [x for x in rows_out if x["aliased"]]
        log(f"\naliased (lever 4): {len(al)} questions, e.g. " + "; ".join(f"{x['aliased']} -> {x['final'] or 'verified'}" for x in al[:8]))
        unk = collections.Counter(u for x in rows_out for u in (x["unknown"] or []) if x["final"] == "unknown_relation")
        log(f"relations still unknown after learning (top 20): {unk.most_common(20)}")

    if session: session.close()
    wall = time.perf_counter() - t0
    out.update({"vm_calls": calls["vm"], "wall_s": round(wall, 1)})
    tag = f"_{out['arm']}{a.tag}"
    (LOGS / f"exp_r11_search_learn{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r11_search_learn{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | VM calls {calls['vm']} | wrote exp_r11_search_learn{tag}.{{json,log}}")


if __name__ == "__main__":
    main()
