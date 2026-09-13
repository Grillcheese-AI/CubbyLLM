"""exp_r14_hippocampus_simpleqa -- the side cortex as the proposer on free text: certified
SHAPES, rebound to the entity a SimpleQA question names, worded by the resolvers, fetched by
the source, verified by the VM -- at 0 wrong?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

exp_r13 measured the hippocampus where the question's words are the episode's (matched
pairs). Free text is the other case: "On what day, month, and year was X born?" names no
episode's entity and none of the label `date of birth`. What memory can still offer is
the chain's SHAPE -- a certified `date of birth` chain, from any entity -- rebound to X:
a plan with a canonical label the question does not spell. That is exactly the plan a
frontier proposer wrote (exp_r11 Gemini runs), and the host already has the rest of the
path: lever 6 words the label from the question ('born' -> `date of birth`), the source
fetches X's facts, the VM certifies. So the proposer here is not a model at all -- it is
the record of what was certified before, and the disposer/source/VM decide as always.

Also the measurement item 4 asked for: the semantic DG (FastWordEncoder bits from the
MoWM codebook, `--word-bits`) against the surface hash, on cues worded differently from
the episodes -- the case the matched pairs could not measure.

  Episodes: every chain a previous exp_r11 run verified (its `verified` list: the trace's
  facts give the canonical relations, the seed is hop 0's subject), plus exp_r13's
  matched-pair episodes (the same wiki world). Cues: the 600-question SimpleQA sample.
  Candidates: `propose(q)` -- recalled plans (same entity) then the nearest shapes
  rebound to the question's entity (its longest run of capitalised tokens; on the record
  as a heuristic). Each candidate goes through `learn_and_answer` (lever 6, the source,
  the walk, the VM); the first verified answer wins; the kill line is a rising WRONG.

  python validation/exp_r14_hippocampus_simpleqa.py --n 600 --resident [--word-bits standin/data/out/word_bits_mowm.json]
         [--episodes-from exp_r11_search_learn_wikidata_lev6b,exp_r11_search_learn_wikidata_gemini4,...] [--k 3]
"""
from __future__ import annotations

import argparse, collections, csv, io, json, pathlib, random, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r11_search_learn import SIMPLEQA, TAU_VM, match  # noqa: E402

DEFAULT_EPISODE_RUNS = ("exp_r11_search_learn_wikidata_lev6b", "exp_r11_search_learn_wikidata_gemini4",
                        "exp_r11_search_learn_wikidata_typed", "exp_r11_search_learn_wikidata_wikitext2",
                        "exp_r11_search_learn_wikidata_alias2", "exp_r11_search_learn_wikidata_lex")
_TOK = r"(?:[A-Z][\w'.-]*|[a-z]{1,3}-[A-Z][\w'.-]*|\d[\w.-]*)"
_JOIN = r"(?:of|de|da|la|le|du|von|van|the|al|el|for|and|&)"
_CAP_RUN = re.compile(r"(?:(?<=\s)|^)(" + _TOK + r"(?:\s+(?:" + _TOK + r"|" + _JOIN + r"))*)")
_STOP_LEAD = {"On", "In", "What", "Which", "Who", "Whom", "When", "Where", "How", "The", "At", "During", "From", "To", "By", "Can", "Is"}
_JOIN_WORDS = set("of de da la le du von van the al el for and &".split())


def question_entity(question: str) -> str | None:
    """The longest run of capitalised tokens (joined by of/de/for/and/...), the question's
    lead word and any parenthetical excluded -- a heuristic entity finder for free text,
    on the record as such. A parenthetical qualifier ('(Missouri politician)') is dropped
    here: binding it to the entity is the hippocampus's second job, not this script's."""
    q = re.sub(r"\([^)]*\)", " ", question.rstrip("?")).replace('"', " ")
    best = None
    for m in _CAP_RUN.finditer(q):
        toks = m.group(1).split()
        while toks and toks[0] in _STOP_LEAD:
            toks = toks[1:]
        while toks and toks[-1].lower() in _JOIN_WORDS:
            toks = toks[:-1]
        if toks and all(t.isdigit() for t in toks):
            continue
        if len(toks) >= 1 and (best is None or len(" ".join(toks)) > len(best)):
            best = " ".join(toks)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--word-bits", default=None)
    ap.add_argument("--episodes-from", default=",".join(DEFAULT_EPISODE_RUNS))
    ap.add_argument("--matched-pairs", default=str(LOGS / "exp_r13_episodes_semantic.jsonl"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.hippocampus import Hippocampus, WordBits, residual_entity
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.lexicon import Lexicon
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize, parse_fact
    from sources import WikidataSource

    wk.ensure_data(("triplets",))
    world = wk.wiki_world()
    known = StoreRelations(world.index._seen)
    rows = list(csv.DictReader(io.StringIO(SIMPLEQA.read_text(encoding="utf-8"))))
    qcol = next(c for c in rows[0] if c.lower() in ("problem", "question"))
    acol = next(c for c in rows[0] if c.lower() in ("answer", "gold", "target"))
    rows = random.Random(a.seed).sample(rows, a.n) if a.n else rows

    # ---- the episodes: what previous runs certified ------------------------------
    wb = WordBits.load(a.word_bits) if a.word_bits else None
    hip = Hippocampus(word_bits=wb)
    n_prev = 0
    for run in [r for r in a.episodes_from.split(",") if r]:
        path = LOGS / f"{run}.json"
        if not path.is_file():
            continue
        for v in json.loads(path.read_text(encoding="utf-8")).get("verified", []):
            # the trace facts came from a source that declared their relations at the time; the
            # wiki world's own vocabulary may split 'date of birth' at 'date' -- parse greedily
            # (last ' of ') and declare the relation to the store and its vocabulary
            triples = [parse_fact(f) for f in v["trace"]]
            if not triples or any(t is None for t in triples):
                continue
            rels = [t.rel for t in triples]; seed = triples[0].subj
            for rel in rels:
                known.declare(rel); world.index.declare_relation(rel)
            if any(normalize(e.question) == normalize(v["q"]) for e in hip.episodes):
                continue
            hip.write(v["q"], rels, seed, v["trace"], v["answer"], {"run": run, "match": v.get("match")})
            n_prev += 1
    n_mp = 0
    if a.matched_pairs and pathlib.Path(a.matched_pairs).is_file():
        mp = Hippocampus.load(a.matched_pairs)
        for e in mp.episodes:
            hip.write(e.question, e.relations, e.seed, e.chain, e.answer, dict(e.provenance, run="exp_r13")); n_mp += 1
    # consolidation: an episode's FACTS live in the world store (they went through the gate when
    # they were certified) -- without them the labels are declared but not held, and a plan
    # 'date of birth of X' splits at the store's own 'date' (caught 2026-09-12, r14 run 1)
    n_facts = 0
    for e in hip.episodes:
        for f in e.chain:
            if f not in world:
                world.add(f); known.add(f); n_facts += 1
    shapes = {tuple(e.relations) for e in hip.episodes}
    log(f"consolidated {n_facts} episode facts into the world store; 'date of birth' held: {'date of birth' in known}")
    log(f"wiki world {len(world)} facts, {len(known)} relations | SimpleQA sample {len(rows)} (seed {a.seed}) | "
        f"episodes {len(hip)} ({n_prev} from exp_r11 runs, {n_mp} matched pairs), {len(shapes)} shapes | "
        f"DG {'semantic (' + str(len(wb)) + ' words)' if wb else 'surface hash'} | k={a.k}")
    log("  SimpleQA-run shapes: " + "; ".join(sorted({" > ".join(tuple(e.relations)) for e in hip.episodes if e.provenance.get("run") != "exp_r13"})))

    # ---- the gate ----------------------------------------------------------------
    src = WikidataSource(offline=a.offline); resolvers = [Lexicon()]
    session = cc.CubelangSession(exe=a.exe) if a.resident else None
    calls = collections.Counter()
    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn) if session else cc.run_program_proto(source, fn=fn, exe=a.exe)
    no_search = lambda q, k: []

    c = collections.Counter(); verified_ex = []; rows_out = []
    for i, r in enumerate(rows):
        q, gold = r[qcol], r[acol]
        ent = question_entity(q)
        # candidates: the hippocampus's own (recalled + rebound to the residual entity), then the
        # nearest shapes rebound to the question's capitalised entity
        # the residual-entity rebinding is for questions in the episode's own words; on free
        # text the capitalised entity is the better seed, so it replaces it when found
        cands = hip.propose(q, k=a.k, rebind=not ent)
        if ent:
            for ep, _d in hip.recall_shape(q, k=a.k):
                plan = QuestionPlan(relations=[None] + list(ep.relations[1:]), tail=f"{ep.relations[0]} of {normalize(ent)}",
                                    n_hop=len(ep.relations))
                if all(not (p.tail == plan.tail and p.relations == plan.relations) for p, _e, _k in cands):
                    cands.append((plan, ep, "rebound_cap"))
        if not cands:
            c["no_candidate"] += 1; continue
        c["planned"] += 1
        done = None; tried = 0; reasons = collections.Counter()
        for plan, ep, kind in cands:
            tried += 1; c["candidates"] += 1
            lr = learn_and_answer(q, no_search, run_fn, store=world, known=known, source=src,
                                  tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan,
                                  resolvers=resolvers)
            if lr.entities: c["fetched_questions_cand"] += 1; c["facts_fetched"] += lr.fetched
            for p in lr.learned: c[f"gate:{p.status}"] += 1
            if lr.result.verified:
                m = match(lr.result.answer, gold, normalize)
                c["verified"] += 1; c[m] += 1; c[f"verified:{kind}"] += 1
                if m == "correct": hip.reinforce(ep)
                log(f"    [{m}/{kind}] {q[:70]!r} -> {lr.result.answer!r} (gold {gold!r}) via {list(ep.relations)} of {plan.tail.rsplit(' of ', 1)[-1]!r}")
                verified_ex.append({"q": q, "kind": kind, "plan": list(ep.relations), "seed": plan.tail, "answer": lr.result.answer,
                                    "gold": gold, "match": m, "aliased": lr.aliased, "via_episode": ep.question,
                                    "trace": [h.fact for h in lr.result.trace]})
                done = lr; break
            reasons[lr.result.reason or "other"] += 1
        if done is None:
            top = reasons.most_common(1)[0][0] if reasons else "none"
            c[f"final:{top}"] += 1
        rows_out.append({"q": q, "gold": gold, "entity": ent, "n_candidates": len(cands), "tried": tried,
                         "verified": done is not None, "answer": done.result.answer if done else None,
                         "reasons": dict(reasons)})
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s, api calls {src.calls}) planned {c['planned']} "
                f"verified {c['verified']} correct {c['correct']} near {c['near']} WRONG {c['WRONG']}")

    log(f"\nHIPPOCAMPUS AS PROPOSER on SimpleQA: {dict(sorted(c.items()))}")
    log(f"api calls {src.calls} | VM calls {calls['vm']} | episodes {len(hip)} (utility>0: {sum(1 for e in hip.episodes if e.utility)})")
    log("\nevery verified answer:")
    for v in verified_ex:
        log(f"  [{v['match']:7s}/{v['kind']}] {v['q'][:80]!r}\n           shape {v['plan']} -> {v['seed']!r} -> {v['answer']!r} (gold {v['gold']!r}) "
            f"aliased {v['aliased']}\n           via episode {v['via_episode'][:70]!r}; chain {v['trace']}")
    if session: session.close()
    wall = time.perf_counter() - t0
    (LOGS / f"exp_r14_hippocampus_simpleqa{a.tag}.json").write_text(json.dumps({
        "n": len(rows), "seed": a.seed, "k": a.k, "word_bits": a.word_bits, "episodes": len(hip), "shapes": len(shapes),
        "counts": dict(c), "verified": verified_ex, "api_calls": src.calls, "vm_calls": calls["vm"], "rows": rows_out,
        "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r14_hippocampus_simpleqa{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r14_hippocampus_simpleqa{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
