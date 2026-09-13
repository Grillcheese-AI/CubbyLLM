"""gen3_llm_wordings -- a frontier model proposes free-text wordings of certified chains; the host
reads its own plan off each wording; the gate certifies or refuses it. Dataset phase only.

Nick's rule (2026-09-12): an LLM's only job is building the dataset, and the final model never calls
one. Here the model writes QUESTIONS, never answers or plans: given a certified fact ('1749-08-28 is
the date of birth of Johann Wolfgang von Goethe', with its sentence when the encyclopedia is the
source), it writes N natural wordings whose answer is exactly the fact's object -- verb forms,
possessives, appositives, the shapes SimpleQA uses and the deterministic table lacks. Then, for each
wording, the HOST does what it does at serve time: the seed is the fact's subject, and the relation
words are the wording's own n-gram that the local property table resolves to the fact's relation
(lever 6's machinery); a wording with no such n-gram is uncertifiable and counted. The plan then goes
through the same gate as build_gen3's (no source to fetch from, coverage, the typed answer class, the
resident VM, the answer equal to the object), and only a certified (wording, plan) becomes a record.
Every verdict is counted by source: that is the measurement -- how far the host is from SimpleQA-style
free text, with the wordings a model writes rather than a table.

Cached by (model, prompt) under standin/data/out/openrouter_cache/; the key is in the gitignored
validation/.env and never in a log.

  python standin/data/gen3_llm_wordings.py --chains 200 --n 3 [--model google/gemini-3.8-flash] [--tag _probe]
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, random, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}

SYSTEM = ("You write natural-language questions for a quiz. You are given one FACT (subject, relation, value) and "
          "sometimes the sentence it was read from. Write questions a curious person might ask whose ONLY correct "
          "answer is exactly the given value. Rules: every question must name the subject exactly as given (you may "
          "add a short appositive after it, e.g. 'the composer'); vary the phrasing across questions -- a verb form "
          "('When was X born?'), a possessive ('What is X's ...?'), a longer form ('On what day, month, and year ...'); "
          "never include the answer; never ask about anything but the given relation; one question per line, no "
          "numbering, no quotes, nothing else.")


def prompt_for(chain: dict, n: int) -> str:
    prov = chain.get("provenance") or {}
    lines = [f"FACT: subject = {chain['seed']}; relation = {chain['rels'][-1]}; value = {chain['obj']}"]
    if prov.get("sentence"):
        lines.append(f"SENTENCE: {prov['sentence'][:300]}")
    lines.append(f"Write {n} questions.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"),
                    help="the certified records; chains are taken from their plan records (seed, gold, provenance)")
    ap.add_argument("--encyclopedia", default=str(ROOT / "validation" / "logs" / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--chains", type=int, default=200)
    ap.add_argument("--n", type=int, default=3, help="wordings asked per chain")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "gen3_llm_wordings.jsonl"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); rng = random.Random(a.seed)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import _FRAME_AND_JOINT, learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource, cot_plan, rec_id
    from openrouter import OpenRouterProposer
    from sources import PropertyAliases

    # the chains: one per distinct certified one-hop fact in the gen-3 set (seed, object); the seed and the
    # relation word come from the plan program (SEED / HOP1) -- the host's plan, on record
    import re
    BIND = re.compile(r'bind\s+frame\s*,\s*(SEED|HOP1)\s*,\s*"((?:[^"\\]|\\.)*)"')
    by_fact: dict[tuple, dict] = {}
    facts_of: dict[str, str] = {}                              # question -> the walked fact, from the chain record
    for l in open(a.gen3, encoding="utf-8"):
        r = json.loads(l)
        if r["task"] == "chain" and r.get("n_hop", 1) == 1:
            q, _, facts = r["prompt"].partition("\nFacts:\n")
            first = facts.split("\n")[0].lstrip("- ").strip()
            if " is the " in first and " of " in first:
                facts_of[q] = first
    for l in open(a.gen3, encoding="utf-8"):
        r = json.loads(l)
        if r["task"] != "plan" or r.get("n_hop", 1) != 1:
            continue
        binds = dict((k, v) for k, v in BIND.findall(r["program"]))
        fact = facts_of.get(r["prompt"])
        if "SEED" not in binds or "HOP1" not in binds or fact is None:
            continue
        label = fact.split(" is the ", 1)[1].rsplit(" of ", 1)[0]          # the store's label the walk used
        fkey = (normalize(binds["SEED"]), normalize(r["gold"]))
        if fkey in by_fact:
            continue
        by_fact[fkey] = {"seed": binds["SEED"], "hop_word": binds["HOP1"], "obj": r["gold"], "provenance": r.get("provenance") or {},
                         "split": r["split"], "rels": [label], "fact": fact}
    chains = list(by_fact.values())
    rng.shuffle(chains); chains = chains[:a.chains]
    print(f"{len(by_fact):,} distinct certified one-hop facts in the gen-3 set; {len(chains)} sampled (seed {a.seed})")
    aliases = PropertyAliases()

    wk.ensure_data(("triplets",))
    world = wk.wiki_world(); known = StoreRelations(world.index._seen)
    n_enc = 0
    if a.encyclopedia and os.path.isfile(a.encyclopedia):
        for r in json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                known.declare(r["rel"]); world.index.declare_relation(r["rel"]); world.add(fact); known.add(fact); n_enc += 1
    src = NullSource(aliases)
    session = cc.CubelangSession(exe=a.exe); calls = collections.Counter()

    def run_fn(source, fn):
        calls["vm"] += 1; return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    # a thinking model's reasoning tokens count against max_tokens (the ceiling probe's run 1): effort low, budget wide
    llm = OpenRouterProposer(a.model, system=SYSTEM, offline=a.offline, max_tokens=1200, reasoning={"effort": "low"})
    print(f"store {len(world):,} facts ({n_enc:,} encyclopedia) | model {a.model}{' (offline cache)' if a.offline else ''}")

    def host_plan(question: str, seed: str, label: str) -> tuple[QuestionPlan | None, str | None, str]:
        """The host's plan for a wording: seed = the fact's subject; the relation words = the shortest n-gram of
        the question (seed removed) that the table resolves to the fact's label (or IS the label / the store's
        word for it). None with a reason when no n-gram does: the wording cannot be planned from its own words."""
        q = normalize(question); s = normalize(seed)
        if f" {s} " not in f" {q} ":
            return None, None, "seed_not_in_question"
        text = f" {q} ".replace(f" {s} ", " ")
        toks = text.split()
        canon = {normalize(l) for l in aliases.relations(label)} | {normalize(label)}   # the store's label and its property's
        want = {normalize(label)} | {normalize(w) for c in canon for w in aliases.wordings(c)}
        for n in range(1, 4):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if all(w in _FRAME_AND_JOINT for w in g) or any(w.isdigit() for w in g):
                    continue
                gram = " ".join(g)
                if gram in want or ({normalize(l) for l in aliases.relations(gram)} & canon):
                    return QuestionPlan(relations=[None], tail=f"{gram} of {seed}", n_hop=1), gram, "ok"
        return None, None, "no_relation_wording"

    c = collections.Counter(); by_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    records: list[dict] = []; examples: dict[str, list] = collections.defaultdict(list); seen_q: set[str] = set()
    for i, ch in enumerate(chains):
        text = llm.chat(prompt_for(ch, a.n), max_tokens=400)
        wordings = [w.strip().strip('"').lstrip("0123456789.-) ").strip() for w in text.splitlines() if w.strip()]
        wordings = [w for w in wordings if w.endswith("?") and len(w) > 8][:a.n]
        c["chains"] += 1; c["wordings"] += len(wordings)
        srcname = (ch["provenance"] or {}).get("source", "?")
        for w in wordings:
            if normalize(w) in seen_q:
                c["duplicate"] += 1; continue
            seen_q.add(normalize(w))
            plan, gram, why = host_plan(w, ch["seed"], ch["rels"][0])
            if plan is None:
                c[why] += 1; by_source[srcname][why] += 1
                if len(examples[why]) < 4:
                    examples[why].append((w, ch["seed"], ch["rels"][0]))
                continue
            lr = learn_and_answer(w, no_search, run_fn, store=world, known=known, source=src,
                                  tau_vm=TAU_VM[1], top_k=3, max_repairs=1, plan=plan)
            res = lr.result
            if res.verified and normalize(res.answer) == normalize(ch["obj"]):
                verdict = "certified"
            elif res.verified:
                verdict = "verified_other_answer"
            else:
                verdict = res.reason or "failed"
            c[verdict] += 1; by_source[srcname][verdict] += 1
            if verdict != "certified":
                if len(examples[verdict]) < 4:
                    examples[verdict].append((w, [gram], res.answer, ch["obj"]))
                continue
            facts = [h.fact for h in res.trace if h.fact]
            base = {"source": f"cubbyllm/gen3_llm_{srcname}", "split": ch["split"], "system": None, "state": None, "repeat": 1,
                    "gold": ch["obj"], "provenance": dict(ch["provenance"] or {}, wording_model=a.model), "wording": "llm",
                    "aliased": lr.aliased, "n_hop": 1}
            recs = [dict(base, task="plan", subtype="n_hop=1", prompt=w, program=cot_plan(ch["seed"], [gram]),
                         vm_ok=None, vm_result=None, vm_error=None, gold_match=None),
                    dict(base, task="chain", subtype="n_hop=1", prompt=w + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts),
                         program=res.source, vm_ok=True, vm_result=res.answer, vm_error=None, gold_match=True)]
            for r in recs:
                r["id"] = rec_id(r["task"], r["prompt"], r["program"]); records.append(r)
            if len(examples["certified"]) < 12:
                examples["certified"].append((w, [gram], res.answer, ch["obj"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(chains)} chains, {c['wordings']} wordings, {c['certified']} certified, live calls {llm.calls}, "
                  f"${llm.usage['cost']:.3f}, {time.perf_counter() - t0:.0f}s", flush=True)
    session.close()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    total = c["wordings"] - c["duplicate"]
    manifest = {"version": "gen3-llm", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_rev": rev, "model": a.model,
                "chains": c["chains"], "wordings": c["wordings"], "certified": c["certified"], "by_verdict": dict(c),
                "by_source": {k: dict(v) for k, v in by_source.items()}, "n_records": len(records),
                "usage": dict(llm.usage, live_calls=llm.calls, finish=llm.finish), "vm_calls": calls["vm"],
                "wall_s": round(time.perf_counter() - t0, 1), "output": a.out}
    mp = a.out.replace(".jsonl", f"{a.tag}.manifest.json") if a.tag else a.out.replace(".jsonl", ".manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n{c['chains']} chains -> {c['wordings']} wordings ({c['duplicate']} duplicates) | certified {c['certified']} "
          f"({c['certified'] / max(1, total):.1%}) | records {len(records)} | live calls {llm.calls}, ${llm.usage['cost']:.3f} | "
          f"VM calls {calls['vm']} | {time.perf_counter() - t0:.0f}s")
    print("by verdict: " + ", ".join(f"{k} {v}" for k, v in c.most_common() if k not in ("chains", "wordings")))
    for s, v in by_source.items():
        print(f"  {s}: " + ", ".join(f"{k} {x}" for k, x in v.most_common()))
    print("\nexamples:")
    for k, ex in examples.items():
        for e in ex:
            print(f"  [{k}] {e}")
    print(f"wrote {a.out}\nwrote {mp}")


if __name__ == "__main__":
    main()
