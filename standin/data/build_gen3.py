"""build_gen3 -- the reverse-built free-text set: certified chains in, questions out, the plan
never the model's.

Wired: STANDALONE (stand-in data; nothing in cubbyllm/ imports this).

Agreed 2026-09-12 (the ceiling probe's reading, and Nick's rule that an LLM's only job is
building the dataset): gen 3 patches the free-text holes the gate exposed -- verb forms ('was
born'), ask-type forms ('on what day, month, and year'), possessives, canonical labels for
plain words -- with questions written the REVERSE way. A certified chain comes first: a fact
the store holds with provenance (the encyclopedia's 10,790 frame-read facts, sentence on
record; a seeded sample of the wiki world's), and for two-hop, a second held fact off the
first's object. The question is a wording of that chain; the plan the emitter must learn is
the HOST's -- the seed and the question's own relation words, which the host resolves
('born' -> `date of birth` / the store's `birth date`, by the local property table and the
ask type) -- and the record enters only when the same gate that serves certifies it:
`learn_and_answer` with NO source to fetch from, the coverage rule, the typed answer class,
the resident VM, and the answer equal to the fact's object. What the gate refuses is counted
by wording and by reason: that count is the coverage rule's own measurement on free text.

Records match gen 2's (`build_gen2_partition.py`): a `plan` record (prompt = the question alone,
program = a CotPlan binding SEED and HOPk as the question's words) and a `chain` record
(prompt = question + Facts, program = the CotChain the walk built), each with the chain's
provenance. Split by entity hash. `--merge` appends them to gen 2 as emitter_sft_v13e.jsonl.

  python standin/data/build_gen3.py --encyclopedia validation/logs/exp_r16_encyclopedia_all.json \
      [--per-relation 300] [--two-hop 200] [--limit 0] [--seed 0] [--out standin/data/out/gen3_free_text.jsonl]
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

# (wording, the relation words the plan binds -- the question's own, resolved by the host)
WORDINGS: dict[str, list[tuple[str, list[str]]]] = {
    "date of birth": [
        ("When was {S} born?", ["born"]),
        ("On what day, month, and year was {S} born?", ["born"]),
        ("In what year was {S} born?", ["born"]),
        ("What is the date of birth of {S}?", ["date of birth"]),
        ("What is {S}'s date of birth?", ["date of birth"]),
        ("What is the birth date of {S}?", ["birth date"]),
    ],                                                   # not "When is X's birthday?": that is P3150, day and month, another property
    "place of birth": [
        ("What is the birthplace of {S}?", ["birthplace"]),
        ("What is the place of birth of {S}?", ["place of birth"]),
        ("Where was {S} born?", ["born"]),
        ("In which city was {S} born?", ["born"]),
    ],
    "date of death": [
        ("When did {S} die?", ["die"]),
        ("In what year did {S} die?", ["die"]),
        ("What is the date of death of {S}?", ["date of death"]),
        ("When did {S} pass away?", ["pass away"]),
    ],
    "place of death": [
        ("Where did {S} die?", ["die"]),
        ("What is the place of death of {S}?", ["place of death"]),
        ("In which city did {S} die?", ["die"]),
    ],
    "inception": [
        ("When was {S} founded?", ["founded"]),
        ("In what year was {S} established?", ["established"]),
        ("When was {S} created?", ["created"]),
        ("What is the inception of {S}?", ["inception"]),
    ],
    "located in the administrative territorial entity": [
        ("In which district is {S} located?", ["district"]),
        ("In what county is {S} located?", ["county"]),
        ("Which administrative territorial entity is {S} located in?", ["administrative territorial entity"]),
    ],
    "capital": [
        ("What is the capital of {S}?", ["capital"]),
        ("Which city is the capital of {S}?", ["capital"]),
        ("What is {S}'s capital?", ["capital"]),
    ],
    "spouse": [
        ("Who is the spouse of {S}?", ["spouse"]),
        ("Who is {S} married to?", ["married"]),
        ("Who was {S}'s husband?", ["husband"]),
        ("Who was {S}'s wife?", ["wife"]),
    ],
    "occupation": [
        ("What is the occupation of {S}?", ["occupation"]),
        ("What was {S}'s profession?", ["profession"]),
        ("What is {S}'s occupation?", ["occupation"]),
    ],
    "educated at": [
        ("Where was {S} educated?", ["educated"]),
        ("What is the alma mater of {S}?", ["alma mater"]),
        ("Which university did {S} attend?", ["attend"]),
        ("Where did {S} study?", ["study"]),
    ],
    "author": [("Who is the author of {S}?", ["author"]), ("Who wrote {S}?", ["wrote"])],
    "director": [("Who is the director of {S}?", ["director"]), ("Who directed {S}?", ["directed"])],
    "child": [("Who is the child of {S}?", ["child"]), ("Who is {S}'s child?", ["child"])],
    "employer": [("Who is the employer of {S}?", ["employer"]), ("Who does {S} work for?", ["work for"])],
    "award received": [("What award did {S} receive?", ["award"]), ("Which award was {S} given?", ["award"])],
    "genre": [("What genre is {S}?", ["genre"]), ("What is the genre of {S}?", ["genre"])],
    "sibling": [("Who is the sibling of {S}?", ["sibling"]), ("Who is {S}'s sibling?", ["sibling"])],
}
# the wiki world's own labels for the relations above (its vocabulary is not Wikidata's)
WORLD_LABELS = {"date of birth": ["birth date"], "place of birth": ["birthplace"], "date of death": ["death date"],
                "place of death": ["place of death"], "capital": ["capital"], "spouse": ["spouse"], "occupation": ["occupation"],
                "educated at": ["alma mater"], "author": ["author"], "director": ["director"], "child": ["child"],
                "employer": ["employer"], "award received": ["award"], "genre": ["genre"], "sibling": ["sibling"]}
# two-hop: the first hop as a noun phrase off the seed ('the spouse of S'), then a wording of the second
HOP1_NOUNS = {"spouse": "the spouse of {S}", "child": "the child of {S}", "parent": "the parent of {S}", "sibling": "the sibling of {S}",
              "author": "the author of {S}", "director": "the director of {S}", "employer": "the employer of {S}"}


class NullSource:
    """A Source with nothing to fetch: the store must already hold the chain. The local
    property table answers the wording questions (relations / kind / wordings)."""
    name = "none"

    def __init__(self, aliases) -> None:
        self.aliases = aliases; self.calls = 0

    def facts(self, entity: str) -> list:
        return []

    def relations(self, text: str) -> list[str]:
        return self.aliases.relations(text)

    def kind(self, label: str) -> str | None:
        return self.aliases.kind(label)

    def wordings(self, label: str) -> list[str]:
        return self.aliases.wordings(label)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def cot_plan(seed: str, rels: list[str]) -> str:
    binds = "\n".join([f'        bind frame, SEED, "{_esc(seed)}";'] +
                      [f'        bind frame, HOP{i + 1}, "{_esc(r)}";' for i, r in enumerate(rels)])
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n"
            "    public function solve(mention: str): str {\n"
            "        create frame: number;\n" + binds + "\n"
            "        return recover(frame, SEED);\n    }\n}\n")


def rec_id(task: str, prompt: str, program: str) -> str:
    return f"{task}-{hashlib.sha256((task + prompt + program).encode('utf-8')).hexdigest()[:12]}"


def cased(entity: str) -> str:
    """The wiki world's entities are lowercase; a question names them the way text does."""
    if entity != entity.lower():
        return entity
    return " ".join(w if w in ("of", "the", "and", "de", "von", "van", "da", "di", "del", "la", "le") else w[:1].upper() + w[1:]
                    for w in entity.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encyclopedia", default=str(ROOT / "validation" / "logs" / "exp_r16_encyclopedia_all.json"),
                    help="exp_r16's facts json (entity, rel, obj, sentence, volume, headword); '' to skip")
    ap.add_argument("--per-relation", type=int, default=300, help="wiki-world facts sampled per relation with wordings")
    ap.add_argument("--two-hop", type=int, default=200, help="two-hop chains sampled per hop-1 relation")
    ap.add_argument("--limit", type=int, default=0, help="cap on encyclopedia facts (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"))
    ap.add_argument("--merge", default=None, help="gen-2 jsonl to append the records to (writes emitter_sft_v13e.jsonl beside --out)")
    ap.add_argument("--merge-cap", type=int, default=12000,
                    help="at most this many certified TRAIN questions (x2 records) enter the merged set, sampled round-robin over "
                         "wordings and sources so gen 3 does not drown gen 2 (12k questions ~ gen 2's own row count); 0 = all")
    ap.add_argument("--from-gen3", default=None, help="skip the build: read an existing gen3 jsonl and only merge")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    if a.from_gen3:
        records = [json.loads(l) for l in open(a.from_gen3, encoding="utf-8")]
        print(f"read {len(records):,} gen-3 records from {a.from_gen3}")
        merge(records, a, rng, rev=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
              built=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize, parse_fact
    from sources import PropertyAliases

    wk.ensure_data(("triplets",))
    world = wk.wiki_world()
    known = StoreRelations(world.index._seen)
    aliases = PropertyAliases()
    src = NullSource(aliases)
    session = cc.CubelangSession(exe=a.exe)
    calls = collections.Counter()

    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    print(f"wiki world {len(world):,} facts, {len(known):,} relations | property table {len(aliases):,} wordings | {time.perf_counter() - t0:.0f}s")

    # ---- 1. the certified pool: (seed, [rel labels], object, provenance) ----------------------------
    chains: list[dict] = []
    n_enc = 0
    if a.encyclopedia and os.path.isfile(a.encyclopedia):
        rows = json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]
        if a.limit:
            rng.shuffle(rows); rows = rows[:a.limit]
        for r in rows:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                known.declare(r["rel"]); world.index.declare_relation(r["rel"])
                world.add(fact); known.add(fact)
            n_enc += 1
            chains.append({"seed": r["entity"], "rels": [r["rel"]], "obj": r["obj"], "n_hop": 1,
                           "prov": {"source": "encyclopedia", "volume": r["volume"], "headword": r["headword"],
                                    "sentence": r["sentence"], "check": r.get("check")}})
    print(f"encyclopedia: {n_enc:,} facts into the store (now {len(world):,})")
    # the wiki world, a seeded sample per relation
    by_rel: dict[str, list] = collections.defaultdict(list)
    wanted = {wl: lab for lab, wls in WORLD_LABELS.items() for wl in wls}
    for facts in world.index._by_subj.values():
        for f, t in facts:
            if t.rel in wanted:
                by_rel[t.rel].append(t)
    n_world = 0
    for wl, ts in sorted(by_rel.items()):
        rng.shuffle(ts)
        for t in ts[:a.per_relation]:
            chains.append({"seed": t.subj, "rels": [wanted[wl]], "obj": t.obj, "n_hop": 1, "world_rels": [wl],
                           "prov": {"source": "wiki", "fact": f"{t.obj} is the {wl} of {t.subj}"}})
            n_world += 1
    # two-hop: hop 1 a noun phrase off the seed, hop 2 a wording off hop 1's object
    n_two = 0
    for r1, noun in HOP1_NOUNS.items():
        cands = []
        for facts in world.index._by_subj.values():
            for f, t in facts:
                if t.rel == r1:
                    for f2, t2 in world.index._by_subj.get(normalize(t.obj), []):
                        if t2.rel in wanted:
                            cands.append((t, t2))
        rng.shuffle(cands)
        for t, t2 in cands[:a.two_hop]:
            chains.append({"seed": t.subj, "rels": [r1, wanted[t2.rel]], "obj": t2.obj, "n_hop": 2, "world_rels": [r1, t2.rel],
                           "hop1_noun": noun, "prov": {"source": "wiki", "facts": [f"{t.obj} is the {r1} of {t.subj}", f"{t2.obj} is the {t2.rel} of {t2.subj}"]}})
            n_two += 1
    print(f"pool: {len(chains):,} chains ({n_enc:,} encyclopedia, {n_world:,} wiki one-hop, {n_two:,} wiki two-hop) | "
          f"relations {sorted(by_rel)}")

    # ---- 2. wordings -> the gate --------------------------------------------------------------
    c = collections.Counter(); by_wording: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    records: list[dict] = []; examples: dict[str, list] = collections.defaultdict(list)
    seen_q: set[str] = set()
    for i, ch in enumerate(chains):
        label = ch["rels"][-1]
        for tmpl, words in WORDINGS.get(label, []):
            if ch["n_hop"] == 1:
                q = tmpl.format(S=cased(ch["seed"]))
                rels = list(words)
                plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {ch['seed']}", n_hop=1)
            else:
                q = tmpl.format(S=ch["hop1_noun"].format(S=cased(ch["seed"])))
                rels = [ch["rels"][0]] + list(words)
                plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {ch['seed']}", n_hop=2)
            if q in seen_q:
                c["duplicate_question"] += 1; continue
            seen_q.add(q)
            c["asked"] += 1
            lr = learn_and_answer(q, no_search, run_fn, store=world, known=known, source=src,
                                  tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan)
            res = lr.result
            if res.verified and normalize(res.answer) == normalize(ch["obj"]):
                verdict = "certified"
            elif res.verified:
                verdict = "verified_other_answer"                      # the store holds another value: not this chain's record
            else:
                verdict = res.reason or "failed"
            c[verdict] += 1; by_wording[tmpl][verdict] += 1
            if verdict != "certified":
                if len(examples[verdict]) < 3:
                    examples[verdict].append((q, rels, res.answer, (res.refused or {}) if isinstance(res.refused, dict) else None))
                continue
            split = "held" if int(hashlib.sha256(normalize(ch["seed"]).encode()).hexdigest(), 16) % 10 == 0 else "train"
            facts = [h.fact for h in res.trace if h.fact]
            base = {"source": f"cubbyllm/gen3_{ch['prov']['source']}", "split": split, "system": None, "state": None,
                    "repeat": 1, "gold": ch["obj"], "provenance": ch["prov"], "wording": tmpl, "aliased": lr.aliased,
                    "n_hop": plan.n_hop}
            plan_rec = dict(base, task="plan", subtype=f"n_hop={plan.n_hop}", prompt=q, program=cot_plan(ch["seed"], rels),
                            vm_ok=None, vm_result=None, vm_error=None, gold_match=None)
            chain_rec = dict(base, task="chain", subtype=f"n_hop={plan.n_hop}",
                             prompt=q + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts), program=res.source,
                             vm_ok=True, vm_result=res.answer, vm_error=None, gold_match=True)
            for r in (plan_rec, chain_rec):
                r["id"] = rec_id(r["task"], r["prompt"], r["program"])
                records.append(r)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{len(chains):,} chains, {c['asked']:,} asked, {c['certified']:,} certified, VM calls {calls['vm']:,}, {time.perf_counter() - t0:.0f}s", flush=True)
    session.close()

    # ---- 3. write ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    wording_table = {t: dict(v) for t, v in by_wording.items()}
    manifest = {"version": "gen3", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_rev": rev, "seed": a.seed,
                "pool": {"chains": len(chains), "encyclopedia": n_enc, "wiki_one_hop": n_world, "wiki_two_hop": n_two},
                "asked": c["asked"], "certified": c["certified"], "by_verdict": dict(c), "by_wording": wording_table,
                "n_records": len(records), "by_task": dict(collections.Counter(r["task"] for r in records)),
                "by_split": dict(collections.Counter(r["split"] for r in records)),
                "by_source": dict(collections.Counter(r["source"] for r in records)),
                "vm_calls": calls["vm"], "wall_s": round(time.perf_counter() - t0, 1), "output": a.out,
                "note": "certified = the gate verified the host's plan for the wording and the VM's answer is the chain's object; "
                        "every other verdict is the gate's refusal reason for that wording, counted"}
    mp = a.out.replace(".jsonl", ".manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nasked {c['asked']:,} | certified {c['certified']:,} ({c['certified'] / max(1, c['asked']):.1%}) | records {len(records):,} "
          f"{manifest['by_task']} split {manifest['by_split']} | VM calls {calls['vm']:,} | {time.perf_counter() - t0:.0f}s")
    print("by verdict: " + ", ".join(f"{k} {v:,}" for k, v in c.most_common()))
    print("\nby wording (certified / asked):")
    for t, v in sorted(wording_table.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(v.values()); print(f"  {v.get('certified', 0):5d} / {n:5d}  {t!r}  " + ", ".join(f"{k} {x}" for k, x in v.items() if k != "certified"))
    print("\nrefusal examples:")
    for k, ex in examples.items():
        for q, rels, ans, ref in ex:
            print(f"  [{k}] {q!r} plan {rels} -> {ans!r} {ref if ref else ''}")
    print(f"wrote {a.out}\nwrote {mp}")

    if a.merge:
        merge(records, a, rng, rev=rev, built=manifest["built"])


def merge(records: list[dict], a, rng: random.Random, rev: str, built: str) -> None:
    """gen 2 unchanged + a capped, stratified sample of the gen-3 TRAIN questions (both records
    of each), all held questions kept for evaluation. The cap keeps gen 3 from drowning gen 2:
    the full build certifies ~52k questions, six wordings of one fact among them; a seeded
    round-robin over (source, wording) takes `merge_cap` questions with every wording and both
    sources represented."""
    gen2 = [json.loads(l) for l in open(a.merge, encoding="utf-8")]
    system = next(r["system"] for r in gen2 if r["task"] == "chain")
    ids = {r["id"] for r in gen2}
    by_q: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_q[r["prompt"].split("\nFacts:")[0]].append(r)             # the chain record's prompt is the question + Facts
    train_qs = [q for q, rs in by_q.items() if rs[0]["split"] == "train"]
    held_qs = [q for q, rs in by_q.items() if rs[0]["split"] != "train"]
    if a.merge_cap and len(train_qs) > a.merge_cap:
        strata: dict[tuple, list[str]] = collections.defaultdict(list)
        for q in train_qs:
            r0 = by_q[q][0]; strata[(r0["source"], r0.get("wording"))].append(q)
        for qs in strata.values():
            rng.shuffle(qs)
        keys = sorted(strata); picked: list[str] = []
        while len(picked) < a.merge_cap and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(picked) < a.merge_cap:
                    picked.append(strata[k].pop())
        train_qs = picked
    merged = list(gen2); n_added = 0
    for q in train_qs + held_qs:
        for r in by_q[q]:
            if r["id"] in ids:
                continue
            merged.append(dict(r, system=system)); n_added += 1
    out13 = os.path.join(os.path.dirname(a.out), "emitter_sft_v13e.jsonl")
    with open(out13, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g3 = [r for r in merged if r["source"].startswith("cubbyllm/gen3")]
    m13 = {"version": "v13e", "built": built, "git_rev": rev, "gen2": a.merge, "gen3": a.from_gen3 or a.out,
           "merge_cap_questions": a.merge_cap, "gen3_train_questions": len(train_qs), "gen3_held_questions": len(held_qs),
           "n_records": len(merged), "by_task": dict(collections.Counter(r["task"] for r in merged)),
           "by_source": dict(collections.Counter(r["source"] for r in merged)),
           "gen3_by_wording_train": dict(collections.Counter(r.get("wording") for r in g3 if r["split"] == "train" and r["task"] == "plan")),
           "train_rows_after_repeat": sum(r.get("repeat", 1) for r in merged if r["split"] == "train"),
           "schema_note": "gen 3 = gen 2 unchanged + a capped, wording-stratified sample of the reverse-built free-text records "
                          "(plan + chain per question, provenance per record); every held question kept for exp_r17"}
    json.dump(m13, open(out13.replace(".jsonl", ".manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"merged with gen 2: {len(merged):,} records ({n_added:,} gen-3: {len(train_qs):,} train questions of {sum(1 for q, rs in by_q.items() if rs[0]['split'] == 'train'):,}, "
          f"{len(held_qs):,} held) {m13['by_task']} | train rows after repeat {m13['train_rows_after_repeat']:,} -> {out13}")


if __name__ == "__main__":
    main()
