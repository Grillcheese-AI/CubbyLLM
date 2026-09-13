"""exp_r18 -- the hdc rephrased set (nace-ai hypernet-scaling-law, Wiki5M 1/2/3-hop QA) through the gate.

`E:\\datasets\\hdc\\ood_validation_rephrased.pq` is 10,000 questions -- 5,000 one-hop, 3,000 two-hop, 2,000
three-hop -- each with its canonical template wording (`question_prompt_original`, the grammar's own basin:
"What is the continent of the country of the country of citizenship of X?"), a free-text rephrasing
(`question_prompt`: "Which continent does the country of citizenship of X belong to?"), the answer, and
the chain's facts in the template's noun form ("A is the country B is in"). That is a matched-pair set
at 10,000 with the facts on record: the store can hold every chain, so a refusal is the coverage rule's
and never the store's.

Two arms, one gate. `--planner host` (the default, no GPU): the plan is the host's, built the gen-3 way
-- the seed is the chain's subject, the relation words are the question's own n-grams the property table
resolves to each hop's label, in walk order -- so the run measures the coverage rule on multi-hop free
text and yields gen-3-shaped records (`--out`) for the certified rows. `--planner emitter --gguf ...`:
the emitter plans, and the run is the generality gate on matched pairs (exp_r9's design at 10,000):
verified / correct / near / WRONG per hop count, rephrased against original. Kill line 0 wrong.

The facts are parsed with the dataset's own `relation_template_mapping.csv` (278 relations, noun
templates); a fact no template parses, or a chain whose hops do not join, is counted and skipped, never
guessed.

  python validation/exp_r18_hdc_rephrased.py --n 1000 --arm both --tag _host
  python validation/exp_r18_hdc_rephrased.py --n 600 --planner emitter --gguf standin/models/emitter_v12e.Q4_K_M.gguf --tag _gen2
"""
from __future__ import annotations

import argparse, collections, csv, hashlib, json, pathlib, random, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
HDC = pathlib.Path("E:/datasets/hdc")
_ARTICLES = frozenset(("the", "a", "an"))


def fact_parsers(mapping_csv: pathlib.Path):
    """(relation label, regex) per template row, plus the generic 'is the <label> of' fallback, longest label first."""
    rows = list(csv.DictReader(open(mapping_csv, encoding="utf-8")))
    rxs = []
    for r in rows:
        word = r["edited_relation"] or r["relation"]
        nt = re.escape(r["noun_template"]).replace(re.escape("{relation}"), re.escape(word)).replace(re.escape("{subject}"), "(?P<subj>.+)")
        rxs.append((r["relation"], word, re.compile("^(?P<obj>.+?) is " + nt + "$")))
    labels = sorted({(r["relation"], r["edited_relation"] or r["relation"]) for r in rows} |
                    {(r["relation"], r["relation"]) for r in rows}, key=lambda x: -len(x[1]))

    def parse(fact: str):
        hits = [(m.group("subj"), rel, m.group("obj"), word) for rel, word, rx in rxs if (m := rx.match(fact))]
        if hits:                                             # the longest relation word is the most specific reading
            hits.sort(key=lambda h: -len(h[3]))
            if len(hits) == 1 or len(hits[0][3]) > len(hits[1][3]):
                return hits[0][:3]
            return None
        for rel, word in labels:                             # 'A is the parent entity of B': the edited word in the generic frame
            k = f" is the {word} of "
            i = fact.find(k)
            if i > 0:
                return fact[:i], rel, fact[i + len(k):]
        return None
    return parse, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HDC / "ood_validation_rephrased.pq"))
    ap.add_argument("--mapping", default=str(HDC / "relation_template_mapping.csv"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arm", choices=("rephrased", "original", "both"), default="both")
    ap.add_argument("--hops", default="1,2,3", help="hop counts to sample")
    ap.add_argument("--planner", choices=("host", "emitter"), default="host")
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v12e.Q4_K_M.gguf"))
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", default=None, help="gen-3-shaped records for the certified rows (host planner only)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--events", default=None, help="write every step of the loop as events (jsonl) for the control panel")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    if a.events:
        from cubbyllm.reasoning import events as ev
        ev.add_sink(ev.JsonlSink(a.events))

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import pyarrow.parquet as pq
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations, _FRAME_AND_JOINT
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource, cot_plan, rec_id
    from exp_r11_search_learn import match
    from sources import PropertyAliases
    from worlds import FactStore

    parse, mapping = fact_parsers(pathlib.Path(a.mapping))
    table = pq.read_table(a.file).to_pylist()
    hops = {int(h) for h in a.hops.split(",")}
    c = collections.Counter()

    # ---- every row's chain, parsed from its facts; the store holds all of them -------------------
    chains: dict[int, list[tuple[str, str, str]]] = {}
    store = FactStore(name="hdc"); known = StoreRelations(store.index._seen)
    for i, row in enumerate(table):
        hs = [parse(f) for f in (row["facts"] or [])]
        if not hs or any(h is None for h in hs):
            c["row:unparsed_fact"] += 1; continue
        if any(normalize(hs[k][2]) != normalize(hs[k + 1][0]) for k in range(len(hs) - 1)):
            c["row:chain_broken"] += 1; continue
        if normalize(hs[-1][2]) != normalize(str(row["answer"])):
            c["row:answer_not_chain_object"] += 1; continue
        if len(hs) != int(row["n_hop"]):
            c["row:hop_count_mismatch"] += 1; continue
        chains[i] = hs
        for subj, rel, obj in hs:
            fact = f"{obj} is the {rel} of {subj}"
            if fact not in store:
                known.declare(rel); store.index.declare_relation(rel); store.add(fact); known.add(fact)
    log(f"hdc {pathlib.Path(a.file).name}: {len(table):,} rows, {len(chains):,} chains parsed "
        f"({', '.join(f'{k[4:]} {v:,}' for k, v in sorted(c.items()) if k.startswith('row:'))}) | store {len(store):,} facts, "
        f"{len(known):,} relations | mapping {len(mapping)} relations | {time.perf_counter() - t0:.0f}s")

    aliases = PropertyAliases(); src = NullSource(aliases)
    session = cc.CubelangSession(exe=a.exe); calls = collections.Counter()

    def run_fn(source, fn):
        calls["vm"] += 1; return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    split_used = [0]                                             # plans whose relation words came by the split tier

    def host_words(question: str, seed: str, labels: list[str]) -> tuple[list[str] | None, str]:
        """The question's own words for each hop's label, in walk order: the longest n-gram (seed and earlier
        picks removed) that IS the label or that the property table resolves to the label's property --
        longest so 'given name' wins over 'name', 'original language of film or tv show' over 'original language'."""
        q = normalize(question); s = normalize(seed)
        if f" {s} " not in f" {q} ":
            return None, "seed_not_in_question"
        text = f" {q} ".replace(f" {s} ", " ")
        picks: dict[int, str] = {}
        # the longest label first, so 'field of this occupation' is read before 'occupation' takes its word
        for k in sorted(range(len(labels)), key=lambda k: -len(labels[k])):
            label = labels[k]
            canon = {normalize(l) for l in aliases.relations(label)} | {normalize(label)}
            want = {normalize(label)} | {normalize(w) for cn in canon for w in aliases.wordings(cn)}
            # NOT hdc's own edited words ('component' for has part, 'instance' for instance of): the gate must
            # resolve the plan's words at serve time, and the table has no such alias -- tried, and it only
            # moved the refusal from here to `unknown_relation`
            toks = text.split(); pick = None
            for n in range(min(len(toks), max(4, len(normalize(label).split()))), 0, -1):   # longest first: 'given name' before 'name'
                for i in range(len(toks) - n + 1):
                    g = toks[i:i + n]
                    if all(w in _FRAME_AND_JOINT for w in g) or any(w.isdigit() for w in g) or g[0] in _ARTICLES or g[-1] in _ARTICLES:
                        continue
                    gram = " ".join(g)
                    # a gram ending in 'of' is the label itself ('instance of') or nothing: Wikidata lists the
                    # inverse reading as an alias ('father of' under child, 'child of' under father)
                    if g[-1] == "of" and gram != normalize(label):
                        continue
                    if gram in want or ({normalize(l) for l in aliases.relations(gram)} & canon):
                        pick = gram; break
                if pick:
                    break
            via_split = False
            label_words = [w for w in normalize(label).split() if w not in _FRAME_AND_JOINT]
            if pick is not None and pick != normalize(label) and len(label_words) > 1 and all(w in set(text.split()) for w in label_words):
                pick = None            # 'country' for `country of citizenship` when 'citizenship' is there too: the label, split, not the alias
            if pick is None:
                # the split wording (covers v6): every content word of the label, or of one of its wordings, is in
                # the question, just not contiguously -- the plan names the label; the gate's split tier covers it
                qw = set(text.split())
                for form in sorted(want, key=lambda f: (f != normalize(label), -len(f))):
                    ws = [w for w in form.split() if w not in _FRAME_AND_JOINT]
                    if ws and all(w in qw for w in ws):
                        pick, via_split = normalize(label), True; split_used[0] += 1
                        for w in ws:
                            text = f" {text.strip()} ".replace(f" {w} ", " ", 1)
                        break
            if pick is None:
                return None, "no_relation_wording"
            picks[k] = pick
            if not via_split:
                text = f" {text.strip()} ".replace(f" {pick} ", " ", 1)
        return [picks[k] for k in range(len(labels))], "ok"

    def hop_worded(question: str, seed: str, label: str) -> bool:
        """Does the question carry any content word of the label, or of one of its table wordings, or a word
        the table resolves to the label's property? ('instance of' -> 'instance', 'is a', 'type' ...)"""
        q = f" {normalize(question)} ".replace(f" {normalize(seed)} ", " ")
        qw = set(q.split())
        canon = {normalize(l) for l in aliases.relations(label)} | {normalize(label)}
        forms = {normalize(label)} | {normalize(w) for cn in canon for w in aliases.wordings(cn)}
        for f in forms:
            ws = [w for w in f.split() if w not in _FRAME_AND_JOINT]
            if ws and all(w in qw for w in ws):
                return True
        return any(({normalize(l) for l in aliases.relations(w)} & canon) for w in qw if w not in _FRAME_AND_JOINT)

    def split_wording(question: str, seed: str, labels: list[str]) -> bool:
        """Diagnostic only: some hop's label (or one of its wordings) has all its content words in the
        question but none contiguously, and every other hop is worded contiguously or the same way --
        the coverage rule's order-bound residue on free text."""
        q = f" {normalize(question)} ".replace(f" {normalize(seed)} ", " ")
        qw = set(q.split()); split = 0
        for label in labels:
            canon = {normalize(l) for l in aliases.relations(label)} | {normalize(label)}
            forms = {normalize(label)} | {normalize(w) for cn in canon for w in aliases.wordings(cn)}
            if any(f" {f} " in q for f in forms):
                continue
            if any(all(w in qw for w in f.split() if w not in _FRAME_AND_JOINT) and any(w not in _FRAME_AND_JOINT for w in f.split())
                   for f in forms):
                split += 1; continue
            return False
        return split > 0

    em = None
    if a.planner == "emitter":
        from emitter import LlamaCppEmitter
        from eval_emitter_vm import strip_fences
        from exp_r9_matched_pairs import emitted_plan
        em = LlamaCppEmitter(a.gguf)

    idx = [i for i in chains if int(table[i]["n_hop"]) in hops]
    rows = random.Random(a.seed).sample(idx, min(a.n, len(idx)))
    arms = ("rephrased", "original") if a.arm == "both" else (a.arm,)
    log(f"sampled {len(rows)} of {len(idx):,} chains (hops {sorted(hops)}, seed {a.seed}) | arms {arms} | planner {a.planner}"
        + (f" {pathlib.Path(a.gguf).name}" if em else ""))
    by: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)      # (arm, n_hop) -> verdicts
    records: list[dict] = []; out_rows: list[dict] = []; wrong_ex: list = []; ex: dict[str, list] = collections.defaultdict(list)
    for j, i in enumerate(rows):
        row = table[i]; hs = chains[i]; n_hop = len(hs)
        seed = hs[0][0]; labels = [h[1] for h in hs]; gold = hs[-1][2]
        for arm in arms:
            q = row["question_prompt"] if arm == "rephrased" else row["question_prompt_original"]
            key = f"{arm} {n_hop}-hop"
            if arm == "rephrased" and n_hop > 1 and not hop_worded(q, seed, labels[-1]):
                # the rephraser dropped the chain's last hop ('Which genre does X belong to?' for 'What is the
                # instance of the genre of X?'): the question as written asks the shorter chain and its gold is the
                # longer one's -- not a question the gate can be right or wrong about; counted, not scored
                by[key]["rephrase_dropped_last_hop"] += 1; c[f"{arm}:rephrase_dropped_last_hop"] += 1
                out_rows.append({"i": i, "arm": arm, "n_hop": n_hop, "q": q, "gold": gold, "final": "rephrase_dropped_last_hop"}); continue
            if em is None:
                words, why = host_words(q, seed, labels)
                if words is None:
                    by[key][why] += 1; c[f"{arm}:{why}"] += 1
                    # diagnostic: the question holds every content word of each label (or of one of its wordings),
                    # just not contiguously -- 'Which country does X hold citizenship in?' for 'country of citizenship'
                    if why == "no_relation_wording" and split_wording(q, seed, labels):
                        by[key]["  of which split_wording"] += 1; c[f"{arm}:split_wording"] += 1
                    if len(ex[why]) < 6:
                        ex[why].append((q, seed, labels))
                    out_rows.append({"i": i, "arm": arm, "n_hop": n_hop, "q": q, "gold": gold, "final": why}); continue
                rels, plan_seed = words, seed
            else:
                try:
                    ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
                except Exception:                                                     # noqa: BLE001
                    ep = None
                if not (ep and ep[0] and ep[1]):
                    by[key]["no_plan"] += 1; c[f"{arm}:no_plan"] += 1
                    out_rows.append({"i": i, "arm": arm, "n_hop": n_hop, "q": q, "gold": gold, "final": "no_plan"}); continue
                rels, plan_seed = ep[0], ep[1]
            plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {plan_seed}", n_hop=len(rels))
            lr = learn_and_answer(q, no_search, run_fn, store=store, known=known, source=src,
                                  tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1, plan=plan)
            res = lr.result
            if res.verified:
                m = match(res.answer, gold, normalize)
                verdict = "certified" if m == "correct" else ("near" if m == "near" else "WRONG")
                if verdict == "WRONG":
                    wrong_ex.append((q, rels, plan_seed, res.answer, gold, [h.fact for h in res.trace]))
            else:
                verdict = res.reason or "failed"
                if len(ex[verdict]) < 6:
                    ex[verdict].append((q, rels, res.answer, (res.refused or {}) if isinstance(res.refused, dict) else None))
                if em is None and verdict == "plan_does_not_cover_question" and split_wording(q, seed, labels):
                    by[key]["  of which split_wording"] += 1; c[f"{arm}:split_wording"] += 1
            by[key][verdict] += 1; c[f"{arm}:{verdict}"] += 1
            out_rows.append({"i": i, "arm": arm, "n_hop": n_hop, "q": q, "gold": gold, "plan": rels, "seed": plan_seed,
                             "aliased": lr.aliased, "snapped": getattr(lr, "snapped", None), "final": verdict, "answer": res.answer})
            if verdict == "certified" and em is None and a.out:
                split = "held" if int(hashlib.sha256(normalize(seed).encode()).hexdigest(), 16) % 10 == 0 else "train"
                facts = [h.fact for h in res.trace if h.fact]
                base = {"source": "cubbyllm/gen3_hdc", "split": split, "system": None, "state": None, "repeat": 1, "gold": gold,
                        "provenance": {"source": "hdc", "file": pathlib.Path(a.file).name, "row": i, "arm": arm,
                                       "facts": [f"{o} is the {r} of {s}" for s, r, o in hs]},
                        "wording": f"hdc {arm} {n_hop}-hop", "aliased": lr.aliased, "n_hop": n_hop}
                plan_rec = dict(base, task="plan", subtype=f"n_hop={n_hop}", prompt=q, program=cot_plan(seed, rels),
                                vm_ok=None, vm_result=None, vm_error=None, gold_match=None)
                chain_rec = dict(base, task="chain", subtype=f"n_hop={n_hop}",
                                 prompt=q + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts), program=res.source,
                                 vm_ok=True, vm_result=res.answer, vm_error=None, gold_match=True)
                for r in (plan_rec, chain_rec):
                    r["id"] = rec_id(r["task"], r["prompt"], r["program"]); records.append(r)
        if (j + 1) % 100 == 0:
            log(f"  {j + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s) " + " | ".join(
                f"{arm} certified {c[f'{arm}:certified']} near {c[f'{arm}:near']} WRONG {c[f'{arm}:WRONG']}" for arm in arms)
                + f" | VM calls {calls['vm']}")
    session.close()

    log(f"\nHDC REPHRASED ({a.planner}): VM calls {calls['vm']} | {time.perf_counter() - t0:.0f}s"
        + (f" | plans worded by the split tier {split_used[0]}" if em is None else ""))
    for key in sorted(by, key=lambda k: (k.split()[1], k.split()[0] != "rephrased")):
        v = by[key]; n = sum(x for k, x in v.items() if not k.startswith("  of which") and k != "rephrase_dropped_last_hop")
        log(f"  {v.get('certified', 0):4d} / {n:4d} certified  {key:18}  near {v.get('near', 0)} WRONG {v.get('WRONG', 0)} | "
            + ", ".join(f"{k} {x}" for k, x in v.most_common() if k not in ("certified", "near", "WRONG")))
    if a.arm == "both":
        # the matched pairs: the same chain, both wordings
        pair = collections.Counter()
        seen: dict[int, dict[str, str]] = collections.defaultdict(dict)
        for r in out_rows:
            seen[r["i"]][r["arm"]] = r["final"]
        for i, d in seen.items():
            pair[(d.get("original") == "certified", d.get("rephrased") == "certified")] += 1
        log(f"\nmatched pairs (original certified, rephrased certified): both {pair[(True, True)]}, original only {pair[(True, False)]}, "
            f"rephrased only {pair[(False, True)]}, neither {pair[(False, False)]}")
    log(f"\nevery WRONG ({len(wrong_ex)}):")
    for q, rels, seed, ans, gold, trace in wrong_ex:
        log(f"  {q!r} plan {rels} seed {seed!r} -> {ans!r} (gold {gold!r}) via {trace}")
    log("\nrefusal examples:")
    for k, items in ex.items():
        for it in items[:4]:
            log(f"  [{k}] {it}")
    if a.out and records:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"\nrecords {len(records):,} ({collections.Counter(r['split'] for r in records)}) -> {a.out}")
    wall = time.perf_counter() - t0
    out = {"file": a.file, "n": len(rows), "seed": a.seed, "arms": arms, "planner": a.planner, "gguf": a.gguf if em else None,
           "counts": dict(c), "by_arm_hop": {k: dict(v) for k, v in by.items()}, "rows": out_rows, "vm_calls": calls["vm"],
           "n_records": len(records), "wall_s": round(wall, 1)}
    (LOGS / f"exp_r18_hdc_rephrased{a.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r18_hdc_rephrased{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r18_hdc_rephrased{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
