"""exp_r26 -- relabelling invariance. WO-2.5, the part that runs without a manifest.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE QUESTION
------------
WO-0.3 measured the role vocabulary and found it 95-98% per-relation: an interface
that mints `H1_DATE_OF_BIRTH` has, by construction, nothing to bind a relation it
never trained on. WO-1.3 dropped the task that minted them, taking the vocabulary
from 412 identifiers to 17, and the gen-3 held split could not see any difference
(599.7 vs 598.7 of 600) because it is at its own ceiling.

So the held split is spent as a discriminator, and this is the test that is not:

    Genuine structural generalization is invariant under bijective renaming.
    Memorization of a relation table collapses.

Every relation is replaced by a fresh opaque token -- `date of birth` -> `r_41027` --
consistently in the question, in the store, and in the vocabulary. Direction and
arity are preserved. Nothing about the STRUCTURE of the problem changes; only the
names do. An emitter that routes the relation it reads in the question is unharmed.
An emitter that recognises the wording and recalls a relation has nothing to recall.

A NOTE ON WHAT THIS CAN AND CANNOT SHOW
---------------------------------------
WO-2.5 records the winner's own self-indictment: once the host supplies the
admissible relations, **a pure copier passes the unseen-relation test**. That is
true here too -- an emitter that blindly copies the token out of the question
scores well, and copying is not understanding. What relabelling rules out is the
*other* failure, the one WO-0.3 measured: a memorized wording-to-relation table.
Ruling out memorization is not the same as demonstrating structure, and the decoy
control that would separate copying from selection needs WO-2.1's manifest.

Stated plainly so that a good score here is not over-read.

THE ARMS
--------
    labelled      the questions and store as they are            (baseline)
    relabelled    every relation -> a fresh opaque token
    null_relation the question names a token with ZERO facts     (the kill-line control)

`null_relation` is the one that can fail badly. The required output is a refusal;
the token is syntactically valid and binds to nothing, so **any spoken answer is a
kill-line breach**, not a scoring miss.

KILL CRITERION, in the two-clause form the amended rule requires
(`docs/WORK_ORDERS.md`, "How to write a kill criterion"):

  * correctness -- any wrong answer under relabelling, or any spoken answer on the
    null control, kills it outright.
  * rate -- relabelled correct below 80% of labelled correct.

Refusals under relabelling are the expected honest failure: recorded, not counted
against the arm.

    python validation/exp_r26_relabelling.py --n 300
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
BIND = re.compile(r'bind\s+frame\s*,\s*(HOP\d+|SEED)\s*,\s*"([^"]*)"')

MODELS = {
    "v13e": "emitter_v13e.Q4_K_M.gguf",
    "v14e": "emitter_v14e_nochain.Q4_K_M.gguf",
}
ARMS = ("labelled", "relabelled", "null_relation")


def relation_of(rec) -> list[str]:
    """The relation strings the gold program binds, outermost first."""
    return [v for k, v in BIND.findall(rec["program"]) if k.startswith("HOP")]


def seed_of(rec) -> str | None:
    for k, v in BIND.findall(rec["program"]):
        if k == "SEED":
            return v
    return None


_C = "bdfgklmnprstvz"
_V = "aeiou"


def _word(rng) -> str:
    """A pronounceable nonsense word: no punctuation to drop, no meaning to leak."""
    return "".join(rng.choice(_C) + rng.choice(_V) for _ in range(3))


def make_map(relations, rng, style: str = "word") -> dict[str, str]:
    """A bijection onto opaque tokens. Sorted first so the map is reproducible.

    `style='opaque'` gives `r_41027`, which was the first cut and is a BAD token:
    the emitter drops the underscore and emits `r 41027`, so a correctly-copied
    relation fails to match the store on formatting alone. Sampled 'other'
    emissions from the first run were 1-in-5 (v13e) and 2-in-5 (v14e)
    de-underscored tokens -- the score was partly measuring the token's spelling.

    `style='word'` gives `vomeka`: pronounceable, punctuation-free, and carrying
    no more meaning than the number did. Both are kept so the artifact stays on
    the record rather than being quietly corrected away.
    """
    rels = sorted(relations)
    if style == "opaque":
        toks = rng.sample(range(10_000, 99_999), len(rels))
        return {r: f"r_{t}" for r, t in zip(rels, toks)}
    out: dict[str, str] = {}
    seen: set[str] = set()
    for r in rels:
        w = _word(rng)
        while w in seen:
            w = _word(rng)
        seen.add(w); out[r] = w
    return out


def relabel_prompt(prompt: str, rels: list[str], mapping: dict[str, str], seed: str | None):
    """Substitute on word boundaries, longest relation first so `date of birth`
    is consumed before `birth`. Returns None when the seed entity did not survive
    -- several relation words ('born', 'die', 'child') occur inside entity names,
    and a substitution that corrupts the entity would measure the corruption."""
    out = prompt
    for r in sorted(rels, key=len, reverse=True):
        tok = mapping.get(r)
        if tok is None:
            return None
        out = re.sub(rf"\b{re.escape(r)}\b", tok, out, flags=re.IGNORECASE)
    if out == prompt:
        return None                                   # nothing was substituted
    if seed and seed.lower() not in out.lower():
        return None                                   # the entity got mangled
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"))
    ap.add_argument("--encyclopedia", default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--models", default="v13e,v14e")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--token-style", default="word", choices=("word", "opaque"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource
    from emitter import LlamaCppEmitter
    from eval_emitter_vm import strip_fences
    from exp_r9_matched_pairs import emitted_plan
    from exp_r11_search_learn import match
    from sources import PropertyAliases
    from worlds import FactStore

    held = [json.loads(l) for l in open(a.gen3, encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    rng = random.Random(a.seed)
    rows = rng.sample(held, min(a.n * 2, len(held)))        # oversample; some drop out below

    # ---- the world, as exp_r17 builds it ----------------------------------
    wk.ensure_data(("triplets",))
    tw = time.perf_counter()
    world = wk.wiki_world(); n_enc = 0
    enc_facts = []
    if a.encyclopedia and pathlib.Path(a.encyclopedia).is_file():
        for r in json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                world.index.declare_relation(r["rel"]); world.add(fact)
                enc_facts.append(fact); n_enc += 1
    known = StoreRelations(world.index._seen)
    log(f"world: {len(world):,} facts ({n_enc:,} encyclopedia) built in {time.perf_counter() - tw:.0f}s")

    # ---- the bijection ----------------------------------------------------
    all_rels = sorted({r for rec in held for r in relation_of(rec)})
    mapping = make_map(all_rels, random.Random(a.seed + 1), a.token_style)
    log(f"relabelling {len(all_rels)} relations ({a.token_style} tokens), e.g. "
        + ", ".join(f"{k} -> {mapping[k]}" for k in all_rels[:4]))

    # ---- keep only records whose prompt survives substitution -------------
    items = []
    for rec in rows:
        rels = relation_of(rec)
        rp = relabel_prompt(rec["prompt"], rels, mapping, seed_of(rec))
        if rp is None:
            continue
        items.append({"rec": rec, "rels": rels, "relabelled_prompt": rp})
        if len(items) >= a.n:
            break
    log(f"{len(items)} of {len(rows)} sampled records survive substitution "
        f"(dropped: entity collision or no substitution)\n")

    # ---- the relabelled world --------------------------------------------
    # Anchored on ' is the {rel} of ' rather than a bare word, so an entity that
    # happens to contain a relation word is never touched.
    tr = time.perf_counter()
    subs = 0
    relabelled_texts = []
    pats = [(re.compile(rf"(?<= is the ){re.escape(r)}(?= of )", re.IGNORECASE), mapping[r])
            for r in all_rels]
    for t in world.texts:
        new = t
        for pat, tok in pats:
            new2 = pat.sub(tok, new)
            if new2 != new:
                new = new2; subs += 1
                break
        relabelled_texts.append(new)
    rworld = FactStore(relabelled_texts, enc=world.enc, name="relabelled")
    for r in all_rels:
        rworld.index.declare_relation(mapping[r])
    rknown = StoreRelations(rworld.index._seen)
    log(f"relabelled world: {len(rworld):,} facts, {subs:,} facts had their relation renamed, "
        f"built in {time.perf_counter() - tr:.0f}s")

    # a fresh token bound to nothing anywhere -- the null control
    null_token = "r_00042" if a.token_style == "opaque" else "zubnog"
    assert not any(null_token in t for t in relabelled_texts[:50_000]), "null token is not free"
    log(f"null control token: {null_token} (zero facts)\n")

    src = NullSource(PropertyAliases())
    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    results: dict[str, dict[str, collections.Counter]] = {}
    breaches: dict[str, list] = {}
    for mname in [m for m in a.models.split(",") if m in MODELS]:
        gguf = ROOT / "standin" / "models" / MODELS[mname]
        if not gguf.is_file():
            log(f"!! {mname}: missing {gguf.name}, skipping"); continue
        em = LlamaCppEmitter(str(gguf))
        results[mname] = {}
        for arm in ARMS:
            c = collections.Counter(); ta = time.perf_counter()
            breaches[f"{mname}/{arm}"] = []
            breaches[f"{mname}/{arm}:bound"] = []
            for it in items:
                rec = it["rec"]
                if arm == "labelled":
                    q, w, kn, gold = rec["prompt"], world, known, rec["gold"]
                elif arm == "relabelled":
                    q, w, kn, gold = it["relabelled_prompt"], rworld, rknown, rec["gold"]
                else:
                    q = relabel_prompt(rec["prompt"], it["rels"],
                                       {r: null_token for r in it["rels"]}, seed_of(rec))
                    w, kn, gold = rworld, rknown, None
                    if q is None:
                        c["skipped"] += 1; continue
                try:
                    ep = emitted_plan(strip_fences(em.emit(q, max_new_tokens=a.max_new)), normalize)
                except Exception:                                   # noqa: BLE001
                    c["emit_error"] += 1; continue
                plan = (QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}",
                                     n_hop=len(ep[0])) if ep and ep[0] and ep[1] else None)
                if plan is None:
                    c["no_plan"] += 1; continue

                # WHAT did it bind? "it collapsed" is a number; "it collapsed because
                # it emitted the relation it was trained on, which no longer exists in
                # the store" is the finding. Three outcomes, and they mean different
                # things: copying the token is routing (and WO-2.5 warns a pure copier
                # passes); re-emitting the original is the memorized wording->relation
                # table WO-0.3 measured; anything else is neither.
                if arm in ("relabelled", "null_relation"):
                    want = (null_token if arm == "null_relation"
                            else {mapping[r] for r in it["rels"]})
                    want = {want} if isinstance(want, str) else want
                    got = {str(x).lower() for x in (ep[0] or [])}
                    orig = {r.lower() for r in it["rels"]}
                    if got & {w.lower() for w in want}:
                        c["bound:copied_token"] += 1
                    elif got & orig:
                        c["bound:recalled_original"] += 1
                    else:
                        c["bound:other"] += 1
                        if len(breaches[f"{mname}/{arm}:bound"]) < 5:
                            breaches[f"{mname}/{arm}:bound"].append(
                                {"q": q[:90], "emitted": list(got)[:4]})
                try:
                    lr = learn_and_answer(q, no_search, run_fn, store=w, known=kn, source=src,
                                          tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3,
                                          max_repairs=1, plan=plan)
                except Exception:                                   # noqa: BLE001
                    c["harness_error"] += 1; continue
                res = lr.result
                if not res.verified:
                    c["refused"] += 1; c[f"why:{res.reason or 'failed'}"] += 1
                    continue
                c["verified"] += 1
                if arm == "null_relation":
                    c["BREACH"] += 1                    # spoke on a relation with no facts
                    if len(breaches[f"{mname}/{arm}"]) < 6:
                        breaches[f"{mname}/{arm}"].append({"q": q, "spoke": res.answer})
                else:
                    m = match(res.answer, gold, normalize)
                    c[m] += 1
                    if m == "WRONG" and len(breaches[f"{mname}/{arm}"]) < 6:
                        breaches[f"{mname}/{arm}"].append(
                            {"q": q, "gold": gold, "got": res.answer})
            results[mname][arm] = c
            log(f"  {mname} {arm:<14} correct {c['correct']:>4}  refused {c['refused']:>4}  "
                f"WRONG {c['WRONG']:>3}  BREACH {c['BREACH']:>3}  no_plan {c['no_plan']:>4}  "
                f"({time.perf_counter() - ta:.0f}s)")
    session.close()

    # ---- the table --------------------------------------------------------
    log(f"\n{'model':<8}{'labelled':>10}{'relabelled':>12}{'ratio':>8}{'wrong':>7}{'breach':>8}   verdict")
    log("-" * 74)
    verdicts = {}
    for mname, arms in results.items():
        lab = arms["labelled"]["correct"]
        rel = arms["relabelled"]["correct"]
        ratio = rel / lab if lab else 0.0
        wrong = arms["relabelled"]["WRONG"]
        breach = arms["null_relation"]["BREACH"]
        if wrong or breach:
            v = "KILLED (correctness clause)"
        elif ratio < 0.80:
            v = "KILLED (rate clause)"
        else:
            v = "survives"
        verdicts[mname] = {"labelled": lab, "relabelled": rel, "ratio": round(ratio, 4),
                           "wrong": wrong, "breach": breach, "verdict": v}
        log(f"{mname:<8}{lab:>10}{rel:>12}{ratio:>8.3f}{wrong:>7}{breach:>8}   {v}")

    log("\nWHAT THE EMITTER BOUND under relabelling -- the finding, not the score:")
    log(f"  {'model':<8}{'copied the token':>18}{'recalled the original':>24}{'other':>8}")
    for mname, arms in results.items():
        c = arms["relabelled"]
        log(f"  {mname:<8}{c['bound:copied_token']:>18}{c['bound:recalled_original']:>24}"
            f"{c['bound:other']:>8}")
    log("  (copying is routing -- and WO-2.5 warns a pure copier passes. Recalling the")
    log("   original is the memorized wording->relation table WO-0.3 measured.)")

    log("\nrefusal reasons, relabelled arm (how it fails matters more than that it fails):")
    for mname, arms in results.items():
        why = {k[4:]: v for k, v in arms["relabelled"].items() if k.startswith("why:")}
        why["no_plan"] = arms["relabelled"]["no_plan"]
        log(f"  {mname:<8}{json.dumps(dict(sorted(why.items(), key=lambda kv: -kv[1])))}")

    for key, ex in breaches.items():
        if ex:
            log(f"\n{key} -- breaches:")
            for e in ex:
                log("   " + json.dumps(e)[:190])

    stem = f"exp_r26_relabelling{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "n": len(items), "seed": a.seed, "relations": len(all_rels), "mapping": mapping,
        "token_style": a.token_style,
        "world_facts": len(world), "relabelled_facts": subs, "null_token": null_token,
        "results": {m: {k: dict(v) for k, v in arms.items()} for m, arms in results.items()},
        "verdicts": verdicts, "breaches": breaches,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
