"""exp_r27 -- does the per-request manifest recover the relabelling loss? WO-2.1.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE SETUP
---------
`exp_r26` relabelled every relation to an opaque token and both emitter arms died:
0.247 and 0.262 of their labelled score. The conclusion was that the
string-argument form does not carry the generalization, which makes WO-2.1's
manifest load-bearing rather than a nicety. This tests that conclusion instead of
assuming it.

Same relabelled world, same questions. One thing changes: **the host supplies the
admissible relations.** Two ways at once, because they are separable and both
matter:

  1. In the PROMPT -- the emitter is told which relations exist for this entity.
  2. As the GATE -- `known=` becomes a `Manifest`, which has exact lookup and no
     fuzzy tier. `StoreRelations.match` will Jaccard-match `date birth` onto
     `date of birth`; a manifest will not. That second tier is what lets a
     memorized relation wear an exact hit's confidence.

THE ARMS
--------
    relabelled        exp_r26's condition, re-run here so the comparison is paired
    manifest          + the admissible set, in the prompt and as the gate
    manifest_decoy3   + 3 plausible relations the entity does NOT have

`manifest_decoy3` is the control WO-2.5 asks for and could not run before a
manifest existed, and it is the one that can embarrass the other two:

    "Once the host supplies the manifest, A PURE COPIER PASSES the
     unseen-relation test, and correctness migrates silently into manifest
     construction."

If the entity has one admissible relation, listing it and asking for a plan is
barely a test -- copy the only line. Decoys put the right relation among k+1
plausible ones, so selection accuracy can be read against the 1/(k+1) chance
baseline. A manifest arm that scores well and a decoy arm that scores at chance
means the manifest is doing the work and the emitter is transcribing.

WHAT A GOOD RESULT WOULD AND WOULD NOT MEAN
-------------------------------------------
Recovery here does not show the emitter generalizes. It shows the INFORMATION is
sufficient when the host supplies it -- which is the premise WO-2.2 then enforces
with a generated grammar, where an unknown relation becomes undecodable rather
than merely discouraged. A prompt asks; a grammar enforces.

    python validation/exp_r27_manifest.py --n 200 --models v14e,v13e
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r26_relabelling import (MODELS, TAU_VM, make_map, relabel_prompt,  # noqa: E402
                                relation_of, seed_of)

ARMS = ("relabelled", "manifest", "manifest_decoy3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", default=str(ROOT / "standin" / "data" / "out" / "gen3_free_text.jsonl"))
    ap.add_argument("--encyclopedia", default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decoys", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--models", default="v14e,v13e")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.manifest import Manifest
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from build_gen3 import NullSource
    from emitter import SYSTEM, LlamaCppEmitter
    from eval_emitter_vm import strip_fences
    from exp_r9_matched_pairs import emitted_plan
    from exp_r11_search_learn import match
    from sources import PropertyAliases
    from worlds import FactStore

    held = [json.loads(l) for l in open(a.gen3, encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    rows = random.Random(a.seed).sample(held, min(a.n * 2, len(held)))

    wk.ensure_data(("triplets",))
    world = wk.wiki_world(); n_enc = 0
    if a.encyclopedia and pathlib.Path(a.encyclopedia).is_file():
        for r in json.loads(pathlib.Path(a.encyclopedia).read_text(encoding="utf-8"))["facts"]:
            fact = f"{r['obj']} is the {r['rel']} of {r['entity']}"
            if fact not in world:
                world.index.declare_relation(r["rel"]); world.add(fact); n_enc += 1
    log(f"world: {len(world):,} facts ({n_enc:,} encyclopedia)")

    all_rels = sorted({r for rec in held for r in relation_of(rec)})
    mapping = make_map(all_rels, random.Random(a.seed + 1), "word")
    log(f"relabelling {len(all_rels)} relations (word tokens), "
        f"e.g. {', '.join(f'{k} -> {mapping[k]}' for k in all_rels[:3])}")

    items = []
    for rec in rows:
        rels = relation_of(rec)
        rp = relabel_prompt(rec["prompt"], rels, mapping, seed_of(rec))
        if rp is None:
            continue
        items.append({"rec": rec, "rels": rels, "q": rp, "seed": seed_of(rec)})
        if len(items) >= a.n:
            break
    log(f"{len(items)} records survive substitution\n")

    # the relabelled world, anchored exactly as exp_r26 builds it
    import re as _re
    pats = [(_re.compile(rf"(?<= is the ){_re.escape(r)}(?= of )", _re.IGNORECASE), mapping[r])
            for r in all_rels]
    texts = []
    for t in world.texts:
        new = t
        for pat, tok in pats:
            n2 = pat.sub(tok, new)
            if n2 != new:
                new = n2; break
        texts.append(new)
    rworld = FactStore(texts, enc=world.enc, name="relabelled")
    for r in all_rels:
        rworld.index.declare_relation(mapping[r])
    rknown = StoreRelations(rworld.index._seen)

    tm = time.perf_counter()
    index = Manifest.index_store(texts)
    pool = [mapping[r] for r in all_rels]
    log(f"manifest index: {len(index):,} entities in {time.perf_counter() - tm:.0f}s | "
        f"decoy pool {len(pool)} relations\n")

    src = NullSource(PropertyAliases())
    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    results: dict[str, dict[str, collections.Counter]] = {}
    for mname in [m for m in a.models.split(",") if m in MODELS]:
        gguf = ROOT / "standin" / "models" / MODELS[mname]
        if not gguf.is_file():
            log(f"!! {mname}: missing {gguf.name}"); continue
        em = LlamaCppEmitter(str(gguf))
        results[mname] = {}
        for arm in ARMS:
            c = collections.Counter(); ta = time.perf_counter()
            rng = random.Random(a.seed + 99)
            for it in items:
                man = None
                if arm != "relabelled":
                    man = Manifest.for_entity(it["seed"] or "", index)
                    if arm == "manifest_decoy3":
                        man = man.with_decoys(pool, a.decoys, rng)
                    c["manifest_size"] += len(man)
                    if len(man) == 0:
                        c["empty_manifest"] += 1
                system = SYSTEM if man is None else SYSTEM + "\n\n" + man.prompt_block()
                try:
                    ep = emitted_plan(strip_fences(
                        em.emit(it["q"], max_new_tokens=a.max_new, system=system)), normalize)
                except Exception:                                   # noqa: BLE001
                    c["emit_error"] += 1; continue
                plan = (QuestionPlan(relations=[None] + ep[0][1:], tail=f"{ep[0][0]} of {ep[1]}",
                                     n_hop=len(ep[0])) if ep and ep[0] and ep[1] else None)
                if plan is None:
                    c["no_plan"] += 1; continue
                want = {mapping[r].lower() for r in it["rels"]}
                got = {str(x).lower() for x in (ep[0] or [])}
                c["bound:right_token" if got & want else "bound:wrong"] += 1
                try:
                    lr = learn_and_answer(it["q"], no_search, run_fn, store=rworld,
                                          known=(man if man is not None else rknown),
                                          source=src, tau_vm=TAU_VM.get(plan.n_hop, 0.2202),
                                          top_k=3, max_repairs=1, plan=plan)
                except Exception:                                   # noqa: BLE001
                    c["harness_error"] += 1; continue
                res = lr.result
                if not res.verified:
                    c["refused"] += 1; c[f"why:{res.reason or 'failed'}"] += 1; continue
                c["verified"] += 1
                c[match(res.answer, it["rec"]["gold"], normalize)] += 1
            results[mname][arm] = c
            log(f"  {mname} {arm:<17} correct {c['correct']:>4}  refused {c['refused']:>4}  "
                f"WRONG {c['WRONG']:>3}  right-token {c['bound:right_token']:>4}  "
                f"({time.perf_counter() - ta:.0f}s)")
    session.close()

    n = len(items)
    log(f"\n{'model':<7}{'arm':<18}{'correct':>9}{'of':>5}{'wrong':>7}{'bound right':>13}"
        f"{'avg manifest':>14}{'chance':>8}")
    log("-" * 82)
    for mname, arms in results.items():
        for arm in ARMS:
            c = arms.get(arm)
            if not c:
                continue
            avg = (c["manifest_size"] / max(1, n)) if arm != "relabelled" else 0
            chance = (1 / avg) if avg else 0
            log(f"{mname:<7}{arm:<18}{c['correct']:>9}{n:>5}{c['WRONG']:>7}"
                f"{c['bound:right_token']:>13}{avg:>14.1f}{chance:>8.3f}")

    log("\nrefusal reasons:")
    for mname, arms in results.items():
        for arm in ARMS:
            c = arms.get(arm)
            if not c:
                continue
            why = {k[4:]: v for k, v in c.items() if k.startswith("why:")}
            why["no_plan"] = c["no_plan"]
            log(f"  {mname} {arm:<18}{json.dumps(dict(sorted(why.items(), key=lambda kv: -kv[1])))}")

    stem = f"exp_r27_manifest{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "n": n, "seed": a.seed, "decoys": a.decoys, "relations": len(all_rels),
        "results": {m: {k: dict(v) for k, v in arms.items()} for m, arms in results.items()},
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
