"""exp_r3_emitter_floor — the pre-harvest FLOOR for the rung-1 generality gate.

Wired: STANDALONE. Needs the emitter GGUF and two logs. NO encoder, NO corpus,
NO VM required. Minutes on a consumer card.

CORRECTION (2026-09-11, after the first run) — READ THIS FIRST
--------------------------------------------------------------
The paragraph below this one was WRONG, and it was mine. I read the manifest's
task NAMES and concluded "no CoT harvest data". `standin/data/build_emitter_sft.py`
says, at line 22, what the `chain` task actually is:

    validation/logs/cot_harvest_v3cf.jsonl: our 517 verified multi-hop chain
    programs (ISolve / recover dialect) ... CHAIN_MULT = 3 (train-only upsampling)

So `chain 1,201` = the grammar's 517 verified chains ×3 (train) + val. v8e IS
the harvest-trained emitter, and this IS the bootstrap test. Checked against
the v8e jsonl itself (1,201 chain records: 1,055 train questions, 63 val):

    A_control      100 questions: 95 in train, 5 in val, 0 unseen  <- the TRAINING SET
    B_misparsed     92 questions:  0 in train, 0 in val, 92 unseen  <- held out by construction
    C_unparseable   37 questions:  0 in train, 0 in val, 37 unseen  <- held out by construction

Arms B and C are clean: the grammar never verified them, so the builder never
saw them. Arm A is not a control; it is the training data, and its numbers
(90/100 hop count, 85/100 exact role chain, 4/100 programs reproducing every
walked object) are what memorising the targets looks like -- the canary's
10.4% object-hit rate is that, not contamination of B or C.

Also: the training prompt was `question + "Facts:" + the walked facts`
(CHAIN_FACTS=1, v3). This probe prompts with the QUESTION ALONE, because a
plan has to be proposed before there are facts. That v8e still emits a
well-formed CotChain program 81/92 times without the facts block is itself a
finding; a facts-free plan-emission task is the obvious next SFT change.

First run (emitter_v8e.Q4_K_M, 2026-09-11):
    B  hop count = gold 45/92 (grammar 0) | disposer-accepted 39 | accepted AND gold 18
    C  hop count = gold 13/37 (grammar 0) | disposer-accepted 10 | accepted AND gold  7
    B, by the grammar's misparse form -> what the emitter did:
       2->1 (24): gold 22, grammar's answer 2        the emitter fixes the under-count
       2->3 (10): gold  9                            ... and the over-count
       3->4  (8): gold  4, 2-hop 4
       3->2 (33): gold 10, grammar's answer 23       ... but inherits THIS one: the
                                                    'contained within the' / 'office
                                                    held by the head of' joints
The number to beat is 18/92, and exp_r7 walks those plans through the VM.

The original (wrong) framing, kept so the mistake is visible:

  1. THE FLOOR. What does an emitter trained on OTHER CubeLang program families
     do on this task, zero-shot? Nobody has that number. It is the baseline the
     harvest-trained run has to beat, and pre-registering it now is what stops
     the later result being graded against a bar invented afterwards.

  2. A TOOLCHAIN SMOKE TEST. If v8e emits well-formed CotChain programs at all,
     the emit -> parse -> score path works and the real run is a data swap.

  3. A TRANSFER READ. v8e has 1,201 `chain` records. If any structure carries
     over to a different chain task, that is evidence the families are not as
     disjoint as the task labels suggest.

Expect low numbers. Low numbers here are INFORMATION, not failure.

v2 (2026-09-11): THE DISPOSER SCORES THE EMITTED PLAN
-----------------------------------------------------
`plan_verify` now exists, so an emitted role chain is not only compared to the
gold hop count -- it is DISPOSED OF exactly as a plan would be at serve time:

  answerable   every relation in the role chain is one the store holds, with
               the walk's own tolerance (exact for H1 = hop 0, relation_matches
               for H2+). Vocabulary from --vocab (a write_vocab_jsonl file the
               harvest already wrote; 165 relations on the real store). No
               corpus needed.
  of_question  every relation in the chain occurs in the question text
               (coverage-lite: the emitted program carries no seed entity, so
               the full covers() cannot be run; this is the half it can).
  accepted     answerable AND of_question -- the plan the disposer would let
               walk.
  escapes      arm B only: accepted AND hop count == gold. The grammar scores
               0 here by definition; every one of these is a question the
               emitter planned correctly where the grammar could not.

That last number is the rung-1 generality gate's first honest emitter reading.

What is scorable without retrieval
----------------------------------
A CotChain program has two separable parts:

    bind frame, H1_ADMINISTRATIVE_TERRITORIAL_ENTITY, "Kanton Genf";
                +---- ROLE: hop index + relation ----+  +-- OBJECT --+

The ROLE chain is the plan — how many hops, which relations, in what order — and
it is a function of the QUESTION alone. That is exactly what the grammar gets
wrong on the 92 `misparsed_chain` failures.

The OBJECT is the walk's output. It cannot be known from the question. Objects
are therefore NOT scored for correctness.

They are used for one thing:

  *** THE CONTAMINATION CANARY ***
  If an emitter returns the CORRECT objects from the question alone, it did not
  reason — it memorised the harvest, and every other number in the run is void.
  A high object-match rate is a FAILED run. For v8e it should be near zero; if
  it is not, something is wrong with the provenance story.

Arms (matched: the same questions the grammar was measured on)
--------------------------------------------------------------
  A  in-basin control   verified questions; gold role chain read from the
                        harvest's own `program_source`. Grammar: 100% here.
  B  misparsed_chain    the 92. Grammar: 0/92 correct hop count, by definition.
  C  unparseable        the 37 the grammar refuses. Grammar: 0/37, emits nothing.

  python validation/exp_r3_emitter_floor.py \
      --gguf standin/models/emitter_v8e.Q4_K_M.gguf [--n-control 100] \
      [--vocab validation/logs/exp_m3_vocab_lookup_vp_res.jsonl]
      -> validation/logs/exp_r3_emitter_floor_{tag}.{json,log}
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys, time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "standin"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BIND = re.compile(r'bind\s+frame\s*,\s*(?P<role>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"(?P<obj>(?:[^"\\]|\\.)*)"\s*;')
ROLE = re.compile(r"^H(?P<hop>\d+)_(?P<rel>.+)$")
CONTROL_ROLE = "ABSENT_CTRL"


HOP = re.compile(r"^HOP(?P<hop>\d+)$")


def parse_plan_program(text: str):
    """gen 2's `plan` task output: a CotPlan binding SEED and HOPk relation strings.
    -> (roles synthesized as H{k}_{REL} so every downstream scorer reads them like a
    CotChain's, seed entity, well_formed) or None if this is not a CotPlan."""
    if "program CotPlan" not in (text or ""):
        return None
    seed, hops = None, {}
    for m in BIND.finditer(text or ""):
        r, o = m.group("role"), m.group("obj")
        if r == "SEED" and seed is None:
            seed = o
        else:
            h = HOP.match(r)
            if h and int(h.group("hop")) not in hops:
                hops[int(h.group("hop"))] = o
    roles = [f"H{k}_" + re.sub(r"[^A-Za-z0-9]+", "_", hops[k]).strip("_").upper() for k in sorted(hops)]
    well = bool(roles) and seed is not None and "function solve" in (text or "")
    return roles, seed, well


def parse_program(text: str):
    """-> (roles in bind order, objects, well_formed). Mirrors programs.build_chain_program.
    A CotPlan (gen 2's plan task) is read through parse_plan_program: its roles come back
    the same way, its objects are empty (a plan has none), and the seed rides on the row."""
    pp = parse_plan_program(text)
    if pp is not None:
        roles, seed, well = pp
        return roles, [], well
    binds = [(m.group("role"), m.group("obj")) for m in BIND.finditer(text or "")]
    seen, roles, objs = set(), [], []
    for r, o in binds:                      # the same binds repeat in every function
        if r in seen or r == CONTROL_ROLE:
            continue
        seen.add(r); roles.append(r); objs.append(o)
    ordered = []
    for r in roles:
        m = ROLE.match(r)
        if m:
            ordered.append((int(m.group("hop")), m.group("rel"), r))
    ordered.sort(key=lambda x: x[0])
    well = bool(ordered) and "implements ISolve" in (text or "") and "function solve" in (text or "")
    return ([r for _, _, r in ordered],
            [objs[roles.index(r)] for _, _, r in ordered],
            well)


def rel_key(role: str) -> str:
    m = ROLE.match(role)
    return re.sub(r"[^a-z0-9]+", "", m.group("rel").lower()) if m else ""


def rel_text(role: str) -> str:
    """H2_COUNTRY_OF_CITIZENSHIP -> 'country of citizenship' (sanitize_role inverted
    up to punctuation, which normalize() discards anyway)."""
    m = ROLE.match(role)
    return m.group("rel").lower().replace("_", " ").strip() if m else ""


def dispose(question: str, roles: list[str], known) -> dict:
    """The plan_verify checks an emitted role chain admits (see docstring v2)."""
    from cubbyllm.reasoning.planner import normalize
    rels = [rel_text(r) for r in roles]                       # H1 first = hop 0's relation
    q = normalize(question)
    of_q = all(normalize(r) in q for r in rels) if rels else False
    unknown, paraphrased = [], []
    for i, r in enumerate(rels):
        if r in known:
            continue
        m = known.match(r) if i > 0 else None                 # hop 0 is exact, like the walk
        (paraphrased if m else unknown).append(r)
    answerable = bool(rels) and not unknown
    return {"answerable": answerable, "of_question": of_q, "accepted": answerable and of_q,
            "unknown_relations": unknown, "paraphrased": paraphrased}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(ROOT / "standin" / "models" / "emitter_v8e.Q4_K_M.gguf"))
    ap.add_argument("--n-control", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude", default=None,
                    help="gen2_exclusions.json: questions whose VM-verified chains entered training -- dropped "
                         "from arms B and C so a later generation is measured on transfer, not recall")
    ap.add_argument("--vocab", default=str(LOGS / "exp_m3_vocab_lookup_vp_res.jsonl"),
                    help="write_vocab_jsonl file: the store's relation vocabulary (no corpus needed)")
    a = ap.parse_args()

    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    harvest, decomp_p = LOGS / "cot_harvest_v3cf.jsonl", LOGS / "exp_m3_exhaustion_decomp.json"
    for p in (harvest, decomp_p, pathlib.Path(a.gguf)):
        if not p.exists():
            sys.exit(f"missing {p}")
    recs = [json.loads(l) for l in harvest.open(encoding="utf-8")]
    decomp = json.loads(decomp_p.read_text(encoding="utf-8"))
    by_q = {r["question"]: r for r in recs}
    mis = [d["question"] for d in decomp["details"] if d["kind"] == "misparsed_chain"]
    unparse = [r["question"] for r in recs if r.get("reason") == "unparseable"]
    control = [r["question"] for r in recs if r.get("verified") and r.get("program_source")][:a.n_control]
    if a.exclude:
        ex = set(json.loads(pathlib.Path(a.exclude).read_text(encoding="utf-8"))["questions"])
        mis = [q for q in mis if q not in ex]; unparse = [q for q in unparse if q not in ex]
        print(f"excluded {len(ex)} trained questions from arms B/C ({a.exclude})")

    from cubbyllm.reasoning.plan_verify import StoreRelations   # noqa: E402
    known = StoreRelations.from_vocab_jsonl(a.vocab) if pathlib.Path(a.vocab).exists() else None
    if known is None:
        sys.exit(f"missing --vocab {a.vocab}: run any exp_m3 --verify-plan vm first, or point at a write_vocab_jsonl file")

    from emitter import LlamaCppEmitter                     # noqa: E402
    sys.path.insert(0, str(ROOT / "standin"))
    from eval_emitter_vm import strip_fences               # noqa: E402  the shipped ``` / <think> strip
    em = LlamaCppEmitter(a.gguf)
    log(f"emitter {pathlib.Path(a.gguf).name} | vocabulary {pathlib.Path(a.vocab).name} ({len(known)} relations)")
    log(f"arms: control {len(control)} | misparsed {len(mis)} | unparseable {len(unparse)}\n")

    results, rows = {}, []
    for arm, qs in (("A_control", control), ("B_misparsed", mis), ("C_unparseable", unparse)):
        c = Counter(); lat = []
        for q in qs:
            r = by_q.get(q, {})
            gold_n = r.get("n_hop")                          # parquet n_hop = GOLD hop count
            t1 = time.perf_counter()
            try:
                out = strip_fences(em.emit(q, max_new_tokens=a.max_new))
            except Exception as e:                            # noqa: BLE001
                c["emit_error"] += 1
                rows.append({"arm": arm, "question": q, "error": str(e)[:200]}); continue
            lat.append(time.perf_counter() - t1)
            roles, objs, well = parse_program(out)
            c["well_formed" if well else "malformed"] += 1
            if not well:
                rows.append({"arm": arm, "question": q, "well_formed": False,
                             "gold_n_hop": gold_n, "emitted": None}); continue
            n = len(roles)
            c["hop_count_correct" if n == gold_n else "hop_count_wrong"] += 1
            c[f"hop_delta_{n - gold_n:+d}"] += 1
            d = dispose(q, roles, known)
            for k in ("answerable", "of_question", "accepted"):
                c[k] += int(d[k])
            if d["paraphrased"]:
                c["paraphrase_tier_used"] += 1
            if d["accepted"] and n == gold_n:
                c["accepted_and_hop_correct"] += 1

            gold_roles = gold_objs = None
            if r.get("program_source"):
                gold_roles, gold_objs, _ = parse_program(r["program_source"])
                if [rel_key(x) for x in roles] == [rel_key(x) for x in (gold_roles or [])]:
                    c["role_chain_exact"] += 1
                # --- the contamination canary ---
                hits = sum(1 for o in objs if o in (gold_objs or []))
                c["object_hits"] += hits
                c["object_slots"] += len(gold_objs or [])
                if gold_objs and hits == len(gold_objs):
                    c["ALL_objects_correct"] += 1
            pp = parse_plan_program(out)
            rows.append({"arm": arm, "question": q, "well_formed": True,
                         "gold_n_hop": gold_n, "emitted_n_hop": n,
                         "roles": roles, "objects": objs, "seed": (pp[1] if pp else None),
                         "program_kind": ("CotPlan" if pp else "CotChain"),
                         "gold_roles": gold_roles, "dispose": d})
        results[arm] = {"n": len(qs), "counts": dict(c),
                        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None}

    # ---------------- report ----------------
    log("=" * 76)
    log("CONTAMINATION CANARY — read this before anything else")
    log("=" * 76)
    ca = results["A_control"]["counts"]
    slots, hits = ca.get("object_slots", 0), ca.get("object_hits", 0)
    allc = ca.get("ALL_objects_correct", 0)
    if slots:
        rate = hits / slots
        log(f"  object slots {slots} | exact object matches {hits}  ({rate:.1%})")
        log(f"  programs with EVERY object correct: {allc}/{results['A_control']['n']}")
        if rate > 0.5 or allc > results["A_control"]["n"] * 0.25:
            log("\n  *** CONTAMINATED. The emitter is reproducing objects it cannot")
            log("  derive from the question. It memorised the harvest. Every other")
            log("  number in this run is void — retrain on a disjoint split, or")
            log("  score only on questions absent from the SFT set. ***")
        else:
            log("\n  Clean: objects are largely wrong, as they must be — they come from")
            log("  the walk, not the question. Role-structure numbers below are usable.")
    log("")

    log("=" * 76)
    log("ZERO-SHOT FLOOR — hop structure by arm (emitter never trained on this task)")
    log("=" * 76)
    log(f"\n{'arm':<16}{'n':>5}{'well-formed':>13}{'hop correct':>13}{'grammar':>10}{'delta':>8}")
    base = {"A_control": None, "B_misparsed": 0.0, "C_unparseable": 0.0}
    for arm in ("A_control", "B_misparsed", "C_unparseable"):
        r = results[arm]; c = r["counts"]; n = r["n"]
        wf = c.get("well_formed", 0); hc = c.get("hop_count_correct", 0)
        g = base[arm]
        gs = "100%" if g is None else f"{g:.0%}"
        d = "" if g is None else f"{hc/max(1,n) - g:+.0%}"
        log(f"{arm:<16}{n:>5}{wf:>13}{hc:>13}{gs:>10}{d:>8}")
    log("\n  grammar baselines are exact, not estimated:")
    log("   A  in-basin, the grammar parses and verifies these        -> 100%")
    log("   B  misparsed_chain IS the class where it miscounts        ->   0%")
    log("   C  unparseable IS the class where it emits nothing        ->   0%")

    for arm in ("A_control", "B_misparsed", "C_unparseable"):
        c = results[arm]["counts"]
        deltas = {k.replace("hop_delta_", ""): v for k, v in c.items() if k.startswith("hop_delta_")}
        log(f"\n  {arm}: hop-count delta (emitted - gold): {dict(sorted(deltas.items()))}")
        if "role_chain_exact" in c:
            log(f"    exact role chain (relations + order): {c['role_chain_exact']}/{results[arm]['n']}")
        if results[arm]["mean_latency_s"]:
            log(f"    mean {results[arm]['mean_latency_s']}s/question")

    log("\n" + "=" * 76)
    log("THE DISPOSER'S VIEW — would plan_verify let the emitted plan walk?")
    log("=" * 76)
    log(f"\n{'arm':<16}{'n':>5}{'answerable':>12}{'of question':>13}{'accepted':>10}{'+hop ok':>9}{'paraphr.':>10}")
    for arm in ("A_control", "B_misparsed", "C_unparseable"):
        c = results[arm]["counts"]; n = results[arm]["n"]
        log(f"{arm:<16}{n:>5}{c.get('answerable',0):>12}{c.get('of_question',0):>13}"
            f"{c.get('accepted',0):>10}{c.get('accepted_and_hop_correct',0):>9}{c.get('paraphrase_tier_used',0):>10}")
    log("\n  answerable  every relation is one the store holds (walk's tolerance; hop 0 exact)")
    log("  of question every relation occurs in the question (coverage without the seed entity)")
    log("  accepted    both -- the plan the disposer would let walk")
    log("  +hop ok     accepted AND hop count == gold. On arm B the grammar scores 0 here.")
    esc = results["B_misparsed"]["counts"].get("accepted_and_hop_correct", 0)
    log(f"\n  ESCAPES: {esc}/{results['B_misparsed']['n']} misparsed questions get an accepted, "
        f"gold-hop-count plan from the emitter. The grammar's number is 0.")

    b = results["B_misparsed"]; bn = b["n"]; bh = b["counts"].get("hop_count_correct", 0)
    log("\n" + "=" * 76)
    if slots and (hits / slots > 0.5):
        log("VERDICT: withheld — the control arm is contaminated (see canary).")
    elif bn and bh / bn >= 0.5:
        log(f"FLOOR: unexpectedly HIGH — {bh}/{bn} of the misparses get a correct hop")
        log("  count from an emitter never trained on this task. Cross-family transfer")
        log("  is real, and the harvest-trained run must beat THIS, not the grammar's 0.")
    elif bn and bh / bn <= 0.15:
        log(f"FLOOR: {bh}/{bn} — low, as expected for an untrained-on-this-task emitter.")
        log("  This is the number the harvest-trained emitter has to beat. Pre-register")
        log("  it now. The gate is: harvest-trained > this floor, on the same questions.")
    else:
        log(f"FLOOR: {bh}/{bn} — partial transfer. Record it; the harvest-trained run")
        log("  is graded against this, not against zero.")
    log("=" * 76)

    tag = a.tag or pathlib.Path(a.gguf).stem
    out = {"gguf": a.gguf, "arms": results, "wall_s": time.perf_counter() - t0, "rows": rows}
    (LOGS / f"exp_r3_emitter_floor_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"exp_r3_emitter_floor_{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {time.perf_counter()-t0:.1f}s | wrote exp_r3_emitter_floor_{tag}.json")


if __name__ == "__main__":
    main()
