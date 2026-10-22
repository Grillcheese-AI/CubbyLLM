"""build_gen2_partition — the gen-2 emitter SFT set: gen 1 + exactly the new signal.

Wired: STANDALONE (stand-in data; nothing in cubbyllm/ imports this).

Gen 1 (`emitter_sft_v8e.jsonl`) is the set emitter_v8e was trained on: the
grammar's verified chains (517 from cot_harvest_v3cf + 684 game chains), each
prompted as question + "Facts:" + the walked facts, plus arithmetic / kernel /
role-binding replay. exp_r3/exp_r7 (2026-09-11) showed that emitter leaving the
grammar's basin on two of three misparse forms, and the VM verifying some of
its plans on questions the grammar could not answer.

Gen 2 = gen 1, unchanged, plus TWO things and nothing else, so the contrast is
attributable:

  1. `chain` records from `cot_harvest_r7*.jsonl` -- chains the emitter
     planned, the grammar walked, and the VM verified. The first training
     records that did not come from the grammar. Included on the VM's word
     alone (`verified`), never filtered by gold: at serve time there is no
     gold, and the loop's premise is that the VM is the truth gate. Gold
     correctness is REPORTED in the manifest as a measurement of that gate.
     Upsampled (R7_MULT) because there are few.

  2. a `plan` task: prompt = the QUESTION ALONE (no facts -- a plan is
     proposed before there are facts), program = a CotPlan that binds the
     seed entity and the relation chain as symbols:

         use vsa;
         program CotPlan implements ISolve {
             public function solve(mention: str): str {
                 create frame: number;
                 bind frame, SEED, "felix joseph widder";
                 bind frame, HOP1, "country of citizenship";
                 bind frame, HOP2, "administrative territorial entity";
                 return recover(frame, SEED);
             }
         }

     Relations are the question's own words (H2_AWARD_RECEIVED stays "award
     received" -- exp_r7 refused 22 gold-hop plans whose role names had
     paraphrased the question). No objects: gen 1's v1 lesson was that a
     facts-free chain target teaches the emitter to bind facts it cannot
     know. One plan record per chain record (gen 1's and r7's).

The questions taken from r7 are written to `gen2_exclusions.json`. THE GEN-2
GATE MUST EXCLUDE THEM: exp_r3 / exp_r7 take `--exclude gen2_exclusions.json`
and drop them from arms B and C, so gen 2 is measured on the questions gen 1
could not verify -- transfer, not recall.

  python standin/data/build_gen2_partition.py \
      [--gen1 standin/data/out/emitter_sft_v8e.jsonl] \
      [--r7 validation/logs/cot_harvest_r7_v2.jsonl] [--r7-mult 6] \
      [--out standin/data/out/emitter_sft_v12e.jsonl]
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, subprocess, sys, time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

BIND = re.compile(r'bind\s+frame\s*,\s*(?P<role>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"(?P<obj>(?:[^"\\]|\\.)*)"\s*;')
ROLE = re.compile(r"^H(?P<hop>\d+)_(?P<rel>.+)$")
CONTROL_ROLE = "ABSENT_CTRL"


def roles_of(program: str) -> list[str]:
    """H-roles in hop order, deduplicated (every function repeats the binds)."""
    seen, out = set(), []
    for m in BIND.finditer(program or ""):
        r = m.group("role")
        if r in seen or r == CONTROL_ROLE or not ROLE.match(r):
            continue
        seen.add(r); out.append(r)
    out.sort(key=lambda r: int(ROLE.match(r).group("hop")))
    return out


def rel_text(role: str) -> str:
    return ROLE.match(role).group("rel").lower().replace("_", " ").strip()


def question_of(prompt: str) -> str:
    return prompt.split("\nFacts:")[0].strip()


def seed_entity(question: str, rel1: str, normalize, parse_question) -> str | None:
    """Text after the LAST '<rel1> of ' in the normalized question; else the last
    ' of '-segment of the grammar's tail; else None (the record is skipped)."""
    q = normalize(question); key = f"{normalize(rel1)} of "
    i = q.rfind(key)
    if i >= 0 and q[i + len(key):].strip():
        return q[i + len(key):].strip()
    p = parse_question(question)
    if p is not None and " of " in p.tail:
        return normalize(p.tail.rsplit(" of ", 1)[1])
    return None


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen1", default=str(ROOT / "standin" / "data" / "out" / "emitter_sft_v8e.jsonl"))
    ap.add_argument("--r7", default=str(ROOT / "validation" / "logs" / "cot_harvest_r7_v2.jsonl"))
    ap.add_argument("--r7-mult", type=int, default=6, help="train-only upsampling of the r7 chains")
    ap.add_argument("--plan-mult", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "emitter_sft_v12e.jsonl"))
    a = ap.parse_args()
    t0 = time.perf_counter()

    from cubbyllm.reasoning.planner import normalize, parse_question

    gen1 = [json.loads(l) for l in open(a.gen1, encoding="utf-8")]
    system = next(r["system"] for r in gen1 if r["task"] == "chain")
    gen1_ids = {r["id"] for r in gen1}
    gen1_questions = {question_of(r["prompt"]) for r in gen1 if r["task"] == "chain"}
    print(f"gen 1: {len(gen1)} records, {dict(Counter(r['task'] for r in gen1))}")

    # ---- 1. r7 chains: the VM's word, gold reported ----------------------------
    r7 = [json.loads(l) for l in open(a.r7, encoding="utf-8")] if os.path.exists(a.r7) else []
    r7_new = [r for r in r7 if r["question"] not in gen1_questions]     # arm A rows are gen-1 questions already
    r7_recs, gold_correct = [], Counter()
    for r in r7_new:
        facts = [t["fact"] for t in r.get("trace", []) if t.get("fact")]
        prompt = r["question"] + ("\nFacts:\n" + "\n".join(f"- {f}" for f in facts) if facts else "")
        gold_correct[bool(r.get("correct"))] += 1
        r7_recs.append({"task": "chain", "subtype": f"n_hop={r['plan']['n_hop']}", "source": "cubbyllm/cot_harvest_r7",
                        "prompt": prompt, "program": r["program_source"], "gold": r.get("gold_answer"),
                        "split": "train", "system": system, "state": None, "repeat": a.r7_mult,
                        "vm_ok": True, "vm_result": r.get("answer"), "vm_error": None,
                        "gold_match": bool(r.get("correct")), "arm": r.get("arm")})
    print(f"r7: {len(r7)} verified records, {len(r7_new)} on questions outside gen 1 -> chain records x{a.r7_mult}; "
          f"gold-correct {gold_correct[True]}/{len(r7_new)} (reported, not filtered)")

    # ---- 2. plan task: question only -> CotPlan ---------------------------------
    plan_recs, skipped = [], Counter()
    for r in [x for x in gen1 if x["task"] == "chain"] + r7_recs:
        roles = roles_of(r["program"])
        if not roles:
            skipped["no_roles"] += 1; continue
        rels = [rel_text(x) for x in roles]
        q = question_of(r["prompt"])
        seed = seed_entity(q, rels[0], normalize, parse_question)
        if seed is None:
            skipped["no_seed_entity"] += 1; continue
        plan_recs.append({"task": "plan", "subtype": f"n_hop={len(rels)}", "source": r["source"],
                          "prompt": q, "program": cot_plan(seed, rels), "gold": None,
                          "split": r["split"], "system": system, "state": None,
                          "repeat": a.plan_mult if r["split"] == "train" else 1,
                          "vm_ok": None, "vm_result": None, "vm_error": None, "gold_match": None})
    print(f"plan: {len(plan_recs)} records ({dict(skipped)} skipped)")

    # ---- assemble -------------------------------------------------------------------
    out = list(gen1)
    for r in r7_recs + plan_recs:
        r["id"] = rec_id(r["task"], r["prompt"], r["program"])
        if r["id"] in gen1_ids:
            continue
        out.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    excl = {"questions": sorted({r["question"] for r in r7_new}),
            "note": "questions whose VM-verified emitter chains entered gen-2 training; exp_r3/exp_r7 --exclude this file"}
    excl_path = os.path.join(os.path.dirname(a.out), "gen2_exclusions.json")
    json.dump(excl, open(excl_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    manifest = {
        "version": "v12e", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_rev": rev,
        "gen1": a.gen1, "gen1_sha256": sha(a.gen1), "r7": a.r7, "r7_sha256": sha(a.r7) if os.path.exists(a.r7) else None,
        "n_records": len(out), "by_task": dict(Counter(r["task"] for r in out)),
        "by_split": {t: dict(Counter(r["split"] for r in out if r["task"] == t)) for t in {r["task"] for r in out}},
        "train_rows_after_repeat": sum(r.get("repeat", 1) for r in out if r["split"] == "train"),
        "r7_chains": len(r7_recs), "r7_mult": a.r7_mult, "r7_gold_correct": gold_correct[True],
        "r7_gold_wrong": gold_correct[False], "plan_records": len(plan_recs), "plan_skipped": dict(skipped),
        "exclusions": excl_path, "n_excluded_questions": len(excl["questions"]),
        "schema_note": "gen 2 = gen 1 unchanged + r7 VM-verified chains (train, x r7_mult) + plan task "
                       "(question-only prompt -> CotPlan binding SEED and HOPk relation symbols). "
                       "The gate for gen 2 excludes the r7 questions.",
        "output": a.out, "output_sha256": sha(a.out), "wall_s": round(time.perf_counter() - t0, 1),
    }
    mp = a.out.replace(".jsonl", ".manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1)
    print(f"gen 2: {len(out)} records {manifest['by_task']} | train rows after repeat {manifest['train_rows_after_repeat']}")
    print(f"wrote {a.out}\nwrote {mp}\nwrote {excl_path} ({len(excl['questions'])} questions to exclude from the gate)")


if __name__ == "__main__":
    main()
