"""build_emitter_sft — assemble the [stand-in] CubeLang-emitter SFT set.

Wired: STANDALONE (stand-in tooling; nothing in cubbyllm/ imports this).

The stand-in trunk (a small open model fine-tuned with Unsloth, served as GGUF)
exists so the serve stack can be built while the 2B trunk trains. Its first
job is emitting CubeLang programs from a question. This script builds that
SFT set from two places and re-verifies EVERY program through the real Rust
cubelang VM before it is allowed in:

  1. cubemind/sandbox/regen (2026-05/06, generator sources lost — .pyc only):
       arith_xl   7,205 GSM8K-derived arithmetic programs with gold
       svc        2,000 role-binding programs (bind ACTION/AGENT, remember)
       atier        600 reasoning kernels (decision/compare/loop/recall) w/ gold
       cubby_aug_v4.txt  38.6k programs = the above + 30k more role-binding
     Dialect drift: the current VM's `ISolver` requires parse(raw) and
     pure verify(input, output); the kernels carry both, the rest do not.
     A two-function shim is inserted (verify is a `return true` stub, exactly
     as the kernels ship it) — "verified" below means EXECUTES on the current
     VM and, when gold exists, MATCHES it. Audit 2026-08-30: arithmetic 300/300
     to gold after the shim, kernels 60/60, role-binding runs (no gold).
  2. validation/logs/cot_harvest_v3cf.jsonl: our 517 verified multi-hop chain
     programs (ISolve / recover dialect) — the programs the oracle actually
     needs; verified by construction (claimed-answer precision 0.9923).

Contamination rule (H-G4): EVERY GSM program in the aug corpus is a GSM8K
*test* question (4,228/4,228 — 1,057 unique) and arith_xl's other 6,148 are
exactly GSM8K *train*. The test set is our general-capability eval, so test
questions are EXCLUDED from the SFT set outright; the exclusion is counted
in the manifest and pinned by a test.

Caps: role-binding is 32k near-templated programs against 6k arithmetic,
600 kernels and 517 chains; it is capped (STANDIN_CAP_ROLE, default 4000,
deduplicated on instruction, sampled with a fixed seed) so the emitter does
not learn to bind ACTION/AGENT and little else. Chains are kept whole.

Output (gitignored): standin/data/out/emitter_sft.jsonl, one record per
program: {id, task, subtype, source, prompt, program, gold, vm_ok, vm_result,
gold_match, split}; and emitter_sft.manifest.json (counts per task/source/
split, exclusions, verification stats, sha256 of every input, git rev).
Split: deterministic 95/5 by sha256(prompt), stratified per task.

  python standin/data/build_emitter_sft.py [--no-verify] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

REGEN = os.environ.get("STANDIN_REGEN", r"C:\Users\grill\Documents\GitHub\cubemind\sandbox\regen")
HARVEST = os.environ.get("STANDIN_HARVEST", os.path.join(ROOT, "validation", "logs", "cot_harvest_v3cf.jsonl"))
GSM_TEST = os.environ.get("STANDIN_GSM_TEST", r"E:\datasets\misc\test_socratic.jsonl")
OUT_DIR = os.environ.get("STANDIN_OUT", os.path.join(ROOT, "standin", "data", "out"))
CAP_ROLE = int(os.environ.get("STANDIN_CAP_ROLE", "4000"))
VAL_FRAC = float(os.environ.get("STANDIN_VAL_FRAC", "0.05"))
SEED = 20260830

SHIM = ("    @external\n    public function parse(raw: str): Input { return raw; }\n\n"
        "    public pure function verify(input: Input, output: Output): bool { return true; }\n\n")
_SOLVE_RE = re.compile(r'(\n\s*@external\s*\n\s*public function solve\()')


# ── pure helpers (unit-pinned in tests/test_build_emitter_sft.py) ──────────
def shim_isolver(src: str) -> str:
    """Insert parse()+verify() before the @external solve() of a program that
    `implements ISolver` but lacks them. No-op when verify() exists or the
    program is not an ISolver (our ISolve chain programs are untouched)."""
    if "implements ISolver" not in src or re.search(r'function verify\(', src):
        return src
    return _SOLVE_RE.sub("\n" + SHIM + r"\1".lstrip("\n"), src, count=1)


def gsm_question(text: str) -> str:
    """The question part of a socratic-format record ('Question: ...\\nAnswer: ...')."""
    return text.split("\nAnswer:")[0].replace("Question: ", "").strip()


def split_of(prompt: str, val_frac: float = VAL_FRAC) -> str:
    h = int(hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def parse_aug_txt(path: str):
    """Yield (instruction, program) from cubby_aug_v4.txt."""
    instr, buf, prog, in_instr = None, [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[INSTRUCTION]"):
                in_instr, buf = True, []
                continue
            if line.startswith("[/INSTRUCTION]"):
                in_instr, instr, prog = False, "\n".join(buf).strip(), []
                continue
            if in_instr:
                buf.append(line.rstrip("\n"))
                continue
            if line.startswith("<|endofdoc|>"):
                if instr is not None and prog:
                    yield instr, "".join(prog)
                instr, prog = None, []
                continue
            if instr is not None:
                prog.append(line)


_HOP_RE = re.compile(r'public function hop_(\d+)\(')


def answer_fn(src: str) -> str:
    """The VM function whose result IS the program's answer. Chain programs
    (ISolve dialect) recover hop 1 in solve() and the final hop in hop_N();
    every other family answers from solve(). Found 2026-08-30: scoring every
    program on solve() dropped all 180 multi-hop chains as 'gold mismatch'."""
    hops = [int(h) for h in _HOP_RE.findall(src)]
    return f"hop_{max(hops)}" if hops else "solve"


def gold_matches(result, gold) -> bool | None:
    if gold is None:
        return None
    try:
        return abs(float(result) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return str(result).strip() == str(gold).strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── assembly ───────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect(limit: int | None):
    # E:\datasets\misc\test_socratic.jsonl carries {question, answer}; the I: copy
    # (socratic_test_text.jsonl) carries {text: "Question: ...\nAnswer: ..."}.
    gsm_test = {gsm_question(r["text"]) if "text" in r else r["question"].strip()
                for r in load_jsonl(GSM_TEST)}
    records, excluded = [], Counter()
    seen_prompts = set()

    def add(task, subtype, source, prompt, program, gold=None):
        key = (task, prompt.strip())
        if key in seen_prompts:
            excluded[f"dup:{task}"] += 1
            return
        seen_prompts.add(key)
        records.append({"task": task, "subtype": subtype, "source": source,
                        "prompt": prompt.strip(), "program": shim_isolver(program), "gold": gold})

    # arithmetic (GSM8K train only)
    for r in load_jsonl(os.path.join(REGEN, "multitask_v4_arith_xl.jsonl")):
        q = r["question"].strip()
        if q in gsm_test:
            excluded["gsm8k_test:arith_xl"] += 1
            continue
        add("arithmetic", f"steps={r.get('n_steps')}", "regen/arith_xl", q, r["cubelang_program"], r.get("gold"))

    # reasoning kernels
    for r in load_jsonl(os.path.join(REGEN, "multitask_v4_atier.jsonl")):
        add("kernel", r.get("subtype", ""), "regen/atier", r["text"], r["cubelang_program"], r.get("gold"))

    # role binding: svc jsonl + the txt's Evt/Ev programs, deduped, capped
    role_pool = []
    for r in load_jsonl(os.path.join(REGEN, "multitask_v4_svc.jsonl")):
        role_pool.append((r["text"], r["cubelang_program"], "regen/svc", r.get("realm_src", "")))
    n_txt_gsm_excluded = 0
    for instr, prog in parse_aug_txt(os.path.join(REGEN, "cubby_aug_v4.txt")):
        m = re.search(r'program ([A-Za-z]+)\d*', prog)
        fam = m.group(1) if m else ""
        if fam in ("Evt", "Ev"):
            role_pool.append((instr, prog, "regen/aug_txt", ""))
        elif fam == "GSM":
            n_txt_gsm_excluded += 1          # all GSM8K test (audit 2026-08-30) — excluded
    excluded["gsm8k_test:aug_txt"] = n_txt_gsm_excluded
    seen_instr = set()
    dedup = []
    for instr, prog, src, realm in role_pool:
        k = instr.strip()
        if k in seen_instr:
            excluded["dup:role_binding"] += 1
            continue
        seen_instr.add(k)
        dedup.append((instr, prog, src, realm))
    rng = random.Random(SEED)
    rng.shuffle(dedup)
    excluded["cap:role_binding"] = max(0, len(dedup) - CAP_ROLE)
    for instr, prog, src, realm in dedup[:CAP_ROLE]:
        add("role_binding", realm, src, instr, prog)

    # our verified multi-hop chains (ISolve dialect) — kept whole
    n_chain = 0
    for r in load_jsonl(HARVEST):
        if r.get("verified") and r.get("program_source"):
            add("chain", f"n_hop={r.get('n_hop')}", "cubbyllm/cot_harvest_v3cf", r["question"], r["program_source"],
                r.get("answer"))
            n_chain += 1

    if limit:
        records = records[:limit]
    for i, r in enumerate(records):
        r["id"] = f"{r['task']}-{hashlib.sha256((r['task'] + r['prompt']).encode('utf-8')).hexdigest()[:12]}"
        r["split"] = split_of(r["prompt"])
    return records, excluded, len(gsm_test)


def verify_all(records, verbose_every=500):
    from cubbyllm.bridges import cubelang_client as cc
    t0 = time.perf_counter()
    stats = Counter()
    for i, r in enumerate(records, 1):
        try:
            out = cc.run_program_proto(r["program"], fn=answer_fn(r["program"]))
            r["vm_ok"] = bool(out.get("ok"))
            r["vm_result"] = None if out.get("result") is None else str(out.get("result"))[:200]
            r["vm_error"] = None
        except Exception as e:                       # compile/run error -> excluded downstream
            r["vm_ok"], r["vm_result"], r["vm_error"] = False, None, str(e).splitlines()[0][:200]
        r["gold_match"] = gold_matches(r["vm_result"], r["gold"]) if r["vm_ok"] else None
        stats[f"{r['task']}:{'ok' if r['vm_ok'] else 'error'}"] += 1
        if r["gold_match"] is not None:
            stats[f"{r['task']}:{'gold_match' if r['gold_match'] else 'GOLD_MISMATCH'}"] += 1
        if i % verbose_every == 0:
            print(f"  verified {i}/{len(records)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return stats, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true", help="skip the VM pass (records carry vm_ok=None)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    print(f"regen {REGEN} | harvest {HARVEST} | gsm8k test {GSM_TEST} | cap role {CAP_ROLE} | val {VAL_FRAC}")
    records, excluded, n_gsm_test = collect(args.limit or None)
    print(f"collected {len(records)} records | excluded {dict(excluded)} | gsm8k test questions {n_gsm_test}")
    by_task = Counter(r["task"] for r in records)
    print(f"  by task: {dict(by_task)}")

    if args.no_verify:
        for r in records:
            r["vm_ok"], r["vm_result"], r["vm_error"], r["gold_match"] = None, None, None, None
        vstats, vwall = {}, 0.0
    else:
        print("=== verifying every program through the real cubelang VM ...", flush=True)
        vstats, vwall = verify_all(records)
        print(f"  done in {vwall:.0f}s: {dict(sorted(vstats.items()))}")

    kept = [r for r in records if r["vm_ok"] is None or (r["vm_ok"] and r["gold_match"] is not False)]
    dropped = Counter(r["task"] for r in records if r not in kept)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "emitter_sft.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    inputs = {p: sha256_file(p) for p in (os.path.join(REGEN, "multitask_v4_arith_xl.jsonl"),
                                          os.path.join(REGEN, "multitask_v4_atier.jsonl"),
                                          os.path.join(REGEN, "multitask_v4_svc.jsonl"),
                                          os.path.join(REGEN, "cubby_aug_v4.txt"), HARVEST, GSM_TEST)}
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_rev = "unknown"
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(), "git_rev": git_rev,
        "config": {"REGEN": REGEN, "HARVEST": HARVEST, "GSM_TEST": GSM_TEST, "CAP_ROLE": CAP_ROLE,
                   "VAL_FRAC": VAL_FRAC, "SEED": SEED, "verified": not args.no_verify, "limit": args.limit},
        "inputs_sha256": inputs,
        "n_records_collected": len(records), "n_records_kept": len(kept),
        "dropped_by_vm": dict(dropped), "excluded": dict(excluded),
        "by_task": dict(Counter(r["task"] for r in kept)),
        "by_source": dict(Counter(r["source"] for r in kept)),
        "by_split": {t: dict(Counter(r["split"] for r in kept if r["task"] == t)) for t in by_task},
        "verification": {"stats": dict(sorted(vstats.items())), "wall_s": vwall,
                         "note": ("vm_ok = executes on the current Rust cubelang VM (run-proto, strict) after the "
                                  "ISolver parse/verify shim; gold_match = result equals the record's gold when "
                                  "one exists. The shim's verify() is a `return true` stub, as the regen kernels "
                                  "ship it. Chain programs are ISolve-dialect and untouched by the shim.")},
        "contamination": {"rule": "GSM8K test questions (H-G4's eval) are excluded from every source",
                          "n_gsm8k_test_questions": n_gsm_test,
                          "excluded_arith_xl": excluded.get("gsm8k_test:arith_xl", 0),
                          "excluded_aug_txt_gsm_programs": excluded.get("gsm8k_test:aug_txt", 0)},
        "output": out_path, "output_sha256": sha256_file(out_path),
        "wall_s": time.perf_counter() - t0,
    }
    mp = os.path.join(OUT_DIR, "emitter_sft.manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nkept {len(kept)}/{len(records)} (dropped by VM: {dict(dropped)})")
    print(f"  by task/split: {manifest['by_split']}")
    print(f"wrote {out_path}\nwrote {mp} ({manifest['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
