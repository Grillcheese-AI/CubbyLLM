"""build_0082_mix - order 0082's mix: S0's file (`emitter_sft_v12e_w_tg30_step.jsonl`) unchanged, plus H-E18's
program-first **plan** and **chain** train records (`emitter_sft_v12e_pf_slots.jsonl`, ids `plan-pf-*` / `chain-pf-*`).
No program-first arithmetic record goes in.

**The repeat is H-E18's mix's own.** In the program-first file each added record has its repeat (1) beside v12e's plan
and chain records at the repeats S0 gives them (byte-identical: `same_form`, so S0 repeats v12e's families exactly as
H-E18's mix did). Keeping each record's own repeat therefore gives the program-first records H-E18's share within
the plan and the chain families; the manifest reports it.

The added records are already in S0's slot form: `emitter_data.convert` made v12e's, v12e_w's and the program-first
slot files with the same options, and every plan and chain record S0 holds is byte-identical to its program-first-file
twin (checked here, `same_form`). What the conversion dropped is counted against the raw file (`--pf-raw`), never
guessed back.

Checks before writing (a record that fails one is left out and counted by reason):
- **the VM:** the filled program and the record's reference, both run in the real VM, give the same value (and a chain's
  value is its gold);
- **held-out (`pf_heldout_eval_w_slots.jsonl`, bucket 0, the other writer):** no shared prompt or question, no shared
  seed (normalized: the chain seeds and the plan seeds), no shared plan shape (the seed and its relations), and the
  seed's bucket is not 0. Seeds and relations come from the raw files' `pf` provenance (`--pf-raw`, `--held-raw`).

    python validation/sft/build_0082_mix.py --s0 <emitter_sft_v12e_w_tg30_step.jsonl> \\
        --pf <emitter_sft_v12e_pf_slots.jsonl> --pf-raw <emitter_sft_v12e_pf.jsonl> \\
        --held <pf_heldout_eval_w_slots.jsonl> --held-raw <pf_heldout_eval.jsonl> --out <emitter_sft_0082_pf.jsonl>

The data stays off the repo; the manifest (counts, SHA-256s) goes to `validation/logs/build_0082_mix_<date>.json`.

Wiring: STANDALONE (order 0082's data builder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

FAMILIES = ("plan", "chain")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: str):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            yield json.loads(ln)


def norm(s: str) -> str:
    from cubbyllm.reasoning.planner import normalize
    return normalize(str(s))


def bucket(seed: str) -> int:
    """`build_program_first.bucket` on the normalized seed: 0 is the held-out eval's bucket."""
    return int(hashlib.sha256(norm(seed).encode("utf-8")).hexdigest()[:8], 16) % 10


def provenance(path: str) -> dict:
    """id -> (normalized seed, normalized relations) from a raw program-first file's `pf` field."""
    out = {}
    for r in rows(path):
        pf = r.get("pf") or {}
        if r.get("task") in FAMILIES and "seed" in pf:
            out[r["id"]] = (norm(pf["seed"]), tuple(norm(x) for x in pf.get("relations", [])))
    return out


def real_vm(program: str):
    from cubbyllm.emit.format import vm_run
    return vm_run(program)


def vm_agrees(r: dict, vm) -> bool:
    from emitter_data import fill
    ok_f, got = vm(fill(r["program"], r["spans"]))
    ok_r, ref = vm(r["reference"])
    if not (ok_f and ok_r) or got is None or norm(got) != norm(ref):
        return False
    return r["task"] != "chain" or r.get("gold") is None or norm(got) == norm(r["gold"])


def build(s0: str, pf: str, pf_raw: str, held: str, held_raw: str, out: str, vm=real_vm) -> dict:
    s0_by_fam = Counter()
    s0_repeat: dict = {f: Counter() for f in FAMILIES}       # S0's own plan / chain repeats, reported
    s0_ids = {}
    for r in rows(s0):
        s0_by_fam[(r["task"], r["split"])] += 1
        if r["task"] in FAMILIES and r["split"] == "train":
            s0_repeat[r["task"]][int(r.get("repeat", 1) or 1)] += 1
            s0_ids[r["id"]] = r

    hp, hprompts, hseeds, hshapes = provenance(held_raw), set(), set(), set()
    for r in rows(held):
        hprompts.update({r["prompt"], r.get("question", "")})
        if r["id"] in hp:
            seed, rels = hp[r["id"]]
            hseeds.add(seed)
            hshapes.add((seed, rels))
    hprompts.discard("")
    tp = provenance(pf_raw)
    raw_added = Counter(r["task"] for r in rows(pf_raw) if r["id"] in tp and r["split"] == "train")

    kept, why, same_form, differs = [], Counter(), 0, 0
    for r in rows(pf):
        if r["split"] != "train":
            continue
        if r["id"] in s0_ids:                              # a v12e record: S0's own, compare the form
            same = r == s0_ids[r["id"]]
            same_form += same
            differs += not same
            continue
        if r["task"] not in FAMILIES:
            continue                                       # program-first arithmetic (and anything else): never added
        assert r["task"] != "arithmetic"
        if r["id"] not in tp:
            why[f"{r['task']}: no provenance"] += 1
            continue
        seed, rels = tp[r["id"]]
        if r["prompt"] in hprompts or r.get("question", "") in hprompts:
            why[f"{r['task']}: prompt shared with the held-out eval"] += 1
        elif seed in hseeds or bucket(seed) == 0:
            why[f"{r['task']}: seed shared with the held-out eval (or its bucket)"] += 1
        elif (seed, rels) in hshapes:
            why[f"{r['task']}: plan shape shared with the held-out eval"] += 1
        elif not vm_agrees(r, vm):
            why[f"{r['task']}: the VM disagrees"] += 1
        else:
            kept.append(r)                                 # its own repeat, as in H-E18's mix
    added = Counter(r["task"] for r in kept)
    added_rows = Counter()
    for r in kept:
        added_rows[r["task"]] += int(r.get("repeat", 1) or 1)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        with open(s0, encoding="utf-8") as g:
            for ln in g:
                f.write(ln if ln.endswith("\n") else ln + "\n")
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s0_rows = {f: sum(n * k for k, n in s0_repeat[f].items()) for f in FAMILIES}
    return {
        "what": "order 0082's mix: S0's file unchanged + the program-first plan and chain train records",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "s0": {"file": os.path.basename(s0), "sha256": sha256(s0),
               "records by family and split": {f"{t}/{s}": n for (t, s), n in sorted(s0_by_fam.items())}},
        "pf": {"file": os.path.basename(pf), "sha256": sha256(pf)},
        "pf_raw": {"file": os.path.basename(pf_raw), "sha256": sha256(pf_raw)},
        "held": {"file": os.path.basename(held), "sha256": sha256(held), "seeds": len(hseeds),
                 "plan shapes": len(hshapes), "prompts": len(hprompts)},
        "s0 plan and chain records identical in the program-first file": {"same": same_form, "differ": differs},
        "raw program-first train records (before slot conversion)": dict(raw_added),
        "dropped by the slot conversion": {f: raw_added[f] - sum(1 for r in rows(pf) if r["task"] == f and
                                                                  r["split"] == "train" and r["id"] in tp)
                                           for f in FAMILIES},
        "left out by the checks": dict(why),
        "added records": dict(added), "added train rows (each record's own repeat)": dict(added_rows),
        "S0's plan / chain train rows": s0_rows,
        "S0's repeats (records by repeat)": {f: dict(sorted(s0_repeat[f].items())) for f in FAMILIES},
        "program-first share within the family (H-E18's mix's)": {
            f: round(added_rows[f] / max(1, added_rows[f] + s0_rows[f]), 4) for f in FAMILIES},
        "arithmetic added": 0,
        "output": {"file": os.path.basename(out), "sha256": sha256(out),
                   "records": sum(s0_by_fam.values()) + len(kept)},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("s0", "pf", "pf-raw", "held", "held-raw", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "validation", "logs",
                                                       f"build_0082_mix_{time.strftime('%Y-%m-%d')}.json"))
    a = ap.parse_args(argv)
    m = build(a.s0, a.pf, a.pf_raw, a.held, a.held_raw, a.out)
    open(a.manifest, "w", encoding="utf-8").write(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
