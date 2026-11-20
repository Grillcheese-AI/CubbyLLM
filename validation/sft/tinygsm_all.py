"""tinygsm_all -- order 0078, step 1: TinyGSM, all of it (17 shards), converted, every kept program run in the VM.

Each shard goes through `import_tinygsm.convert` (``--p6``: with cubelang P6-5's ``//`` and ``%`` as steps; off, as in
September), with the eval overlap filter built once from: v12e's val split and the 236 program-first problems (the
emitter's reads), GSM8K test (1,319) and GSM8K's 329 dev problems (``$CUBBY_DATASETS/gsm8k``). Then **every** kept
program runs in the VM (one ``CubelangSession``) and must give the converter's value: a disagreement stops the run
(the order's kill) and is kept by id. Resumable: a shard whose manifest exists is skipped.

  CUBELANG_EXE=<cubelang with P6-3..7> python validation/sft/tinygsm_all.py [--p6] [--shards 0 1 ...]
      [--out-dir $CUBBY_DATA/tinygsm_0078] [--log validation/logs/tinygsm_0078_<date>.json]
-> per shard ``tinygsm_s<k>.jsonl`` + ``.manifest.json`` off-repo; one log (ids and counts only) under validation/logs.

Wired: STANDALONE (order 0078's data build).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE, os.path.join(ROOT, "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"


def eval_banned(data: str, datasets: str) -> tuple[set, dict]:
    """The shingles of every eval question the emitter is read on, and how many questions each source gave."""
    import import_tinygsm as T
    rd = lambda p: [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    src = {
        "v12e val": [r for r in rd(os.path.join(data, "emitter_sft_v12e_w_slots.jsonl")) if r.get("split") == "val"],
        "program-first 236": rd(os.path.join(data, "pf_heldout_eval_w_slots.jsonl")),
        "gsm8k test": rd(os.path.join(datasets, "gsm8k", "test.jsonl")),
        "gsm8k dev": [r for r in rd(os.path.join(datasets, "gsm8k", "train.jsonl")) if r.get("dev")],
    }
    banned: set = set()
    for rows in src.values():
        for r in rows:
            banned |= T.shingles(r.get("question") or r.get("prompt", ""))
    return banned, {k: len(v) for k, v in src.items()}


def vm_all(path: str, session) -> dict:
    """Every kept program in the VM: its value must be the converter's."""
    from validation.sft.build_emitter_sft import answer_fn, shim_isolver
    from cubbyllm.reasoning.slots import num_value
    n, agree, bad = 0, 0, []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        src = shim_isolver(r["program"])
        try:
            out = session.run(src, fn=answer_fn(src))
            v = num_value(str(out.get("result"))) if out.get("ok") else None
        except Exception as e:                           # noqa: BLE001 -- a VM stop is a disagreement, kept
            v, out = None, {"error": str(e)[:120]}
        n += 1
        if v is not None and abs(v - float(r["gold"])) <= 1e-6 * max(1.0, abs(v)):
            agree += 1
        else:
            bad.append({"id": r["id"], "gold": r["gold"], "vm": v if v is not None else out.get("error")})
    return {"n": n, "agree": agree, "disagree": len(bad), "first": bad[:20]}


def main() -> None:
    from cubbyllm import local_paths
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.bridges import vm_identity
    import import_tinygsm as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--p6", action="store_true", help="cubelang P6-5's // and % as steps")
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(17)))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    data, datasets = local_paths.value("CUBBY_DATA") or ".", local_paths.value("CUBBY_DATASETS") or "."
    out_dir = a.out_dir or os.path.join(data, "tinygsm_0078" + ("" if a.p6 else "_nop6"))
    os.makedirs(out_dir, exist_ok=True)
    if not a.p6:
        import ast
        T.OPS.pop(ast.FloorDiv, None)
        T.OPS.pop(ast.Mod, None)
    parquets = sorted(glob.glob(os.path.join(datasets, "tinygsm", "data", "train-*-of-00017.parquet")))
    banned, eval_counts = eval_banned(data, datasets)
    session, shards = cc.CubelangSession(), {}
    try:
        for k in a.shards:
            out = os.path.join(out_dir, f"tinygsm_s{k}.jsonl")
            man = out.replace(".jsonl", ".manifest.json")
            if not os.path.exists(man):
                t = time.time()
                m = T.convert(parquets[k], out, banned=banned, log=lambda s: print(f"s{k}{s}", flush=True))
                m["vm_all"] = vm_all(out, session)
                m["wall_s"] = round(time.time() - t, 1)
                json.dump(m, open(man, "w", encoding="utf-8"), indent=1)
                print(f"s{k}: kept {m['kept']:,} of {m['rows']:,}; vm {m['vm_all']['agree']:,}/{m['vm_all']['n']:,}", flush=True)
            shards[k] = json.load(open(man, encoding="utf-8"))
            if shards[k]["vm_all"]["disagree"]:
                print(f"s{k}: the VM disagrees on {shards[k]['vm_all']['disagree']} programs: stopped (the kill)")
                break
    finally:
        session.close()
    tot = lambda f: sum(f(m) for m in shards.values())
    rep = {"what": "order 0078 step 1: TinyGSM converted shard by shard, every kept program run in the VM",
           "date": _dt.date.today().isoformat(), "p6_forms": a.p6, "vm": vm_identity.identity(),
           "eval_questions": eval_counts, "shards": {f"s{k}": m for k, m in sorted(shards.items())},
           "total": {"rows": tot(lambda m: m["rows"]), "kept": tot(lambda m: m["kept"]),
                     "vm_runs": tot(lambda m: m["vm_all"]["n"]), "vm_disagree": tot(lambda m: m["vm_all"]["disagree"]),
                     "overlaps_dropped": tot(lambda m: m["rejects"].get("overlaps an eval question", 0))}}
    log = a.log or os.path.join(ROOT, "validation", "logs", f"tinygsm_0078{'' if a.p6 else '_nop6'}_{rep['date']}.json")
    json.dump(rep, open(log, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep["total"]), "->", log)


if __name__ == "__main__":
    main()
