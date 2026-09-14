"""emitter_sft_v13f: v13e (gen 2 + the reverse-built gen-3 records + the LLM wordings) + a capped sample
of the hdc records (`gen3_hdc.jsonl`, two- and three-hop free-text wordings, 2026-09-13) + the retrieval
shape (`profile_sft.jsonl`: who / what / where is X -> SEED + ASK, 2026-09-14).

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

Every added record keeps its own provenance and split; ids de-duplicate; the chain records get gen 2's
system line like the gen-3 merge. Caps keep the new shapes from drowning the rest: `--hdc-cap` questions
(both records of each, stratified by arm and hop count) and `--profile-cap` questions (stratified by ask
kind); every held question is kept for evaluation.

  python standin/data/build_v13f.py --copy-to "<drive folder>"
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, random, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "standin" / "data" / "out"


def stratified(by_q: dict, cap: int, key, rng: random.Random) -> list[str]:
    train = [q for q, rs in by_q.items() if rs[0]["split"] == "train"]
    if not cap or len(train) <= cap:
        return train
    strata: dict = collections.defaultdict(list)
    for q in train:
        strata[key(by_q[q][0])].append(q)
    for qs in strata.values():
        rng.shuffle(qs)
    keys = sorted(strata, key=str); picked: list[str] = []
    while len(picked) < cap and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < cap:
                picked.append(strata[k].pop())
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(OUT / "emitter_sft_v13e.jsonl"))
    ap.add_argument("--hdc", default=str(OUT / "gen3_hdc.jsonl"))
    ap.add_argument("--profile", default=str(OUT / "profile_sft.jsonl"))
    ap.add_argument("--hdc-cap", type=int, default=8000, help="hdc TRAIN questions kept (both records of each)")
    ap.add_argument("--profile-cap", type=int, default=6000, help="profile TRAIN questions kept")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT / "emitter_sft_v13f.jsonl"))
    ap.add_argument("--copy-to", default=None, help="a folder to copy the jsonl + manifest into (the Drive folder Colab reads)")
    a = ap.parse_args()
    t0 = time.perf_counter(); rng = random.Random(a.seed)
    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    system = next(r["system"] for r in base if r["task"] == "chain" and r.get("system"))
    ids = {r["id"] for r in base}
    merged = list(base); added = collections.Counter()

    def add(records: list[dict], cap: int, key, name: str) -> tuple[int, int]:
        by_q: dict = collections.defaultdict(list)
        for r in records:
            by_q[r["prompt"].split("\nFacts:")[0]].append(r)
        train_qs = stratified(by_q, cap, key, rng)
        held_qs = [q for q, rs in by_q.items() if rs[0]["split"] != "train"]
        for q in train_qs + held_qs:
            for r in by_q[q]:
                if r["id"] in ids:
                    continue
                merged.append(dict(r, system=system if r["task"] == "chain" else r.get("system"))); ids.add(r["id"]); added[name] += 1
        return len(train_qs), len(held_qs)

    hdc = [json.loads(l) for l in open(a.hdc, encoding="utf-8")]
    hdc_train, hdc_held = add(hdc, a.hdc_cap, lambda r: (r["provenance"].get("arm"), r.get("n_hop")), "hdc")
    prof = [json.loads(l) for l in open(a.profile, encoding="utf-8")]
    prof_train, prof_held = add(prof, a.profile_cap, lambda r: r.get("subtype"), "profile")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    m = {"version": "v13f", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_rev": rev, "base": os.path.basename(a.base),
         "hdc": {"file": os.path.basename(a.hdc), "cap_questions": a.hdc_cap, "train_questions": hdc_train, "held_questions": hdc_held, "records_added": added["hdc"]},
         "profile": {"file": os.path.basename(a.profile), "cap_questions": a.profile_cap, "train_questions": prof_train, "held_questions": prof_held, "records_added": added["profile"]},
         "n_records": len(merged), "by_task": dict(collections.Counter(r["task"] for r in merged)),
         "by_source": dict(collections.Counter(r["source"] for r in merged)),
         "by_split": dict(collections.Counter(r["split"] for r in merged)),
         "train_rows_after_repeat": sum(r.get("repeat", 1) for r in merged if r["split"] == "train"),
         "schema_note": "v13f = v13e unchanged + a capped, (arm, hops)-stratified sample of the hdc two/three-hop free-text records + "
                        "a capped, kind-stratified sample of the retrieval shape (who / what / where is X -> SEED + ASK); every "
                        "held question kept. The A/B bar for v13e (exp_r17 531 / 600 at 0 wrong) is pre-registered and untouched."}
    json.dump(m, open(a.out.replace(".jsonl", ".manifest.json"), "w", encoding="utf-8"), indent=1)
    (ROOT / "validation" / "logs" / "emitter_sft_v13f.manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"v13f: {len(merged):,} records (+{added['hdc']:,} hdc: {hdc_train:,} train q / {hdc_held:,} held; +{added['profile']:,} profile: "
          f"{prof_train:,} train q / {prof_held:,} held) {m['by_task']} | train rows {m['train_rows_after_repeat']:,} | {time.perf_counter() - t0:.0f}s -> {a.out}")
    if a.copy_to:
        dst = pathlib.Path(a.copy_to); dst.mkdir(parents=True, exist_ok=True)
        for p in (a.out, a.out.replace(".jsonl", ".manifest.json"), a.base, a.base.replace(".jsonl", ".manifest.json")):
            if os.path.exists(p):
                shutil.copy2(p, dst / os.path.basename(p))
        print(f"copied v13f + v13e (jsonl + manifest) -> {dst}")


if __name__ == "__main__":
    main()
