"""H-E19 generalisation split: is the 236-question loop score earned on problem STRUCTURES the emitter trained on, or
on new ones? The held-out pf questions are rewordings of GSM8K problems (program GSM<n>); this tags each held-out
question by whether its template's program name occurs anywhere in the emitter's training data, then re-reads every
existing step-loop result json split seen / unseen. CPU, no model.

    python validation/exp_he19_seen_split.py
"""
from __future__ import annotations

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "standin", "data", "out")
PAT = re.compile(r"program (GSM\d+)")


def main() -> None:
    held = [json.loads(l) for l in open(os.path.join(D, "pf_heldout_eval_w_slots.jsonl"), encoding="utf-8")]
    tmpl = {r["id"]: (PAT.findall((r.get("reference") or "") + (r.get("program") or "")) or ["?"])[0] for r in held}
    train = set()
    for name in ("emitter_sft_v12e_w_slots.jsonl", "emitter_sft_v12e_w_tg30_step.jsonl"):
        with open(os.path.join(D, name), encoding="utf-8") as f:
            for line in f:
                if '"split": "train"' in line:
                    train.update(PAT.findall(line))
    seen = {i: t in train for i, t in tmpl.items()}
    ns = sum(seen.values())
    lines = [f"held-out: {len(held)} questions, {len(set(tmpl.values()))} templates; {ns} questions on a template the "
             f"training data names ({len(train)} train templates), {len(held) - ns} on an unseen one", "",
             f"  {'loop run':<28} {'all':>7} {'seen':>7} {'unseen':>7}"]
    out = {"n_seen": ns, "n_unseen": len(held) - ns, "runs": {}}
    for p in sorted(glob.glob(os.path.join(HERE, "logs", "exp_he19_step_loop_*_held.json"))):
        res = json.load(open(p, encoding="utf-8")).get("results") or []
        res = [r for r in res if r.get("id") in tmpl]
        if not res:
            continue
        tag = os.path.basename(p)[len("exp_he19_step_loop_"):-len(".json")]
        acc = {}
        for k, keep in (("all", lambda r: True), ("seen", lambda r: seen.get(r["id"], False)),
                        ("unseen", lambda r: not seen.get(r["id"], False))):
            rr = [r for r in res if keep(r)]
            acc[k] = sum(bool(r.get("correct")) for r in rr) / max(1, len(rr))
        n_s = sum(seen.get(r["id"], False) for r in res)
        acc["n"], acc["n_seen"] = len(res), n_s
        out["runs"][tag] = acc
        lines.append(f"  {tag:<28} {acc['all']:>7.3f} {acc['seen']:>7.3f} {acc['unseen']:>7.3f}   (n {len(res)}: {n_s} seen)")
    print("\n".join(lines))
    base = os.path.join(HERE, "logs", "exp_he19_seen_split")
    open(base + ".log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(out, open(base + ".json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
