"""Order 0067, line 3's held-out read (Nick, 2026-11-13: "then read the held-out kinds"): the reader on instruction
kinds no row was built from.

The kinds are the pre-registered split's held-out IFEval kinds (`validation/logs/rule_kinds_split_2026-11-12.json`).
Each instruction is IFEval's own wording of a rule, with arguments IFEval's checker draws from a seed
(`ifeval_check.py --describe`). The reader (an adapter, the instruction alone, slots constrained) reads each; where it
writes a rule, K replies built near the rule's edge (`rule_replies.held_out`) are labelled by IFEval's checker and
judged by the VM running the rule.
Line 3 (held-out): a rule written for at least half; where written, agreement at least 95%; a rule passing a reply
the checker fails at most 2% (kill above 5%). Read on the 7 kinds untouched by the reply world's starter relations;
forbidden words and bullet lists reported apart (Cowork's ruling). M-IFEval's five French kinds need its prompts,
which are not on this machine: not read.

  python validation/exp_rule_heldout.py --export <450M export> --tokenizer <tokenizer.json> --adapter <rules adapter>
      [--per 30] [--k 5] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import exp_rule_agree as AG  # noqa: E402
import rule_replies  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402
from cubbyllm.speech import rule_reader as rr  # noqa: E402

__wiring__ = "STANDALONE"

APART = ("keywords:forbidden_words", "detectable_format:number_bullet_lists")


def describe(specs: list[dict]) -> list[dict]:
    env = dict(os.environ, CUBBY_IFEVAL_CHECKER=local_paths.value("CUBBY_IFEVAL_CHECKER"),
               CUBBY_IFEVAL_NLTK=local_paths.value("CUBBY_IFEVAL_NLTK") or "")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "specs.jsonl", Path(d) / "out.jsonl"
        src.write_text("".join(json.dumps(s) + "\n" for s in specs), encoding="utf-8")
        subprocess.run([local_paths.value("CUBBY_IFEVAL_PYTHON"), str(HERE / "ifeval_check.py"), "--describe", str(src), str(dst)],
                       check=True, env=env)
        return [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    split = json.loads((HERE / "logs" / "rule_kinds_split_2026-11-12.json").read_text(encoding="utf-8"))
    specs = [{"id": f"{iid}#{s}", "instruction_id": iid, "seed": s} for iid in split["held_out"] for s in range(a.per)]
    items = describe(specs)
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"programs": a.adapter})
    tk, eos = trunk.tokenizer, trunk.eos

    def gen(ids, allowed, max_new):
        ids = ids[-trunk.max_ctx:]
        return trunk.generate("programs", [ids], max_new, prefix_allowed_tokens_fn=lambda b, s: allowed(s.tolist()[len(ids):]))[0]

    pairs, read = [], {}
    rng = random.Random(0)
    for it in items:
        iid = it["id"].split("#")[0]
        got = rr.read(it["instruction"], gen, lambda t: tk.encode(t).ids, eos)
        read[it["id"]] = {"kind": iid, "instruction": it["instruction"], "kwargs": it["kwargs"], "how": got["how"],
                          "plan": got["plan"], "surface": got["surface"]}
        if got["how"] == "written":
            for j, reply in enumerate(rule_replies.replies(iid, it["kwargs"] or {}, "en", rng, a.k)):
                pairs.append({"id": f"{it['id']}:{j}", "item": it["id"], "instruction_id": iid, "kwargs": it["kwargs"] or {},
                              "reply": reply})
    truth = AG.checker(pairs) if pairs else {}
    vm = cc.CubelangSession()
    run = lambda src, fn: vm.run(src, fn=fn)              # noqa: E731
    per = defaultdict(Counter)
    for it in read.values():
        per[it["kind"]].update({"instructions": 1, it["how"]: 1})
    misses = []
    for p in pairs:
        it = read[p["item"]]
        held = rp.run(it["plan"], p["reply"], run)["held"]
        want = truth[p["id"]]
        per[it["kind"]].update({"pairs": 1, "agree": int(held == want), "false passes": int(bool(held) and not want),
                                "checker passes": int(want)})
        if held != want and len(misses) < 30:
            misses.append({"kind": it["kind"], "instruction": it["instruction"][:140], "rule": it["surface"],
                           "reply": p["reply"][:120], "checker": want, "rule_held": held})

    def total(kinds):
        t = Counter()
        for k in kinds:
            t.update(per[k])
        n = max(1, t["pairs"])
        return {"instructions": t["instructions"], "written": t["written"],
                "written_share": round(t["written"] / max(1, t["instructions"]), 4), "pairs": t["pairs"],
                "agreement": round(t["agree"] / n, 4), "false_passes": round(t["false passes"] / n, 4),
                "checker_pass_share": round(t["checker passes"] / n, 4)}

    kinds7 = [k for k in split["held_out"] if k not in APART]
    out = {"order": "0067", "what": "line 3's held-out read: IFEval's held-out kinds, IFEval's own wording, the reader",
           "adapter": Path(a.adapter).name, "per_kind_instructions": a.per, "replies_per_rule": a.k,
           "line": {"written_share": ">= 0.5", "agreement": ">= 0.95", "false_passes": "<= 0.02 (kill > 0.05)"},
           "seven_untouched_kinds": total(kinds7), "apart": {k: total([k]) for k in APART},
           "per_kind": {k: dict(per[k], **total([k])) for k in split["held_out"]},
           "french_kinds": "not read: M-IFEval's prompts are not on this machine (licence unread)",
           "misses": misses, "vm": vm_identity.identity(),
           "reads": [dict(v, id=k) for k, v in read.items()]}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("seven_untouched_kinds", "apart")}, indent=1))
    for k in split["held_out"]:
        v = out["per_kind"][k]
        print(f"  {k:45s} written {v['written_share']:.2f}  agree {v['agreement']:.3f}  false {v['false_passes']:.3f}  "
              f"how {dict((h, per[k][h]) for h in ('written', 'asked', 'none') if per[k][h])}")


if __name__ == "__main__":
    main()
