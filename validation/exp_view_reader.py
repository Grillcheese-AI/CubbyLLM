"""Order 0067, the second reader's read (step 5's lines): the adapter reads held-out long instructions, its answer held to
the grammar (`cubbyllm/speech/view_grammar.py`, through grilly2's logits hook), and its program is run on each row's
replies against the generator's verdict.

Per row: the split (each check's span and kind against the row's own), the text no segment covers (flagged "asked":
nothing is dropped silently), the program compiled and run on the 8 replies (each reply's views bound by the host;
a reply too long for the VM or a program that does not compile or does not run is "asked", never a verdict).
Lines (0067's addendum): every sentence accounted for (0 dropped silently, kill); each part's span and kind for at least
90% of parts on held-out compositions; the join's agreement at least 95%, false passes at most 2% (kill above 5%);
5-8 parts at least 90% of the in-length agreement.

  python validation/exp_view_reader.py --export <450M export> --tokenizer <json> --adapter <view reader adapter>
      [--splits test,test_long] [--n 100] [--max-new 1800] [--out <json>] [--keep <answers.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_grammar as VG  # noqa: E402
from cubbyllm.speech import view_reader as VR  # noqa: E402

__wiring__ = "STANDALONE"

HEAD = "public function solve(mention: str): number {"


def bind(program: str, reply: str) -> str:
    """The views the program reads, bound from `reply` at the top of ``solve`` (raises `RV.TooLong`)."""
    views = sorted({m for m in re.findall(r"\bn_([a-z]+)\b", program) if m in RV.COLUMNS})
    return program.replace(HEAD, HEAD + "\n        " + "\n        ".join(RV.literals(reply, views)), 1)


def verdict(vm, program: str, reply: str):
    try:
        src = bind(program, reply)
    except RV.TooLong:
        return None, "a view too long for the VM"
    try:
        res = vm.run(src, fn="solve").get("result")
    except Exception as e:                                # does not compile or does not run: asked
        return None, f"the VM: {str(e)[:100]}"
    return (None, "the VM returned nothing") if res is None else (str(res).strip() in ("1", "1.0"), None)


def covered(instruction: str, segs: list) -> int:
    """Characters (not spaces) no segment covers."""
    mask = [False] * len(instruction)
    for s in segs:
        for k in range(s["start"], s["end"]):
            mask[k] = True
    return sum(1 for k, c in enumerate(instruction) if not mask[k] and not c.isspace())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--splits", default="test,test_long")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=1800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067" / "rows.jsonl")
    rows = [json.loads(ln) for ln in open(rows_p, encoding="utf-8")]
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    from cubbyllm.ops.grilly_seats import ranked_processor
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"programs": a.adapter})
    tk, eos, vm = trunk.tokenizer, trunk.eos, cc.CubelangSession()
    pieces: dict = {}

    def piece(t):
        if t not in pieces:
            pieces[t] = tk.decode([t])
        return pieces[t]
    report, kept = {"order": "0067", "what": "the second reader read on held-out instructions", "adapter": Path(a.adapter).name,
                    "vm": vm_identity.identity(), "splits": {}}, []
    for split in a.splits.split(","):
        pool = [r for r in rows if r["split"] == split]
        sample = random.Random(0).sample(pool, min(a.n, len(pool)))
        t = Counter()
        for r in sample:
            t0 = time.time()
            ids = tk.encode(VR.prompt(r["instruction"])).ids
            proc = ranked_processor(VG.chooser(VG.Grammar(r["instruction"]), piece, eos, len(ids)))
            out = trunk.generate("programs", [ids], a.max_new, logits_processor=proc)[0]
            answer = tk.decode([x for x in out if x != eos])
            t["seconds"] += time.time() - t0
            got = VR.parse(answer, r["instruction"])
            t["rows"] += 1
            gold = [(s["kind"], s.get("part"), s["start"], s["end"]) for s in r["segments"] if s["kind"] == "check"]
            t["parts"] += len(gold)
            if "error" in got:
                t["answers not parsed (all asked)"] += 1
                kept.append({"id": r["id"], "split": split, "answer": answer, "error": got["error"]})
                continue
            mine = {(s["kind"], s["part"], s["start"], s["end"]) for s in got["segments"]}
            t["parts with span and kind"] += sum(1 for g in gold if g in mine)
            t["characters left uncovered (flagged asked)"] += covered(r["instruction"], got["segments"])
            t["segments"] += len(got["segments"])
            t["segment kinds right"] += sum(1 for s in got["segments"] for g in r["segments"]
                                            if (s["start"], s["end"]) == (g["start"], g["end"]) and s["kind"] == g["kind"])
            for rep in r["replies"]:
                v, why = verdict(vm, got["program"], rep["text"])
                t["replies"] += 1
                if v is None:
                    t["asked: " + why.split(":")[0]] += 1
                    continue
                t["checked"] += 1
                t["agree"] += int(v == rep["held"])
                t["false passes"] += int(v and not rep["held"])
            kept.append({"id": r["id"], "split": split, "answer": answer})
        n = max(1, t["checked"])
        report["splits"][split] = {"counts": dict(t), "span_and_kind": round(t["parts with span and kind"] / max(1, t["parts"]), 4),
                                   "agreement": round(t["agree"] / n, 4), "false_passes": round(t["false passes"] / n, 4),
                                   "checked_share": round(t["checked"] / max(1, t["replies"]), 4),
                                   "seconds_per_instruction": round(t["seconds"] / max(1, t["rows"]), 1)}
        print(split, json.dumps(report["splits"][split]), flush=True)
    if "test" in report["splits"] and "test_long" in report["splits"]:
        a_in, a_long = report["splits"]["test"]["agreement"], report["splits"]["test_long"]["agreement"]
        report["long_over_in_length"] = round(a_long / a_in, 4) if a_in else None
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.keep:
        Path(a.keep).write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")


if __name__ == "__main__":
    main()
