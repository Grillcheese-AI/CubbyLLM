"""Order 0073, line 8: seconds per instruction at home. The reader's read as the host would run it: one
instruction at a time on grilly2, the answer held to the machine of its instruction through grilly2's logits hook
(``prefix_allowed_tokens_fn``: the ids the machine allows next), greedy. Timed on dev rows only; nothing is scored
and no answer is kept (the counted reads are Colab's).

A read's time is all of it: the prompt encoded, the machine built, every step of the model and of the grammar.
One untimed read comes first (the model's first call loads and compiles). Reported: seconds per instruction,
median and 95th percentile, by number of parts and over all the rows timed (dev has 2 to 4 parts). **Pass: a
median of at most 8 s for 2 to 4 parts.** Above round 2's 25 s the target is not served.

  python validation/exp_view_split_time.py --arm m|w [--own-text] --export <450M export> --tokenizer <json>
      --adapter <dir> [--rows <rows.jsonl>] [--n 60] [--seed 0] [--max-new 900] --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402
from cubbyllm.speech.view_machine import Machine  # noqa: E402

__wiring__ = "STANDALONE"

PASS_SECONDS, ROUND_2_SECONDS = 8.0, 25.0


def holder(reader: T.Reader, prompt_len: int, clock: dict):
    """The callback for grilly2's ``prefix_allowed_tokens_fn``, for one instruction: the ids written so far are
    pushed into its machine, and the ids it allows next are returned (``eos`` alone once the answer has ended)."""
    def allowed(_batch_id, input_ids) -> list:
        written = input_ids.tolist()                       # waits for the device: the model's time, not the grammar's
        t0 = time.perf_counter()
        for t in [int(x) for x in written[prompt_len:]][reader.n:]:
            reader.push(t)
        ids = sorted(reader.allowed()) or [reader.eos]
        clock["grammar"] = clock.get("grammar", 0.0) + time.perf_counter() - t0
        return ids
    return allowed


def time_one(trunk, role: str, vocab: T.Vocab, instruction: str, arm: str, own_text: bool, max_new: int) -> dict:
    """One instruction read: {"seconds", "grammar" (seconds of it in the grammar), "ids", "end", "whole"}."""
    t0, clock = time.perf_counter(), {}
    ids = trunk.tokenizer.encode(V.prompt(instruction)).ids
    reader = T.Reader(Machine(instruction, arm, own_text), vocab, trunk.eos)
    out = trunk.generate(role, [ids], max_new, prefix_allowed_tokens_fn=holder(reader, len(ids), clock))[0]
    seconds = time.perf_counter() - t0
    for t in out[reader.n:]:                               # what was written after the hook's last call
        if reader.end:
            break
        reader.push(int(t))
    answer = "".join(vocab.pieces[int(t)] for t in out if int(t) != trunk.eos)
    end = reader.end or "limit"
    whole = end == "finished" and "lines" in V.parse(answer, instruction, arm, own_text=own_text)
    return {"seconds": seconds, "grammar": clock.get("grammar", 0.0), "ids": sum(1 for t in out if int(t) != trunk.eos),
            "end": end, "whole": whole}


def pick(rows: list, n: int, seed: int, split: str = "dev") -> list:
    """`n` rows of a dev split, a seeded sample (its own mix of parts). ``split``: order 0077 times dev_real."""
    dev = [r for r in rows if r["split"] == split]
    return random.Random(seed).sample(dev, min(n, len(dev)))


def summary(timed: list) -> dict:
    """`timed`: [{"n_parts", "seconds", "grammar", "ids", "end", "whole"}] -> the line's numbers."""
    def stats(xs: list) -> dict:
        secs = sorted(x["seconds"] for x in xs)
        return {"rows": len(xs), "median seconds": round(statistics.median(secs), 2),
                "95th percentile seconds": round(secs[math.ceil(0.95 * len(secs)) - 1], 2),
                "median ids": int(statistics.median(x["ids"] for x in xs)),
                "ended": dict(Counter(x["end"] for x in xs)), "answers whole": sum(1 for x in xs if x["whole"])}
    by = {str(k): stats([x for x in timed if x["n_parts"] == k]) for k in sorted({x["n_parts"] for x in timed})}
    short, long = ([x for x in timed if lo <= x["n_parts"] <= hi] for lo, hi in ((2, 4), (5, 8)))
    out = {"all rows": stats(timed), "by number of parts": by, "2 to 4 parts": stats(short) if short else None,
           "5 to 8 parts": stats(long) if long else None,                    # order 0074's line 7
           "share of the time in the grammar": round(sum(x["grammar"] for x in timed) / max(1e-9, sum(x["seconds"] for x in timed)), 4)}
    med = out["2 to 4 parts"]["median seconds"] if short else None
    out["line 8"] = {"needs": f"a median of at most {PASS_SECONDS:g} s for 2 to 4 parts", "median seconds": med,
                     "passes": med is not None and med <= PASS_SECONDS, "served": med is not None and med <= ROUND_2_SECONDS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=V.ARMS)
    ap.add_argument("--own-text", action="store_true")
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="dev", choices=("dev", "dev_real"), help="dev_real: order 0077's real prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cubbyllm import local_paths
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073" / "rows.jsonl")
    sample = pick([json.loads(ln) for ln in open(rows_p, encoding="utf-8")], a.n, a.seed, a.split)
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"split": a.adapter})
    vocab = T.Vocab.of(trunk.tokenizer)
    time_one(trunk, "split", vocab, sample[0]["instruction"], a.arm, a.own_text, a.max_new)       # untimed: the first call
    timed = []
    for k, r in enumerate(sample, start=1):
        got = time_one(trunk, "split", vocab, r["instruction"], a.arm, a.own_text, a.max_new)
        timed.append(dict(got, n_parts=r["n_parts"]))
        print(f"  {k}/{len(sample)}: {r['n_parts']} parts, {got['ids']} ids, {got['seconds']:.1f} s ({got['end']})", flush=True)
    report = dict({"order": "0073", "what": "line 8: seconds per instruction at home, one at a time on grilly2, dev rows",
                   "arm": a.arm, "own_text": a.own_text, "adapter": Path(a.adapter).name, "export": Path(a.export).parent.name,
                   "rows": len(timed), "split": a.split, "seed": a.seed, "first": "one untimed read before the timed ones"}, **summary(timed))
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["line 8"]), flush=True)
    print("timed:", a.out, flush=True)


if __name__ == "__main__":
    main()
