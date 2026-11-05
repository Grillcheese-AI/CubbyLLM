"""validation/selfimprove/run.py -- order 0049 on the GPU, in three steps (the notebook's cells; Nick runs it):

    python validation/selfimprove/run.py base-read --records R --ckpt C --tokenizer T --adapter A --props P --out O
    python validation/selfimprove/run.py t1 --level K ...      (the stream with level K planted, its nights)
    python validation/selfimprove/run.py t3 --level K ...      (T3's arms on the planted gaps)
The stream is rebuilt from the records and the seed each time (``stream.build``); its held problems do not depend on
the level, and the base read's are checked against T1's. The numbers below are the ones written down before the run
(0049, Cowork's checks of 2026-11-04 and 2026-11-05): the gap rule's 20 failures in 2,000 requests; a probe trained on
up to 100 failed problems' right programs and read on up to 100 of the key's other problems, flagged at +5 points of
greedy right rate; a re-probe after 3 nights; practice's 8 attempts, up to 32 where 8 find fewer than 10 right. The rest are the harness's and
are recorded in each output (``args``).

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"


def args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("base-read", "t1", "t3"))
    for k in ("records", "ckpt", "tokenizer", "adapter", "props", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--level", type=int, default=0, help="the planted arithmetic level (from the base read)")
    ap.add_argument("--shuffled", default="arithmetic/steps=2", help="value-computing step rows only (0049, T1 #2)")
    ap.add_argument("--per-family", type=int, default=300)
    ap.add_argument("--held", type=int, default=50)
    ap.add_argument("--night", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--min-failures", type=int, default=20)
    ap.add_argument("--probe-train", type=int, default=100)
    ap.add_argument("--probe-held", type=int, default=100, help="the key's other problems the probe's rate is read on")
    ap.add_argument("--min-gain", type=float, default=0.05, help="the probe's rise in the held right rate (5 points)")
    ap.add_argument("--recheck", type=int, default=3)
    ap.add_argument("--probe-steps", type=int, default=10, help="at least; one pass over the right programs")
    ap.add_argument("--probe-min-right", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0, help="practice's sampling (top-k 50)")
    ap.add_argument("--tries", type=int, default=8, help="practice's attempts a problem")
    ap.add_argument("--max-tries", type=int, default=32, help="where --tries find fewer than --bootstrap-min right")
    ap.add_argument("--bootstrap-min", type=int, default=10)
    ap.add_argument("--practice-max", type=int, default=300)
    ap.add_argument("--min-right", type=int, default=20)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ul", type=float, default=1.0, help="T3: the unlikelihood term's weight")
    ap.add_argument("--latent-after", type=int, default=5000)
    ap.add_argument("--t3-problems", type=int, default=300)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    from cubbyllm.selfimprove.gaps import Rule
    from selfimprove import stream as S
    from selfimprove.loop import Loop, base_read
    from selfimprove.mark import Marker
    from selfimprove.trunk import Trunk
    a = args(argv)
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(x) for x in open(a.records, encoding="utf-8") if x.strip()]
    t0 = time.time()
    trunk = Trunk(a.ckpt, a.tokenizer, a.adapter, seed=a.seed)
    marker = Marker(recs, a.props)
    marker.selfcheck(recs)
    print(f"loaded {len(recs):,} records, the trunk and the VM in {time.time() - t0:.0f}s", flush=True)
    level = a.level or 2                                      # the held problems do not depend on the level
    header, stream, held = S.build(recs, level, a.shuffled, a.per_family, a.held, a.night, a.seed)
    held_hash = hashlib.sha256("\n".join(sorted(r["id"] for r in held)).encode()).hexdigest()[:16]
    stamp = {"args": vars(a), "held_ids_sha": held_hash}
    if a.step == "base-read":
        val = [r for r in recs if r["split"] == "val"]
        res = {**base_read(trunk, marker, held, val), **stamp, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(os.path.join(a.out, "base_read.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    if not a.level:
        raise SystemExit("--level: the planted level, from the base read (written into 0049 before night 1)")
    br = os.path.join(a.out, "base_read.json")
    if os.path.exists(br) and json.load(open(br))["held_ids_sha"] != held_hash:
        raise SystemExit("the held problems differ from the base read's: the records or the seed changed")
    rule = Rule(a.min_failures, a.probe_train, a.probe_held, a.min_gain, a.recheck)
    with open(os.path.join(a.out, "stream_header.json"), "w") as f:
        json.dump({**header, **stamp}, f, indent=1)
    lp = Loop(trunk, marker, stream, held, a.out, rule, a)
    if a.step == "t1":
        res = lp.run(stream)
    else:
        from selfimprove import t3
        res = t3.run(lp, stream, held, header["planted"], a.out, a)
    print(json.dumps({"step": a.step, "minutes": round((time.time() - t0) / 60, 1),
                      **({"gaps": res.get("gaps"), "routing": res.get("routing")} if a.step == "t1" else {})}, indent=1))


if __name__ == "__main__":
    main()
