"""validation/selfimprove/stream.py -- order 0049's stream: train splits only, the planted gaps known by construction,
one family whose golds are shuffled, served in a fixed order, a night every --night requests.

Sources: 0048's audited records (``emitter_sft_v12e_wd.jsonl``, its train split): Wikidata relation-word programs (EN,
FR), the arithmetic levels (``steps=k``) and the emitter families (kernel, role_binding, chain, plan). Nothing from
0046's dev or test (0048's build already left those out; its manifest says so).
  planted   wd/en, wd/fr (the programs adapter was never trained on them: 0048's gap) and arithmetic at --level (the
            level one above the base adapter's; fixed before the run from the base's read, written in the stream's
            header)
  shuffled  a copy of --shuffled's records with their golds permuted among themselves: no rule to learn; must never
            be flagged. The original family leaves the stream (its held problems stay): the copy's prompts are the
            original's, so both would share one signature, and the copy must be the only family on its key. For an
            arithmetic family only the rows that compute a value are copied (no stop row, ``return $S<K>``, which the
            judge reads by its form, not its gold: the first T1's copy of level 1 was half stop rows), and the marker
            gives the copy no neutral values (another step's value would excuse the original's own answer)
  the rest  every other family, as it comes
Each request carries its family, whether it is planted or shuffled, the prompt, the gold and the reference program
(the signature is read from the reference, ``cubbyllm.selfimprove.signature``), and the night it falls in. Practice
problems and held-out problems per family are kept apart (``--held`` per family, never in the stream).

    python validation/selfimprove/stream.py --records <0048 jsonl> --level 4 --out $CUBBY_DATA_ROOT/selfimprove/stream.jsonl
-> the stream (one request a line, a header first) and <out>.held.jsonl

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

__wiring__ = "STANDALONE"


def family(r: dict) -> str:
    if r["task"] == "wd":
        return f"wd/{r.get('lang') or 'en'}"
    if r["task"] == "arithmetic":
        return "arithmetic/" + (r.get("subtype") or "").split(";")[-1].strip()
    return r["task"]


def build(records: list[dict], level: int, shuffled: str, per_family: int, held: int, night: int, seed: int) -> tuple:
    """(header, stream, held-out) from the train records."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in records:
        if r.get("split") == "train":
            by[family(r)].append(r)
    planted = {"wd/en", "wd/fr", f"arithmetic/steps={level}"}
    missing = planted - set(by)
    if missing:
        raise SystemExit(f"no train records for the planted families {sorted(missing)}")
    if shuffled not in by:
        raise SystemExit(f"no family {shuffled!r} to shuffle")
    stream, held_out = [], []
    for fam in sorted(by):
        rows = rng.sample(by[fam], len(by[fam]))
        held_out += [dict(r, family=fam, planted=fam in planted) for r in rows[:held]]
        if fam == shuffled:
            continue
        stream += [dict(r, family=fam, planted=fam in planted, shuffled=False) for r in rows[held:held + per_family]]
    pool = [r for r in by[shuffled] if not (r["task"] == "arithmetic" and r["program"].lstrip().startswith("return"))]
    src = rng.sample(pool, min(len(pool), per_family + held))
    golds = [r["gold"] for r in src]
    rng.shuffle(golds)
    stream += [dict(r, id=r["id"] + "#shuffled", gold=g, family=shuffled + "#shuffled", planted=False, shuffled=True)
               for r, g in zip(src, golds)][:per_family]
    rng.shuffle(stream)
    for i, r in enumerate(stream):
        r["night"] = i // night + 1
    header = {"what": "order 0049's stream", "seed": seed, "level": level, "planted": sorted(planted),
              "shuffled": shuffled + "#shuffled", "original_in_stream": False, "per_family": per_family, "held": held, "night": night,
              "requests": len(stream), "families": dict(collections.Counter(r["family"] for r in stream))}
    return header, stream, held_out


def main() -> None:
    from cubbyllm.local_paths import expand
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--level", type=int, required=True, help="the arithmetic level planted (one above the base's)")
    ap.add_argument("--shuffled", default="arithmetic/steps=2", help="the family copied with its golds shuffled")
    ap.add_argument("--per-family", type=int, default=300)
    ap.add_argument("--held", type=int, default=50, help="held-out problems per family (never in the stream)")
    ap.add_argument("--night", type=int, default=500, help="requests a night")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="$CUBBY_DATA_ROOT/selfimprove/stream.jsonl")
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(expand(a.records), encoding="utf-8") if line.strip()]
    header, stream, held = build(recs, a.level, a.shuffled, a.per_family, a.held, a.night, a.seed)
    out = expand(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": header}) + "\n")
        for r in stream:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out + ".held.jsonl", "w", encoding="utf-8") as f:
        for r in held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(header, indent=1), "->", out)


if __name__ == "__main__":
    main()
