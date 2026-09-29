"""make_base_cache — a smaller token cache for the 450M base run (numpy only).

The 450M run draws ~3B tokens; the pinned cache is 17.74B (71 GB of uint32).
Uploading and staging all of it to a Colab VM costs hours for data the run
never reads. This copies, per shard:

* a fraction ``--frac`` of its training region as evenly spaced 1M-token
  blocks (so every file concatenated into the shard is represented), and
* its held-out tail, byte for byte: the same ``min(val_tokens, 5%)`` tail
  ``train_base.Corpus`` scores as validation. ``_holdout.json`` records each
  tail's length so the subset validates on exactly the original tail and
  never trains on it, and ``_sizes.json`` the original training sizes, so the
  data mix is the one the full cache would give.

Shards under ``--keep-small`` tokens are copied whole. The table printed at
the end shows how many epochs of each shard a run of ``--run-tokens`` would
draw under train_base's default mix (keep every row well under ~4).

  python validation/make_base_cache.py --src <token_cache> --dst <drive>/cubbyllm/token_cache_base

A LATER subset for a continued run takes its blocks only from the gaps between
the blocks earlier subsets took, so the continued run reads tokens the earlier
one never saw. The held-out tails stay byte for byte the same, so validation is
comparable across runs. ``--exclude-frac F`` names an earlier subset by its
``--frac`` (the first base cache, built before ``_subset.json`` existed);
``--exclude-dir D`` reads the blocks an earlier subset recorded in its
``_subset.json``. Shards under ``--keep-small`` were copied whole before and are
again: they have no unseen blocks.

  python validation/make_base_cache.py --src <token_cache> --dst <drive>/cubbyllm/token_cache_cont \
      --frac 0.2 --exclude-frac 0.2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BLOCK = 1 << 20


def holdout(n: int, val_tokens: int) -> int:
    return min(int(val_tokens), int(0.05 * n))       # == train_base.Corpus


def spread(train: int, frac: float) -> list:
    """The evenly spaced 1M-token block starts a subset of ``frac`` takes."""
    k = max(1, int(np.ceil(frac * train / BLOCK)))
    return [int(i * (train - BLOCK) // max(k - 1, 1)) for i in range(k)]


def gap_blocks(train: int, used: list) -> list:
    """Whole-block starts inside [0, train) that overlap none of the blocks at ``used``."""
    out, pos = [], 0
    for s in sorted(used) + [train]:
        out.extend(range(pos, s - BLOCK + 1, BLOCK))
        pos = max(pos, s + BLOCK)
    return out


def pick(cands: list, k: int) -> list:
    """``k`` of ``cands``, evenly spread (all of them if there are fewer)."""
    k = min(k, len(cands))
    return [cands[i * (len(cands) - 1) // max(k - 1, 1)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of <source>.u32 shards")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--keep-small", type=int, default=200_000_000)
    ap.add_argument("--run-tokens", type=float, default=3.2e9)
    ap.add_argument("--exclude-frac", type=float, action="append", default=[],
                    help="an earlier subset of the same --src, by its --frac; take no block it took")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="an earlier subset's directory (its _subset.json); take no block it took")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    prev = []
    for d in a.exclude_dir:
        with open(os.path.join(d, "_subset.json"), encoding="utf-8") as f:
            prev.append(json.load(f)["shards"])
    hold_map, sizes, record = {}, {}, {}
    for p in sorted(glob.glob(os.path.join(a.src, "*.u32"))):
        name = os.path.splitext(os.path.basename(p))[0]
        src = np.memmap(p, dtype=np.uint32, mode="r")
        n = len(src)
        hold = holdout(n, a.val_tokens)
        train = n - hold
        out = os.path.join(a.dst, name + ".u32")
        note = ""
        if n <= a.keep_small or a.frac >= 1.0:
            starts, blk = [0], train
            if a.exclude_frac or prev:
                note = "  (whole again: no unseen blocks)"
        else:
            k = max(1, int(np.ceil(a.frac * train / BLOCK)))
            used = [s for f in a.exclude_frac for s in spread(train, f)]
            for p in prev:                             # a whole-shard record covers every block
                r = p.get(name, {"starts": [], "block": BLOCK})
                used += [s + j for s in r["starts"] for j in range(0, int(r["block"]), BLOCK)]
            if used:
                cands = gap_blocks(train, used)
                starts = pick(cands, k)
                note = f"  ({len(starts):,} fresh blocks of {len(cands):,} unseen)"
                if len(starts) < k:
                    note += f"  <-- wanted {k:,}: this shard has run out of unseen blocks"
            else:
                starts = spread(train, a.frac)
            blk = BLOCK
        record[name] = {"starts": starts, "block": blk}
        want = len(starts) * blk + hold
        if os.path.exists(out) and os.path.getsize(out) == 4 * want:
            print(f"  {name:16s} already built ({want:,} tokens)")
        else:
            with open(out + ".tmp", "wb") as f:
                for s in starts:
                    f.write(np.asarray(src[s:s + blk]).tobytes())
                f.write(np.asarray(src[train:]).tobytes())      # the original val tail
            os.replace(out + ".tmp", out)
            print(f"  {name:16s} {n:>14,} -> {want:>14,} tokens{note}", flush=True)
        hold_map[name] = hold
        sizes[name] = (n, want - hold)
    with open(os.path.join(a.dst, "_holdout.json"), "w", encoding="utf-8") as f:
        json.dump(hold_map, f, indent=1)
    # the mix must follow the ORIGINAL shard sizes: a subset that keeps small
    # shards whole would otherwise up-weight them by 1/frac
    with open(os.path.join(a.dst, "_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({k: n - hold_map[k] for k, (n, kept) in sizes.items()}, f, indent=1)
    # which blocks this subset took, so a later subset can avoid them (--exclude-dir)
    with open(os.path.join(a.dst, "_subset.json"), "w", encoding="utf-8") as f:
        json.dump({"frac": a.frac, "exclude_frac": a.exclude_frac, "exclude_dir": a.exclude_dir,
                   "shards": record}, f)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from train_base import DECAY, QUALITY, DECAY_FRAC
    except Exception:                                   # no torch here: same defaults
        QUALITY, DECAY, DECAY_FRAC = {}, {}, 0.2
    tot = {k: v[0] for k, v in sizes.items()}

    def probs(mult):
        w = {k: tot[k] * mult.get(k, 1.0) for k in tot}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}
    ps, pd = probs(QUALITY), probs(DECAY)
    st, dc = a.run_tokens * (1 - DECAY_FRAC), a.run_tokens * DECAY_FRAC
    print(f"\n{'shard':16s} {'subset train':>14s}  epochs of the subset in a {a.run_tokens/1e9:.1f}B run")
    for k, (n, kept) in sizes.items():
        ep = (st * ps[k] + dc * pd[k]) / kept
        print(f"{k:16s} {kept:>14,}  {ep:6.2f}{'  <-- raise --frac' if ep > 4 else ''}")
    gb = sum(4 * (kept + hold_map[k]) for k, (n, kept) in sizes.items()) / 1e9
    print(f"\ntotal {gb:.1f} GB in {a.dst}")


if __name__ == "__main__":
    main()
