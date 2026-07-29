"""H-G1 — cost out the next fixed-vocab step (128k-256k BPE) with a real measurement.

The trained-tokenizer history (19,947 -> 32,000 -> 65,536) proves the pipeline;
its reports record corpus/subsample sizes but NO wall-clock. H-G1's validate
clause asks for a costing of the next step before committing to it. So: train
byte-level BPE (the same HF `tokenizers` machinery the 65k build used) on a
measured slice of the same kind of corpus, at 65,536 and 131,072 vocab, and
report wall-clock + throughput + the scaling ratio. The point is an order-of-
magnitude anchor ("hours, not weeks, on this machine"), not a production build.

Corpus: first 200MB of cubby-lm's corpus_hf.txt (real training text).
Standalone; writes tokenizer JSONs to the job temp dir, not the repo.
"""
from __future__ import annotations

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tokenizers import ByteLevelBPETokenizer  # noqa: E402

CORPUS = r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\trunk_torch\data\corpus_hf.txt"
TMP = os.environ.get("CLAUDE_JOB_DIR", os.environ.get("TEMP", ".")) + r"\tmp"
SLICE_MB = 200


def main():
    os.makedirs(TMP, exist_ok=True)
    slice_path = os.path.join(TMP, f"g1_slice_{SLICE_MB}mb.txt")
    if not os.path.exists(slice_path):
        with open(CORPUS, "rb") as f, open(slice_path, "wb") as out:
            out.write(f.read(SLICE_MB * 1_000_000))
    sz = os.path.getsize(slice_path) / 1e9

    print("=" * 76)
    print(f"H-G1  BPE build costing  ({SLICE_MB}MB slice of corpus_hf.txt,"
          f" HF tokenizers {__import__('tokenizers').__version__})")
    print("=" * 76)
    results = {}
    for vocab in (65_536, 131_072):
        tok = ByteLevelBPETokenizer()
        t0 = time.time()
        tok.train(files=[slice_path], vocab_size=vocab, min_frequency=2)
        wall = time.time() - t0
        results[vocab] = wall
        out = os.path.join(TMP, f"g1_bbpe{vocab//1024}k.json")
        tok.save(out)
        print(f"  vocab {vocab:>7,}: {wall:7.1f}s  "
              f"({sz/wall*3600:.1f} GB/h at this vocab)")

    r = results[131_072] / results[65_536]
    print(f"\n[costing]")
    print(f"  vocab 65k->131k cost ratio on identical data: {r:.2f}x")
    # the 65k production build subsampled ~5M lines (~6.3GB) from 115GB
    est_hours = results[131_072] * (6.3 / sz) / 3600
    print(f"  extrapolated 131k build on the production-size subsample"
          f" (~6.3GB, linear-in-data assumption): ~{est_hours:.1f}h"
          f" single-machine CPU")
    print("  => the next fixed-vocab step is an HOURS-scale, zero-new-tooling"
          " job. The real work is choosing/pinning the subsample mix (H-G3)"
          " — note the 65k build's own report shows the NSFW-labeled sources"
          " contributed ~136k lines to tokenizer training.")


if __name__ == "__main__":
    main()
