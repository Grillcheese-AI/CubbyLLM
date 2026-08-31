"""probe_v3 — is the local GGUF the v3 emitter? Two discriminating probes.

Wired: STANDALONE (stand-in tooling; run from the repo root).

  (a) no-prefill: v2/v3 were trained with an EMPTY think block in the target,
      so the model closes it immediately (or never opens it); v1 reasons at
      length before answering.
  (b) invented-facts chain: v3's whole point is copying the prompt's Facts
      block into the bindings (chain text-EM 0.893). Facts about Zorblax the
      Painter cannot be in any corpus — if the program binds Fnordovia /
      Quuxville, the model is reading the prompt, not its memory: v3.

  python standin/scripts/probe_v3.py --gguf standin/models/emitter_maybe_v3.Q4_K_M.gguf --n-gpu-layers -1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from standin.emitter import LlamaCppEmitter  # noqa: E402

CHAIN_PROMPT = ("What is the capital of the homeland of Zorblax the Painter?\n"
                "Facts:\n"
                "- Fnordovia is the homeland of Zorblax the Painter\n"
                "- Quuxville is the capital of Fnordovia")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=2048)
    args = ap.parse_args()

    print(f"loading {args.gguf} (n_gpu_layers={args.n_gpu_layers}, n_ctx={args.n_ctx}) ...", flush=True)
    e = LlamaCppEmitter(args.gguf, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, prefill="")

    t0 = time.perf_counter()
    a = e.emit("What is 4 plus 3?", max_new_tokens=120)
    print(f"\n(a) no-prefill, {time.perf_counter() - t0:.1f}s — head of output:")
    print("   ", repr(a[:200]))
    thinks = "</think>" in a and len(a.split("</think>")[0].split()) > 8

    t0 = time.perf_counter()
    b = e.emit(CHAIN_PROMPT, max_new_tokens=300)
    print(f"\n(b) invented-facts chain, {time.perf_counter() - t0:.1f}s:")
    print(b[:700])
    binds = ("Fnordovia" in b) and ("Quuxville" in b)

    print("\n=== verdict ===")
    print(f"  reasons at length without prefill : {thinks}   (v1 trait)")
    print(f"  binds the prompt's invented facts : {binds}   (v3 trait)")
    if binds and not thinks:
        print("  -> this is the v3 emitter.")
    elif binds:
        print("  -> reads the prompt's facts (v3 data) but still thinks: likely v3 weights, judge by (b).")
    else:
        print("  -> does NOT copy the Facts block: this is v1 (or v2). Re-export the GGUF from the v3 Colab session.")


if __name__ == "__main__":
    main()
