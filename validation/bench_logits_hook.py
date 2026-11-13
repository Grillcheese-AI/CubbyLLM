"""Order 0071: what grilly2's logits hook costs a decode step of the 450M on the home card.

Arms, run interleaved (one round = every arm once, in order), greedy, batch 1, no EOS:
  none      -- generate() as the seats call it today;
  identity  -- a processor that returns its input (the hook's own cost: the penalty kernel apart from the sampling
               kernel, the token history kept, one Python call);
  allow1k   -- AllowedTokensLogitsProcessor over 1,000 ids (a mask built once, added each step);
  allow18k  -- the same over 18,000 ids;
  prefix1k  -- prefix_allowed_tokens_fn returning 1,000 ids (a mask built on the device every step; informative).
A step's time = (time to write 1 + N tokens - time to write 1) / N: the prompt's pass cancels. The median over rounds.
Line (pre-registered in the order): identity adds at most 2% a step. Kill: more than 10%.

  python validation/bench_logits_hook.py --export <the 450M's grilly export> [--rounds 5] [--new 64] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.ops import grilly_seats  # noqa: E402

__wiring__ = "STANDALONE"


def arms(vocab: int):
    from grilly.infer.logits_process import AllowedTokensLogitsProcessor
    rng = random.Random(0)
    a1k, a18k = sorted(rng.sample(range(vocab), 1000)), sorted(rng.sample(range(vocab), 18000))
    return {"none": {}, "identity": {"logits_processor": [lambda i, s: s]},
            "allow1k": {"logits_processor": [AllowedTokensLogitsProcessor(a1k)]},
            "allow18k": {"logits_processor": [AllowedTokensLogitsProcessor(a18k)]},
            "prefix1k": {"prefix_allowed_tokens_fn": lambda b, s: a1k}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = grilly_seats.grilly()
    model, cfg = grilly_seats.load_export(a.export)
    vocab = int(cfg["vocab_size"])
    prompt = g.tensor([[(i * 7919) % vocab for i in range(32)]])
    kinds = arms(vocab)

    def run(kw, n):
        t = time.perf_counter()
        with g.no_grad():
            model.generate(prompt, max_new_tokens=n, eos_token_id=None, **kw).tolist()
        return time.perf_counter() - t

    for kw in kinds.values():                                # warm every path once (pipelines, masks)
        run(kw, 4)
    steps = {k: [] for k in kinds}
    for r in range(a.rounds):
        for k, kw in kinds.items():
            one, full = run(kw, 1), run(kw, 1 + a.new)
            steps[k].append((full - one) / a.new * 1000)
        print(f"round {r + 1}: " + ", ".join(f"{k} {v[-1]:.2f} ms" for k, v in steps.items()), flush=True)
    med = {k: statistics.median(v) for k, v in steps.items()}
    over = {k: (med[k] / med["none"] - 1) * 100 for k in kinds if k != "none"}
    verdict = "met" if over["identity"] <= 2 else "killed" if over["identity"] > 10 else "missed, under the kill"
    g2 = Path(grilly_seats.grilly().__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(g2), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    out = {"order": "0071", "what": "per-step decode time of the 450M with grilly2's logits hook", "model": "base450m (grilly export)",
           "vocab": vocab, "prompt_tokens": 32, "new_tokens": a.new, "rounds": a.rounds, "batch": 1, "greedy": True,
           "grilly2": head, "machine": platform.platform(), "step_ms_each_round": steps, "step_ms_median": med,
           "added_percent_vs_none": over, "line": "identity <= 2%", "kill": "identity > 10%", "verdict": verdict}
    print(json.dumps({k: out[k] for k in ("step_ms_median", "added_percent_vs_none", "verdict")}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
