"""validation/head_lab_generate.py -- order 0068, step 2f: route 5 wired into grilly2's ``generate``, timed there against a
full 32k head and the full 128k head, on step 2e's prompts.

Step 2e timed route 5 in a hand-written decode loop. Here the seats' own call runs: grilly2's ``model.generate``,
greedy, batch 1, with ``grilly_seats.attach_route`` setting ``model.shortlist`` for route 5 and the full head when it
is not set. A streamer stamps each token as it reaches the host; a step is the time between two stamps after the first
token (the prompt's pass is not in it), the first ``WARM`` of each run left out.
  full128   the trained 128k head          head32k   ``lm_head`` replaced by its first 32,000 rows
  route128  route 5 over the 128k head (``attach_route``: K 8,192, N 2,048, rank bound 256, as 2d and 2e)
Also, untimed: route 5 in a hand loop (``hidden`` and ``ShortlistHead.argmax``, as 2e decoded), with the full head's
token at every step; the wired ``generate`` must write the hand loop's tokens.

    python validation/head_lab_generate.py --export <base export> --dumps <head_shortlist folder> --cache <token cache>
-> validation/logs/head_lab_generate_<date>_<grilly2 commit>.json

Wired: STANDALONE (a timing; order 0068's report reads it). Imports grilly (grilly2), never torch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

__wiring__ = "STANDALONE"

KINDS = ("full128", "head32k", "route128")
WARM = 2                                           # steps of each run left out


class Clock:
    """A ``generate`` streamer that keeps the time each put reaches it (the prompt's put first)."""

    def __init__(self) -> None:
        self.stamps: list = []

    def put(self, ids) -> None:
        self.stamps.append(time.perf_counter())

    def end(self) -> None:
        pass


def step_times(stamps: list, warm: int = WARM) -> list:
    """Milliseconds from each token to the next, after the first token (stamps[0] is the prompt's put, stamps[1] the
    first token, after the prompt's pass), the first ``warm`` steps left out."""
    return [(b - a) * 1e3 for a, b in zip(stamps[1:-1], stamps[2:])][warm:]


def first_difference(a: list, b: list):
    """The first position where two token lists differ, or None."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def main() -> None:
    import grilly as g
    from head_dump import SHARDS
    from head_lab_decode import _commit, _stats, prompt_tokens
    from head_shortlist_eval import head_rows

    from cubbyllm.ops.grilly_seats import attach_route, load_export
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index", default="index_router_succ.npz")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ranks", type=int, default=256)
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    F = g.nn.functional
    model, cfg = load_export(a.export)
    full = model.lm_head
    W = full.weight
    v, dim = (int(s) for s in W.shape)
    head32 = g.nn.Linear(dim, 32000, bias=False)
    with g.no_grad():
        head32.weight.copy_(g.from_numpy(np.ascontiguousarray(head_rows(a.export)[:32000])))
    route = attach_route(model, os.path.join(a.dumps, a.index), near=a.n, context=a.k, ranks=a.ranks)
    if route is None:
        raise SystemExit("this grilly2 has no shortlist decode (grilly2 branch cowork/shortlist-generate)")

    def gen(kind, prompt, steps):
        model.lm_head = head32 if kind == "head32k" else full
        model.shortlist = route if kind == "route128" else None
        clock = Clock()
        out = model.generate(g.tensor([prompt.tolist()]), max_new_tokens=steps, streamer=clock)
        model.lm_head, model.shortlist = full, None
        return out.tolist()[0][len(prompt):], step_times(clock.stamps)

    def hand(prompt, steps):
        """Route 5 decoding on its own choices, as 2e did, and the full head's token at each step."""
        cache = model.make_cache(1, len(prompt) + steps)
        x = model.hidden(g.tensor([prompt.tolist()]), past_key_values=cache, logits_to_keep=1)
        prev, toks, same = int(prompt[-1]), [], 0
        for i in range(steps):
            tok = route.argmax(x, prev)
            same += int(g.argmax(F.linear(x, W).reshape(1, -1), dim=-1).numpy().reshape(-1)[0]) == tok
            toks.append(tok)
            prev = tok
            if i < steps - 1:
                x = model.hidden(g.tensor([[tok]]), past_key_values=cache, logits_to_keep=1)
        return toks, same

    prompts = prompt_tokens(a.cache, SHARDS, a.prompts, a.prompt_len)
    times = {k: [] for k in KINDS}
    tokens = {k: {} for k in KINDS}
    with g.no_grad():
        log("warm-up")
        for k in KINDS:
            gen(k, prompts[0][1], 8)
        route.picks = route.short_picks = 0
        for i, (name, toks) in enumerate(prompts):
            for k in KINDS[i % len(KINDS):] + KINDS[:i % len(KINDS)]:
                tokens[k][name], ts = gen(k, toks, a.steps)
                times[k] += ts
            log(f"{name}: " + ", ".join(f"{k} {statistics.median(times[k]):.3f}" for k in KINDS))
        picks, shorts = route.picks, route.short_picks
        log("hand loop (untimed)")
        checks = []
        for name, toks in prompts:
            ht, same = hand(toks, a.steps)
            checks.append({"shard": name, "generate_writes_the_hand_loops_tokens": tokens["route128"][name] == ht,
                           "hand_loop_same_token_as_full_head": same,
                           "route_generate_first_difference_from_full128": first_difference(
                               tokens["route128"][name], tokens["full128"][name])})
    st = {k: _stats(ts) for k, ts in times.items()}
    res = {"what": "order 0068 step 2f: route 5 wired into grilly2's generate (greedy, batch 1), timed",
           "date": _dt.date.today().isoformat(), "grilly2": _commit(g), "machine": platform.platform(),
           "batch": 1, "precision": "fp32", "vocab": v, "hidden": dim, "k": a.k, "n": a.n, "ranks": a.ranks,
           "prompts": [n for n, _ in prompts], "prompt_len": a.prompt_len, "steps": a.steps,
           "warm_steps_left_out": WARM, "generate_step": st,
           "route_picks_timed": picks, "route_short_picks_timed": shorts,
           "hand_loop": {"same_token_as_full_head": sum(c["hand_loop_same_token_as_full_head"] for c in checks),
                         "steps": a.steps * len(checks), "per_prompt": checks}}
    res["lines"] = {"route128_at_most_head32k": st["route128"]["p50_ms"] <= st["head32k"]["p50_ms"],
                    "generate_writes_the_hand_loops_tokens": all(c["generate_writes_the_hand_loops_tokens"]
                                                                 for c in checks)}
    path = os.path.join(HERE, "logs", f"head_lab_generate_{res['date']}_{res['grilly2']}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    log(json.dumps(res, indent=1))
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
