"""validation/head_lab_decode.py -- order 0068, step 2e: route 5 inside a real decode step on a card (grilly2), at 128k and
on a synthetic 256k head, against a full 32k head's step.

Step 2d timed the head alone, token after token, where the route's codes and centroids can stay in the card's
last-level cache. Here each head runs inside the model's own decode step: the base export decodes greedily, batch 1,
and a step is timed from the token going in to the next token on the host.
  trunk     no head: one value of the state read back, the text's next token fed in (the trunk alone)
  full128   the trained 128k head          head32k   its first 32,000 rows (a full 32k head)
  route128  route 5 over the 128k head (grilly2's ``ShortlistHead``)
  full256   a synthetic 256k head          route256  route 5 over it
The 256k head is the trained rows and a copy of them with Gaussian noise at --noise of each row's spread (seed
--seed); its cluster lists hold each token and its copy, its successors are repeated. It measures time only: a
chosen copy is fed back as its original. Prompts are the middle of the first --prompts shards' held-in text.
Also, untimed: route 5 at 128k decoding on its own choices, against the full head's token at every step.

    python validation/head_lab_decode.py --export <base export> --dumps <head_shortlist folder> --cache <token cache>
-> validation/logs/head_lab_decode_<date>_<grilly2 commit>.json

Wired: STANDALONE (a timing; order 0068's report reads it). Imports grilly (grilly2), never torch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

__wiring__ = "STANDALONE"

KINDS = ("trunk", "full128", "head32k", "route128", "full256", "route256")
WARM = 2                                           # steps of each run left out


def doubled_lists(ids: np.ndarray, offsets: np.ndarray, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Every cluster's list with each token followed by its copy (token + vocab), and the new offsets."""
    lens = np.diff(offsets)
    out = np.empty(2 * len(ids), dtype=np.int64)
    out[0::2], out[1::2] = ids, ids + vocab
    return out, np.concatenate([[0], np.cumsum(2 * lens)]).astype(np.int64)


def noisy_copy(w: np.ndarray, noise: float, seed: int, chunk: int = 8192) -> np.ndarray:
    """``w`` plus Gaussian noise at ``noise`` of each row's standard deviation, in row chunks (float32)."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(w)
    for s in range(0, len(w), chunk):
        part = w[s:s + chunk]
        out[s:s + chunk] = part + noise * part.std(axis=1, keepdims=True) * rng.standard_normal(part.shape,
                                                                                               dtype=np.float32)
    return out


def prompt_tokens(cache: str, names, n: int, length: int) -> list:
    """``length`` tokens from the middle of the held-in text of the first ``n`` shards (no held-out token)."""
    hold = json.load(open(os.path.join(cache, "_holdout.json"), encoding="utf-8"))
    out = []
    for name in names[:n]:
        path = os.path.join(cache, f"{name}.u32")
        total = os.path.getsize(path) // 4
        held_in = total - int(hold[name])
        start = held_in // 2
        if start + length > held_in:
            raise SystemExit(f"{name}: too little held-in text for {length} tokens")
        out.append((name, np.memmap(path, dtype=np.uint32, mode="r")[start:start + length].astype(np.int64)))
    return out


def _stats(ts: list) -> dict:
    s = sorted(ts)
    return {"p50_ms": round(statistics.median(s), 3), "mean_ms": round(statistics.fmean(s), 3),
            "p90_ms": round(s[int(0.9 * (len(s) - 1))], 3), "n": len(s)}


def _commit(g) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(g.__file__),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    import grilly as g
    from grilly.infer.shortlist import ShortlistHead
    from head_dump import SHARDS
    from head_shortlist_eval import head_rows

    from cubbyllm.model.vocab.shortlist import ShortlistIndex
    from cubbyllm.ops.grilly_seats import load_export
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
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=68)
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)                              # noqa: E731
    F = g.nn.functional
    model, cfg = load_export(a.export)
    W = model.lm_head.weight
    model.lm_head = g.nn.Identity()                                     # the step ends at the state; heads below
    w = head_rows(a.export)
    v, dim = w.shape
    idx = ShortlistIndex.load(os.path.join(a.dumps, a.index))
    s, succ = idx.screen, idx.succ
    W32 = g.from_numpy(np.ascontiguousarray(w[:32000]))
    card128 = ShortlistHead(W, s.centroids, s.ids, s.offsets, succ, near=a.n, context=a.k, ranks=a.ranks)
    log("building the synthetic 256k head")
    W256 = g.from_numpy(np.concatenate([w, noisy_copy(w, a.noise, a.seed)]))
    ids2, off2 = doubled_lists(s.ids, s.offsets, v)
    card256 = ShortlistHead(W256, s.centroids, ids2, off2, None if succ is None else np.concatenate([succ, succ]),
                            near=a.n, context=a.k, ranks=a.ranks)
    del w

    def best(logits) -> int:
        return int(g.argmax(logits.reshape(1, -1), dim=-1).numpy().reshape(-1)[0])

    def pick(kind, x, prev, text_next):
        if kind == "trunk":
            x.reshape(-1)[:1].numpy()
            return text_next
        if kind == "full128":
            return best(F.linear(x, W))
        if kind == "head32k":
            return best(F.linear(x, W32))
        if kind == "route128":
            return card128.argmax(x, prev)
        if kind == "full256":
            return best(F.linear(x, W256)) % v
        return card256.argmax(x, prev) % v

    prompts = prompt_tokens(a.cache, SHARDS, a.prompts, a.prompt_len + a.steps)

    def run(kind, toks, steps, check=False):
        prompt, cont = toks[:a.prompt_len], toks[a.prompt_len:]
        cache = model.make_cache(1, a.prompt_len + steps + 8)
        out = model(g.tensor(prompt.reshape(1, -1).tolist()), past_key_values=cache, logits_to_keep=1)
        tok = pick(kind, out.logits, int(prompt[-1]), int(cont[0]))
        times, agree, first_diff, short = [], 0, None, 0
        for i in range(steps):
            t0 = time.perf_counter()
            out = model(g.tensor([[tok]]), past_key_values=cache, logits_to_keep=1)
            nxt = pick(kind, out.logits, tok, int(cont[min(i + 1, len(cont) - 1)]))
            times.append((time.perf_counter() - t0) * 1e3)
            if kind.startswith("route"):
                short += (card128 if kind == "route128" else card256).last.short
            if check:
                same = best(F.linear(out.logits, W)) == nxt
                agree += same
                if not same and first_diff is None:
                    first_diff = i
            tok = nxt
        return times[WARM:], agree, first_diff, short

    times = {k: [] for k in KINDS}
    shorts = {"route128": 0, "route256": 0}
    with g.no_grad():
        log("warm-up")
        for k in KINDS:
            run(k, prompts[0][1], 8)
        for i, (name, toks) in enumerate(prompts):
            order = KINDS[i % len(KINDS):] + KINDS[:i % len(KINDS)]
            for k in order:
                ts, _, _, sh = run(k, toks, a.steps)
                times[k] += ts
                if k in shorts:
                    shorts[k] += sh
            log(f"{name}: " + ", ".join(f"{k} {statistics.median(times[k]):.3f}" for k in KINDS))
        log("agreement pass (untimed)")
        checks = []
        for name, toks in prompts:
            _, agree, first_diff, _ = run("route128", toks, a.steps, check=True)
            checks.append({"shard": name, "steps": a.steps, "same_token_as_full_head": agree,
                           "first_difference_at_step": first_diff})
    st = {k: _stats(ts) for k, ts in times.items()}
    trunk = st["trunk"]["p50_ms"]
    res = {"what": "order 0068 step 2e: route 5 inside a real decode step (grilly2), 128k and a synthetic 256k head",
           "date": _dt.date.today().isoformat(), "grilly2": _commit(g), "machine": platform.platform(),
           "batch": 1, "precision": "fp32", "vocab": v, "hidden": dim, "k": a.k, "n": a.n, "ranks": a.ranks,
           "prompts": [n for n, _ in prompts], "prompt_len": a.prompt_len, "steps": a.steps, "warm_steps_left_out": WARM,
           "synthetic_256k": {"rows": 2 * v, "noise_of_row_std": a.noise, "seed": a.seed, "time_only": True},
           "decode_step": st,
           "head_inside_the_step_ms": {k: round(st[k]["p50_ms"] - trunk, 3) for k in KINDS if k != "trunk"},
           "short_lists": shorts,
           "agreement_128k_route_decoding_on_its_own": {
               "same_token_as_full_head": sum(c["same_token_as_full_head"] for c in checks),
               "steps": sum(c["steps"] for c in checks), "per_prompt": checks}}
    res["lines"] = {f"{r}_at_most_head32k": st[r]["p50_ms"] <= st["head32k"]["p50_ms"] for r in ("route128", "route256")}
    path = os.path.join(HERE, "logs", f"head_lab_decode_{res['date']}_{res['grilly2']}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    log(json.dumps(res, indent=1))
    log("->", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
