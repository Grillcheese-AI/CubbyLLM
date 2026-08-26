"""H-P1 — a decode state is a FORKABLE world snapshot: branching is exact and
costs O(1) in context length.

CLAIM. In a recurrent model the state at step t IS the latent situation, so
"go back to a choice point and choose differently" costs one state copy plus
the continuation — never a re-run of the prefix. That is the abduction step of
a counterfactual (Pearl's rung three) for the price of a tensor clone, and it
is the structural reason to build branching-futures machinery on this backbone
rather than bolt tree search onto a KV cache that grows with context.

Three checks, each with a kill:
  (a) EXACT: k branches rolled from ONE cloned state equal k independent full
      passes over prefix+branch to fp tolerance. A fork that drifts is a
      different model, and every downstream number would be about that model.
  (b) BOUNDED: the floats a fork must copy do not grow with prefix length
      (pure MinGRU: n_layers x d always; hybrid: attention caches trimmed to
      ``window``). The episodic store is excluded — it grows off the recurrent
      path by design, and is deep-copied, not carried.
  (c) ISOLATED: rolling a branch never mutates the snapshot it came from, and
      a branch's episodic-store writes never leak into a sibling branch.

Also reports the work saved: 1 prefix + k continuations vs k full passes.

Standalone, CPU, seconds. With CB_CKPT the same three checks run on the real
checkpoint (shapes from its meta; random token ids, since exactness does not
depend on the tokens being meaningful).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import _common as C  # noqa: E402

CONFIGS = [
    ("mingru", dict(backbone="mingru", d=32, L=3)),
    ("hybrid", dict(backbone="hybrid", d=32, L=3, window=8)),
    ("hybrid+mem", dict(backbone="hybrid", d=32, L=3, window=8, mem_every=2)),
]


@torch.no_grad()
def check_model(model, V: int, P: int, n: int, k: int, prefix_lengths, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(0, V, (P,), generator=g)
    branches = [torch.randint(0, V, (n,), generator=g) for _ in range(k)]

    t0 = time.perf_counter()
    _, st0 = C.prefix_state(model, prefix)
    snap = C.gru_states(st0).clone()
    store_len = _store_lengths(st0)
    forked = [C.continue_from(model, st0, br)[0] for br in branches]     # (a) branches
    t_fork = time.perf_counter() - t0

    t0 = time.perf_counter()
    refs = []
    for br in branches:                                                    # independent passes
        s, out = None, []
        for t in torch.cat([prefix, br]):
            lg, s = model.step(t.view(1), s)
            out.append(lg[0])
        refs.append(torch.stack(out[P:]))
    t_full = time.perf_counter() - t0

    max_diff = max(float((f - r).abs().max()) for f, r in zip(forked, refs))
    snapshot_moved = float((C.gru_states(st0) - snap).abs().max())        # (c) origin untouched
    store_moved = _store_lengths(st0) != store_len

    sizes = []
    for P_ in prefix_lengths:                                              # (b) state size vs prefix
        _, st = C.prefix_state(model, torch.randint(0, V, (P_,), generator=g))
        sizes.append(C.state_numel(st))

    return dict(max_diff=max_diff, snapshot_moved=snapshot_moved, store_moved=store_moved,
                sizes=sizes, steps_fork=P + k * n, steps_full=k * (P + n),
                t_fork=t_fork, t_full=t_full)


def _store_lengths(state):
    bb = state["bb"]
    if isinstance(bb, dict) and "mem" in bb:
        return tuple(len(v["store"]) for v in bb["mem"].values())
    return ()


def run(quick: bool = False, verbose: bool = True) -> dict:
    V = 40
    P, n, k = (24, 6, 3) if quick else (48, 12, 4)
    lengths = [8, 24] if quick else [8, 32, 64]                 # all >= window=8
    results = {}
    for name, kw in CONFIGS:
        model = C.build_model(V=V, **kw)
        r = check_model(model, V, P, n, k, lengths)
        results[name] = r
        if verbose:
            C.banner(f"[{name}]", [
                f"(a) max |forked - independent| over {k} branches x {n} steps: {r['max_diff']:.2e}",
                f"(b) state floats after prefixes {lengths}: {r['sizes']}",
                f"(c) origin snapshot moved by {r['snapshot_moved']:.1e}; store mutated: {r['store_moved']}",
                f"work: {r['steps_fork']} step()s forked vs {r['steps_full']} independent "
                f"({r['steps_full']/r['steps_fork']:.2f}x); wall {r['t_fork']:.3f}s vs {r['t_full']:.3f}s",
            ])
        assert r["max_diff"] < 1e-4, f"{name}: fork drifts from independent pass ({r['max_diff']:.2e})"
        assert len(set(r["sizes"])) == 1, f"{name}: state grew with prefix length {r['sizes']}"
        assert r["snapshot_moved"] == 0.0 and not r["store_moved"], f"{name}: branch mutated its origin"
        assert r["steps_fork"] < r["steps_full"]

    if C.CKPT:
        model, meta = C.load_checkpoint(C.CKPT)
        V = int(meta["vocab"])
        # boundedness is only testable PAST the attention window: below it the
        # KV cache is still filling, which is growth by design, not a leak
        w = int(meta.get("window", 512)) if meta.get("backbone") == "hybrid" else 8
        lengths = [w, w + 64, w + 256]
        r = check_model(model, V, P, n, k, lengths)
        results["ckpt"] = r
        if verbose:
            C.banner(f"[checkpoint {os.path.basename(C.CKPT)} d={meta['D']} L={meta['L']} window={w}]", [
                f"(a) max |forked - independent|: {r['max_diff']:.2e}",
                f"(b) state floats after prefixes {lengths}: {r['sizes']}",
                f"(c) origin moved {r['snapshot_moved']:.1e}; store mutated: {r['store_moved']}",
                f"wall {r['t_fork']:.2f}s forked vs {r['t_full']:.2f}s independent",
            ])
        assert r["max_diff"] < 1e-3          # fp32 at real width; bf16 ckpts may need looser
        assert len(set(r["sizes"])) == 1, f"checkpoint state grew past the window: {r['sizes']}"
        assert r["snapshot_moved"] == 0.0 and not r["store_moved"]
    return results


def main():
    print("H-P1 fork equivalence — is the decode state an exact, bounded, isolated snapshot?\n")
    run()
    print("PASS: forking a state is exact, its cost is context-independent, and branches "
          "are isolated. Counterfactual re-entry (H-P5) and keyframe recompute (H-P4) "
          "may rely on it.")


if __name__ == "__main__":
    main()
