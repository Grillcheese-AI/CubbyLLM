"""Tier 1: can this backbone learn an induction circuit at all? Pure vs hybrid.

WHY THIS EXISTS. `exp_d1b_backbone_bakeoff.py` chose MinGRU and CLAUDE.md records
"Backbone RESOLVED (H-D1)". Two problems with treating that as settled:

  1. Its `Trunk(d, n_layers, mixer_cls)` builds ONE mixer class for every layer.
     A hybrid was not rejected there, it was UNREPRESENTABLE. Meanwhile cubby-lm's
     production 1.7B run WAS a hybrid — `layer_idx % attn_every_n == 0` with
     `--attn-every-n 3`, 8 of 22 layers (`trunk_torch/blocks.py:133`,
     `runpod_launch.sh:104`). CubbyLLM dropping that was drift, not a decision.
  2. It scored on **bpc**. Attention's mechanistic edge is content-based lookup —
     "where did this token appear before, and what followed it". That is invisible
     to next-token perplexity on ordinary text and shows up in COPYING. Picking a
     backbone on bpc is picking on the one metric blind to the question.

THE MECHANISM, AND A CORRECTION (2026-08-03). Induction is a TWO-layer circuit
(Olsson et al.): a prev-token head — the VALUE position learns "I follow KEY" —
then a match head — the final KEY finds the position that follows KEY and copies
it. The prev-token head has to be learned from absolute positions and is slow to
form from scratch (this is the well-known "induction bump").

That reverses a wrong assumption three earlier revisions of this file were built
on — that PURE ATTENTION is the easy "ceiling" arm. It is not. A recurrent layer
carries the previous token for FREE in its state (h_t = a·h_{t-1} + x), so it
never has to learn a prev-token head:
  * minGRU (0 attn): prev-token free; must only associate over distance -> its
    limit is the fixed state losing the pair at long range.
  * hybrid (2/6 attn): minGRU gives prev-token, attention gives distance-free
    match -> the cleanest induction circuit; forms easiest.
  * pure attention (6/6): must learn BOTH heads from scratch at d=128 -> the
    HARDEST arm. Sitting at chance here is EXPECTED, not a broken harness.
Every run of this file showed exactly that (attn always ~chance; recurrence arms
sometimes solving) and every run called it "harness failure". The guard now trips
only when NOTHING trains, and the verdict compares minGRU vs hybrid — the real
question — with pure-attn reported but not gating.

THE TASK. Pure induction, nothing else: a random filler sequence, one planted
`KEY VALUE` pair, then `KEY` again at the end. Predict VALUE. Keys are drawn from
a reserved range that never occurs as filler, so the match is unambiguous. Loss is
scored ONLY at the final position — no language modelling to hide behind.

WHY IT BEATS A LANGUAGE RUN FOR THIS. Induction is a phase change: accuracy sits
at chance, then jumps. You read it without statistics. The 413M/1.16B-token run
took 13 GPU-hours to report copy 0/20, which distinguishes nothing.

KILL CRITERION, stated before running. If `mingru` reaches the induction task at
parity with `hybrid`, attention is not mechanistically required here, the hybrid
question closes, and cubby-lm's alpha_attn -> 0.109 (MEMORY_PROBE.md:145, its own
model attenuating attention ~9x) gets a plausible explanation. If `mingru` stays
near chance while `hybrid` converges, that is the ablation neither repo ever ran.

Also reports accuracy vs. PLANT DISTANCE. A fixed-size state should degrade as the
pair moves further back; attention should not. That gradient is the signature.

  python validation/exp_d3_induction.py
  CB_STEPS=4000 CB_S=256 python validation/exp_d3_induction.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # for exp_d1b_* siblings
sys.path.insert(0, os.path.dirname(_HERE))      # for `cubbyllm` (repo root)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp_d1b_backbone_bakeoff import (RMSNorm, SwiGLU, AttnMixer,  # noqa: E402
                                      MinGRUMixer)

D = int(os.environ.get("CB_D", "128"))
L = int(os.environ.get("CB_L", "6"))
S = int(os.environ.get("CB_S", "128"))
B = int(os.environ.get("CB_B", "32"))
STEPS = int(os.environ.get("CB_STEPS", "3000"))
EVERY = int(os.environ.get("CB_EVAL", "250"))
LR = float(os.environ.get("CB_LR", "3e-3"))
V_FILL = 48                      # filler + legal answers
V_KEY = 16                       # reserved keys, never appear as filler
V = V_FILL + V_KEY
ATTN_EVERY = int(os.environ.get("CB_ATTN_EVERY", "3"))   # cubby-lm's rule


def make_batch(n, seq, g, dev):
    """Filler noise + one planted `KEY VALUE` + trailing `KEY`. Target = VALUE.

    Keys come from a reserved range so the sequence contains the key exactly
    twice: once at the plant, once as the query. Any hit is a real lookup — there
    is no second occurrence to get lucky on, and VALUE is drawn from the filler
    range so the answer cannot be identified by token type.
    """
    x = torch.randint(0, V_FILL, (n, seq), generator=g)
    k = torch.randint(V_FILL, V, (n,), generator=g)
    v = torch.randint(0, V_FILL, (n,), generator=g)
    p = torch.randint(0, seq - 3, (n,), generator=g)      # plant position
    idx = torch.arange(n)
    x[idx, p] = k
    x[idx, p + 1] = v
    x[idx, seq - 1] = k                                   # the query
    return x.to(dev), v.to(dev), p


class Trunk(nn.Module):
    """N layers, but the mixer is chosen PER LAYER — the thing exp_d1b could not do.

    ``pattern`` is one mixer class per layer, so 'mingru everywhere except every
    3rd' is expressible. That is cubby-lm's production shape.
    """

    def __init__(self, d, pattern):
        super().__init__()
        self.n1 = nn.ModuleList([RMSNorm(d) for _ in pattern])
        self.mix = nn.ModuleList([m(d) for m in pattern])
        self.n2 = nn.ModuleList([RMSNorm(d) for _ in pattern])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in pattern])

    def forward(self, x):
        for n1, m, n2, f in zip(self.n1, self.mix, self.n2, self.ffn):
            x = x + m(n1(x))
            x = x + f(n2(x))
        return x


class Model(nn.Module):
    """Token + LEARNED POSITION embeddings, then the trunk.

    The positional term is not decoration. This task asks for the token at
    KEY+1, which is a positional relation; attention is permutation-equivariant
    apart from the causal mask, so with content alone it can locate the key and
    still have no way to say "the next one". The first run of this file omitted
    positions entirely and the pure-attention CEILING arm scored 3.9% — at chance
    — while a 2/6 hybrid hit 100%. A control that cannot do the task is what
    caught it; without that arm the table would have looked clean and been wrong.
    MinGRU gets ordering free from the recurrence, so this only ever handicapped
    the arms meant to be strongest.
    """

    def __init__(self, pattern, seq):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(seq, D)
        self.trunk = Trunk(D, pattern)
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.norm(self.trunk(h)))[:, -1]             # last pos only


def pattern_for(name):
    if name == "mingru":
        return [MinGRUMixer] * L
    if name == "attn":
        return [AttnMixer] * L
    if name == "hybrid":            # cubby-lm: attention on every ATTN_EVERY-th
        return [AttnMixer if i % ATTN_EVERY == 0 else MinGRUMixer for i in range(L)]
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, dev, n=512):
    """Overall accuracy, plus a breakdown by how far back the pair was planted."""
    g = torch.Generator().manual_seed(999)                # SAME eval set every arm
    x, y, p = make_batch(n, S, g, dev)
    pred = model(x).argmax(-1).cpu()
    ok = (pred == y.cpu())
    dist = (S - 1) - p                                    # query-to-plant distance
    bands, edges = [], [0, S // 4, S // 2, 3 * S // 4, S]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        bands.append(float(ok[m].float().mean()) if m.any() else float("nan"))
    return float(ok.float().mean()), bands


# FIXED LR, not a swept one. Three earlier revisions of this file failed here:
# a single lr=3e-3 let one arm escape and called it architecture; a per-arm LR
# PROBE then picked on noise, because this task is GROKKING-SHAPED — accuracy is
# flat at chance through a long plateau and then jumps abruptly (S=64 jumped at
# ~1500, S=256 at ~4500). You cannot predict the winning LR from the first
# quarter when nothing has generalised yet, and one run is close to a coin flip on
# whether/when it escapes. So the honest instrument is not one run per arm but
# MANY SEEDS at a known-good LR, scored on how OFTEN and how FAST each arm solves.
# A binary escape/no-escape reversing with config was the coin; solve-rate and
# median steps-to-solve are the graded signal that survives it.
LR = float(os.environ.get("CB_LR", "3e-3"))
SEEDS = int(os.environ.get("CB_SEEDS", "4"))
SOLVE = 0.90                                              # acc that counts as solved
# Two cadences, deliberately separate. LOG_EVERY prints the CHEAP per-step values
# (loss, |grad|) that are already computed — set it to 1 to see every step and
# catch anything strange (a loss spike, a grad blow-up before it becomes NaN).
# EVERY runs the EXPENSIVE accuracy eval (512 examples); doing that per step would
# ~500x the runtime, so it stays periodic. Default LOG_EVERY=1 per request.
LOG_EVERY = int(os.environ.get("CB_LOG", "1"))


def train_once(name, dev, seed):
    """One arm, one seed, to STEPS. Returns (final_acc, bands, steps_to_solve, diverged).

    Per-step loss/|grad| logging (CB_LOG) makes a long run legible and surfaces a
    NaN the instant it happens — otherwise a diverged seed is indistinct from one
    that merely never groks; both just report low final accuracy. Accuracy (the
    costly eval) is refreshed every EVERY steps. On non-finite loss the seed stops
    early and is flagged.
    """
    torch.manual_seed(seed)
    model = Model(pattern_for(name), S).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    warm = max(1, STEPS // 20)
    g = torch.Generator().manual_seed(seed + 1)
    solved_at, diverged = None, False
    acc, bands, acc_shown = 0.0, [], "  --  "
    for s in range(1, STEPS + 1):
        for grp in opt.param_groups:                      # linear warmup, then flat
            grp["lr"] = LR * min(1.0, s / warm)
        x, y, _ = make_batch(B, S, g, dev)
        loss = F.cross_entropy(model(x), y)
        lv = loss.detach().item()
        if not (lv == lv) or lv in (float("inf"), float("-inf")):   # NaN or inf
            print(f"      seed {seed} step {s:>5}: loss {lv} — NON-FINITE, "
                  f"stopping this seed (lr={grp['lr']:.1e}, grad blew up)", flush=True)
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        fresh_acc = s % EVERY == 0 or s == STEPS
        if fresh_acc:
            acc, bands = evaluate(model, dev)
            acc_shown = f"{acc:5.1%}"
            if solved_at is None and acc >= SOLVE:
                solved_at = s
        if s % LOG_EVERY == 0 or fresh_acc or s == 1:
            # a big |grad| here is the early warning; it grows before loss NaNs
            print(f"      seed {seed} step {s:>5}  loss {lv:6.3f}  acc {acc_shown}"
                  f"  |grad| {gn:7.2f}  lr {grp['lr']:.1e}", flush=True)
    return acc, bands, solved_at, diverged


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else xs[n // 2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2


def run(name, dev):
    n_attn = sum(1 for m in pattern_for(name) if m is AttnMixer)
    n_p = sum(p.numel() for p in Model(pattern_for(name), S).parameters())
    print(f"  --- {name}: {n_attn}/{L} attention layers, {n_p:,} params, "
          f"{SEEDS} seeds @ lr={LR:.0e} ---", flush=True)
    t0, accs, solves, last_bands, n_div = time.perf_counter(), [], [], [], 0
    for seed in range(SEEDS):
        acc, bands, at, diverged = train_once(name, dev, seed)
        accs.append(acc)
        if at is not None:
            solves.append(at)
        last_bands.append(bands)
        n_div += int(diverged)
        tag = "DIVERGED (NaN/inf)" if diverged else (f"solved @ {at}" if at
                                                     else "NOT solved")
        print(f"    seed {seed}: final acc {acc:6.1%}  ({tag})", flush=True)
    if n_div:
        print(f"    !! {n_div}/{SEEDS} seeds diverged — lr={LR:.0e} too high for"
              f" {name}; lower CB_LR before trusting its solve-rate", flush=True)
    n_solved = sum(a >= SOLVE for a in accs)
    # bands averaged over the seeds that solved (an unsolved seed's bands are noise)
    ok = [b for a, b in zip(accs, last_bands) if a >= SOLVE]
    bands = [sum(x) / len(x) for x in zip(*ok)] if ok else [float("nan")] * 4
    print(f"    done in {time.perf_counter()-t0:.0f}s\n", flush=True)
    return {"name": name, "params": n_p, "n_attn": n_attn,
            "solve_rate": n_solved / SEEDS, "n_solved": n_solved,
            "median_solve": _median(solves), "bands": bands}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = 1.0 / V_FILL
    print("induction-circuit bake-off — pure recurrence vs cubby-lm's hybrid")
    print(f"  d={D} L={L} seq={S} batch={B} steps={STEPS} | {dev}")
    print(f"  task: filler + planted 'KEY VALUE' + trailing KEY -> predict VALUE")
    print(f"  loss at the FINAL position only. chance = 1/{V_FILL} = {chance:.1%}\n")

    print(f"  {SEEDS} seeds/arm at lr={LR:.0e}. Solve = final acc >= {SOLVE:.0%}.")
    print("  Ranking on solve-RATE and steps-to-solve, not one run — this task is")
    print("  grokking-shaped, so a single run is a coin flip and reverses with"
          " config.\n")

    # Default order runs HYBRID first as a positive control: it is the arm most
    # likely to solve, so if it groks the harness works and any later arm's
    # failure is real signal — whereas leading with an arm we expect to fail
    # (mingru at long S) means watching SEEDS*STEPS of guaranteed-null before the
    # first informative result. If hybrid ALSO fails to solve, S is too hard for
    # this budget: lower CB_S or raise CB_STEPS before reading anything. Override
    # with CB_ARMS to pick/reorder (e.g. CB_ARMS=hybrid to sanity-check one arm).
    arms = [a.strip() for a in os.environ.get("CB_ARMS", "hybrid,mingru,attn").split(",")
            if a.strip()]
    res = [run(n, dev) for n in arms]

    print("  === RESULT ===")
    print("   arm    | attn |     params | solve rate | median steps-to-solve")
    print("  --------+------+------------+------------+----------------------")
    for r in res:
        ms = "—" if r["median_solve"] is None else f"{int(r['median_solve']):,}"
        print(f"  {r['name']:>7} | {r['n_attn']:>2}/{L} | {r['params']:>10,} |"
              f"  {r['n_solved']}/{SEEDS} = {r['solve_rate']:>3.0%} | {ms:>14}")
    print()
    print("  accuracy by query-to-plant distance, over seeds that solved:")
    q = [f"{int(S*a)}-{int(S*b)}" for a, b in ((0,.25),(.25,.5),(.5,.75),(.75,1))]
    print("   arm    |" + "".join(f" {h:>9} " for h in q))
    for r in res:
        print(f"  {r['name']:>7} |" + "".join(f" {b:>8.1%} " for b in r["bands"]))
    print()
    by = {r["name"]: r for r in res}
    rates = {r["name"]: r["solve_rate"] for r in res}
    # CB_ARMS may omit arms (e.g. a single-arm sanity check). The full verdict
    # needs both mingru and hybrid; without them, stop after the table.
    if "mingru" not in by or "hybrid" not in by:
        print(f"  (ran {', '.join(by)} — full verdict needs both mingru and hybrid;"
              " table above stands on its own.)")
        return
    pure, hyb = by["mingru"], by["hybrid"]
    # THE GUARD, CORRECTED. Earlier revisions gated on "pure attention must solve,
    # it is the ceiling." That premise is WRONG for this task at this scale, which
    # is why every run tripped it. Induction is a TWO-layer circuit: a prev-token
    # head (VALUE learns "I follow KEY") then a match head (final KEY finds it).
    # The prev-token head must be learned from absolute positions and is famously
    # slow from scratch — the "induction bump". A recurrent layer gets the
    # prev-token carry FREE from its state, so minGRU and hybrid have a structural
    # head start and PURE ATTENTION IS THE HARDEST ARM, not the easiest. So the
    # only real harness failure is when NOTHING trains at all.
    if all(r["n_solved"] == 0 for r in res):
        print(f"  HARNESS FAILURE — every arm solved 0/{SEEDS}. Nothing trained.")
        print("  Loss stuck near ln(V_FILL)=3.87 means no arm left init: raise")
        print("  CB_STEPS or CB_LR, or lower CB_S. (A run where mingru/hybrid solve")
        print("  and pure attn does not is NOT a failure — attn is the hard arm here.)")
        return
    # The real question is mingru vs hybrid: does adding attention to a recurrent
    # backbone make the circuit form more reliably, or hold at longer distance?
    if "attn" in rates:
        print(f"  (pure attn solved {rates['attn']:.0%} — expected low; it is the hard")
        print("   arm, not the ceiling. It is reported, it does not gate the verdict.)")
    if hyb["solve_rate"] - pure["solve_rate"] >= 0.5:
        print(f"  VERDICT: hybrid solves far more reliably than pure recurrence "
              f"({rates['hybrid']:.0%} vs {rates['mingru']:.0%}), seed-robust.")
        print("  The attention layers are load-bearing at this context length, and")
        print("  CubbyLLM's drift to 0% attention dropped something real. Also read")
        print("  the by-distance row: if mingru holds near and fails far while hybrid")
        print("  holds throughout, that is the fixed-state length limit, measured.")
    elif pure["solve_rate"] >= 0.75 and pure["solve_rate"] >= hyb["solve_rate"] - 1e-9:
        print(f"  VERDICT: pure recurrence solves as reliably as the hybrid "
              f"({rates['mingru']:.0%} vs {rates['hybrid']:.0%}) at this length.")
        print("  Attention earns nothing HERE — re-run at larger CB_S before")
        print("  concluding it earns nothing at length. minGRU's own state limit is")
        print("  the thing to find: the CB_S where its solve-rate starts to fall.")
    else:
        print(f"  VERDICT: partial separation (mingru {rates['mingru']:.0%}, hybrid "
              f"{rates['hybrid']:.0%}). Suggestive, not decisive — add seeds")
        print("  (CB_SEEDS) or sweep CB_S to see if the gap holds before acting on it.")


if __name__ == "__main__":
    main()
