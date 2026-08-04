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

THE TASK (canonical, dense — 2026-08-03). A random block repeated once:
`[r_0..r_{T-1}  r_0..r_{T-1}]`. Predict the next token at every position; score
the SECOND half, where the only way to predict is to recall what followed this
same token T positions back. That is induction, and it gives T supervised
examples PER SEQUENCE. This replaced a single-needle task that scored ONE final
position — a signal so sparse that whether a model escaped its init basin was a
coin flip (an arm would solve S=256 but not S=64, which is optimisation noise, not
architecture). The dense form is how induction heads are actually studied (Olsson
et al.) and is reliably learnable at this scale, so the arms can be RANKED.

WHY IT BEATS A LANGUAGE RUN FOR THIS. It isolates the circuit: no vocabulary
frequency, no register, just "did the model learn to copy-by-content-match". The
413M/1.16B-token run took 13 GPU-hours to report copy 0/20, which distinguishes
nothing about the architecture.

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
if S % 2:
    S += 1                                               # sequence is [half, half]
T = S // 2                                               # repeat offset = induction distance
B = int(os.environ.get("CB_B", "32"))
STEPS = int(os.environ.get("CB_STEPS", "3000"))
EVERY = int(os.environ.get("CB_EVAL", "250"))
LR = float(os.environ.get("CB_LR", "3e-3"))
V = int(os.environ.get("CB_V", "128"))                   # a larger vocab -> fewer
CHANCE = 1.0 / V                                          # within-half collisions
ATTN_EVERY = int(os.environ.get("CB_ATTN_EVERY", "3"))   # cubby-lm's rule


def make_seq(n, g):
    """The CANONICAL induction task: a random block, then the SAME block again.

    x = [r_0 .. r_{T-1}  r_0 .. r_{T-1}]. In the SECOND copy every next-token is
    determined only by having seen the first copy — the sole way to predict token
    i (i>=T) is to recall what followed this same token T positions back. That is
    induction, and here it supplies T supervised examples PER SEQUENCE instead of
    the single end-of-sequence signal the old single-needle task gave. The sparse
    old signal is why nothing learned reliably; this is how induction heads are
    actually studied (Olsson et al.), and it is reliably learnable at this scale.
    """
    base = torch.randint(0, V, (n, T), generator=g)
    return torch.cat([base, base], dim=1)                # (n, 2T) = (n, S)


def induction_loss_acc(model, x):
    """Next-token loss/accuracy on the SECOND half only — the induction region.

    First-half tokens are genuinely random (chance-only), so scoring or training
    on them just dilutes the gradient. Predicting position i uses logits at i-1,
    so the second half's targets x[:, T:] are read from logits[:, T-1:-1]."""
    logits = model(x)                                    # (B, S, V), all positions
    pred = logits[:, T - 1:-1]                           # -> predicts x[:, T:]
    tgt = x[:, T:]
    loss = F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1))
    acc = (pred.argmax(-1) == tgt).float().mean()
    return loss, float(acc)


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
    """Token + learned position embeddings, trunk, per-position head.

    The prev-token head that induction needs is learned from these absolute
    positions; recurrent layers also get ordering free from their state. Returns
    logits at EVERY position (B, S, V) — the dense task scores next-token over the
    whole second half, not one final slot as the old single-needle version did.
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
        return self.head(self.norm(self.trunk(h)))                   # (B, S, V)


WINDOW = int(os.environ.get("CB_WINDOW", "96"))                       # cubby-lm: 512


class WindowedAttnMixer(nn.Module):
    """Causal attention restricted to the last WINDOW tokens — cubby-lm's
    LocalCausalAttention (attention.py:33, W=512).

    THE POINT, and why the two package tests matter here. Full attention keeps a
    KV cache that GROWS with context, so a full-attention hybrid would break
    tests/model/test_mingru_decode.py::test_state_size_is_independent_of_context
    _length and kill the O(1)-inference claim. A bounded window keeps state at
    O(WINDOW) = constant, so decode stays context-independent. The catch that
    buys: a token can only reach back WINDOW-1, so this solves induction only when
    the induction distance T (= S/2 here) is within the window. Sweeping S past
    2*WINDOW is where it falls back to chance — the deployable arm's real limit.
    """

    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, Sx, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, Sx, self.h, d // self.h).transpose(1, 2)
                   for t in (q, k, v))
        i = torch.arange(Sx, device=x.device)
        keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < WINDOW)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)   # causal+window
        return self.o(o.transpose(1, 2).reshape(B, Sx, d))


ATTN_TYPES = (AttnMixer, WindowedAttnMixer)                          # for counting


def pattern_for(name):
    if name == "mingru":
        return [MinGRUMixer] * L
    if name == "attn":                                    # pure full attention
        return [AttnMixer] * L
    if name == "hybrid":            # MinGRU + FULL attention every ATTN_EVERY-th
        return [AttnMixer if i % ATTN_EVERY == 0 else MinGRUMixer for i in range(L)]
    if name == "whybrid":           # MinGRU + WINDOWED attention — the deployable
        return [WindowedAttnMixer if i % ATTN_EVERY == 0 else MinGRUMixer
                for i in range(L)]
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, dev, n=512):
    """Second-half induction accuracy on a FIXED eval set (same for every arm)."""
    g = torch.Generator().manual_seed(999)
    x = make_seq(n, g).to(dev)
    return induction_loss_acc(model, x)[1]


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
    acc, acc_shown = 0.0, "  --  "
    for s in range(1, STEPS + 1):
        for grp in opt.param_groups:                      # linear warmup, then flat
            grp["lr"] = LR * min(1.0, s / warm)
        x = make_seq(B, g).to(dev)
        loss, _ = induction_loss_acc(model, x)
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
            acc = evaluate(model, dev)
            acc_shown = f"{acc:5.1%}"
            if solved_at is None and acc >= SOLVE:
                solved_at = s
        if s % LOG_EVERY == 0 or fresh_acc or s == 1:
            # a big |grad| here is the early warning; it grows before loss NaNs
            print(f"      seed {seed} step {s:>5}  loss {lv:6.3f}  ind-acc {acc_shown}"
                  f"  |grad| {gn:7.2f}  lr {grp['lr']:.1e}", flush=True)
    return acc, solved_at, diverged


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else xs[n // 2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2


def run(name, dev):
    n_attn = sum(1 for m in pattern_for(name) if m in ATTN_TYPES)
    n_p = sum(p.numel() for p in Model(pattern_for(name), S).parameters())
    print(f"  --- {name}: {n_attn}/{L} attention layers, {n_p:,} params, "
          f"{SEEDS} seeds @ lr={LR:.0e} ---", flush=True)
    t0, accs, solves, n_div = time.perf_counter(), [], [], 0
    for seed in range(SEEDS):
        acc, at, diverged = train_once(name, dev, seed)
        accs.append(acc)
        if at is not None:
            solves.append(at)
        n_div += int(diverged)
        tag = "DIVERGED (NaN/inf)" if diverged else (f"solved @ {at}" if at
                                                     else "NOT solved")
        print(f"    seed {seed}: final ind-acc {acc:6.1%}  ({tag})", flush=True)
    if n_div:
        print(f"    !! {n_div}/{SEEDS} seeds diverged — lr={LR:.0e} too high for"
              f" {name}; lower CB_LR before trusting its solve-rate", flush=True)
    n_solved = sum(a >= SOLVE for a in accs)
    print(f"    done in {time.perf_counter()-t0:.0f}s\n", flush=True)
    return {"name": name, "params": n_p, "n_attn": n_attn,
            "solve_rate": n_solved / SEEDS, "n_solved": n_solved,
            "median_solve": _median(solves),
            "best_acc": max(accs) if accs else 0.0}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("induction bake-off — pure recurrence vs cubby-lm's hybrid")
    print(f"  d={D} L={L} seq={S} (repeat offset T={T}) batch={B} steps={STEPS}"
          f" vocab={V} | {dev}")
    print("  task: [random block | same block], predict next token; scored on the")
    print(f"  SECOND half where only induction can. chance = 1/{V} = {CHANCE:.1%}\n")

    print(f"  {SEEDS} seeds/arm at lr={LR:.0e}. Solve = 2nd-half acc >= {SOLVE:.0%}.")
    print("  Dense signal (T examples/seq) so it learns reliably; the discriminator")
    print("  is STEPS-TO-SOLVE (speed) and, as T grows, whether an arm solves at all.")
    print(f"  arms: mingru & whybrid keep BOUNDED, context-independent state (whybrid")
    print(f"  window={WINDOW}, so it solves only while T={T} < window); hybrid & attn")
    print(f"  use FULL attention — solve at any T but state GROWS with context, which")
    print(f"  would break test_mingru_decode's O(1) guarantee. Only the bounded arms")
    print(f"  are deployable. Sweep CB_S across 2*window={2*WINDOW} to see whybrid fall.\n")

    # Default order runs HYBRID first as a positive control: it is the arm most
    # likely to solve, so if it groks the harness works and any later arm's
    # failure is real signal — whereas leading with an arm we expect to fail
    # (mingru at long S) means watching SEEDS*STEPS of guaranteed-null before the
    # first informative result. If hybrid ALSO fails to solve, S is too hard for
    # this budget: lower CB_S or raise CB_STEPS before reading anything. Override
    # with CB_ARMS to pick/reorder (e.g. CB_ARMS=hybrid to sanity-check one arm).
    arms = [a.strip() for a in
            os.environ.get("CB_ARMS", "hybrid,whybrid,mingru").split(",")
            if a.strip()]
    res = [run(n, dev) for n in arms]

    print(f"  === RESULT (T={T}) ===")
    print("   arm    | attn |     params | solve rate | median steps | best acc")
    print("  --------+------+------------+------------+--------------+---------")
    for r in res:
        ms = "—" if r["median_solve"] is None else f"{int(r['median_solve']):,}"
        print(f"  {r['name']:>7} | {r['n_attn']:>2}/{L} | {r['params']:>10,} |"
              f"  {r['n_solved']}/{SEEDS} = {r['solve_rate']:>3.0%} | {ms:>12} |"
              f" {r['best_acc']:>7.1%}")
    print()
    by = {r["name"]: r for r in res}
    rates = {r["name"]: r["solve_rate"] for r in res}
    # Only real harness failure is when NOTHING trains — the dense task is
    # reliably learnable, so a zero across every arm means the budget/LR is wrong,
    # not the architecture.
    if all(r["n_solved"] == 0 for r in res):
        print(f"  HARNESS FAILURE — every arm solved 0/{SEEDS}. Nothing trained.")
        print(f"  Loss stuck near ln(V)={__import__('math').log(V):.2f} means no arm")
        print("  left init: raise CB_STEPS or CB_LR, or lower CB_S.")
        return
    # The DEPLOYABLE comparison is mingru vs whybrid: both keep bounded,
    # context-independent state, so both preserve test_mingru_decode's O(1)
    # guarantee. Full hybrid/attn are the unbounded reference — they show what is
    # achievable if you were willing to pay a growing KV cache (you are not).
    if "attn" in rates:
        print(f"  (attn/full = {rates.get('attn', 0):.0%}; full-hybrid = "
              f"{rates.get('hybrid', 0):.0%} — UNBOUNDED state, reference only.)")
    elif "hybrid" in rates:
        print(f"  (full-hybrid = {rates['hybrid']:.0%} — UNBOUNDED state, the")
        print("   'if we could pay a growing cache' reference. Not deployable as-is.)")
    if "whybrid" not in by or "mingru" not in by:
        print(f"  (ran {', '.join(by)} — the deployable verdict needs mingru AND")
        print("   whybrid; table stands on its own.)")
        return
    pure, wat = by["mingru"], by["whybrid"]
    mp, wp = pure["median_solve"], wat["median_solve"]
    reach = "within" if T < WINDOW else "BEYOND"
    print(f"  (induction distance T={T} is {reach} the window={WINDOW}.)")
    if wat["solve_rate"] - pure["solve_rate"] >= 0.5:
        print(f"  VERDICT: windowed attention is load-bearing and DEPLOYABLE — "
              f"whybrid {rates['whybrid']:.0%} vs mingru {rates['mingru']:.0%} at "
              f"T={T}, both at bounded state.")
        print(f"  A window={WINDOW} attention layer holds the association where the")
        print("  recurrent state loses it. This is the hybrid rung CubbyLLM skipped,")
        print("  and it keeps O(1)-ish decode. Confirm: raise CB_S past 2*window and")
        print("  whybrid should collapse to mingru (the association leaves the window).")
    elif pure["solve_rate"] >= 0.75 and wat["solve_rate"] >= 0.75 and mp and wp:
        faster = "whybrid" if wp < mp else "mingru"
        ratio = max(mp, wp) / max(1, min(mp, wp))
        print(f"  VERDICT: both bounded arms solve at T={T}; {faster} groks "
              f"{ratio:.1f}x faster ({int(min(mp,wp)):,} vs {int(max(mp,wp)):,} steps).")
        print("  Windowed attention earns only speed here, not capability. Raise CB_S")
        print("  toward 2*window: the length where mingru drops but whybrid still")
        print("  holds is where the window becomes load-bearing.")
    else:
        print(f"  VERDICT: mixed (mingru {rates['mingru']:.0%}, whybrid "
              f"{rates['whybrid']:.0%}). Add CB_SEEDS or adjust CB_STEPS/CB_S.")


if __name__ == "__main__":
    main()
