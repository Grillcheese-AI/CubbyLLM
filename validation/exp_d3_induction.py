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

THE MECHANISM AT ISSUE. MinGRU's gate is computed from the current input alone:

    a = 0.001 + 0.998 * sigmoid(proj_d(x_t));  h = a * h_prev + x_scan

`h_prev` never enters the gate, so within a layer there is no operation comparing
the current token against stored content. Attention does exactly that comparison.
Across depth there is indirect state-dependence, so this is a real question rather
than a proof — which is why it gets an experiment instead of an argument.

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


def train_once(name, lr, dev, steps, quiet=False):
    """One arm at one learning rate. Warmup included — see LRS for why."""
    torch.manual_seed(0)                                  # identical init per arm
    model = Model(pattern_for(name), S).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warm = max(1, steps // 20)
    g = torch.Generator().manual_seed(1)
    best = 0.0
    for s in range(1, steps + 1):
        for grp in opt.param_groups:                      # linear warmup, then flat
            grp["lr"] = lr * min(1.0, s / warm)
        x, y, _ = make_batch(B, S, g, dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % EVERY == 0 or s == steps:
            acc, bands = evaluate(model, dev)
            best = max(best, acc)
            if not quiet:
                print(f"    step {s:>5}  loss {float(loss):6.3f}  acc {acc:6.1%}"
                      f"   by distance {' '.join(f'{b:5.1%}' for b in bands)}",
                      flush=True)
    acc, bands = evaluate(model, dev)
    return model, acc, bands, best


# One shared LR across three architectures is not a fair comparison, and the first
# two runs of this file proved it the hard way: at a single lr=3e-3 the pure-ATTN
# ceiling arm sat at loss 3.87 = ln(48) for 8000 steps — exactly uniform, i.e. it
# never left initialisation — while a 2/6 hybrid hit 100%. Pure attention can
# certainly do induction, so that was an optimisation artifact being read as an
# architectural result. Each arm now gets its own best LR.
LRS = [float(x) for x in os.environ.get("CB_LRS", "3e-3,1e-3,3e-4").split(",")]


def run(name, dev):
    n_attn = sum(1 for m in pattern_for(name) if m is AttnMixer)
    print(f"  --- {name}: {n_attn}/{L} attention layers ---", flush=True)
    t0 = time.perf_counter()
    probe = max(1, STEPS // 4)                            # short LR probe per arm
    scores = []
    for lr in LRS:
        _, _, _, best = train_once(name, lr, dev, probe, quiet=True)
        scores.append((best, lr))
        print(f"    probe lr={lr:.0e} ({probe} steps) -> best acc {best:6.1%}", flush=True)
    lr = max(scores)[1]
    print(f"    -> full run at lr={lr:.0e}", flush=True)
    model, acc, bands, _ = train_once(name, lr, dev, STEPS)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"    done in {time.perf_counter()-t0:.0f}s\n", flush=True)
    return {"name": name, "acc": acc, "bands": bands, "params": n_p,
            "n_attn": n_attn, "lr": lr}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = 1.0 / V_FILL
    print("induction-circuit bake-off — pure recurrence vs cubby-lm's hybrid")
    print(f"  d={D} L={L} seq={S} batch={B} steps={STEPS} | {dev}")
    print(f"  task: filler + planted 'KEY VALUE' + trailing KEY -> predict VALUE")
    print(f"  loss at the FINAL position only. chance = 1/{V_FILL} = {chance:.1%}\n")

    res = [run(n, dev) for n in ("mingru", "hybrid", "attn")]

    print("  === RESULT ===")
    print("   arm    | attn |     params |     lr | induction acc | vs chance")
    print("  --------+------+------------+--------+---------------+----------")
    for r in res:
        print(f"  {r['name']:>7} | {r['n_attn']:>2}/{L} | {r['params']:>10,} |"
              f" {r['lr']:.0e} | {r['acc']:>12.1%}  | {r['acc']/chance:>6.1f}x")
    print()
    print("  accuracy by query-to-plant distance (near -> far):")
    q = [f"{int(S*a)}-{int(S*b)}" for a, b in ((0,.25),(.25,.5),(.5,.75),(.75,1))]
    print("   arm    |" + "".join(f" {h:>9} " for h in q))
    for r in res:
        print(f"  {r['name']:>7} |" + "".join(f" {b:>8.1%} " for b in r["bands"]))
    print()
    pure = next(r for r in res if r["name"] == "mingru")
    hyb = next(r for r in res if r["name"] == "hybrid")
    ceil = next(r for r in res if r["name"] == "attn")
    # THE GUARD. Pure attention provably can do induction — it is the textbook
    # task for it. If the ceiling arm is at chance, the harness is broken and
    # nothing below it can be ranked, however clean the other rows look. Two
    # earlier runs of this file printed a confident verdict over exactly this
    # failure; the guard exists so a third cannot.
    if ceil["acc"] < 0.5:
        print("  HARNESS FAILURE — the pure-attention CEILING arm scored"
              f" {ceil['acc']:.1%}.")
        print("  Attention can solve induction; at chance it means this harness is")
        print("  not training, so NO comparison below it is valid. Do not read the")
        print("  mingru/hybrid rows. Check: loss stuck near ln(V_FILL)=3.87 means")
        print("  the model never left init — widen CB_LRS or raise CB_STEPS.")
        return
    if pure["acc"] > 0.5 and pure["acc"] >= hyb["acc"] * 0.9:
        print("  VERDICT: pure recurrence learns induction at parity with the hybrid.")
        print("  Attention is NOT mechanistically required for this circuit, the")
        print("  hybrid question closes, and cubby-lm's alpha_attn -> 0.109 is")
        print("  consistent with attention simply not earning its place.")
    elif hyb["acc"] > 0.5 and pure["acc"] < 0.5:
        print("  VERDICT: the hybrid learns induction and pure recurrence does not.")
        print("  This is the ablation neither repo ran. It says CubbyLLM's 0%-attention")
        print("  backbone cannot form the circuit that copying and needle recall need,")
        print("  and that the drift away from cubby-lm's shape was load-bearing.")
    else:
        print("  VERDICT: inconclusive — neither arm cleared 50%. Raise CB_STEPS or")
        print("  shorten CB_S before reading anything into the comparison; a task no")
        print("  arm can learn ranks nothing.")


if __name__ == "__main__":
    main()
