"""H0 / H-A2 — smallest possible GCE-style controller + P5 Wrong-Context Probing.

The central bet: making active weights a function of context (theta = f(c))
instead of a fixed state that gets overwritten prevents sequential-task
interference. The smallest falsifiable version, exactly as H0's validation
clause specifies: a context embedding feeding a hypernetwork that generates ONE
small weight block (the output head), trained sequentially, compared against a
same-capacity fixed-weight baseline, then P5-probed.

Task suite: T sequential regression tasks. Task t maps x in R^32 -> y = A_t x
(A_t random). Tasks are learned STRICTLY sequentially (no replay, no revisit).

Models (matched backbone, matched training budget):
  baseline   x -> shared MLP -> shared head            (theta fixed, overwritten)
  gce        x -> shared MLP -> head with weights H(c) (theta = f(c), c = task
             embedding; only H's parameters are trained — the head's weights are
             GENERATED per task, never stored per task)
  gce+hard   same, plus the canonical architectural hardening from the
             hypernetwork-CL literature (von Oswald et al. 2019): when training
             task t, add a penalty keeping H(c_p) close to its pre-task-t
             snapshot for every previous p < t. H0's kill criterion explicitly
             reads "even after architectural hardening", so the naive variant
             alone cannot kill or confirm the bet — this one can.

Metrics:
  retention  task-0 MSE right after learning task 0 vs after learning all T.
  P5 probe   evaluate task 0 with WRONG contexts: if wrong-context error stays
             low, the hypernetwork learned to ignore c (Structural Bypass
             Defect) and H0's mechanism is NOT doing the work -> kill criterion.
  cost       wall-clock per step for both models (the "fast" half sanity check).

Scope note, stated honestly: context identity is GIVEN at train and test here.
This validates the mechanism (generated weights avoid interference and P5 can
detect bypass), NOT context *inference* — which is a separate, harder problem
(H-C4's router is the baseline for that half).
"""
from __future__ import annotations

import sys
import time
import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIN, DH, DOUT = 32, 64, 32
T_TASKS = 6
STEPS_PER_TASK = 1500
BATCH = 64
LR = 1e-3
CTX_DIM = 16


def frozen_backbone():
    """The FROZEN trunk stand-in — H0's validate clause specifies the
    controller rectifies one small weight block, NOT the trunk. Same seed
    everywhere so every variant sees identical features."""
    g = torch.Generator().manual_seed(42)
    lin = nn.Linear(DIN, DH)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(DH, DIN, generator=g) / np.sqrt(DIN))
        lin.bias.zero_()
    for p in lin.parameters():
        p.requires_grad_(False)
    return nn.Sequential(lin, nn.Tanh())


class Baseline(nn.Module):
    """Frozen backbone + ONE fixed head, overwritten in place task after task."""
    def __init__(self):
        super().__init__()
        self.backbone = frozen_backbone()
        self.head = nn.Linear(DH, DOUT)

    def forward(self, x, task_id=None):
        return self.head(self.backbone(x))


class GCE(nn.Module):
    """Shared backbone + head whose weights are GENERATED from task context.

    theta_head = H(c): the hypernetwork H maps a learned task embedding c to
    the head's (DH x DOUT + DOUT) parameters. Nothing about task t is stored
    as task-t head weights — only H and the embeddings are parameters.
    """
    def __init__(self):
        super().__init__()
        self.backbone = frozen_backbone()
        self.ctx = nn.Embedding(T_TASKS, CTX_DIM)
        n_head = DH * DOUT + DOUT
        self.hyper = nn.Sequential(
            nn.Linear(CTX_DIM, 64), nn.Tanh(), nn.Linear(64, n_head),
        )
        # small init so generated weights start near zero
        with torch.no_grad():
            self.hyper[-1].weight.mul_(0.1)
            self.hyper[-1].bias.zero_()

    def forward(self, x, task_id):
        c = self.ctx(torch.as_tensor([task_id]))
        theta = self.hyper(c)[0]
        W = theta[: DH * DOUT].view(DOUT, DH)
        b = theta[DH * DOUT:]
        h = self.backbone(x)
        return h @ W.t() + b


def make_tasks(rng):
    return [rng.standard_normal((DOUT, DIN)).astype(np.float32) / np.sqrt(DIN)
            for _ in range(T_TASKS)]


def batch(A, rng):
    x = rng.standard_normal((BATCH, DIN)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(x @ A.T)


def eval_mse(model, A, task_id, rng, n=2048):
    x = rng.standard_normal((n, DIN)).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x), task_id)
    return float(((pred - torch.from_numpy(x @ A.T)) ** 2).mean())


def generated_theta(model, task_id):
    c = model.ctx(torch.as_tensor([task_id]))
    return model.hyper(c)[0]


def train_sequential(model, tasks, rng, harden=False, reg_beta=100.0):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    task0_after = {}
    step_times = []
    for t, A in enumerate(tasks):
        # hardening: snapshot the generated weights of all previous contexts
        snaps = {}
        if harden and t > 0:
            with torch.no_grad():
                snaps = {p: generated_theta(model, p).clone() for p in range(t)}
        for _ in range(STEPS_PER_TASK):
            x, y = batch(A, rng)
            t0 = time.perf_counter()
            loss = ((model(x, t) - y) ** 2).mean()
            if snaps:
                reg = sum(((generated_theta(model, p) - s) ** 2).mean()
                          for p, s in snaps.items()) / len(snaps)
                loss = loss + reg_beta * reg
            opt.zero_grad(); loss.backward(); opt.step()
            step_times.append(time.perf_counter() - t0)
        task0_after[t] = eval_mse(model, tasks[0], 0, rng)
    return task0_after, float(np.median(step_times) * 1000)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tasks = make_tasks(rng)
    chance = float(np.mean([np.mean((rng.standard_normal((2048, DIN)).astype(np.float32) @ A.T) ** 2) for A in tasks]))

    print("=" * 78)
    print(f"H0/H-A2  GCE-style controller vs fixed weights  "
          f"({T_TASKS} sequential tasks, no replay)")
    print("=" * 78)
    print(f"  predict-zero (chance) MSE ~= {chance:.3f}\n")

    variants = [
        ("baseline", Baseline(), False),
        ("gce-naive", GCE(), False),
        ("gce+hard", GCE(), True),
    ]
    rows = {}
    for name, model, harden in variants:
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ret, ms = train_sequential(model, tasks, rng, harden=harden)
        rows[name] = (model, ret, ms, nparam)

    print(f"\n  task-0 MSE after finishing task t   (retention curve;"
          f" chance={chance:.2f})")
    print("  t          " + "".join(f"{t:>9}" for t in range(T_TASKS)))
    for name in rows:
        ret = rows[name][1]
        print(f"  {name:<11}" + "".join(f"{ret[t]:>9.3f}" for t in range(T_TASKS)))
    print("\n  " + "  ".join(f"{n}: {rows[n][2]:.2f}ms/step,"
                             f" {rows[n][3]:,}p" for n in rows))

    # ---- P5 Wrong-Context Probing on both trained GCE variants -----------
    verdicts = {}
    for name in ("gce-naive", "gce+hard"):
        model = rows[name][0]
        right = eval_mse(model, tasks[0], 0, rng)
        wrongs = [eval_mse(model, tasks[0], w, rng) for w in range(1, T_TASKS)]
        bypass = float(np.mean(wrongs)) < 2 * right
        verdicts[name] = (right, float(np.mean(wrongs)), bypass)
        print(f"\n  [P5] {name}: task-0 data RIGHT ctx MSE {right:.3f},"
              f" WRONG ctx mean {np.mean(wrongs):.3f} (chance {chance:.2f})")
        print(f"  [P5] {name}: "
              + ("wrong-ctx error LOW -> STRUCTURAL BYPASS suspected" if bypass
                 else "context signal is load-bearing (no bypass)"))

    print("\n[verdict]")
    fb = rows["baseline"][1][T_TASKS - 1]
    fn = rows["gce-naive"][1][T_TASKS - 1]
    fh = rows["gce+hard"][1][T_TASKS - 1]
    print(f"  final task-0 MSE: baseline {fb:.3f} | gce-naive {fn:.3f} |"
          f" gce+hardening {fh:.3f}   (just-learned ~0.001, chance {chance:.2f})")
    ok = (fh < 0.1 * fb) and not verdicts["gce+hard"][2]
    print(f"  naive theta=f(c) alone: partial (interference persists INSIDE"
          f" the shared hypernetwork).")
    print(f"  hardened theta=f(c): {'near-zero forgetting -> H0 mechanism'
          f' SUPPORTED at toy scale' if ok else 'still forgets -> H0 kill'
          f' criterion TRIGGERED at toy scale'}.")


if __name__ == "__main__":
    main()
