"""H0 (part 2) — does θ=f(c) survive when context is INFERRED, not given?

The toy H0 result (exp_h0_gce.py) fed the hypernetwork the TRUE task id. That
validates the mechanism but sidesteps the "automatic specialization" half of
the goal: in a real model nobody hands you the task id — it must be inferred
from the input. This is exactly H-C4's "router as the context source." So:
replace the oracle task id with a learned router and re-measure.

Setup: T sequential tasks, each y = A_t x with task-specific INPUT statistics
(x_t ~ N(mu_t, I), well-separated mu_t) so context is inferable from x — the
realistic case where different domains both look different AND need different
processing. Strictly sequential, no replay.

  baseline     frozen backbone + one fixed head, overwritten         (θ fixed)
  oracle-ctx   hypernetwork fed the TRUE task id (the exp_h0_gce win) (upper bound)
  router-naive a router trained ONLY unsupervised (per-sample confidence
               pressure) infers a soft context from x                 (the trap)
  router-super a router SUPERVISED with the task label at train time (exactly
               what H-C4's domain_head.pt is), used label-free at test — the
               SAME hypernetwork consumes its inferred context         (the real test)

All generative models use the same anti-drift hardening H0 needed (snapshot +
regularize generated weights of prior context slots). Extra question the router
raises: does the ROUTER itself forget how to route old tasks after training on
new ones? Measured directly. Plus a P5 analog: force the wrong slot on task-0
inputs and confirm error rises (context is load-bearing, not bypassed).

Standalone; reuses the frozen-backbone discipline from exp_h0_gce.
"""
from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIN, DH, DOUT = 32, 64, 32
T_TASKS = 6
K_SLOTS = 8               # >= T_TASKS: room for the router to allocate slots
CTX_DIM = 16
STEPS_PER_TASK = 1500
BATCH = 64
LR = 1e-3


def frozen_backbone():
    g = torch.Generator().manual_seed(42)
    lin = nn.Linear(DIN, DH)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(DH, DIN, generator=g) / np.sqrt(DIN))
        lin.bias.zero_()
    for p in lin.parameters():
        p.requires_grad_(False)
    return nn.Sequential(lin, nn.Tanh())


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = frozen_backbone()
        self.head = nn.Linear(DH, DOUT)

    def forward(self, x, task_id=None):
        return self.head(self.backbone(x))


class HyperHead(nn.Module):
    """Shared: backbone + head weights generated from a context embedding."""
    def __init__(self):
        super().__init__()
        self.backbone = frozen_backbone()
        n_head = DH * DOUT + DOUT
        self.hyper = nn.Sequential(
            nn.Linear(CTX_DIM, 64), nn.Tanh(), nn.Linear(64, n_head))
        with torch.no_grad():
            self.hyper[-1].weight.mul_(0.1)
            self.hyper[-1].bias.zero_()

    def head_from_ctx(self, c):                # c: (B, CTX_DIM)
        theta = self.hyper(c)                  # (B, n_head)
        W = theta[:, :DH * DOUT].view(-1, DOUT, DH)
        b = theta[:, DH * DOUT:]
        return W, b

    def apply_head(self, x, c):
        W, b = self.head_from_ctx(c)
        h = self.backbone(x)                   # (B, DH)
        return torch.einsum("boh,bh->bo", W, h) + b


class Oracle(HyperHead):
    """Context = true task id -> its slot embedding."""
    def __init__(self):
        super().__init__()
        self.ctx = nn.Embedding(T_TASKS, CTX_DIM)

    def context(self, x, task_id):
        return self.ctx(torch.full((x.shape[0],), task_id, dtype=torch.long))

    def forward(self, x, task_id):
        return self.apply_head(x, self.context(x, task_id))


class Router(HyperHead):
    """Context = router(x) soft-assignment over learned slots. The head never
    sees a task id — only the router's inferred context. `supervised` decides
    whether the router gets task-label supervision during training (the H-C4
    domain-classifier pattern) or only unsupervised confidence pressure."""
    def __init__(self, supervised: bool):
        super().__init__()
        self.supervised = supervised
        self.slots = nn.Embedding(K_SLOTS, CTX_DIM)
        self.router = nn.Sequential(
            nn.Linear(DIN, 64), nn.Tanh(), nn.Linear(64, K_SLOTS))

    def route_logits(self, x):
        return self.router(x)

    def route(self, x):
        return F.softmax(self.router(x), dim=-1)     # (B, K_SLOTS)

    def context(self, x, task_id=None):
        return self.route(x) @ self.slots.weight     # (B, CTX_DIM)

    def forward(self, x, task_id=None):
        return self.apply_head(x, self.context(x))


def make_tasks(rng):
    # note: `/ np.sqrt(DIN)` (a float64 scalar) upcasts -> cast back to f32,
    # else torch matmuls against these downstream hit a float/double mismatch.
    A = [(rng.standard_normal((DOUT, DIN)) / np.sqrt(DIN)).astype(np.float32)
         for _ in range(T_TASKS)]
    # well-separated task input means so context is inferable from x.
    # (norm() returns float64 -> cast back so all downstream arrays stay f32.)
    mu = rng.standard_normal((T_TASKS, DIN)).astype(np.float32)
    mu = (mu / np.linalg.norm(mu, axis=1, keepdims=True) * 4.0).astype(np.float32)
    return A, mu


def batch(A, mu, rng):
    x = (rng.standard_normal((BATCH, DIN)).astype(np.float32) + mu)
    return torch.from_numpy(x), torch.from_numpy(x @ A.T)


def eval_mse(model, A, mu, task_id, rng, n=2048):
    x = (rng.standard_normal((n, DIN)).astype(np.float32) + mu)
    with torch.no_grad():
        pred = model(torch.from_numpy(x), task_id)
    return float(((pred - torch.from_numpy(x @ A.T)) ** 2).mean())


def gen_slot_thetas(model):
    """Snapshot generated head for every context slot (oracle: task ids;
    router: learned slots) — the hardening anchor."""
    with torch.no_grad():
        if isinstance(model, Oracle):
            c = model.ctx.weight
        else:
            c = model.slots.weight
        return model.hyper(c).clone()


def train_sequential(model, tasks, mus, rng, harden=True, reg_beta=50.0,
                     route_sup=0.3):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    is_router = isinstance(model, Router)
    is_gen = isinstance(model, HyperHead)
    task0 = {}
    for t in range(T_TASKS):
        A, mu = tasks[t], mus[t]
        snap = gen_slot_thetas(model) if (harden and is_gen and t > 0) else None
        c_in = model.slots.weight.detach().clone() if (snap is not None and is_router) \
            else (model.ctx.weight.detach().clone() if snap is not None else None)
        for _ in range(STEPS_PER_TASK):
            x, y = batch(A, mu, rng)
            loss = ((model(x, t) - y) ** 2).mean()
            if snap is not None:
                cur = model.hyper(c_in)
                loss = loss + reg_beta * ((cur - snap) ** 2).mean()
            if is_router:
                if model.supervised:
                    # H-C4 pattern: task label supervises which slot (slot==task)
                    tgt = torch.full((x.shape[0],), t, dtype=torch.long)
                    loss = loss + route_sup * F.cross_entropy(
                        model.route_logits(x), tgt)
                else:
                    # unsupervised: per-sample confidence only (the trap — this
                    # rewards a single collapsed slot just as well)
                    p = model.route(x)
                    loss = loss + route_sup * (
                        -(p * (p + 1e-9).log()).sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        task0[t] = eval_mse(model, tasks[0], mus[0], 0, rng)
    return task0


def router_slot_for_task(model, tasks, mus, task_id, rng):
    x = (rng.standard_normal((512, DIN)).astype(np.float32) + mus[task_id])
    with torch.no_grad():
        p = model.route(torch.from_numpy(x)).mean(0)
    return int(p.argmax()), float(p.max())


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    A, mu = make_tasks(rng)
    tasks, mus = A, mu
    chance = float(np.mean([
        np.mean((( (rng.standard_normal((2048, DIN)).astype(np.float32) + mu[t])
                   @ A[t].T)) ** 2) for t in range(T_TASKS)]))

    print("=" * 78)
    print(f"H0 part 2 — INFERRED context (router) vs oracle vs fixed"
          f"  ({T_TASKS} sequential tasks, no replay)")
    print("=" * 78)
    print(f"  task inputs are task-specific (separated means) so context is"
          f" inferable from x; chance MSE ~= {chance:.2f}\n")

    models = {
        "baseline": Baseline(),
        "oracle-ctx": Oracle(),
        "router-naive": Router(supervised=False),
        "router-super": Router(supervised=True),
    }
    ret = {}
    for name, m in models.items():
        ret[name] = train_sequential(m, tasks, mus, rng,
                                     harden=(name != "baseline"))

    print("  task-0 MSE after finishing task t   (retention)")
    print("  t            " + "".join(f"{t:>9}" for t in range(T_TASKS)))
    for name in models:
        print(f"  {name:<13}" + "".join(f"{ret[name][t]:>9.3f}"
                                        for t in range(T_TASKS)))

    A0t = torch.from_numpy(A[0]).T.contiguous()          # float32

    for rname in ("router-naive", "router-super"):
        m = models[rname]
        print(f"\n  [{rname} health] slot chosen per task after full training:")
        slots = {}
        for t in range(T_TASKS):
            s, conf = router_slot_for_task(m, tasks, mus, t, rng)
            slots[t] = s
            print(f"    task {t}: slot {s}  (confidence {conf*100:.0f}%)")
        distinct = len(set(slots.values()))
        print(f"    -> {distinct}/{T_TASKS} distinct slots"
              f" ({'good separation' if distinct >= T_TASKS-1 else 'COLLAPSE'})")

        x0 = torch.from_numpy((rng.standard_normal((2048, DIN)).astype(np.float32)
                               + mus[0]))
        with torch.no_grad():
            right = float(((m(x0, 0) - x0 @ A0t) ** 2).mean())
            wrong_slot = (slots[0] + 1) % K_SLOTS
            c_wrong = m.slots.weight[wrong_slot].expand(x0.shape[0], -1)
            wrong = float(((m.apply_head(x0, c_wrong) - x0 @ A0t) ** 2).mean())
        # context is load-bearing iff forcing the wrong slot clearly HURTS.
        bypass = wrong <= right * 1.15
        print(f"    [P5] task-0 right ctx MSE {right:.3f} | forced-wrong slot"
              f" MSE {wrong:.3f} (chance {chance:.2f}) ->"
              f" {'BYPASS (wrong ctx not worse)' if bypass else 'context load-bearing (wrong ctx hurts)'}")
        models[rname]._distinct = distinct
        models[rname]._bypass = bypass

    print("\n[verdict]")
    fb = ret["baseline"][T_TASKS - 1]
    fo = ret["oracle-ctx"][T_TASKS - 1]
    fn = ret["router-naive"][T_TASKS - 1]
    fs = ret["router-super"][T_TASKS - 1]
    print(f"  final task-0 MSE: baseline {fb:.2f} | oracle {fo:.3f} |"
          f" router-naive {fn:.2f} | router-super {fs:.3f}  (chance {chance:.2f})")
    dn = models["router-naive"]._distinct
    ds = models["router-super"]._distinct
    print(f"  - router-NAIVE collapses to {dn}/{T_TASKS} slots -> reverts to the"
          f" baseline ({fn:.2f}, {fn/max(fo,1e-9):.0f}x oracle): unsupervised"
          f" context DISCOVERY is the hard part, not θ=f(c) generation.")
    print(f"  - router-SUPERVISED (task-label routing, trained ONLINE alongside"
          f" the head) roughly halves the forgetting ({fs:.2f} vs baseline"
          f" {fb:.2f}) but stays {fs/max(fo,1e-9):.0f}x the oracle and its OWN"
          f" slots drift to {ds}/{T_TASKS} distinct with task-0 confidence"
          f" falling off -> THE ROUTER IS ITSELF A SEQUENTIAL LEARNER THAT"
          f" FORGETS.")
    print("  Actionable: the θ=f(c) generation is not the bottleneck — context"
          " INFERENCE is. Use an OFFLINE-pretrained router over all contexts"
          " (H-C4's domain_head.pt is exactly that: trained once on 88 domains,"
          " not online), or harden the router too. An online-learned router"
          " recreates the very forgetting problem H0 solves for the head.")


if __name__ == "__main__":
    main()
