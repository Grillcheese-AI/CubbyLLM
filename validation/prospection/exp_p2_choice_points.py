"""H-P2 — choice points are SPARSE, and the model's own entropy localizes them.

CLAIM. Most next-token predictions are not choices. The branching landscape
only opens where the predictive distribution is genuinely spread, and if the
model's entropy marks exactly those positions then "branch only at high-entropy
steps" is a budget rule with a real signal behind it (rats do vicarious trial
and error only at decision points, and less as the task becomes habitual). On
the branch grammar the ground truth is known: exactly one free choice per
segment (which B_k follows SEP, entropy ln K), every other next-token
deterministic (entropy 0). So this is a detector test with labels.

MEASURE. Train the toy model; over held-out text compute per-position entropy
on BOTH paths (parallel forward, and the decode ``step`` path a budget rule
would actually run on); AUROC of entropy against the true choice positions;
the fraction of positions above ln(K)/2 (the sparsity a prospection budget
sees); training loss against the grammar's floor ln(K)/seg_len (is the model
at the floor, or is "entropy" just untrained noise?).

KILL. AUROC well under 0.9, or the flagged fraction far from 1/seg_len: entropy
does not localize choices and the budget rule needs a different signal.

With CB_CKPT + CUBBY_SPM + CB_TEXT (or CB_CORPUS): the entropy quantiles and
the fraction of positions above 1 / 2 / 3 nats on real text. No labels there,
but that fraction IS the number that sets a real prospection budget.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _common as C  # noqa: E402


@torch.no_grad()
def entropy_profiles(model, toks: np.ndarray):
    """Per-position next-token entropy: (parallel forward, decode step)."""
    x = torch.from_numpy(toks[:-1]).view(1, -1)
    h_fwd = C.entropy(model.forward(x)[0]).numpy()
    s, hs = None, []
    for t in x[0]:
        lg, s = model.step(t.view(1), s)
        hs.append(float(C.entropy(lg[0])))
    return h_fwd, np.asarray(hs)


def run(quick: bool = False, verbose: bool = True) -> dict:
    steps = 150 if quick else 400
    model, g, losses = C.trained_toy(steps=steps)
    toks, _ = g.sample(60 if quick else 200, np.random.default_rng(123))   # held-out
    mask = g.choice_mask(toks)[:-1]
    h_fwd, h_step = entropy_profiles(model, toks)
    thr = 0.5 * math.log(g.K)
    r = dict(
        auroc_fwd=C.auroc(h_fwd, mask), auroc_step=C.auroc(h_step, mask),
        frac_flagged=float((h_step > thr).mean()), true_frac=1.0 / g.seg_len,
        h_choice=float(h_step[mask].mean()), h_other=float(h_step[~mask].mean()),
        loss=float(np.mean(losses[-20:])), floor=g.entropy_floor, ln_k=math.log(g.K),
    )
    if verbose:
        C.banner(f"[branch grammar K={g.K} seg_len={g.seg_len}, toy model, {steps} steps]", [
            f"train loss {r['loss']:.3f} nats/token vs grammar floor {r['floor']:.3f} "
            f"(ln K = {r['ln_k']:.3f} once per segment)",
            f"entropy at true choice points {r['h_choice']:.3f} vs elsewhere {r['h_other']:.3f} (step path)",
            f"AUROC(entropy -> choice point): forward {r['auroc_fwd']:.3f} | step {r['auroc_step']:.3f}",
            f"fraction flagged above ln(K)/2: {r['frac_flagged']:.3f} (true choice fraction {r['true_frac']:.3f})",
        ])
    floor_auc, floor_gap = (0.85, 0.3) if quick else (0.95, 0.6)
    assert r["auroc_step"] > floor_auc, f"entropy does not localize choices (AUROC {r['auroc_step']:.3f})"
    assert r["h_choice"] - r["h_other"] > floor_gap, "no entropy contrast between choice and non-choice"
    assert abs(r["frac_flagged"] - r["true_frac"]) < (0.12 if quick else 0.06), "flagged fraction off"

    if C.CKPT:
        model, meta = C.load_checkpoint(C.CKPT)
        ids = C.real_tokens(4096)
        if ids is None:
            print("CB_CKPT set but no CB_TEXT/CB_CORPUS (+CUBBY_SPM): skipping real-text profile")
        else:
            hs = []
            with torch.no_grad():
                for i in range(0, len(ids) - 1, 512):
                    x = torch.from_numpy(ids[i:i + 513][:-1]).view(1, -1)
                    hs.append(C.entropy(model.forward(x)[0]).numpy())
            h = np.concatenate(hs)
            r["real"] = dict(q50=float(np.quantile(h, .5)), q90=float(np.quantile(h, .9)),
                             above=[float((h > t).mean()) for t in (1.0, 2.0, 3.0)])
            if verbose:
                C.banner(f"[checkpoint {os.path.basename(C.CKPT)} on {len(h)} real positions]", [
                    f"entropy median {r['real']['q50']:.2f} nats, p90 {r['real']['q90']:.2f}",
                    "fraction above 1 / 2 / 3 nats: " + " / ".join(f"{v:.3f}" for v in r["real"]["above"]),
                    "-> that fraction is the share of steps a prospection budget would open branches at",
                ])
    return r


def main():
    print("H-P2 choice points — does entropy localize the real choices, and how sparse are they?\n")
    run()
    print("PASS: entropy is a usable choice-point detector on the toy; the budget rule has a signal.")


if __name__ == "__main__":
    main()
