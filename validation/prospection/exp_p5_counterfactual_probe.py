"""H-P5 — counterfactual re-entry: snapshot at a choice point, choose
differently, and does the continuation switch to what WOULD have happened?

CLAIM. With a forkable state (H-P1) a rung-three counterfactual is mechanical:
abduce the situation at the choice point (it IS the state), intervene (feed the
other branch token), re-run. On the branch grammar the counterfactual has a
ground truth — template_j instead of template_k — so it is scoreable, which
history never is. Three things a counterfactual engine has to get right, each
measured with labels here:
  * FRAGILE outcomes flip. The first token after the shared prefix is
    branch-specific: under the intervention the model must predict branch j's
    token, not branch k's.
  * OVERDETERMINED outcomes do not. The shared-prefix tokens follow either
    branch; the model's predictions there must be unchanged by the
    intervention. (Alexander or no Alexander, Philip's army still marches.)
  * WASHOUT. At the next SEP the next choice is independent of the branch:
    the factual and counterfactual predictive distributions there must agree
    (KL ~ 0) even though the states that produced them differ.

MEASURE. One pass over held-out text, snapshotting the state at every SEP.
From each snapshot: a factual run (B_k + template_k + SEP) and a
counterfactual run (B_j + template_j + SEP, j != k). Teacher-forced greedy
accuracy on the template for both (factual is the CONTROL — if it is low the
probe is broken, not the model); agreement on overdetermined positions; flip
rate on the fragile position; KL between the two next-choice distributions at
the following SEP; the step-by-step state-distance profile between the runs.

KILL. Counterfactual accuracy near chance, or overdetermined predictions
changing under the intervention, or no washout: the state does not support
re-entry, and the counterfactual-history capability cannot be built on it
as-is. A model that passes here has rung two and the mechanics of rung three
on a world it fully knows; what it does NOT have is a causal model of any
world it does not fully know — that is the graph's job, not the state's.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import _common as C  # noqa: E402


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp, lq = F.log_softmax(p_logits.float(), -1), F.log_softmax(q_logits.float(), -1)
    ok = torch.isfinite(lp) & torch.isfinite(lq)
    return float((lp[ok].exp() * (lp[ok] - lq[ok])).sum())


@torch.no_grad()
def probe(model, g: C.BranchGrammar, toks: np.ndarray, ks: np.ndarray, rng) -> dict:
    ids = torch.from_numpy(toks)
    n_seg = len(ks)
    acc_f = acc_cf = over_agree = fragile_flip = fragile_right = 0
    n_over = n_frag = n_tpl = 0
    kls, dist_prof = [], []
    state, lg = None, None
    for t in range(ids.shape[0]):
        lg, state = model.step(ids[t].view(1), state)
        seg, off = divmod(t, g.seg_len)
        if off != 0 or seg == 0 or seg >= n_seg - 1:
            continue                                            # snapshot only at SEP, skipping ends
        k = int(ks[seg])
        j = (k + 1 + int(rng.integers(0, g.K - 1))) % g.K          # a different branch
        run_f = torch.from_numpy(np.concatenate([[1 + k], g.templates[k], [g.SEP]]))
        run_cf = torch.from_numpy(np.concatenate([[1 + j], g.templates[j], [g.SEP]]))
        tr_f, tr_cf = [], []
        out_f, _ = C.continue_from(model, state, run_f, track=tr_f)
        out_cf, _ = C.continue_from(model, state, run_cf, track=tr_cf)
        # out[i] is the prediction made AFTER feeding run[i]: out[0] predicts t1, ...,
        # out[M-1] predicts tM, out[M] predicts SEP, out[M+1] predicts the next B.
        pred_f, pred_cf = out_f.argmax(-1), out_cf.argmax(-1)
        tgt_f, tgt_cf = run_f[1:], run_cf[1:]
        acc_f += int((pred_f[:g.M] == tgt_f[:g.M]).sum())
        acc_cf += int((pred_cf[:g.M] == tgt_cf[:g.M]).sum())
        n_tpl += g.M
        over_agree += int((pred_f[:g.shared] == pred_cf[:g.shared]).sum())   # overdetermined
        n_over += g.shared
        fragile_flip += int(pred_f[g.shared] != pred_cf[g.shared])           # fragile
        fragile_right += int(pred_cf[g.shared] == tgt_cf[g.shared])
        n_frag += 1
        kls.append(kl(out_f[-1], out_cf[-1]))                                 # washout at SEP
        dist_prof.append([float((a - b).norm()) for a, b in zip(tr_f, tr_cf)])
    prof = np.mean(np.asarray(dist_prof), axis=0)
    return dict(acc_factual=acc_f / n_tpl, acc_counterfactual=acc_cf / n_tpl,
                overdetermined_agree=over_agree / n_over, fragile_flip=fragile_flip / n_frag,
                fragile_correct=fragile_right / n_frag, washout_kl=float(np.mean(kls)),
                dist_profile=prof, n=n_frag)


def run(quick: bool = False, verbose: bool = True) -> dict:
    steps = 150 if quick else 400
    model, g, _ = C.trained_toy(steps=steps)
    rng = np.random.default_rng(321)
    toks, ks = g.sample(40 if quick else 120, rng)
    r = probe(model, g, toks, ks, rng)
    if verbose:
        prof = " ".join(f"{v:.2f}" for v in r["dist_profile"])
        C.banner(f"[toy model, {r['n']} choice points, K={g.K}, shared prefix {g.shared}, template {g.M}]", [
            f"factual teacher-forced accuracy (control): {r['acc_factual']:.3f}",
            f"COUNTERFACTUAL accuracy vs template_j:      {r['acc_counterfactual']:.3f}  (chance ~ {1/g.C:.3f})",
            f"overdetermined positions unchanged:        {r['overdetermined_agree']:.3f}",
            f"fragile position flipped / correct:        {r['fragile_flip']:.3f} / {r['fragile_correct']:.3f}",
            f"washout KL(factual || counterfactual) at next SEP: {r['washout_kl']:.4f} nats",
            f"|h_f - h_cf| per step (B, t1..tM, SEP):    {prof}",
        ])
    floor = 0.8 if quick else 0.9
    assert r["acc_factual"] > floor, "factual control failed — the probe, not the claim, is broken"
    assert r["acc_counterfactual"] > floor, "counterfactual continuation does not switch branch"
    assert r["overdetermined_agree"] > floor, "intervention changed overdetermined predictions"
    assert r["fragile_flip"] > floor and r["fragile_correct"] > floor, "fragile outcome did not flip"
    # Washout is a READOUT property here, not a state property: the distance
    # profile above shows |h_f - h_cf| is still large at the next SEP (the toy
    # gates decay at tau ~ 3, but the template inputs differ for 6 steps), yet
    # the next-choice distributions agree to a few hundredths of a nat. The
    # honest number to carry forward is how far an intervention's residue
    # persists in state past the boundary — measured, not assumed to be zero.
    assert r["washout_kl"] < (0.25 if quick else 0.15), "next choice still depends on the intervened branch"
    return r


def main():
    print("H-P5 counterfactual probe — snapshot, intervene, re-run: does the branch switch?\n")
    run()
    print("PASS: re-entry from a snapshot with a substituted choice yields the ground-truth "
          "counterfactual; overdetermined tokens stay put, the fragile one flips, and the next "
          "choice washes out.")


if __name__ == "__main__":
    main()
