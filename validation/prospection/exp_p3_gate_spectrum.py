"""H-P3 — the retention gate's time-constant spectrum: what the substrate can
hold, at init, after training, and under chrono init.

CLAIM. Learning across timescales needs a state that can RETAIN across those
timescales; otherwise neither an eligibility trace nor an episodic write-back
can credit a choice that happened long ago, because the state no longer
carries it. MinGRU's gate sets a per-unit time constant tau = -1/ln(a). Three
substrate facts to pin, all properties of ``mingru.py`` as written:
  (1) at init (proj_d bias = 1 -> a ~ 0.73) tau ~ 3 steps for EVERY unit —
      nothing is long-memory until training happens to move it there;
  (2) a = 0.001 + 0.998*sigmoid(.) caps a at 0.999, so tau can never exceed
      ~1000 steps — a ceiling of the parameterization, not of the data. If a
      dependency needs a slower unit than that, this gate cannot represent it;
  (3) chrono init (Tallec & Ollivier 2018) spreads tau log-uniformly over
      decades at init — the shape cortex shows (Murray et al. 2014, a hierarchy
      of intrinsic timescales) — without touching the scan or the decode path.

MEASURE. Per-unit tau from the geometric-mean retention over real inputs, per
layer: p10/p50/p90 and the fraction of units slower than 8 and 100 steps, for
(a) init, (b) the toy-trained model — its longest dependency is only shared+1
= 3 steps (branch id -> first divergent token), so the honest expectation is
that training moves NO unit past tau ~ 8: gates lengthen only as far as the
data demands, which is exactly why a long-horizon substrate has to be set by
init or by a long-horizon objective — (c) chrono init. Plus a pin that the
retention formula used here equals what ``_MinGRUMixer.step`` multiplies by.

Optional (CB_P3_TRAIN=1): delayed-copy at lags 8/32/128, default vs chrono
init, matched steps — prints final loss per lag. Informational at this scale;
the chrono claim's kill (no gain at long lags) belongs to a scale run.

KILL for what IS asserted: any of (1)-(3) failing means the substrate story in
the Group P write-up is wrong about this backbone and must be corrected.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import _common as C  # noqa: E402


def describe(taus: dict) -> tuple[list[str], dict]:
    lines, stats = [], {}
    for i, tau in taus.items():
        s = dict(p10=C.pct(tau, .10), p50=C.pct(tau, .50), p90=C.pct(tau, .90),
                 slow8=float((tau >= 8).float().mean()), slow100=float((tau >= 100).float().mean()))
        stats[i] = s
        lines.append(f"layer {i}: tau p10 {s['p10']:6.1f} | p50 {s['p50']:6.1f} | p90 {s['p90']:6.1f}"
                     f" | >=8 steps {s['slow8']:.2f} | >=100 steps {s['slow100']:.2f}")
    return lines, stats


@torch.no_grad()
def formula_pin(model) -> float:
    """|a_from_formula - a_the_step_uses|: h = a*h_prev + x_scan with h_prev = 1."""
    mixer = C.mingru_mixers(model.backbone)[0][1]
    xt = torch.randn(4, model.config.d_model)
    h, _ = mixer.step(xt, torch.ones_like(xt))
    x_scan = torch.sigmoid(mixer.proj_g(xt)) * torch.tanh(mixer.proj_v(xt))
    return float(((h - x_scan) - C.retention_from_gate_logits(mixer.proj_d(xt))).abs().max())


def delayed_copy(lag: int, chrono: bool, steps: int, V: int = 16, d: int = 32, L: int = 2,
                 seed: int = 0) -> float:
    """Predict the token from ``lag`` steps back. Returns the mean of the last 10 losses."""
    model = C.build_model(V=V, d=d, L=L, seed=seed)
    if chrono:
        for _, mx in C.mingru_mixers(model.backbone):
            C.chrono_init_(mx, 1.0, 500.0, seed=seed)
    opt = torch.optim.Adam(list(model.parameters()), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    S, hist = lag + 32, []
    for _ in range(steps):
        x = torch.randint(1, V, (8, S), generator=g)
        y = torch.roll(x, lag, dims=1)
        loss = F.cross_entropy(model.forward(x)[:, lag:].reshape(-1, V), y[:, lag:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
    return float(np.mean(hist[-10:]))


def run(quick: bool = False, verbose: bool = True) -> dict:
    g = C.BranchGrammar()
    toks, _ = g.sample(64, np.random.default_rng(5))
    x = torch.from_numpy(toks[:512]).view(8, 64)

    m_init = C.build_model(V=g.V, d=32, L=2)
    pin = formula_pin(m_init)
    tau_init = C.unit_time_constants(C.retention_profile(m_init, x))
    m_tr, _, _ = C.trained_toy(steps=150 if quick else 400)
    tau_tr = C.unit_time_constants(C.retention_profile(m_tr, x))
    m_ch = C.build_model(V=g.V, d=32, L=2)
    for _, mx in C.mingru_mixers(m_ch.backbone):
        C.chrono_init_(mx, 1.0, 500.0)
    tau_ch = C.unit_time_constants(C.retention_profile(m_ch, x))
    cap = float(C.time_constants(torch.tensor([0.999999], dtype=torch.float64))[0])   # clamps at A_MAX

    li, si = describe(tau_init)
    lt, st = describe(tau_tr)
    lc, sc = describe(tau_ch)
    r = dict(pin=pin, init=si, trained=st, chrono=sc, cap=cap)
    if verbose:
        C.banner("[pin] retention formula vs _MinGRUMixer.step:", [f"max |diff| = {pin:.1e}"])
        C.banner("[a] at init (proj_d bias = 1):", li)
        C.banner(f"[b] after toy training (grammar's longest dependency: {g.shared + 1} steps, "
                 "branch id -> first divergent token):", lt)
        C.banner("[c] chrono init, target tau log-uniform in [1, 500]:", lc)
        C.banner("[ceiling] a <= 0.999 =>", [f"tau can never exceed {cap:.1f} steps on this gate "
                                            f"(TAU_CAP = {C.TAU_CAP:.1f})"])
    assert pin < 1e-5, "retention formula drifted from mingru.py — fix _common first"
    for i, s in si.items():
        assert 1.5 < s["p50"] < 8.0, f"init tau median {s['p50']} outside the bias=1 expectation"
    for i, s in sc.items():
        assert math.log10(s["p90"]) - math.log10(s["p10"]) >= 1.5, "chrono spectrum too narrow"
        assert s["slow100"] >= 0.2, "chrono init produced too few slow units"
    assert abs(cap - C.TAU_CAP) / C.TAU_CAP < 1e-6, "gate ceiling moved — mingru.py's a-range changed?"

    if os.environ.get("CB_P3_TRAIN") == "1":
        steps = 300
        lines = []
        for lag in (8, 32, 128):
            base = delayed_copy(lag, False, steps)
            chr_ = delayed_copy(lag, True, steps)
            r[f"copy_lag{lag}"] = (base, chr_)
            lines.append(f"lag {lag:>3}: default init {base:.3f} | chrono {chr_:.3f}  (ln V = {math.log(16):.3f} = chance)")
        if verbose:
            C.banner(f"[delayed copy, {steps} steps, d=32 L=2 — informational]", lines)

    if C.CKPT:
        # The learned spectrum at scale: did real training move any unit to a
        # long time constant, and how close does it get to the ~1000-step cap?
        model, meta = C.load_checkpoint(C.CKPT)
        ids = C.real_tokens(2048)
        if ids is None:
            print("CB_CKPT set but no CB_TEXT/CB_CORPUS (+CUBBY_SPM): skipping checkpoint spectrum")
        else:
            S = 1024
            xr = torch.from_numpy(ids[: (len(ids) // S) * S]).view(-1, S)
            prof = C.retention_profile(model, xr)
            tau_ck = C.unit_time_constants(prof)
            latch = C.unit_latch_fraction(prof, 100.0)
            lk, sk = describe(tau_ck)
            allt = torch.cat(list(tau_ck.values()))
            alll = torch.cat(list(latch.values()))
            r["ckpt"] = dict(layers=sk, max_tau=float(allt.max()),
                             slow100=float((allt >= 100).float().mean()),
                             slow8=float((allt >= 8).float().mean()),
                             latch10=float((alll >= 0.10).float().mean()),
                             latch50=float((alll >= 0.50).float().mean()),
                             latch_p90=float(torch.quantile(alll, .90)))
            if verbose:
                C.banner(f"[checkpoint {os.path.basename(C.CKPT)} d={meta['D']} L={meta['L']} "
                         f"(recurrent layers only) on {xr.numel()} real tokens]", lk + [
                    f"all recurrent units: >=8 steps {r['ckpt']['slow8']:.3f} | >=100 steps "
                    f"{r['ckpt']['slow100']:.3f} | slowest unit tau = {r['ckpt']['max_tau']:.1f} "
                    f"(cap {C.TAU_CAP:.0f})",
                    "conditional latching (instantaneous tau>=100): units doing it on >=10% of steps "
                    f"{r['ckpt']['latch10']:.3f} | on >=50% of steps {r['ckpt']['latch50']:.3f} | "
                    f"p90 unit latches on {r['ckpt']['latch_p90']:.3f} of steps",
                ])
    return r


def main():
    print("H-P3 gate spectrum — what timescales can the MinGRU state hold?\n")
    run()
    print("PASS: init is uniformly short-memory, the gate ceiling is ~1000 steps, chrono init "
          "spreads the spectrum. Set CB_P3_TRAIN=1 for the delayed-copy arm.")


if __name__ == "__main__":
    main()
