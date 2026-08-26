# validation/prospection/ — Group P: branching futures, timescales, keyframe compression

The "5th dimension" thread, made falsifiable. The design conversation framed
it as a landscape of branching futures indexed by choices; operationally that
is path space, and the question is what the *architecture* buys toward
representing, searching, storing and learning from branches. Every claim
below is a `CUBBYLLM_HYPOTHESES.md` Group P entry with a kill criterion, and
every number in those entries links a log in `../logs/`.

Same rules as the parent directory: standalone scripts, never imported by
`cubbyllm/`, each prints a results block and asserts its own headline numbers,
CPU in seconds. Every script also runs on a real checkpoint — same
`{"meta", "params"}` layout `exp_needle_recall.py` reads — via

    CB_CKPT=D:\...\ckpt.pt CUBBY_SPM=data\grillcheese_bbpe128k.json CB_TEXT=some.txt \
    python validation/prospection/exp_p2_choice_points.py

(`CB_CORPUS`/`CB_SOURCES` work too, exactly as in the needle script.)

| Script | Hypothesis | What it measures | Kill |
|---|---|---|---|
| `exp_p1_fork_equivalence.py` | H-P1 | Forked decode state == independent full pass (fp tol); state floats vs prefix length; branch isolation incl. the episodic store; step()s saved | any drift, any growth with prefix, any leak |
| `exp_p2_choice_points.py` | H-P2 | Per-position entropy as a detector of ground-truth choice points (AUROC), flagged fraction vs true fraction, loss vs grammar floor; on a checkpoint: entropy quantiles + fraction above 1/2/3 nats | AUROC well under 0.9 / fraction off |
| `exp_p3_gate_spectrum.py` | H-P3 | Per-unit retention time constant tau at init, after training, under chrono init; the gate's tau ceiling; formula pin vs `_MinGRUMixer.step`. `CB_P3_TRAIN=1`: delayed-copy default vs chrono at lags 8/32/128 | any substrate fact wrong |
| `exp_p4_keyframe_interp.py` | H-P4 | Linear-interpolation error between keyframes vs hold-last, per gap; per-unit error vs tau (Spearman); recompute exactness; bytes/step; numerical pins on the AURA note's Hilbert/Fourier/Hamiltonian operators | interpolation not under hold-last at gap 4; recompute not exact |
| `exp_p5_counterfactual_probe.py` | H-P5 | Snapshot at SEP, substitute the branch token, re-run: counterfactual accuracy vs ground-truth template, overdetermined positions unchanged, fragile position flipped, next-choice washout KL, state-distance profile | counterfactual at chance; overdetermined changing; no washout |
| `exp_p6_multihorizon_pilot.py` | H-P6 | **GPU / Colab** (`notebooks/multihorizon_pilot.ipynb`). Two matched arms, baseline vs + a multi-horizon (successor-feature) head at γ = 0 / 0.875 / 0.98 / 0.998: held-out CE at matched tokens, per-horizon cosine skill vs the trivial predictor, a linear probe on frozen features of both arms, the H-P3 gate spectrum of both, the H-B6 health gates. Reuses `train_colab.py`; falls back to tokenizing the Wikipedia jsonl on Drive | head costs CE beyond noise; skill only at horizon 1 |
| `test_prospection.py` | — | `run(quick=True)` of all five in one pytest session (`python -m pytest validation/prospection -q`, ~15 s) | — |
| `make_text_sample.py` | — | Builds the `CB_TEXT` plain-text sample from the domain-tagged Wikipedia jsonl next to the checkpoints (writes to `%TEMP%`, never into the repo) | — |

`_common.py` holds the shared pieces: the `BranchGrammar` toy corpus with
*known* choice points (one free choice per segment, templates share a prefix so
the branch id must survive in state), tiny `CubbyModel` builders through the
real `TrainLoop`, exact decode-state forking (MinGRU tensors, hybrid KV tuples,
deep-copied `EpisodicStore`), entropy/AUROC, and the retention-gate math.

## Headline numbers (2026-08-25, toy scale — CPU, torch 2.13 cpu)

- **H-P1** fork == independent pass at `0.00e+00` on mingru / hybrid /
  hybrid+mem; state floats constant over prefixes 8/32/64 (96 / 577 / 1345);
  no origin mutation, no store leak; 96 step()s forked vs 240 independent.
- **H-P2** toy loss 0.198 nats/token vs floor 0.173; entropy 1.375 (≈ ln 4) at
  true choice points vs 0.002 elsewhere; AUROC 1.000 on both paths; flagged
  fraction 0.125 = the true 1/8.
- **H-P3** init tau ≈ 2.8 steps for every unit; **the gate caps tau at 999.5
  steps** (`a ≤ 0.999`); toy training moved no unit past tau ≈ 4 (its longest
  dependency is 3 steps — gates lengthen only as far as the data demands);
  chrono init: tau p10/p50/p90 ≈ 2.4 / 15 / 260, 22% of units slower than 100
  steps. Delayed copy (300 steps, informational): lag 8 → 2.09 default vs 2.15
  chrono; lag 32 → 2.71 vs 2.52; lag 128 → 2.71 vs 2.60 (chance 2.77).
- **H-P4** on the toy (tau ≈ 3): linear interpolation rel-RMS 0.59 / 0.83 /
  1.16 at gaps 2 / 4 / 8 vs hold-last 0.80 / 1.03 / 1.26 — it barely beats
  doing nothing and is worse than the unit mean from gap 8 up; Spearman(log tau,
  error) = −0.38 (slow units interpolate better); recompute exact (0.0e+00);
  256 B/step of states vs 32.5 B/step keyframes+tokens at gap 8. Hilbert and
  Fourier strategies equal linear interpolation to 4e-16; the Hamiltonian flow
  conserves norm (drift 4e-10) and misses keyframe 1 by 0.715 relative.
- **H-P5** factual control 1.000; counterfactual accuracy 1.000 (chance 0.125);
  overdetermined positions unchanged 1.000; fragile position flipped 1.000 and
  correct 1.000; washout KL 0.050 nats at the next SEP while the state
  distance is still 4.6 (washout is a readout property here, not a state one).

Toy-scale VERIFIED is not scale-VERIFIED (`VALIDATION_REPORT.md` caveats
apply). The checkpoint arms below are the measurements that decide the store
design and the prospection budget on the real model.

## Checkpoint arms (2026-08-25) — `D:\My Drive\cubbyllm\ab_hybrid.pt` (d=512, L=8, hybrid 1:3, window 512, ~0.98B tokens; the H-D4 winner) and `hd5_mem2.pt` (same + episodic memory every 2nd layer)

Text: 80 Wikipedia articles across the 20 domain files next to the checkpoints,
built by `make_text_sample.py` (writes to `%TEMP%`, never into the repo), first
4096 tokens through `grillcheese_bbpe128k.json`. Logs: `../logs/exp_p*_ab_hybrid.log`,
`../logs/exp_p1_fork_equivalence_hd5_mem2.log`.

- **H-P1** fork == independent pass at `0.00e+00` on both checkpoints; state
  bounded past the window (1,575,427 floats ≈ 6.3 MB on `ab_hybrid`, of which
  1,572,864 are the three attention layers' KV caches and 2,560 the five
  recurrent layers; 2,755,075 on `hd5_mem2`, the extra being the four memory
  layers' FIFO buffers); no origin mutation, no store leak; 2.4 s forked vs
  5.2 s independent for 4 branches. **The fork cost of the hybrid is the
  attention window, not the recurrence** — a keyframe that must be
  re-enterable without the preceding 512 tokens is ~6 MB, not ~10 KB.
- **H-P2** real-text entropy: median 4.32 nats, p90 6.14; fraction of positions
  above 1 / 2 / 3 nats = **0.914 / 0.830 / 0.701**. At this model's scale
  choice points are not sparse — an absolute-entropy threshold would open
  branches at 70–90% of steps. The budget rule needs a *relative* criterion
  (top decile within a window, or entropy above the model's running median)
  or a stronger model; the toy result (1/8 flagged) does not transfer as-is.
- **H-P3** the learned spectrum: recurrent-layer median τ = 0.5 / 0.7 / 1.1 /
  1.5 / 1.8 steps (layers 1, 2, 4, 5, 7) — **shorter than init (2.8)**; 0.5% of
  recurrent units have τ ≥ 8, none ≥ 100, slowest single unit 111 (cap 999).
  Conditional latching (instantaneous τ ≥ 100) is not hiding long memory
  either: 1.2% of units latch on ≥10% of steps, the p90 unit on 0.4%. After
  ~1B tokens the MinGRU layers are 1–2-step mixers and every long-range
  dependency lives in the windowed attention — consistent with the H-D4
  needle result (pure MinGRU ~32% recall, hybrid ~98% inside its window).
- **H-P4** on the real trajectory: linear interpolation rel-RMS 0.74 / 0.87 /
  1.00 / 1.10 / 1.17 at gaps 2 / 4 / 8 / 16 / 32 vs hold-last 0.92 / 1.10 /
  1.21 / 1.30 / 1.34 — from gap 8 interpolation is worse than the unit mean;
  Spearman(log τ, error) = **−0.88**; recompute exact (0.0e+00); recurrent
  state 10,240 B/step vs 1,282 B/step keyframes + tokens at gap 8 (attention
  KV on top, see H-P1). Interpolation-based compression is dead on this model;
  recompute-from-keyframe is the store's regenerator.

What these four say together, for the "learn across timescales" thread: the
trained substrate has essentially no slow subspace to put a multi-timescale
scheme on. Long time constants will have to be *made* — chrono init, an
explicit long-horizon objective, or the LRU-style rotation-block recurrence —
or the timescale machinery has to live in attention and the episodic store,
where the H-D5 memory rung already is.

## H-P6 — the pilot arm that follows from the above (built 2026-08-25, UNRUN at scale)

`exp_p6_multihorizon_pilot.py` + `notebooks/multihorizon_pilot.ipynb` implement
the recommendation: leave the backbone alone, and put the medium horizon where
the model already keeps it — an auxiliary head on the trunk feature that
predicts the discounted average of the *future* tokens' (centered, unit)
embeddings at γ = 0 / 0.875 / 0.98 / 0.998 (mean horizons 1 / 8 / 50 / 500
tokens), targets computed inside the training window by the package's own
log-domain scan run time-reversed (pinned against the naive recursion in
`test_prospection.py`). Two arms at matched tokens, then: held-out CE, per-horizon
cosine skill vs the trivial mean-direction predictor, a linear probe on frozen
features of *both* arms, the H-P3 gate spectrum of both, and the H-B6 health
gates. CPU-smoked end to end at d=64 (tokenize → train → checkpoint → eval →
probe → spectrum → JSON); the numbers that matter need the A100 run.

Reading it: Δ held-out CE within ±0.01 nats = the head is free (keep it as a 2B
runbook arm); probe-multihorizon above probe-baseline = the objective changed
the representation; recurrent `p50`/`slow8` moving above the baseline's = an
explicit long-horizon objective *does* lengthen the recurrence, the first
evidence for the "made, not inherited" route.
