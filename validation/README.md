# validation/ — hypothesis validation experiments

Standalone experiment scripts, one per hypothesis (or hypothesis cluster) from
`../CUBBYLLM_HYPOTHESES.md`. **Nothing in this directory is CubbyLLM package
code or wired into any model path** — these are the cheap, falsifiable
experiments the hypothesis doc calls for, kept in-repo so every number in
`VALIDATION_REPORT.md` traces to a runnable script. The target package layout
(H-F1) is a separate spec; this directory is deliberately outside it.

All scripts run on CPU (torch 2.10 cpu / numpy), Python 3.12, no GPU needed
(per H-E3: no validation may require special hardware). Sibling-repo code is
imported directly from its source tree (paths at the top of each script) —
candidates under test, not dependencies.

| Script | Hypotheses | What it measures |
|---|---|---|
| `exp_a1_drift.py` | H-A1 | C_ctx structural check + sequential-distribution drift of cubby-lm's actual `HebbianGrowthLayer` |
| `exp_a_forgetting.py` | H-A3, H-A4 | Zero-Forgetting Stability Benchmark: sequential key→value retention for Hebbian (±decay), NLMS, SDM |
| `exp_b_algebra.py` | H-B1, H-B2 | Binarization compatibility of cubby-lm's head vectors; HRR vs BSC bind/unbind/bundle capacity at D=10240 |
| `exp_c7_capacity.py` | H-C7 | Required codebook dimension D vs vocab size: direct-readout crosstalk margin + resonator P_cap law |
| `exp_c2_lzw.py` | H-C2 | LZW hypertoken sequence-length reduction on real tokenizer output |
| `exp_c3_latency.py` | H-C3 | Output-head latency: full-vocab matmul+softmax vs IVF top-K retrieval at V=32k…1M |
| `exp_c1_hyperencoder.py` | H-C1 | Generated (surface-form) token embeddings vs static lookup table at matched vocab |
| `exp_h0_gce.py` | H0, H-A2 | Sequential-task retention: shared-weight baseline vs context-generated head (θ=f(c)) + P5 wrong-context probing |
| `exp_h0b_learned_context.py` | H0, H-C4 | θ=f(c) with INFERRED context: naive vs supervised router vs oracle; router-collapse finding |
| `exp_b5_grilly_3way.py` | H-B1/B2/B5 | 3-way duties matrix over grilly's real BinaryOps/HolographicOps/BlockCodeOps |
| `exp_g2_nyt_forgetting.py` | H-G2 | Forgetting benchmark over real dated NYT headline keys (correlated, chronological) |
| `exp_g1_bpe_cost.py` | H-G1 | Wall-clock BPE build cost at 65k vs 131k vocab (the next fixed-vocab step) |
| `exp_a5_sparse.py` | H-A5 | DG sparse expansion vs size-matched dense control over a Hebbian store |
| `exp_d1_backbone.py` | H-D1 | Char-LM viability bake-off: cubby-lm MinGRU vs BDH-style Q=K recurrence vs attention |
| `exp_d2_ttt.py` | H-D2 | TTT fast-weights vs KV-cache: per-token compute + state size vs context length |
| `exp_g3_overlap.log` | H-G3 | (inline check) Gutenberg ID overlap between factual/ and gbooks/ |
| `prospection/exp_p1_fork_equivalence.py` | H-P1 | Decode-state forking: forked branches vs independent passes (exact), state size vs prefix length (bounded), branch/store isolation |
| `prospection/exp_p2_choice_points.py` | H-P2 | Next-token entropy as a detector of ground-truth choice points (AUROC, flagged fraction); real-text entropy profile with `CB_CKPT` |
| `prospection/exp_p3_gate_spectrum.py` | H-P3 | MinGRU retention time-constant spectrum at init / trained / chrono init; the gate's ~1000-step ceiling; `CB_P3_TRAIN=1` delayed-copy arm |
| `prospection/exp_p4_keyframe_interp.py` | H-P4 | Keyframe compression: linear interpolation vs hold-last vs exact recompute, per-unit error vs tau; numerical pins on the AURA note's Hilbert/Fourier/Hamiltonian operators |
| `prospection/exp_p5_counterfactual_probe.py` | H-P5 | Snapshot at a choice point, substitute the choice, re-run: counterfactual accuracy with ground truth, overdetermined vs fragile positions, washout |
| `prospection/exp_p6_multihorizon_pilot.py` | H-P6 | GPU pilot arm (`notebooks/multihorizon_pilot.ipynb`) and the Group P pilot-arm runner: baseline vs + multi-horizon successor-feature head at matched tokens — held-out CE, per-horizon skill, linear probe, gate spectrum at init/end, beyond-window impulse response, H-B6 health gates. Ran 2026-08-26: free but redundant |
| `exp_a7_learning_gate.py` (+ `test_a7_gate.py`) | H-A7 | The learning gate's first experiment: on a trained checkpoint (`hd5_mem21.pt`), a deliberately-forgetting delta (fine-tune on one source, no replay), a matched clean delta (full mixture = replay) and the null delta are scored on identical held-out windows — mix CE, per-source CE, GSM8K-text CE (H-G4), the H-B6 retrieval probe — with paired-SE thresholds; the gate must reject the forgetting delta and pass the other two. Also records which of the proposed six terms cannot see a trunk delta at all (CoT precision, routing precision: model-free surfaces; NYT-vs-NLMS: a memory-method benchmark). CPU ~50 min; **the recorded run is Colab** (`notebooks/a7_learning_gate.ipynb`, A100, 5.4 min, 2026-08-30: forgetting rejected + named, null passes, replay arm rejected for uniform drift — its train loss rose at LR 1e-4; LR ladder queued via `CB_A7_ARMS=B` / `CB_A7_LR` / `CB_A7_RESUME_OPT=1`). Unit pins: `python -m pytest validation/test_a7_gate.py -q` |
| `exp_m3_cot_pipeline.py` (+ `test_cot_counterfactuals.py`) | H-F2/H-B3 CoT M1 | The CoT kill-criterion eval through the real cubelang VM: retrieval-only vs chase-only vs verified CoT, frame-size-conditional τ floors, planted-fault calibration card, harvest per `docs/schemas/cot-harvest-schema.md` — and since 2026-08-28 the per-verified-chain **counterfactual neighborhood** (`counterfactuals[]`; fake-VM unit pins in `test_cot_counterfactuals.py`, `python -m pytest validation/test_cot_counterfactuals.py -q`). Needs `cubelang.exe` (sibling repo release build) + the corpus `.pq` + the v4 table |
| `prospection/exp_p7_chrono_pilot.py` | H-P7 | GPU pilot arm (`notebooks/chrono_pilot.ipynb`): chrono init on the recurrent gates vs default — held-out CE, survival of the init spectrum, beyond-window impulse response, needle recall at 256/512/1024/4096. **Ran 2026-08-26 (A100, 65.5M tokens/arm): KILLED** — +0.444 nats held-out CE (threshold 0.01); τ spectrum eroded ~4× but survived (5–10% of units ≥ 100 steps); beyond-window impulse ~1e-3 (9–45× baseline). `logs/exp_p7_{baseline,chrono}.{log,json}` |

`prospection/` is a subfolder because its five scripts share a toy corpus and
helpers (`prospection/_common.py`, `prospection/README.md`); run
`python -m pytest validation/prospection -q` for the quick pass of all five.

Run any script directly: `python validation/exp_*.py`. Each prints its own
results block and asserts its headline numbers, so a silent regression fails
loudly.
