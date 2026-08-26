# Colab validation notebooks

One notebook per validation, each self-contained: **intro → setup → config → run
→ how-to-read**. They wrap the `validation/` scripts (the notebooks are thin; the
logic and the honesty live in the scripts and their captured logs).

Generated from [`_build.py`](_build.py) — **edit that and re-run
`python notebooks/_build.py`**, don't hand-edit the `.ipynb` JSON.

## Open in Colab

Each notebook clones the repo, installs deps, and mounts Drive in its first cell.
Open straight from GitHub:

```
https://colab.research.google.com/github/Grillcheese-AI/CubbyLLM/blob/master/notebooks/<name>.ipynb
```

Then edit the **paths cell** (`DRIVE`, `TOKENIZER`, `CORPUS_DRIVE`) to your Drive
layout before running.

| notebook | validates | needs |
|---|---|---|
| [`hd3_backbone_ab.ipynb`](hd3_backbone_ab.ipynb) | **H-D3** — windowed-attention hybrid vs pure MinGRU on needle recall (two matched training runs + eval) | corpus + tokenizer, ~5h/both arms |
| [`induction_probe.ipynb`](induction_probe.ipynb) | the toy behind H-D3 — pure recurrence vs windowed/full hybrid on canonical induction, within vs beyond the window | GPU only, ~20 min |
| [`decode_throughput.ipynb`](decode_throughput.ipynb) | the O(1)-state inference claim — incremental `step()` vs naive, and vs a real peer | a trained checkpoint |
| [`ffn_spectrum.ipynb`](ffn_spectrum.ipynb) | the low-rank FFN gate — is the trained FFN compressible? | a trained checkpoint, minutes |
| [`multihorizon_pilot.ipynb`](multihorizon_pilot.ipynb) | **H-P6** — the multi-horizon (successor-feature) prediction head as a pilot arm: baseline vs +head at matched tokens, held-out CE, per-horizon skill, linear probe, gate spectrum | tokenizer + corpus (falls back to the Wikipedia jsonl on Drive), ~15 min/arm on an A100 |

## Lessons baked in (so they don't bite again)

- **Stage the corpus to local SSD.** Drive-FUSE memmap reads bottleneck the GPU —
  measured **3,000 vs ~100,000 tok/s** (2026-08-04). The paths cell does this.
- **Real-corpus LR ≈ 3e-4 with warmup.** `3e-3` cold diverges to loss ~40 in three
  steps (finite, so it isn't a NaN — the divergence guard in `train_colab.py`
  catches it now, but the config here is already correct).
- **The backbone is recorded in the checkpoint meta**, so eval reconstructs the
  trained model (a hybrid scored as pure MinGRU would read its attention as noise).
- **Read the right metric.** For H-D3 that's the *needle table at distance*, not
  bpc or the live `copy` floor — both arms look alike on those; only recall at
  distance separates them.
