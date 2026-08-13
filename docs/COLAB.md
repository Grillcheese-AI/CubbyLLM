# Training CubbyLLM on Colab (RTX PRO 6000 Blackwell / any CUDA GPU)

The package is device-agnostic: `core.device.resolve_device()` returns CUDA when
present, `move_model()` moves every component onto it, and on CUDA the MinGRU
backbone uses the fast O(log T) **parallel** scan (the sequential fallback is
DirectML-only). So **nothing in the model changes** to run on the Blackwell card
— you upload the package + data and run `validation/train_colab.py`.

`grilly` (the Vulkan VSA tier) is **not needed on Colab** — `ops/vsa.py` falls
back to the numpy reference, and the binding head isn't in the training forward
path anyway.

## 1. Verify the GPU

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
print(torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name())
print("capability", torch.cuda.get_device_capability())   # Blackwell -> (12, 0)
```

**Blackwell caveat (sm_120):** if capability is `(12, x)` and you later get
`RuntimeError: no kernel image is available for execution on the device`, the
torch build predates Blackwell. Install a CUDA-12.8 build:

```python
!pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
# (or a nightly cu128 wheel if the stable one lacks sm_120)
```

## 2. Get the code + data onto Colab

Pick one. **(a) zip upload** — simplest for a first run:

```python
# On your PC: zip the package + this script + the two small data files, e.g.
#   cubbyllm/ , validation/train_colab.py ,
#   grillcheese_spm32k_v2.model (0.8 MB) , tinystory_50k.json (44 MB)
from google.colab import files; up = files.upload()          # pick cubby_colab.zip
!unzip -q cubby_colab.zip -d /content/CubbyLLM
```

**(b) GitHub** — best for iterating (needs you to push the repo once):

```python
!git clone https://github.com/<you>/CubbyLLM.git /content/CubbyLLM
# upload the two data files separately (they shouldn't live in git)
```

**(c) Google Drive** — for large/real corpora later:

```python
from google.colab import drive; drive.mount('/content/drive')
# point CUBBY_STORIES / a corpus DataPipeline at /content/drive/...
```

## 3. Install deps

```python
!pip install -q sentencepiece            # torch + numpy already on Colab
```

## 4. Train

```python
import os
os.environ.update(
    CUBBY_SPM="/content/CubbyLLM/grillcheese_spm32k_v2.model",
    CUBBY_STORIES="/content/CubbyLLM/tinystory_50k.json",
    CB_D="512", CB_L="8", CB_B="64", CB_S="256", CB_STEPS="1000",  # ~50M params
)
!cd /content/CubbyLLM && python validation/train_colab.py
```

Expected: it prints the device + capability, param count, then per-eval
`loss / ppl / bpc / tok-s`, peak VRAM, and wall time. Every `CB_GEN` steps
(default 500) it also prints `CB_GEN_N` (default 5) **sample generations**
(temperature + top-k, seeded from EOS) so you can watch text quality improve
qualitatively alongside the loss. **bf16 mixed precision is on by default on CUDA**
(`CB_AMP=0` to disable) — ~2–3× faster and ~half the activation memory, which is
what makes a 2 B trunk (`CB_D=2048 CB_L=32`, ~2.0 B params) trainable here. On a
96 GB Blackwell you can push `CB_D`/`CB_L`/`CB_B` far higher.

### Resume (survive Colab timeouts)

Point `CB_CKPT` at a **Drive** path and training checkpoints there every
`CB_CKPT_EVERY` steps (default 500); rerun the *same* command and it resumes
exactly where it stopped — weights, optimizer, and step — so a disconnect costs
at most `CB_CKPT_EVERY` steps. The save is atomic (`.tmp`→replace), so a kill
mid-write can't corrupt it, and it refuses to load a checkpoint whose
architecture (`CB_D`/`CB_L`/vocab) doesn't match.

```python
os.environ.update(
    CB_CKPT="/content/drive/MyDrive/cubbyllm/ckpt.pt",  # resume across sessions
    CB_CKPT_EVERY="500",
    CB_STEPS="200000",                                  # keep the same target across restarts
)
!cd /content/CubbyLLM && python validation/train_colab.py   # first run: trains + saves
# ...session dies at step 8,300...
!cd /content/CubbyLLM && python validation/train_colab.py   # rerun: "resumed ... @ step 8000"
```

## Scaling on 96 GB

| knob | what it grows | notes |
| --- | --- | --- |
| `CB_D` | width (dominant cost: backbone ~9·D²·L + the memory generator's `Linear(ctx, D²)`) | 512 → 1024 → 2048 all fit |
| `CB_L` | backbone depth | linear in params/compute |
| `CB_B` / `CB_S` | batch / sequence (throughput + activation memory) | raise `CB_B` first for GPU utilization |
| `CB_STEPS` | training length | |

Watch the printed **peak VRAM** and step up until you're using the card. The
memory layer generates a single `(D, D)` matrix per step (`x + tanh(x @ Wᵀ)`), so
there is **no per-position D×D blowup** — large shapes are safe.

## Real corpus (WeightedCorpusPipeline)

`train_colab.py` uses `InMemoryDataPipeline` over TinyStories by default, but it
also drives the real **`WeightedCorpusPipeline`** (the H-G3 corpus source) when
you point it at a token cache. Colab can't see the local `E:`/`D:` corpus, so
the flow is **tokenize locally → upload the compact `uint16` cache → train on
GPU reading the cache**:

**1. Tokenize locally (one-time, CPU).** Already built (2026-07-30): 14.37 B tokens
across 10 sources in `D:\grillcheese_training_data\token_cache` as `uint32` `<name>.u32`
shards, using the **128 k byte-level BPE** `grillcheese_bbpe128k.json` (opcodes atomic,
byte-exact — 128 k exceeds `uint16`, hence `uint32`). To (re)build, use the parallel,
resumable tokenizer (multi-core, streams big files, skips fresh shards):

```bash
python E:\datasets\_pipeline\tokenize_parallel.py            # all sources
python E:\datasets\_pipeline\tokenize_parallel.py --only wiki_full   # one source
```

The package's `WeightedCorpusPipeline.prepare()` is the single-thread reference; the
tokenizer is chosen by extension (`.json` → HF byte-level BPE, `.model` → SentencePiece).

**2. Upload** `token_cache/*.u32` + `corpus_sources.json` + `grillcheese_bbpe128k.json`
(the tokenizer, needed for vocab size) to Drive, e.g. `/content/drive/MyDrive/cubbyllm/`.

**3. Train on Colab** — set the two env vars and run:

```python
import os
os.environ.update(
    CB_CORPUS="/content/drive/MyDrive/cubbyllm/token_cache",  # shards on Drive
    CB_STAGE="/content/token_cache",                          # copy to LOCAL disk first (see below)
    CB_SOURCES="/content/drive/MyDrive/cubbyllm/corpus_sources.json",  # for names+weights
    CUBBY_SPM="/content/drive/MyDrive/cubbyllm/grillcheese_bbpe128k.json",  # 128k BBPE (.json)
    CB_D="512", CB_L="8", CB_B="64", CB_S="256", CB_STEPS="2000",
)
!cd /content/CubbyLLM && python validation/train_colab.py
```

**Always set `CB_STAGE` for real runs.** Training does random-window `memmap`
reads; served straight off the Drive FUSE mount each read is a network round-trip
and throughput collapses. `CB_STAGE` copies the `.u32` shards to local disk once
(sequentially — Drive's fast path) and trains from there, so random access hits
local disk / page cache. The copy is **idempotent** (skips shards already staged
at the right size), so a Colab timeout mid-copy just resumes on rerun. Needs
~55 GB free local disk for the full cache; omit `CB_STAGE` only for a quick smoke.

The pipeline runs in **`cache_only`** mode (memmaps the shards, no source files
or tokenizer needed), samples windows **by weight**, and prints the source mix +
`manifest_hash` so the run is reproducible. Weights, dedup, and the gated sources
(`unified/`, arxiv-latex, temporal) are documented in
`E:\datasets\_pipeline\corpus_manifest.md`. Swapping to the 128k BPE tokenizer
later is just a new `spm_path` + re-tokenize.

## MFU pilot (run BEFORE renting anything for the 2B cycle)

The panel-settled decision chain (docs/research/2026-08-model-panel-agenda.md)
starts with a ~2-hour measured-MFU pilot: hybrid vs pure-MinGRU vs transformer
comparator, eager vs torch.compile, at the 150M shape. It needs **no data
files** (random tokens) — just the package + the script:

```python
# after step 2 (code on Colab) — no sentencepiece, no corpus needed
!cd /content/CubbyLLM && python validation/exp_t1_mfu_pilot.py --tag _colab
# 2B-shape memory/throughput probe (grad-ckpt on):
!cd /content/CubbyLLM && MFU_D=2048 MFU_L=32 MFU_B=4 MFU_S=1024 MFU_CKPT=1 \
    python validation/exp_t1_mfu_pilot.py --tag _colab_2b --arms hybrid,attn
```

It prints an arm x mode table (ms/step, tok/s, achieved TFLOPS, MFU vs the
card's dense-bf16 peak, peak memory) and a projected A100-hours/$ for the
14.37B-token cycle-one run at each arm's best measured MFU. Unknown card names:
set `MFU_PEAK_TFLOPS`. Download `validation/logs/exp_t1_mfu_pilot_colab*.json`
back into `validation/logs/` when done — the decision record lives there.

**What decides what** (3/3 panel-convergent): hybrid MFU ≥ ~2/3 of the
transformer arm → cycle-one dense hybrid as planned; a large gap that
torch.compile does not close → budget the Triton MinGRU-scan port (days,
saves hundreds of $) before renting; ratio shift (1:3 → more attention) is
the last resort, not the first.
