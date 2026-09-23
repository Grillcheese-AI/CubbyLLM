"""Generate base450m_pretrain.ipynb: the 450M CubbyLLM base on one Colab A100 in
~20 hours (Route A, phase 1 of docs/research/2026-09-23-training-acceleration.md).
The notebook is thin; the logic is validation/train_base.py. Re-run after editing:
    python notebooks/_build_base.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "base450m_pretrain.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# CubbyLLM base: 450M on one A100 in ~20 hours

**What this is.** Route A, phase 1 of `docs/research/2026-09-23-training-acceleration.md`. A d1024/L32 hybrid
(MinGRU, window-512 attention every 3rd layer, 4 heads × 256) trained on ~3B tokens: at 20 A100-hours this
size beats the 2B shape at equal compute (fitted loss ~3.0 vs ~3.2). It is the prototype that talks and
emits, and the seed for the 2B: doubling its width (HyperCloning) gives d2048/L32 with 8 × 256 heads exactly.

| | |
|---|---|
| trainable parameters | 576M = 314M trunk + 131M embedding + 131M head (vocab 127,996 padded to 128,000) |
| FLOPs per token | 2.72 G; the 128k head is 29% of it |
| A100 at 35-45% MFU | 40-52k tok/s: 2.7-3.5B tokens in 20 h, ~21-27k optimizer steps of 131k tokens |

**What `validation/train_base.py` does** (every knob is an env var, echoed at start):
- AdamW (weight decay 0.1 on matrices, β2 0.95); WSD schedule (300-step warmup, flat, 1-sqrt decay to zero
  over the last 20%); ~131k tokens per step by accumulation; `torch.compile` of the whole loss.
- **Causal context**: the router context at position t is the running mean of tokens 0..t, and the adapter is
  per sample, so training sees exactly what decode sees.
- **Data mix**: size-proportional with quality multipliers (FineWeb-Edu ×2, books/wikibooks/textbooks ×1.5;
  encyclopedic, fact and legal shards down-weighted since knowledge lives in the store). The decay phase
  up-weights FineWeb-Edu, QA, code and books. The pinned `corpus_sources.json` weights are per *source*
  (all 1.0): taken literally they would repeat `pretrain_ext` ~295 times and `arxiv` ~21 times in this run
  (`CB_MIX=pinned` still does that, if you want it).
- **Validation** on each shard's held-out tail every 250 steps, per source, plus an in-context copy probe.
- **Samples every 500 steps** (EN, FR, QA, code prompts).
- **Checkpoints** to Drive every 45 min (two alternating slots). A resume draws exactly
  the windows the uninterrupted run would have: an interrupted-and-resumed CPU smoke run finished
  bit-identical to an uninterrupted one.
- **Sizes itself**: after ~40 steps it measures s/step and sets the step count to fit `CB_HOURS`.

**Before you run**
1. `master` on GitHub has `validation/train_base.py` and the 2026-09-23 causal-context change
   (`cubbyllm/model/assembly.py`, `cubbyllm/core/generation.py`). This notebook clones `master` and checks both.
2. The token cache. The full cache is 71 GB and the run reads ~3B tokens, so `validation/make_base_cache.py`
   builds a subset on the machine that holds the cache. It keeps every shard's original validation tail and the
   full cache's mix. The one this notebook expects was built with `--frac 0.2`: 14.7 GB, at most 1.75 epochs of
   any shard in a 3.2B-token run (`validation/logs/make_base_cache.log`). It sits on Drive at
   `cubbyllm/token_cache_base/`, tokenizer included under `tokenizer/`, and is staged to local disk below.
   `DATA_SOURCE = 'hf'` reads the same folder from a private Hugging Face dataset instead (Colab secret `HF_TOKEN`).
3. ~17 GB free on Drive for checkpoints, on top of the cache: two 7 GB slots with optimizer state and the 2.3 GB
   final. The extra 7 GB pre-decay copy is off here (`CB_STABLE_END='0'`); turn it on if Drive has room.
4. Colab Pro+ for 24 h sessions on an A100-80G. Pro and pay-as-you-go stop at 12 h: set `SESSION_H = 12` below;
   the run resumes in the next session.
"""),
    code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, time, subprocess, signal, json, glob
SESSION_T0 = time.time()
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers numpy
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print('torch', torch.__version__)
assert os.path.exists(f'{REPO}/validation/train_base.py'), 'train_base.py is not on master yet: push it first'
assert 'causal running mean' in open(f'{REPO}/cubbyllm/model/assembly.py').read(), \
    'master has the old (non-causal) context: push the 2026-09-23 assembly.py/generation.py change first'
"""),
    code(r"""
# --- EDIT to your layout ---
SESSION_H    = 24                                  # Colab Pro+; 12 on Pro / pay-as-you-go
DATA_SOURCE  = 'drive'                             # 'drive' or 'hf' (private dataset)
HF_REPO      = 'grillcheese/cubbyllm-token-cache-base'
DRIVE        = '/content/drive/MyDrive/cubbyllm'
CORPUS_DRIVE = f'{DRIVE}/token_cache_base'         # make_base_cache.py output, tokenizer/ inside
CORPUS       = '/content/token_cache'              # LOCAL SSD: random reads off Drive-FUSE run ~30x slower
TOKENIZER    = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
RUN_DIR      = f'{DRIVE}/runs/base450m'            # checkpoints, metrics, logs
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)
if DATA_SOURCE == 'hf':
    from huggingface_hub import snapshot_download
    try:
        from google.colab import userdata; tok = userdata.get('HF_TOKEN')
    except Exception:
        tok = None                                 # falls back to huggingface_hub's own login
    snapshot_download(HF_REPO, repo_type='dataset', local_dir=CORPUS, token=tok)
else:
    assert glob.glob(f'{CORPUS_DRIVE}/*.u32'), f'no .u32 shards in {CORPUS_DRIVE}'
    !rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
    if not os.path.exists(TOKENIZER):
        os.makedirs(os.path.dirname(TOKENIZER), exist_ok=True)
        !cp {DRIVE}/grillcheese_bbpe128k.json {TOKENIZER}
assert os.path.exists(TOKENIZER), TOKENIZER
assert glob.glob(f'{CORPUS}/*.u32'), f'no .u32 shards in {CORPUS}'
!du -sh {CORPUS}; df -h /content | tail -1
"""),
    code(r"""
# --- config shared by every run below, and a runner that tees the log to Drive ---
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR=RUN_DIR, CB_NAME='base450m',
    CB_D='1024', CB_L='32', CB_HEADS='4', CB_WINDOW='512', CB_ATTN_EVERY='3',
    CB_S='1024', CB_MICRO='16', CB_TOKENS_PER_STEP='131072',   # set CB_MICRO from PREFLIGHT A
    CB_LR='6e-4', CB_BETA2='0.95', CB_WD='0.1',                 # set CB_LR from PREFLIGHT B
    CB_WARMUP='300', CB_DECAY_FRAC='0.2',
    CB_EVAL='250', CB_GEN='500', CB_CKPT_MIN='45',
    CB_STABLE_END='0',                                          # '1' keeps the pre-decay copy (+7 GB on Drive)
)

def run(env_extra, log_name):
    env = dict(os.environ, **BASE, **env_extra)
    log = f'{RUN_DIR}/{log_name}'
    p = None
    try:
        with open(log, 'a', encoding='utf-8', buffering=1) as f:
            f.write(f"\n=== {time.strftime('%F %T')} "
                    + ' '.join(f'{k}={v}' for k, v in sorted(env_extra.items())) + '\n')
            p = subprocess.Popen([sys.executable, '-u', 'validation/train_base.py'], cwd=REPO, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end=''); f.write(line)
            p.wait()
    finally:
        if p and p.poll() is None:       # a stopped cell: let the script save its checkpoint first
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                p.terminate()
"""),
    md(r"""
### Preflight A: throughput (~10 min, random tokens, no data needed)

Full stack, head included (the 2B-shape MFU pilot measured the trunk alone). Set `CB_MICRO` in the config cell
to the fastest micro-batch that fits and rerun that cell. The last line says what 20 h buys. **If MFU is under
~30%, stop and look before spending 20 hours.** If compilation fails, `CB_COMPILE='0'` runs eager, much slower
(flex attention unfused).
"""),
    code(r"""
run(dict(CB_MODE='bench', CB_BENCH_MICRO='8,16,24,32'), 'bench.log')
"""),
    md(r"""
### Preflight B, optional: learning rate (~30-40 min)

Three short arms on real data, same windows per arm, each ~20M tokens with a 30-step warmup and no decay. Short
probes favour high learning rates: if the next-lower LR is within ~0.02 of the best, take the lower one. A
diverged arm is excluded. The repo's earlier d512 runs trained at 3e-4 (Adam, no weight decay).
"""),
    code(r"""
run(dict(CB_MODE='lrprobe', CB_PROBE_LRS='3e-4,6e-4,1.2e-3', CB_PROBE_STEPS='150'), 'lrprobe.log')
"""),
    md(r"""
### The run (~20 h)

**Rerun this same cell after a disconnect**: it resumes from the newest Drive slot, with the schedule it was
sized to. Stopping the cell saves a checkpoint first. The hard stop is set before the session's end so the
last checkpoint always lands on Drive.
"""),
    code(r"""
left_h = SESSION_H - 0.4 - (time.time() - SESSION_T0) / 3600
HOURS  = min(20.0, left_h - 0.8)            # only used by a fresh run, to size itself
print(f'{left_h:.1f} h left in this session -> CB_HOURS={HOURS:.1f}, hard stop after {left_h - 0.3:.1f} h')
run(dict(CB_HOURS=f'{HOURS:.2f}', CB_WALL_H=f'{left_h - 0.3:.2f}'), 'train.log')
"""),
    md(r"""
### Only if the run cannot continue

Out of sessions while still in the stable phase? Cool the newest checkpoint down into a finished model of its own
(decay over a further 25% of its steps, under its own name so the main slots are untouched). The stable phase
exists to make this possible.
"""),
    code(r"""
slots = [s for s in glob.glob(f'{RUN_DIR}/base450m_slot?.pt.json') if json.load(open(s)).get('step', -1) >= 0]
newest = max(slots, key=lambda s: json.load(open(s))['step'])
step = json.load(open(newest))['step']
print('cooling down', newest[:-5], 'from step', step)
run(dict(CB_RESUME_FROM=newest[:-5], CB_DECAY_START=str(step), CB_NAME='base450m_cool'), 'cooldown.log')
"""),
    code(r"""
# --- curves (any time, also mid-run from another session) ---
import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open(f'{RUN_DIR}/base450m_metrics.jsonl')]
tr = [r for r in rows if 'loss' in r]
va = [r for r in rows if 'val' in r]
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].plot([r['tokens'] / 1e9 for r in tr], [r['loss'] for r in tr], lw=.6)
ax[0].set(title='train loss', xlabel='B tokens', ylim=(None, min(8, max(r['loss'] for r in tr))))
for k in [k for k in va[-1]['val'] if not k.startswith('_')] if va else []:
    ax[1].plot([r['step'] for r in va], [r['val'][k] for r in va], label=k, lw=1)
if va:
    ax[1].plot([r['step'] for r in va], [r['val']['_mix'] for r in va], 'k', lw=2, label='mix')
    ax[1].legend(fontsize=7, ncol=2); ax[1].set(title='held-out loss per source', xlabel='step')
    ax[2].plot([r['step'] for r in va], [r['val']['_copy'] for r in va])
    ax[2].set(title='in-context copy loss (repeated half)', xlabel='step')
plt.tight_layout(); plt.show()
"""),
    md(r"""
### How to read

- **Loss** starts at ~11.76 (= ln 128,000) and falls fast. `gnorm` settles under ~1 after the warmup (the clip
  is 1.0). A spike that does not recover within ~100 steps means the LR is too high: stop the cell, then rerun
  with a lower `CB_LR`. A non-finite loss stops the run *without* saving, so the last checkpoint stays clean.
- **`copy`** is the loss on the second copy of a random token sequence. It falls toward 0 once an
  induction/copy circuit exists: the capability gate this repo uses (ckpt_v21 never reached it by 11k steps).
  If it hasn't moved by ~1B tokens, retrieval and long-context claims aren't testable on this model yet.
- **Per-source validation.** FineWeb-Edu, books, code and QA should keep falling through the decay. Wiki, facts
  and legal are down-weighted on purpose (knowledge lives in the store), so they fall less.
- **The decay** (last 20%) drops the loss noticeably. That is WSD's cooldown gain, not overfitting.
- **Samples** (every 500 steps): word salad early, fluent sentences somewhere around 1B tokens, format (`Answer:`,
  a function body) by the decay. The French prompt is the one to watch: French is a minority of the mix.
- **MFU** on the step line counts the full stack, head included. Under ~30%: check the corpus is on local SSD,
  and the micro-batch.

### After the run

`{RUN_DIR}/base450m_final.pt` holds the weights only (fp32, ~2.3 GB) plus meta, schedule and token count.
With `CB_STABLE_END='1'`, `base450m_stable_end.pt` also keeps the pre-decay state, to re-run the decay with another
data mix. Load the final model for decode:

```python
sys.path.insert(0, f'{REPO}/validation'); from train_base import load_base
from cubbyllm.core.decoding import DecodeConfig, generate
m = load_base(f'{RUN_DIR}/base450m_final.pt', 'cuda')
```

Record the verdict (final val per source, `copy`, samples, tokens, hours) in `CUBBYLLM_HYPOTHESES.md`, with the
Drive logs copied into `validation/logs/`. Next is phase 2: width-clone to d2048 on an H100.
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "machine_shape": "hm",
                             "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
