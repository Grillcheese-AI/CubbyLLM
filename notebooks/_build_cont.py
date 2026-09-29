"""Generate base450m_continue.ipynb: base450m keeps training on tokens it has never
seen (the kill branch of the growth pilot, docs/research/2026-09-25-grow-450m.md).
The notebook is thin; the logic is validation/train_base.py (CB_INIT_FROM, CB_STEPS)
and the data is validation/make_base_cache.py --exclude-frac. Re-run after editing:
    python notebooks/_build_cont.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "base450m_continue.ipynb"


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
# CubbyLLM base: continue the 450M on fresh data

**What this is.** The kill branch of the growth pilot (`docs/research/2026-09-25-grow-450m.md`, "Pilot result").
base450m has seen ~9 tokens per trunk parameter, and after an equal hour every grown arm was behind it on every
source they shared. So the base keeps training, on tokens it has never seen, before growth is tried again.

| | |
|---|---|
| starts from | `runs/base450m/base450m_final.pt`: step 21,103, 2.77B tokens, held-out mix 2.3615. Weights only, a fresh optimizer |
| data | a fresh subset of the same 17.7B-token cache: only the 1M-token blocks base450m's subset did not take, the same 13 held-out tails, the same mix |
| schedule | re-warm to 3e-4 over 300 steps, flat, then 1-sqrt decay to zero over the last 20%. The step count is fixed by `TARGET_TOKENS`, so the run spans sessions |
| default budget | 3.2B tokens = 24,414 steps of 131k tokens: ~19 h on an A100 or an RTX PRO 6000 (46-48k tok/s measured), less on an H100 |
| after it | ~6B tokens seen, ~19 per trunk parameter |

**Why 3e-4, half the base's peak.** The data is the same distribution, so the re-warm only has to unlock the
annealed weights, not adapt them. In the pilot, the first control at 6e-4 climbed back to the base's own
pre-decay loss (mix 2.54) within 300 steps; at 3e-4 the climb was ~0.06.

**Pre-registered (the readout cell applies it).** Against base450m, on the same 13 held-out tails:
1. the held-out mix ends at least 0.03 below 2.3615;
2. FineWeb-Edu, books, QA and code each end below the base's final;
3. `copy` ends no more than 0.1 above the base's 1.20.

If 1 fails, more pretraining is not what the base lacks: stop spending here and look at the mix. If all three pass,
the talk adapter is retrained on the new base and gates E6-E9 are re-run. The new base replaces base450m only if
every gate is at least talk_v2.1's, with 0 spoken wrong.

**Before you run**
1. **Build the fresh subset** on the machine that holds the full token cache, straight into the Drive folder:
   ```
   python validation/make_base_cache.py --src <full token cache> --dst <Drive>/cubbyllm/token_cache_cont --frac 0.2 --exclude-frac 0.2
   ```
   Each shard's blocks come only from the gaps between the blocks `token_cache_base` took (built with `--frac 0.2`).
   The held-out tails are copied byte for byte, and `_subset.json` records the blocks, so a later leg can avoid
   both subsets (`--exclude-dir`). 14.7 GB, like the first cache. The three small shards (arxiv, pretrain_ext,
   wikipedia) were copied whole the first time and are again; base450m drew ~0.17 epochs of each.
2. **Push `master`** with `validation/train_base.py`'s `CB_INIT_FROM` (the growth commit has it; checked below).
3. **Drive space:** 14.7 GB for the cache, plus two 7 GB checkpoint slots, the 7 GB pre-decay copy and the 2.3 GB
   final, ~38 GB in all. `CB_STABLE_END='0'` saves 7 GB, but a later leg would then have to re-warm again.
4. **Colab Pro+** for 24 h sessions; set `SESSION_H = 12` on Pro. Rerun the run cell in each new session.
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
assert 'CB_INIT_FROM' in open(f'{REPO}/validation/train_base.py').read(), \
    'master has no CB_INIT_FROM in train_base.py: push the growth commit first'
"""),
    code(r"""
# --- EDIT to your layout ---
SESSION_H     = 24                                  # Colab Pro+; 12 on Pro / pay-as-you-go
TARGET_TOKENS = 3.2e9                               # this leg's budget; fixed at the first session
DATA_SOURCE   = 'drive'                             # 'drive' or 'hf' (private dataset)
HF_REPO       = 'grillcheese/cubbyllm-token-cache-cont'
DRIVE         = '/content/drive/MyDrive/cubbyllm'
BASE_CKPT     = f'{DRIVE}/runs/base450m/base450m_final.pt'   # the model this run continues
BASE_RUN      = f'{DRIVE}/runs/base450m'                     # its metrics: the bar the readout uses
BASE_CACHE    = f'{DRIVE}/token_cache_base'                  # the cache it trained on: the checks compare
CORPUS_DRIVE  = f'{DRIVE}/token_cache_cont'                  # make_base_cache.py --exclude-frac 0.2
CORPUS        = '/content/token_cache_cont'                  # LOCAL SSD (Drive-FUSE reads are ~30x slower)
TOKENIZER     = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
RUN_DIR       = f'{DRIVE}/runs/base450m_cont'                # checkpoints, metrics, logs
assert os.path.exists(BASE_CKPT), BASE_CKPT
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)

def check_shards():
    # Every shard the cache lists must be on Drive and on local disk at full size: the growth pilot's
    # arms trained on 9, 10 and 12 of the 13 shards because the local copy was incomplete for each.
    names = sorted(json.load(open(f'{CORPUS_DRIVE}/_sizes.json')))
    bad = []
    for n in names:
        p, q = f'{CORPUS_DRIVE}/{n}.u32', f'{CORPUS}/{n}.u32'
        if not os.path.exists(p):
            bad.append(f'{n} (not on Drive yet: still uploading?)')
        elif not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
            bad.append(n)
    assert not bad, f'missing or short, rerun this cell: {bad}'
    return len(names)

if DATA_SOURCE == 'hf':
    from huggingface_hub import snapshot_download
    try:
        from google.colab import userdata; tok = userdata.get('HF_TOKEN')
    except Exception:
        tok = None                                  # falls back to huggingface_hub's own login
    snapshot_download(HF_REPO, repo_type='dataset', local_dir=CORPUS, token=tok)
    check_shards = lambda: len(glob.glob(f'{CORPUS}/*.u32'))
else:
    assert os.path.exists(f'{CORPUS_DRIVE}/_sizes.json'), f'no cache at {CORPUS_DRIVE}: build it first (see above)'
    !rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
print(check_shards(), 'shards on local disk, all at full size')
if not os.path.exists(TOKENIZER):                   # the fresh cache may not carry its own copy
    os.makedirs(os.path.dirname(TOKENIZER), exist_ok=True)
    for t in (f'{BASE_CACHE}/tokenizer/grillcheese_bbpe128k.json', f'{DRIVE}/grillcheese_bbpe128k.json'):
        if os.path.exists(t):
            !cp "{t}" "{TOKENIZER}"
            break
assert os.path.exists(TOKENIZER), TOKENIZER

# Same validation, fresh training data: the held-out tails and the mix sizes must be base450m's,
# and the subset must say it excluded the first one's blocks.
same = lambda f: json.load(open(f'{CORPUS}/{f}')) == json.load(open(f'{BASE_CACHE}/{f}'))
if os.path.exists(f'{BASE_CACHE}/_holdout.json'):
    assert same('_holdout.json'), 'held-out tails differ from base450m\'s cache: validation would not compare'
    assert same('_sizes.json'), 'mix sizes differ from base450m\'s cache: the data mix would change'
    print('held-out tails and mix sizes: the same as base450m\'s')
sub = json.load(open(f'{CORPUS}/_subset.json')) if os.path.exists(f'{CORPUS}/_subset.json') else {}
assert sub.get('exclude_frac') or sub.get('exclude_dir'), \
    'this cache does not record excluding base450m\'s blocks: build it with --exclude-frac 0.2'
print(f"fresh subset: --frac {sub['frac']}, excluding --frac {sub['exclude_frac']} {sub['exclude_dir'] or ''}")
!du -sh {CORPUS}; df -h /content | tail -1
"""),
    code(r"""
# --- config, and a runner that tees the log to Drive ---
STEPS = int(TARGET_TOKENS // 131072)
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR=RUN_DIR, CB_NAME='base450m_cont',
    CB_INIT_FROM=BASE_CKPT,                                     # weights only; ignored once a slot exists
    CB_D='1024', CB_L='32', CB_HEADS='4', CB_WINDOW='512', CB_ATTN_EVERY='3',
    CB_S='1024', CB_MICRO='32', CB_TOKENS_PER_STEP='131072',   # micro set below from the card's memory
    CB_LR='3e-4', CB_BETA2='0.95', CB_WD='0.1',
    CB_WARMUP='300', CB_DECAY_FRAC='0.2', CB_STEPS=str(STEPS), CB_SEED='2',
    CB_EVAL='250', CB_GEN='500', CB_CKPT_MIN='45',
    CB_STABLE_END='1',                                          # keep the pre-decay state for a later leg
)
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 2**30
BASE['CB_MICRO'] = '32' if gb >= 70 else ('16' if gb >= 38 else '8')   # divides 128: 131,072 tokens a step
# flex attention's compiled backward at head size 256 needs ~112 KB of shared memory per block. A100, H100 and
# B200 have it; RTX PRO 6000 Blackwell, L4 and other sm_86/89/120 cards stop at 99 KB. There the attention
# layers take the reference path (dense-mask SDPA, the same maths).
smem = getattr(props, 'shared_memory_per_block_optin', 0) or (
    160 * 1024 if (props.major, props.minor) in ((8, 0), (9, 0), (10, 0)) else 0)
if smem < 114688:
    BASE['CB_NO_FLEX'] = '1'
print(f"{props.name}, {gb:.0f} GB, {smem // 1024} KB shared memory per block -> micro {BASE['CB_MICRO']}, "
      f"{'reference attention (CB_NO_FLEX=1)' if smem < 114688 else 'flex attention'}")
print(f"{STEPS:,} steps = {STEPS * 131072 / 1e9:.2f}B tokens, decay from step {int(STEPS * 0.8):,}")

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
### The run

**Rerun this same cell in every session.** The first session starts from base450m's weights; every later one
resumes from the newest Drive slot, with the schedule fixed at the start (`TARGET_TOKENS` is read only then).
Stopping the cell saves a checkpoint first. The hard stop is set before the session ends, so the last checkpoint
always lands on Drive. The first log lines should say `initialised from ... base450m_final.pt` (first session)
or `resumed ... @ step N` (later ones).
"""),
    code(r"""
check_shards()                                     # a new runtime starts with an empty local disk
left_h = SESSION_H - 0.4 - (time.time() - SESSION_T0) / 3600
print(f'{left_h:.1f} h left in this session: hard stop after {left_h - 0.3:.1f} h')
run(dict(CB_WALL_H=f'{left_h - 0.3:.2f}'), 'train.log')
"""),
    code(r"""
# --- readout: against base450m on the same tails (any time, from any session; the checks count at the end) ---
import matplotlib.pyplot as plt
STEPS = globals().get('STEPS') or int(TARGET_TOKENS // 131072)   # the config cell needs a GPU; this one does not
def vals(path):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r for r in rows if 'val' in r]
bva = vals(f'{BASE_RUN}/base450m_metrics.jsonl')
base = bva[-1]['val'] if bva else dict(_mix=2.3615, _copy=1.20, fineweb_edu=3.316, books=3.115,
                                       nemotron_qa=3.189, nemotron_code=1.510)   # H-E5's recorded finals
va = vals(f'{RUN_DIR}/base450m_cont_metrics.jsonl')
assert va, 'no validation yet: the first comes at step 250'
v, step = va[-1]['val'], va[-1]['step']
KEY = ('fineweb_edu', 'books', 'nemotron_qa', 'nemotron_code')
done = step >= STEPS
print(f"step {step:,} of {STEPS:,}: {'finished' if done else 'in progress, the checks count at the end'}\n")
print(f"{'source':16s} {'base450m':>9s} {'now':>9s} {'change':>8s}")
for k in sorted(k for k in v if not k.startswith('_')):
    b = base.get(k, float('nan'))
    print(f"{k:16s} {b:9.4f} {v[k]:9.4f} {v[k] - b:+8.4f}{'   (check 2)' if k in KEY else ''}")
print(f"{'mix':16s} {base['_mix']:9.4f} {v['_mix']:9.4f} {v['_mix'] - base['_mix']:+8.4f}   (check 1: <= -0.03)")
print(f"{'copy':16s} {base['_copy']:9.4f} {v['_copy']:9.4f} {v['_copy'] - base['_copy']:+8.4f}   (check 3: <= +0.10)")
checks = {'1. mix at least 0.03 below the base': v['_mix'] <= base['_mix'] - 0.03,
          '2. FineWeb-Edu, books, QA, code each below the base': all(v[k] < base[k] for k in KEY),
          '3. copy no more than 0.1 above the base': v['_copy'] <= base['_copy'] + 0.1}
print()
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'fail'}  {k}")
if done:
    verdict = ('PASS: retrain the talk adapter on base450m_cont_final.pt and re-run gates E6-E9'
               if all(checks.values()) else
               'FAIL: ' + ('more pretraining is not what the base lacks; look at the mix'
                           if not checks['1. mix at least 0.03 below the base'] else
                           'the mix fell but a check failed; read the per-source table before using this base'))
    print('\nverdict:', verdict)
    with open(f'{RUN_DIR}/readout.json', 'w') as f:
        json.dump({'verdict': verdict, 'step': step, 'checks': checks, 'base': base, 'final': v}, f, indent=1)

tok = [r['step'] * 131072 / 1e9 for r in va]
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].plot(tok, [r['val']['_mix'] for r in va], 'k'); ax[0].axhline(base['_mix'], color='k', ls='--', lw=.8)
ax[0].axhline(base['_mix'] - 0.03, color='g', ls=':', lw=.8)
ax[0].set(title='held-out mix (dashed: base450m, dotted: the bar)', xlabel='B tokens this run')
for k in KEY:
    line, = ax[1].plot(tok, [r['val'][k] for r in va], label=k)
    ax[1].axhline(base[k], color=line.get_color(), ls='--', lw=.8)
ax[1].legend(fontsize=8); ax[1].set(title='the four checked sources (dashed: base450m)', xlabel='B tokens this run')
ax[2].plot(tok, [r['val']['_copy'] for r in va]); ax[2].axhline(base['_copy'] + 0.1, color='g', ls=':', lw=.8)
ax[2].set(title='in-context copy (dotted: the bar)', xlabel='B tokens this run')
plt.tight_layout(); plt.show()
"""),
    md(r"""
### How to read

- **The loss rises first.** The re-warm lifts the annealed weights off their minimum: in the pilot's first hour at
  3e-4, every source rose ~0.06 and `copy` went from 1.20 to 1.46. That is the price of continuing, not damage.
- **The stable phase** should then fall, slowly: by ~1B tokens the mix should be clearly below its first-hour
  level. It may still sit above 2.3615 when the decay starts (step 19,531 at the default budget); that is normal.
  The base's own plateau was 2.569, and its decay took 0.21 off. Flat for a whole billion tokens is the signal to
  stop and look.
- **The decay** (last 20%) gives the big drop, and it up-weights FineWeb-Edu, QA, code and books, the four checked
  sources. Wiki, facts and legal are down-weighted on purpose (knowledge lives in the store), so they fall less.
- **Samples** every 500 steps (EN, FR, QA, code) should stay fluent throughout; the facts in them stay unreliable,
  by design.

### After the run

- `RUN_DIR/base450m_cont_final.pt` holds the weights (fp32, ~2.3 GB). `validation/export_base.py --ckpt ... --out ...`
  exports it for grilly2.
- **If the readout passes:** point `talk_sft.ipynb`'s `CKPT` at the new final, retrain the talk adapter (~30 min),
  and re-run gates E6-E9 (`validation/exp_e6_talk_gate.py`, `exp_e7_explicit_gate.py`, `exp_e8_history_retrieval.py`,
  `exp_e9_context_followups.py`) unchanged.
- **A later leg** continues from `base450m_cont_stable_end.pt`, the pre-decay state, so it does not re-warm again.
  Build a newer subset with `--exclude-dir` for both earlier caches, then run with a new `CB_NAME`, `CB_CORPUS`
  pointing at it, `CB_STEPS` set to the new total, `CB_DECAY_START` at 80% of it, and `CB_RESUME_FROM` set to the
  stable-end file **in the first session only** (with it set, every session would restart from that file).
- Record the result under H-E5 in `CUBBYLLM_HYPOTHESES.md`, with `train.log` and the metrics copied into
  `validation/logs/`.
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
