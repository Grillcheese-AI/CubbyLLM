"""Generate grow_mem_continue.ipynb: base450m_cont grows (depth x2 + FFN x2, zero exits) with the episodic
memory inserted (H-D5's read, exits zeroed), and keeps training on fresh data with instruction following, agentic
trajectories and the emitter's own programs in the decay; the control arm is the 450M itself on the same
data and hours. The notebook is thin; the logic is validation/grow_base.py (the step-0 build and its exactness
check), validation/train_base.py (CB_INIT_FROM, CB_MEM_EVERY, CB_S) and validation/make_decay_shards.py (the
three shards). Re-run after editing:
    python notebooks/_build_grow_mem.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "grow_mem_continue.ipynb"


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
# CubbyLLM base: grow, add the memory, and keep training

**What this is.** Two arms on the same data, the same hours and the same card, from the same weights
(`base450m_cont_stable_end.pt`: the continued 450M before its decay, so neither arm re-warms):

| arm | shape | what it tests |
|---|---|---|
| `grown` | depth x2 + FFN x2 with zero exits (d1024, L64, FFN 4096; the pilot's one surviving shape) **plus the episodic memory** (`cubbyllm/model/recall`, H-D5) on every 4th layer = 16 reads, the count "every 2 layers" gives on the 32-layer base; exits zeroed. 1.36B parameters, the same function as the 450M at step 0 (checked) | whether a bigger trunk and a read over the past beyond the 512-token window pay for their cost |
| `control` | the 450M as it is (d1024, L32, 576M) | the bar: the grow note's rule, "growth is worth it only if a grown arm beats the control at equal compute" |

Both train at **S = 2048** (the memory has nothing to read inside the window; the agent trajectories run 3-5k
tokens), and both get the **three new shards** in the decay (`validation/make_decay_shards.py`): `instruct`
(the host's guarded instruction-following chats, in the talk format), `agent` (observe → plan → act → observe
trajectories in a strict JSON action format) and `program` (the emitter's own slotted programs). So the
control also measures what the decay data alone brings to the 450M.

**Pre-registered (the readout cell applies it; the numbers are the owner's to move).** On the 13 held-out tails
the base validates on, plus the three new ones:
1. **Growth:** at equal hours, the grown arm's held-out mix ends at or below the control's. Above it by more than
   0.02 → the 450M still wants data before parameters; growth stays killed.
2. **Memory:** `validation/exp_needle_recall.py` on the grown arm, lengths 1024 / 2048 / 4096: the beyond-window
   depths rise off chance (> 25% where the control is at chance) with in-window recall unchanged. Flat →
   the memory read did not train at this scale; serve without it.
3. **The decay data:** each arm ends below its own first reading on `instruct`, `agent` and `program`, and no
   checked pretraining source (FineWeb-Edu, books, QA, code) ends more than 0.02 above the control's.
4. **Serving (after the adapters are retrained on the winner):** gates E6-E9 each at least talk_v2_2's with 0
   spoken wrong; the emitter on exp_r17 and the step loop at least the 450M's own numbers; the
   instruction-following gate (`validation/exp_if_follow.py`, to be built: rule-checked constraints over held chats)
   above the 450M's. The gap to the stand-in on the same benches is reported as measured, whichever way it goes.

**Round 2 (added 2026-10-20).** Round 1's grown arm (0.42B tokens, 20 h on a 96 GB RTX 6000 at 77,824-token
steps) learned the new shards and matched the 450M's trend, but gate 2 failed flat: beyond-window needle recall
15.9% against 16.8% for the shuffled control -- the 16 reads never trained. The small-scale hypertoken reads of
the same day (`validation/exp_k10_hypertoken.py`) say why: a read that must find its key from the hidden state
learns nothing from text that never pays for a token 512 back; given a reason, it learns (0.00 → 0.82). Round 2
gives the memory its reason: **`facts_text_far`** (`validation/make_fact_shard.py --far`: 8-16 wiki-world facts,
then ~1,500 tokens of real corpus text, then questions about the facts -- beyond the window, where only the
episodic store can answer) at ~5% of the tokens, continuing the grown arm from its `stable_end`. Gate 2's
readout becomes `exp_hypertoken_recall.py` on `facts_text_far` (the questions answered, teacher-forced; the
450M's own number beside it) plus the needle probe as before. Set `ROUND = 2` below.

**Before you run**
1. **Add the three shards** to the cache on the machine that holds the data, then copy them and the two index
   files to Drive:
   ```
   python validation/make_decay_shards.py --cache <local token_cache_cont> --agent-tokens 150000000
   copy instruct.u32 agent.u32 program.u32 _holdout.json _sizes.json -> <Drive>/cubbyllm/token_cache_cont/
   ```
2. **Push `master`** with `grow_base.py --mem-every` and `train_base.py`'s `CB_MEM_EVERY` (checked below).
3. **Drive space:** the grown arm's slots are ~16 GB each with the optimizer state (two of them), plus the 5.5 GB
   step-0 file and the 5.5 GB final: ~45 GB. The control's are the 7 GB of the continuation.
4. **Colab Pro+** for 24 h sessions; set `SESSION_H = 12` on Pro. Rerun the run cell in each new session; the
   grown arm will take several.
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
src = open(f'{REPO}/validation/train_base.py').read()
assert 'CB_MEM_EVERY' in src and 'CB_INIT_FROM' in src, 'master has no CB_MEM_EVERY in train_base.py: push the memory commit first'
assert '--mem-every' in open(f'{REPO}/validation/grow_base.py').read(), 'master has no --mem-every in grow_base.py'
"""),
    code(r"""
# --- EDIT to your layout ---
ARM           = 'grown'                             # 'grown' (depth x2 + FFN x2 + memory) or 'control' (the 450M)
SESSION_H     = 24                                  # Colab Pro+; 12 on Pro / pay-as-you-go
HOURS_PER_ARM = 20.0                                # equal compute: the same hours on the same card for both arms
MEM_EVERY     = 4                                   # on the 64-layer grown model: 16 reads, one per original pair
SEQ           = 2048                                # > the 512 window, so the memory has a past to read
TOKENS_PER_STEP = 80_000                            # the optimizer step (micro x accumulation, train_base derives the
                                                    # accumulation): 131,072 as pre-registered; 80,000 is what the grown arm
                                                    # ran at on the RTX 6000 (2026-10-18: 19 forwards of micro 2 a step
                                                    # instead of 32). BOTH ARMS MUST MATCH -- the readout checks it.
DRIVE         = '/content/drive/MyDrive/cubbyllm'
START_CKPT    = f'{DRIVE}/runs/base450m_cont/base450m_cont_stable_end.pt'   # the continued 450M, before its decay
CONT_RUN      = f'{DRIVE}/runs/base450m_cont'                               # its metrics: the first reading
CORPUS_DRIVE  = f'{DRIVE}/token_cache_cont'                  # + instruct.u32, agent.u32, program.u32
CORPUS        = '/content/token_cache_cont'                  # LOCAL SSD (Drive-FUSE reads are ~30x slower)
TOKENIZER     = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
ROUND         = 2                                   # 1: the first pair of arms; 2: the memory's round -- the grown arm continues
                                                    # from round 1's stable_end (before its decay, so the run picks up at the
                                                    # constant LR and decays at its own end) with facts_text_far in the mix
MEM_SHARD, MEM_MULT = 'facts_text_far', 23          # round >= 2: ~5% of the tokens (41M-token shard x 23 against the 17.7B cache:
                                                    # ~0.5 epoch at 0.42B tokens); the same share in the decay
PREV          = '' if ROUND <= 2 else f'_{ROUND - 1}'
RUN_SUFFIX    = '' if ROUND == 1 else f'_{ROUND}'
CONTINUE_FROM = '' if ROUND == 1 else f'{DRIVE}/runs/grow_mem_{ARM}{PREV}/grow_mem_{ARM}{PREV}_stable_end.pt'
RUN_DIR       = f'{DRIVE}/runs/grow_mem_{ARM}{RUN_SUFFIX}'   # checkpoints, metrics, logs, the step-0 file
assert ARM in ('grown', 'control'), ARM
if CONTINUE_FROM:
    assert os.path.exists(CONTINUE_FROM), f"round {ROUND} of the {ARM} arm needs round {ROUND - 1}'s stable_end: {CONTINUE_FROM}"
assert os.path.exists(START_CKPT), START_CKPT
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)

def check_shards():
    # Every shard the cache lists must be on Drive and on local disk at full size (the growth pilot's arms
    # trained on 9, 10 and 12 of the 13 shards because the local copy was incomplete for each).
    names = sorted(json.load(open(f'{CORPUS_DRIVE}/_sizes.json')))
    for n in ('instruct', 'agent', 'program') + ((MEM_SHARD,) if ROUND >= 2 else ()):
        assert n in names, f'{n}.u32 is not in the cache yet: build it and copy it to Drive'
    bad = []
    for n in names:
        p, q = f'{CORPUS_DRIVE}/{n}.u32', f'{CORPUS}/{n}.u32'
        if not os.path.exists(p):
            bad.append(f'{n} (not on Drive yet: still uploading?)')
        elif not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
            bad.append(n)
    assert not bad, f'missing or short, rerun this cell: {bad}'
    return len(names)

assert os.path.exists(f'{CORPUS_DRIVE}/_sizes.json'), f'no cache at {CORPUS_DRIVE}'
!rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
print(check_shards(), 'shards on local disk, all at full size')
assert os.path.exists(TOKENIZER), TOKENIZER
!du -sh {CORPUS}; df -h /content | tail -1
"""),
    md(r"""
### Step 0: the grown model, exact (the `grown` arm only; CPU, no GPU cost)

`grow_base.py` fills the depth x2 + FFN x2 shape from the start checkpoint with every copy's exit zeroed, inserts
a `MemoryRead` on every `MEM_EVERY`th layer with its output projection zeroed, and scores base and grown on
the same held-out windows: the losses must agree to 1e-4 and the logits to 1e-2 in float32. The saved file
is what the run initialises from. ~10 min on Colab's CPU; skip the cell on later sessions once the file exists.
"""),
    code(r"""
GROWN = CONTINUE_FROM or f'{RUN_DIR}/grow_step0_d2f2m{MEM_EVERY}.pt'
if CONTINUE_FROM:
    assert os.path.exists(CONTINUE_FROM), CONTINUE_FROM
    print(f'round {ROUND}: continuing the {ARM} arm from', CONTINUE_FROM)
if ARM == 'grown' and not CONTINUE_FROM and not os.path.exists(GROWN):
    !cd {REPO} && python validation/grow_base.py --ckpt "{START_CKPT}" --cache {CORPUS} --depth 2 --ffn 2 \
        --mem-every {MEM_EVERY} --sources arxiv,books,fineweb_edu,nemotron_code --windows 1 --seq 1024 \
        --save "{GROWN}" --tag colab_d2f2m{MEM_EVERY}
    rep = json.load(open(f'{REPO}/validation/logs/grow_step0_colab_d2f2m{MEM_EVERY}.json'))
    assert rep['exact'], 'the grown model is NOT exact: do not train it; read the log'
    print(f"exact: {rep['params_src']/1e6:,.0f}M -> {rep['params_dst']/1e6:,.0f}M parameters, "
          f"max |logit diff| {rep['max_logit_diff']:.1e}")
elif ARM == 'grown':
    print('step-0 file present:', GROWN)
"""),
    code(r"""
# --- config, and a runner that tees the log to Drive ---
shape = (dict(CB_L='64', CB_FFN_MULT='4', CB_MEM_EVERY=str(MEM_EVERY), CB_INIT_FROM=GROWN) if ARM == 'grown'
         else dict(CB_L='32', CB_INIT_FROM=CONTINUE_FROM or START_CKPT))
# the three decay shards: multipliers on size-proportional draws (15.5M, 79.5M and 7.3M tokens against a
# 17.7B cache), solved so that in the decay instruct is ~5%, agent ~10% and program ~2% of the tokens -- 0.8,
# 0.3 and 0.7 epochs of each at a 1.2B-token run -- and a trace of each in the stable phase (0.5 / 1 / 0.3%)
# so the base meets the formats before the decay. train_base prints the epoch table; keep every row under ~4.
MIX_JSON   = json.dumps({'instruct': 6, 'agent': 2, 'program': 8, **({MEM_SHARD: MEM_MULT} if ROUND >= 2 else {})})
DECAY_JSON = json.dumps({'instruct': 91, 'agent': 36, 'program': 78, **({MEM_SHARD: MEM_MULT} if ROUND >= 2 else {})})
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR=RUN_DIR, CB_NAME=f'grow_mem_{ARM}{RUN_SUFFIX}',
    CB_D='1024', CB_HEADS='4', CB_WINDOW='512', CB_ATTN_EVERY='3',
    CB_S=str(SEQ), CB_TOKENS_PER_STEP=str(TOKENS_PER_STEP),
    CB_LR='3e-4', CB_BETA2='0.95', CB_WD='0.1',
    CB_WARMUP='100', CB_DECAY_FRAC='0.2', CB_HOURS=str(HOURS_PER_ARM), CB_SEED='3',
    CB_MIX_JSON=MIX_JSON, CB_DECAY_JSON=DECAY_JSON,
    CB_EVAL='250', CB_GEN='500', CB_CKPT_MIN='45',            # samples every 500 steps (Nick, 2026-10-17)
    CB_STABLE_END='1',
    **shape,
)
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 2**30
# micro-batch: sequences per forward at S=2048 (x accumulation = TOKENS_PER_STEP, whatever the micro:
# train_base derives the accumulation from CB_TOKENS_PER_STEP, so a smaller micro changes throughput, never
# the batch or the schedule). The grown arm is 64 layers: without activation recompute even a 96 GB RTX 6000
# held micro 2 only (2026-10-18) -- the saved activations of 64 layers at S=2048 and the 128k-vocab logits in
# float32 fill the card long before the weights do (22 GB with Adam). So the grown arm always recomputes
# (CB_GRAD_CKPT=1: ~1/3 more compute a step, a fraction of the memory) and runs micro 8; the memory layers'
# (B, S, S) similarity and (B, S, K, d) gathered values are then transient too. train_base prints peak GB;
# raise micro if there is room (16 should fit at 96 GB).
if ARM == 'grown':
    BASE['CB_MICRO'] = '8'
    BASE['CB_GRAD_CKPT'] = '1'
else:
    BASE['CB_MICRO'] = '16' if gb >= 70 else '8'
smem = getattr(props, 'shared_memory_per_block_optin', 0) or (
    160 * 1024 if (props.major, props.minor) in ((8, 0), (9, 0), (10, 0)) else 0)
if smem < 114688:
    BASE['CB_NO_FLEX'] = '1'
print(f"{ARM}: {props.name}, {gb:.0f} GB -> micro {BASE['CB_MICRO']} at S={SEQ}"
      f"{' with activation recompute (CB_GRAD_CKPT=1)' if BASE.get('CB_GRAD_CKPT') else ''}, "
      f"{'reference attention (CB_NO_FLEX=1)' if smem < 114688 else 'flex attention'}; "
      f"{HOURS_PER_ARM} h, the step count is sized after the first measured steps and then fixed")

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

**Rerun this same cell in every session.** The first session initialises from the step-0 file (`grown`) or the
continuation's pre-decay weights (`control`) with a fresh optimizer and a 100-step warmup at the same peak,
measures its throughput, and fixes the step count to `HOURS_PER_ARM`; every later session resumes from the
newest Drive slot. Stopping the cell saves a checkpoint first. The first log lines should say
`initialised from ... grow_step0_... (grown from base450m_cont_stable_end.pt)` or `... not grown`.
Samples come every 500 steps.
"""),
    code(r"""
check_shards()                                     # a new runtime starts with an empty local disk
left_h = SESSION_H - 0.4 - (time.time() - SESSION_T0) / 3600
print(f'{left_h:.1f} h left in this session: hard stop after {left_h - 0.3:.1f} h')
run(dict(CB_WALL_H=f'{left_h - 0.3:.2f}'), 'train.log')
"""),
    code(r"""
# --- readout: both arms side by side, against the continuation's last reading (any time, from any session) ---
import matplotlib.pyplot as plt
def rows_of(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []
def vals(path):
    return [r for r in rows_of(path) if 'val' in r]
def step_size(path):
    # tokens per optimizer step, from the training rows (tokens seen / steps): the arms must agree
    tr = [r for r in rows_of(path) if 'tokens' in r and r.get('step')]
    return round(tr[-1]['tokens'] / tr[-1]['step']) if tr else None
mpath = lambda a: f'{DRIVE}/runs/grow_mem_{a}{RUN_SUFFIX}/grow_mem_{a}{RUN_SUFFIX}_metrics.jsonl'
arms = {a: vals(mpath(a)) for a in ('grown', 'control')}
sizes = {a: step_size(mpath(a)) for a in ('grown', 'control')}
hours = {a: max((r['hours'] for r in rows_of(mpath(a)) if 'hours' in r), default=None) for a in ('grown', 'control')}
print('tokens per step:', sizes, '  hours:', {a: (round(h, 2) if h else None) for a, h in hours.items()})
cont_rows = rows_of(f'{CONT_RUN}/base450m_cont_metrics.jsonl')
cont = [r for r in cont_rows if 'val' in r]
cont_decay = min((r['step'] for r in cont_rows if r.get('phase') == 'decay'), default=10**9)
start = ([r for r in cont if r['step'] <= cont_decay] or cont or [{'val': {}}])[-1]['val']   # the arms start from stable_end
NEW, KEY = ('instruct', 'agent', 'program'), ('fineweb_edu', 'books', 'nemotron_qa', 'nemotron_code')
have = {a: v for a, v in arms.items() if v}
assert have, 'no validation yet: the first comes at step 250'
print(f"{'source':16s} " + ' '.join(f'{a:>10s}' for a in have) + '   (start = base450m_cont at its stable_end, S=1024)')
print('note: _copy repeats a half-sequence; at S=2048 the repeat is 1024 tokens back, beyond the window --')
print('      only a memory read can lower it (floor ~ ln(vocab) = 11.8); at S=1024 it is in-window copy (~1.2).')
keys = sorted({k for v in have.values() for k in v[-1]['val'] if not k.startswith('_')})
for k in keys + ['_mix', '_copy']:
    row = ' '.join(f"{v[-1]['val'].get(k, float('nan')):10.4f}" for v in have.values())
    tag = '  new shard' if k in NEW else ('  checked' if k in KEY else '')
    print(f"{k:16s} {row}   start {start.get(k, float('nan')):8.4f}{tag}")
checks = {}
if len(have) == 2:
    g, c = have['grown'][-1]['val'], have['control'][-1]['val']
    same_step = sizes['grown'] and sizes['control'] and abs(sizes['grown'] - sizes['control']) <= 2 * SEQ
    same_hours = hours['grown'] and hours['control'] and abs(hours['grown'] - hours['control']) <= 0.5
    checks['0. equal compute: the same tokens per step and the same hours on both arms'] = bool(same_step and same_hours)
    checks['1. growth: grown mix <= control mix (+0.02 tolerance)'] = g['_mix'] <= c['_mix'] + 0.02
    checks['3b. checked sources: grown within 0.02 of the control'] = all(g[k] <= c[k] + 0.02 for k in KEY)
for a, v in have.items():
    first, last = v[0]['val'], v[-1]['val']
    checks[f'3a. {a}: instruct, agent, program each below their first reading'] = all(
        last.get(k, 9) < first.get(k, 0) for k in NEW if k in first)
print()
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'fail'}  {k}")
print('\ncheck 2 (the memory): the facts_text_far probe below on each final, then exp_needle_forward; check 4 is the serving gates.')
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for a, v in have.items():
    tok = [r['step'] * (sizes.get(a) or TOKENS_PER_STEP) / 1e9 for r in v]      # tokens as measured, not as planned
    ax[0].plot(tok, [r['val']['_mix'] for r in v], label=a)
    for k in NEW:
        if k in v[-1]['val']:
            ax[1].plot(tok, [r['val'][k] for r in v], label=f'{a}: {k}')
    ax[2].plot(tok, [r['val']['_copy'] for r in v], label=a)
if start:
    ax[0].axhline(start['_mix'], color='k', ls='--', lw=.8)
ax[0].legend(); ax[0].set(title='held-out mix (dashed: the start)', xlabel='B tokens')
ax[1].legend(fontsize=7); ax[1].set(title='the three new shards', xlabel='B tokens')
ax[2].legend(); ax[2].set(title=f'in-context copy ({"beyond the window" if SEQ > 2 * 512 else "in the window"})', xlabel='B tokens')
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Check 2, round 2: the facts answered past the window

`exp_hypertoken_recall.py` on `facts_text_far`'s held-out documents: the questions sit ~1,500 tokens past the
facts, beyond the 512 window, so a right answer can only have come through the episodic store (grown) or not at
all (the control and the 450M as it is: the floor -- the format learned or not, the fact out of reach either way).
Pass: the grown final above both floors by 10 points with the in-window `_copy` floor unchanged; then
`exp_needle_forward.py` at 1024 / 2048 / 4096 as pre-registered.
"""),
    code(r"""
probes = [(a, f'{DRIVE}/runs/grow_mem_{a}{RUN_SUFFIX}/grow_mem_{a}{RUN_SUFFIX}_final.pt',
           f'{DRIVE}/runs/grow_mem_{a}{RUN_SUFFIX}/hypertoken_recall_facts_far.json') for a in ('grown', 'control')]
probes.append(('450M as it is (the floor)', f'{CONT_RUN}/base450m_cont_final.pt', f'{CONT_RUN}/hypertoken_recall_facts_far.json'))
for a, final, out in probes:
    if not os.path.exists(final):
        print(a, 'has no final yet'); continue
    print('===', a)
    env = dict(os.environ, CB_CKPT=final, CB_CORPUS=CORPUS, CUBBY_SPM=TOKENIZER, CB_SHARDS=MEM_SHARD, CB_DOCS='400',
               CB_BATCH='8', CB_OUT=out)
    subprocess.run([sys.executable, 'validation/exp_hypertoken_recall.py'], cwd=REPO, env=env, check=True)
"""),
    md(r"""
### After the run

- `RUN_DIR/grow_mem_<arm>_final.pt` holds the weights. `validation/export_base.py --ckpt ... --out ...` exports
  for grilly2 — **the grown arm needs grilly2's cubby model to carry the memory first** (not there as of
  2026-10-17; the torch path serves it meanwhile).
- **Check 2:** `python validation/exp_needle_recall.py` on the grown final at 1024 / 2048 / 4096, the control
  beside it.
- **Check 4:** retrain the emitter LoRA (`notebooks/emitter_450m.ipynb`) and the talk LoRA (`talk_sft.ipynb`) on
  the winner (its `CB_L` / `CB_FFN_MULT` / `CB_MEM_EVERY` in the adapter notebooks), then gates E6-E9,
  exp_r17, the step loop, and the instruction-following gate once it is built.
- Record the result under its H entry in `CUBBYLLM_HYPOTHESES.md`, with `train.log` and the metrics copied into
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
