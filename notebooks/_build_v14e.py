"""Build notebooks/standin_v14e_nochain_sft.ipynb -- the WO-1.3 arm.

Same pattern as `_build_gen3.py`: the notebook is generated from this file so
the source of truth is reviewable Python, not JSON with embedded newlines.

    python notebooks/_build_v14e.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "standin_v14e_nochain_sft.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src(text)}


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


INTRO = r"""
# [stand-in] CubeLang emitter — **v14e_nochain**: the WO-1.3 arm (drop `chain`)

Trains `LiquidAI/LFM2.5-2.6B` on `emitter_sft_v14e_nochain.jsonl` = **v13e with every `chain`
row removed** and the decorative attributes (`@external` / `@system` / `@once`) stripped.
Everything else — LoRA rank, targets, schedule, seed, `MAX_SEQ`, the masking arm — is byte-identical
to `standin_gen3_masked_sft.ipynb`, because the whole point is that the DATA is the only variable.

## What this tests

`chain` is 19,404 of v13e's 38,808 rows. It is work the host already does deterministically
(`build_chain_program`), and in the training data the facts are handed to the model in the prompt —
so the model is transcribing, not discovering. It is also the source of **~99% of the per-relation
role identifiers**: dropping it takes the role vocabulary from **412 distinct roles to 17**, and
what survives is the generalizing shape (`SEED`, `HOP1`, `HOP2`, `ASK`, `ACTION`, `AGENT`).

**Falsifiable prediction:** plan accuracy and cross-task retention improve, with **no loss of
VM-verified answers**.

**Kill criterion:** if the VM-verified answer rate drops, `chain` was doing something the audit did
not see, and this arm is abandoned rather than explained.

## ⚠️ Read before spending GPU time: this arm needs a control that does not exist yet

`validation/exp_r17_gen3_heldout.py`'s **531 / 600 at 0 wrong** is often quoted as "the v13e bar".
It is not. That number was produced by **`emitter_v12e.Q4_K_M.gguf`** — the gen-2 model, scored on
the gen-3 held split (`validation/logs/exp_r17_gen3_heldout_gen2_lev8.log`, first line). It is the
incumbent's score, i.e. the bar v13e was supposed to beat.

`standin/models/` contains no `emitter_v13e` or `v13f` GGUF: **the with-`chain` corpus has never
been trained.** So comparing this arm against 531/600 would confound two changes at once — dropping
`chain`, and adding all of gen 3's free-text data on top of gen 2.

For a clean read of the prediction, run **two** sessions with the same `STANDIN_ARM`:

| run | `STANDIN_VERSION` | role |
|---|---|---|
| control | `v13e` | with `chain` |
| treatment | `v14e_nochain` | without `chain` |

Then score both with the same commands (cell 4 prints them). Comparing treatment to `v12e`'s
531/600 alone answers a different question than the one WO-1.3 asks.

## The recipe is v12e's, verbatim

The LoRA and SFT settings in the training cell are read off `standin/models/train_card.json` — the
card of `emitter_v12e`, the model that holds the 531/600 reference:

    lora: {r: 64, alpha: 128, targets: "q,k,v,out_proj,in_proj,w1,w2,w3"}
    sft:  {epochs: 2, batch: 32, lr: 1e-4, sched: cosine, max_seq: 4096}

An earlier draft of this notebook copied `standin_gen3_masked_sft.ipynb`'s block instead
(r=32, alpha=32, lr=2e-4, and `target_modules` naming `o_proj` / `gate_proj` / `up_proj` /
`down_proj`, which **LFM2.5 does not have** — its attention is `in_proj`/`out_proj` and its MLP is
`w1`/`w2`/`w3`). Two reasons that was wrong: it names submodules that do not exist, and a different
rank and LR from the baseline means the comparison measures the *recipe* as well as the data.
**Do not tune the training cell.** The control run uses it unedited.

## Arms

`STANDIN_ARM` defaults to **`full`** — loss on the whole chat text, the v8e/v12e recipe — because
v12e was trained that way and the chain of comparison has to be consistent. `masked` trains on the
assistant response only; if you use it, **use it for the control run too**. The adapter, merged
model and GGUF land in an arm-named folder, so the two never collide.
"""

SETUP = r"""
# --- setup (run once per session) ---
import os, json, time, random, re
!pip -q install unsloth trl datasets
import sys
from google.colab import drive; drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/cubbyllm/standin'
if not os.path.exists('/content/CubbyLLM/.git'):
    !rm -rf /content/CubbyLLM && git clone https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM 2>&1 | tail -2
else:
    !cd /content/CubbyLLM && git fetch origin 2>&1 | tail -1 && git reset -q --hard origin/master && git log -1 --format='repo at %h %s'
sys.path.insert(0, '/content/CubbyLLM/standin/data'); sys.path.insert(0, '/content/CubbyLLM')
try:
    from identity import EMITTER_SYSTEM
    print('identity: from the repo checkout')
except ImportError as e:
    assert os.path.exists(f'{DRIVE}/identity.py'), f'identity.py not in the repo checkout nor at {DRIVE} ({e})'
    sys.modules.pop('identity', None); sys.path.insert(0, DRIVE)
    from identity import EMITTER_SYSTEM
    print('identity: from Drive')
# v14e_nochain is the WO-1.3 treatment arm; set STANDIN_VERSION=v13e to train the CONTROL
# (same everything, `chain` rows present) so the two are comparable.
VERSION = os.environ.get('STANDIN_VERSION', 'v14e_nochain')
# Default 'full', NOT 'masked': emitter_v12e -- the model holding the 531/600 reference -- was
# trained full (loss on the whole chat text), and the control run must match whatever is used here.
# Set STANDIN_ARM=masked to run the masking variant, but then run the CONTROL masked too.
ARM = os.environ.get('STANDIN_ARM', 'full')            # 'full': the v8e/v12e recipe | 'masked': loss on the assistant response only
assert ARM in ('masked', 'full'), ARM
DATA = f'{DRIVE}/emitter_sft_{VERSION}.jsonl'
MANIFEST = f'{DRIVE}/emitter_sft_{VERSION}.manifest.json'
OUT = f'{DRIVE}/emitter_lfm25_2p6b_{VERSION}'         # adapter + merged + GGUF land here
MODEL = os.environ.get('STANDIN_MODEL', 'LiquidAI/LFM2.5-2.6B')   # the v8e base; keep it, or the contrast is not attributable
MODEL_TAG = '' if MODEL == 'LiquidAI/LFM2.5-2.6B' else '_' + MODEL.split('/')[-1].lower().replace('.', 'p')
OUT = OUT + MODEL_TAG + '_' + ARM
print('arm:', ARM, '| out:', OUT)
EVAL_ONLY = os.environ.get('STANDIN_EVAL_ONLY') == '1'
if EVAL_ONLY: print('EVAL_ONLY: will load', f'{OUT}/merged', '(exists:', os.path.exists(f'{OUT}/merged'), ')')
MAX_SEQ = 4096   # the v8e setting (owner, 2026-09-03: 4096 trains better than 2048 on the A100-80G)
!nvidia-smi --query-gpu=name,memory.total --format=csv
for f in (DATA, MANIFEST):
    print(('ok      ' if os.path.exists(f) else 'MISSING ') + f)
m = json.load(open(MANIFEST))
assert m.get('version') == VERSION, f"manifest is {m.get('version')!r}, STANDIN_VERSION is {VERSION!r}"
print('manifest', m['version'], ':', m['by_task'], '| records', m['n_records'], '| train rows after repeat', m['train_rows_after_repeat'])
print('by source:', m['by_source'])
if 'chain' in m['by_task']:
    print(f"\n  NOTE: this corpus still has {m['by_task']['chain']:,} `chain` rows -- this is the CONTROL run.")
else:
    print('\n  no `chain` rows: this is the WO-1.3 treatment arm.')
"""

DATA_CELL = r"""
# --- data: chat-format the records; train/val from the builder's deterministic split ---
SYSTEM = EMITTER_SYSTEM
from collections import Counter
recs = [json.loads(l) for l in open(DATA, encoding='utf-8')]
# the v8e filter (VM-ok, and not gold-wrong) -- EXCEPT the r7 chains, which enter on the VM's word alone:
# gold is not available at serve time, and filtering by it would make gen 2's data cleaner than the loop
# can ever produce.
R7 = 'cubbyllm/cot_harvest_r7'
recs = [r for r in recs if r.get('vm_ok') in (True, None) and (r.get('gold_match') is not False or r.get('source') == R7)]
train = [r for r in recs if r['split'] == 'train']; val = [r for r in recs if r['split'] in ('val', 'held')]
print('by source (train):', Counter(r.get('source') for r in train).most_common())
print('train', len(train), Counter(r['task'] for r in train)); print('val  ', len(val), Counter(r['task'] for r in val))

# --- WO-1.3's mechanism, checked in-place: the role vocabulary should be FLAT ---
# A generalizing role interface does not grow one identifier per relation. v13e mints 412 distinct
# roles, ~99% of them `H<hop>_<RELATION>` and all from `chain`; this arm should be in the teens.
BINDS_ANY = re.compile(r'\bbind\s+[A-Za-z_][\w.]*\s*,\s*([A-Za-z_][\w.]*)\s*,')
roles = Counter(r_ for rec in recs for r_ in BINDS_ANY.findall(rec['program']))
per_rel = [k for k in roles if re.match(r'^H\d+_[A-Z0-9_]{3,}$', k)]
print(f'\nrole vocabulary: {len(roles)} distinct | {len(per_rel)} look per-relation | top:',
      [k for k, _ in roles.most_common(8)])
print('  (v13e measures 412 distinct / ~403 per-relation; this arm should be ~17 / ~4)')

NO_THINK = '<think>\n</think>\n'   # LFM2.5 opens <think> on its own; train it to close immediately (v2, 2026-08-30)
def to_messages(r):
    return [{'role': 'system', 'content': r.get('system') or SYSTEM},
            {'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': NO_THINK + r['program'].strip() + '\n'}]

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained((f'{OUT}/merged' if EVAL_ONLY else MODEL), max_seq_length=MAX_SEQ, load_in_4bit=False, dtype=None)
print('model:', f'{OUT}/merged (trained, from Drive)' if EVAL_ONLY else MODEL)

def fmt(r):
    return {'text': tokenizer.apply_chat_template(to_messages(r), tokenize=False, add_generation_prompt=False)}
from datasets import Dataset
random.Random(0).shuffle(train)
ds_train = Dataset.from_list([fmt(r) for r in train for _ in range(int(r.get('repeat', 1)))])   # builder's `repeat`
print('train rows after repeat weights:', len(ds_train))
lens = [len(tokenizer(x['text']).input_ids) for x in ds_train.select(range(min(500, len(ds_train))))]
print('token lengths (sample of 500): max', max(lens), 'p95', sorted(lens)[int(0.95*len(lens))], '-> MAX_SEQ', MAX_SEQ)
plan_example = next(r for r in train if r['task'] == 'plan')
print('\na plan record, formatted:\n', fmt(plan_example)['text'][:700])
"""

TRAIN = r"""
# --- LoRA + SFT: THE v12e RECIPE, verbatim. Do not tune anything here. ---
#
# These are not fresh choices. They are read off `standin/models/train_card.json` -- the card of
# emitter_v12e, the model that produced the 531/600 at 0 wrong that everything is measured against:
#
#     lora: {r: 64, alpha: 128, targets: "q,k,v,out_proj,in_proj,w1,w2,w3"}
#     sft:  {epochs: 2, batch: 32, lr: 1e-4, sched: cosine, max_seq: 4096}
#
# Two things this fixes versus an earlier draft of this notebook, which copied
# standin_gen3_masked_sft.ipynb's r=32 / alpha=32 / lr=2e-4 block:
#
#  1. `target_modules` named `o_proj`, `gate_proj`, `up_proj`, `down_proj`. LFM2.5 has no such
#     submodules -- its attention is in_proj/out_proj and its MLP is w1/w2/w3. Worth noting that
#     no `emitter_v13e` or `v13f` GGUF exists in standin/models/, so that config plausibly never
#     ran to completion.
#  2. A different LoRA rank and LR than the baseline means the comparison measures the recipe as
#     well as the data. The whole point of WO-1.3 is that DATA is the only variable.
#
# The CONTROL run (STANDIN_VERSION=v13e) must use this same cell, unedited.
if EVAL_ONLY:
    print('EVAL_ONLY: skipping LoRA + training; the merged model from Drive is already loaded')
else:
    from trl import SFTTrainer, SFTConfig
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj', 'w1', 'w2', 'w3'],
        use_gradient_checkpointing='unsloth', random_state=3407)
    cfg = SFTConfig(output_dir='/content/emitter_ckpt', per_device_train_batch_size=32, gradient_accumulation_steps=1,
                    num_train_epochs=2, learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.05,
                    logging_steps=10, optim='adamw_8bit', save_strategy='no', bf16=True, max_seq_length=MAX_SEQ,
                    dataset_text_field='text', packing=False, report_to='none', seed=0)
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds_train, args=cfg)
    if ARM == 'masked':
        # loss on the assistant response only: the chat template's user / assistant headers, read off a
        # rendered probe rather than assumed (LFM2.5 is ChatML-shaped: <|im_start|>user ... <|im_start|>assistant)
        from unsloth.chat_templates import train_on_responses_only
        probe = tokenizer.apply_chat_template([{'role': 'user', 'content': 'XQX'}, {'role': 'assistant', 'content': 'YQY'}], tokenize=False)
        i, j = probe.find('XQX'), probe.find('YQY'); k = probe.rfind('<|im_start|>', 0, i)
        INSTR = probe[k:i]; between = probe[i + 3:j]; RESP = between[between.rfind('<|im_start|>'):] if '<|im_start|>' in between else between
        assert INSTR and RESP and INSTR != RESP, (INSTR, RESP)
        print('masking: instruction_part', repr(INSTR), '| response_part', repr(RESP))
        trainer = train_on_responses_only(trainer, instruction_part=INSTR, response_part=RESP)
        ex = trainer.train_dataset[0]; n_lab = sum(1 for x in ex['labels'] if x != -100); n_tok = len(ex['labels'])
        assert 0 < n_lab < n_tok, (n_lab, n_tok)
        print(f'masking check on row 0: {n_lab}/{n_tok} tokens carry loss ({n_lab / n_tok:.0%}); the rest is context')
        print('  loss tokens decode to:', repr(tokenizer.decode([t for t in ex['labels'] if t != -100])[:200]))
    else:
        print('full: loss on the whole chat text, as v8e / v12e -- the baseline arm')
    t0 = time.time(); stats = trainer.train(); print(f'trained in {(time.time()-t0)/60:.1f} min; final loss', stats.training_loss)
    os.makedirs(OUT, exist_ok=True); model.save_pretrained(f'{OUT}/adapter'); tokenizer.save_pretrained(f'{OUT}/adapter')
    # `.get`, never `m['...']`. The v12e-era card read m['gen1_sha256'] / m['r7_sha256'], keys that
    # exist ONLY in v12e's manifest -- on any other corpus that is a KeyError raised AFTER training
    # finished and the adapter was written, so the run "succeeds" and still loses its provenance.
    # The hyperparameters below are read back off `cfg`/the LoRA call rather than retyped, so the
    # card cannot drift from what actually ran (the v12e card records "warmup: 20" while this cell
    # uses warmup_ratio=0.05 -- exactly that kind of stale copy).
    json.dump({'version': VERSION, 'arm': ARM, 'base': MODEL,
               'work_order': 'WO-1.3 (drop chain + decorative attributes)',
               'recipe': 'v12e (standin/models/train_card.json), unchanged',
               'corpus_sha256': m.get('output_sha256'), 'corpus_base': m.get('base'),
               'data_sha256': m.get('output_sha256'), 'gen1_sha256': m.get('gen1_sha256'),
               'r7_sha256': m.get('r7_sha256'), 'gen3': m.get('gen3'),
               'train_rows': len(ds_train), 'final_loss': stats.training_loss,
               'lora': {'r': 64, 'alpha': 128, 'targets': 'q,k,v,out_proj,in_proj,w1,w2,w3',
                        'random_state': 3407},
               'sft': {'epochs': cfg.num_train_epochs, 'batch': cfg.per_device_train_batch_size,
                       'lr': cfg.learning_rate, 'sched': cfg.lr_scheduler_type,
                       'warmup_ratio': cfg.warmup_ratio, 'optim': str(cfg.optim),
                       'max_seq': MAX_SEQ, 'seed': cfg.seed}},
              open(f'{OUT}/train_card.json', 'w'), indent=1)
    print('adapter ->', f'{OUT}/adapter', '| train card ->', f'{OUT}/train_card.json')
"""

EXPORT = r"""
# --- export FIRST: merged fp16 + GGUF (q8_0 for the VM eval, q4_k_m for the 12 GB Vulkan box) ---
if EVAL_ONLY:
    print('EVAL_ONLY: skipping export; the merged model from Drive is already loaded')
else:
    model.save_pretrained_merged(f'{OUT}/merged', tokenizer, save_method='merged_16bit')
    model.save_pretrained_gguf(f'{OUT}/gguf', tokenizer, quantization_method=['q8_0', 'q4_k_m'])
    import glob as _glob
    GGUFS = sorted(_glob.glob(f'{OUT}/gguf*/*.gguf'))
    print('GGUF files:'); [print('  ', g, f'{os.path.getsize(g)/1e9:.2f} GB') for g in GGUFS]
    TAG = f'{VERSION}_{ARM}'
    print(f'\nlocally, as standin/models/emitter_{TAG}.Q4_K_M.gguf, the gate:')
    print(f'  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_{TAG}.Q4_K_M.gguf --tag _{TAG}')
    print(f'  python validation/exp_r11_search_learn.py --wikidata --n 600 --seed 7 --lexicon --gguf standin/models/emitter_{TAG}.Q4_K_M.gguf --tag _{TAG}')
    print('\n  the WO-1.3 comparison needs the CONTROL scored the same way:')
    print(f'  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v13e_{ARM}.Q4_K_M.gguf --tag _v13e_{ARM}')
    print('\n  reference: emitter_v12e (gen 2) scores 531/600 at 0 WRONG on this split')
    print('  -- that is the INCUMBENT, not the control for this experiment.')
"""

SMOKE = r"""
# --- OPTIONAL format-level smoke eval on the val split (the verified read is local: exp_r17 / exp_r11) ---
# `plan` is scored by exact match on the CotPlan text (deterministic given the question), and ALSO by
# "role chain matches" -- the relations and their order, ignoring the seed spelling -- which is the part the
# disposer reads. Set STANDIN_EVAL_N=0 to skip.
FastLanguageModel.for_inference(model)
import transformers; transformers.logging.set_verbosity_error()
def strip_think(s):
    return re.sub(r'^\s*(?:<think>)?.*?</think>\s*', '', s, count=1, flags=re.S) if '</think>' in s else s
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'#.*', '', strip_think(s))).strip()
BINDS = re.compile(r'bind\s+frame\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"((?:[^"\\]|\\.)*)"\s*;')
def role_chain(prog):   # CotPlan: the HOPk strings in order; CotChain: the H-roles in order
    b = BINDS.findall(strip_think(prog))
    hops = sorted((int(k[3:]), v.lower()) for k, v in b if k.startswith('HOP') and k[3:].isdigit())
    if hops: return [v for _, v in hops]
    return [k for k, _ in b if re.match(r'^H\d+_', k)]
def emit(prompt, max_new=768, system=None):
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': system or SYSTEM}, {'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True) + NO_THINK
    enc = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
N_PER_TASK = int(os.environ.get('STANDIN_EVAL_N', '8'))
rng = random.Random(1); by_task = {}
for r in val: by_task.setdefault(r['task'], []).append(r)
sample = [r for t, rs in sorted(by_task.items()) for r in rng.sample(rs, min(N_PER_TASK, len(rs)))]
CAP = {'chain': 500, 'plan': 300, 'kernel': 600, 'arithmetic': 500, 'role_binding': 400}
print('eval sample', len(sample), {t: min(N_PER_TASK, len(rs)) for t, rs in sorted(by_task.items())})
hits = Counter(); tot = Counter(); chain_ok = Counter(); outputs = []
t0 = time.time()
for i, r in enumerate(sample):
    gen = emit(r['prompt'], max_new=CAP.get(r['task'], 600)); ok = norm(gen) == norm(r['program'])
    tot[r['task']] += 1; hits[r['task']] += int(ok)
    rc = None
    if r['task'] in ('plan', 'chain'):
        rc = role_chain(gen) == role_chain(r['program']); chain_ok[r['task']] += int(rc)
    outputs.append({'id': r['id'], 'task': r['task'], 'subtype': r.get('subtype', ''), 'source': r.get('source'),
                    'prompt': r['prompt'], 'reference': r['program'], 'generated': gen, 'exact_match': ok,
                    'role_chain_match': rc, 'gold': r.get('gold')})
    if (i+1) % 25 == 0: print(f'  {i+1}/{len(sample)} ({time.time()-t0:.0f}s)')
print('[stand-in] val exact-match by task:', {t: f'{hits[t]}/{tot[t]}' for t in tot}, '| overall', round(sum(hits.values())/max(1, sum(tot.values())), 3))
print('[stand-in] role-chain match (relations + order, what the disposer reads):', {t: f'{chain_ok[t]}/{tot[t]}' for t in chain_ok})
# `.get`, not `[...]`: only v12e's manifest ever carried output_sha256, so the gen-3 notebook's
# `m['output_sha256']` raises KeyError on v13e/v13f -- and this cell runs by default.
json.dump({'model': MODEL, 'version': VERSION, 'arm': ARM, 'n': len(sample),
           'exact_match_by_task': {t: hits[t]/tot[t] for t in tot},
           'role_chain_by_task': {t: chain_ok[t]/tot[t] for t in chain_ok}, 'outputs': outputs,
           'manifest_output_sha256': m.get('output_sha256')}, open(f'{OUT}/val_generations.json', 'w'), indent=1)
print('generations ->', f'{OUT}/val_generations.json')
"""

HOWTO = r"""
### How to read

- **This notebook proves nothing about the gate.** The smoke eval above is a *format* read — does the
  model reproduce the reference program text. The gate is on your machine:
  `python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v14e_nochain_<arm>.Q4_K_M.gguf --tag _v14e_<arm>`
  (the gen-3 **held** split: free-text wordings over entities no training record used), and
  `python validation/exp_r11_search_learn.py --wikidata --n 600 --seed 7 --lexicon --gguf ... --tag ...`
  (the same 600 SimpleQA gen 2 ran).

- **The comparison is treatment vs control, not treatment vs 531/600.** Score
  `v14e_nochain_<arm>` and `v13e_<arm>` on the *same* commands. `emitter_v12e`'s 531/600 at 0 wrong
  is the gen-2 incumbent — useful context, not the control, because v13e also added all of gen 3's
  free-text data.

- **What WO-1.3 predicts:** plan accuracy and cross-task retention up, **VM-verified answers not
  down**. Read three numbers on each run — plans the disposer accepts, verified-and-correct, and the
  honest one, **WRONG**, which must not rise.

- **The kill criterion is real.** If VM-verified answers drop, `chain` was doing something the audit
  did not see. Abandon the arm; do not explain the drop away.

- **A second thing this arm changes, worth watching separately:** the role vocabulary collapses from
  412 distinct identifiers to ~17. If cross-task retention improves while plan accuracy is flat, that
  is the vocabulary result, not the transcription result — they are different claims and the audit
  (`validation/sft_mix_audit.py`) separates them.

- **Everything here is `[stand-in]`.** It goes in `standin/README.md` and the research doc, never in
  `CUBBYLLM_HYPOTHESES.md` except as a pointer.
"""

nb = {
    "cells": [
        md(INTRO),
        code(SETUP),
        code(DATA_CELL),
        code(TRAIN),
        code(EXPORT),
        code(SMOKE),
        md(HOWTO),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(nb['cells'])} cells)")
