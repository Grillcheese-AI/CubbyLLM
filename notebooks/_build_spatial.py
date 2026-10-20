"""Generate spatial_facts_450m.ipynb: the continued 450M keeps training with the spatial link on the two
hypertoken shards -- villages as places (``spatial_here``) and the wiki world's facts at addresses (``facts``) --
against the same trunk on the same villages and facts as text, at the same tokens (H-S12 gates 1-2, H-K10
gate 4, H-K7 gate 5 in one run). The notebook is thin; the logic is validation/train_base.py (CB_SPATIAL,
CB_SPATIAL_QUERY, CB_ROUTE_LOSS, the Corpus side files), validation/make_spatial_shard.py --ask placed,
validation/make_fact_shard.py and validation/exp_hypertoken_recall.py (the readout). Re-run after editing:
    python notebooks/_build_spatial.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "spatial_facts_450m.ipynb"


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
# CubbyLLM base: the spatial link and the fact link, trained

**What this is.** Two arms from the same weights (`base450m_cont_stable_end.pt`, the continued 450M before
its decay), the same tokens, the same hours, the same card:

| arm | what the trunk sees | what it tests |
|---|---|---|
| `link` | the villages as **places** (`spatial_here`: every name token bound with its cell, every question carrying the cell it is about) and the wiki world's facts at **addresses** (`facts`: the object's tokens bound at the subject's cell + the relation's step, the question's `here` at the same address), through `SpatialLink(query="placed")` with the route loss on | whether a trunk reads the scene it was shown -- a fact seen **once**, answered -- and whether the learned route (the trunk's own query, taught by `CB_ROUTE_LOSS`) comes along |
| `text` | the same villages with cells as words (`spatial_text_here`) and the same facts as plain text (`facts_text`), nothing placed | the bar: attention on the same questions at the same tokens |

Both arms train at **S = 1024** (the shards' documents are 512 tokens, aligned, so every window holds whole
documents) with the two shards at ~5% of the tokens each, the rest the cache the 450M always trains on.

**Pre-registered (the readout cell applies it).** `validation/exp_hypertoken_recall.py` on each arm's final, every
held-out document of its two shards, teacher-forced exact answers:
1. **H-S12 gate 1:** `link` answers the villages' held-out questions at ≥ 90% where `text` sits below 60%.
2. **H-K10 gate 4 / H-K7 gate 5:** `link` answers the held-out facts' questions at ≥ 2× `text` (and ≥ 50%).
3. **H-S12 gate 2:** the text held-out mix of `link` within 0.01 of `text` (the link costs the trunk nothing on text).
4. **The route:** `link`'s training log carries the route loss inside the total; a second probe with the places
   given but the *questions'* `here` removed is the learned route's own number (reported, no gate yet).
Kill: gate 1 below 75%, or gate 3 worse by 0.02.

**Run 1 (2026-10-20, G4, 2,808 steps = 0.37B tokens an arm) and what run 2 changes.** The link arm answered the
villages' held-out questions at **0.793** with the address on the question and **0.700** with the trunk's own query
(the route, taught by the route loss); the facts at **0.016** (first token 0.27) -- the read failed on the facts, for
reasons in the shard and the scene, not the algebra: every placed token wrote into one scene per 1,024-token
window (two documents, every name piece, *and the questions' own tokens at the addresses they asked about*:
~150-250 items in d = 1024, past the capacity law's ~55 for 90%), and the facts sat inside the attention window
(so `text` copies them and "link ≥ 2× text" tested attention, not the read). **Run 2:** a thing WRITES into the
scene and a question only ASKS there (`.role.u8` beside the places; `train_base.Tokens.flags`); the scene restarts
at every document (`SpatialLink.scene(starts=)`); the facts move **beyond the window** -- `facts_far1k`
(`make_fact_shard.py --far --doc 1024 --facts 6 10`: 6-10 facts, ~800 tokens of real text, then the questions;
~24 written items a document) against `facts_far1k_text`, the same documents as words, where only the trunk's
recurrence could carry a fact 800 tokens; micro 32.

**The small-scale reads this rests on (2026-10-20, `validation/exp_k10_hypertoken.py`):** a 3.4M trunk, facts shown
once, questions beyond the window -- 0.94 with the address on the question, 0.82 for the learned route taught by
the route loss, 0.00 for text, 0.03 for attention over the whole document, 0.00 with a mismatched address.

**Before you run**
1. The shards are on Drive under `token_cache_cont/` -- run 1: `spatial_here` (+ `.pos.i16`, `.role.u8`),
   `spatial_text_here`, `facts` (+ `.pos.i16`), `facts_text`; run 2 adds `facts_far1k` (+ `.pos.i16`, `.role.u8`)
   and `facts_far1k_text` -- built 2026-10-20, with the three index files.
2. **Push `master`** with `CB_SPATIAL_QUERY` and `CB_ROUTE_LOSS` in `train_base.py` (checked below).
3. Each arm is ~2 h on an A100; the two can run in two sessions or two runtimes. No activation recompute
   (the route loss taps the query in the forward: `CB_GRAD_CKPT=0`).
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
for k in ('CB_SPATIAL', 'CB_SPATIAL_QUERY', 'CB_ROUTE_LOSS'):
    assert k in src, f'master has no {k} in train_base.py: push the hypertoken commits first'
assert os.path.exists(f'{REPO}/validation/exp_hypertoken_recall.py'), 'push exp_hypertoken_recall.py first'
"""),
    code(r"""
# --- EDIT to your layout ---
ARM           = 'link'                              # 'link' (places + the route loss) or 'text' (the control)
SESSION_H     = 12                                  # Colab Pro; 24 on Pro+
HOURS_PER_ARM = 2.0                                 # equal compute: the same hours on the same card for both arms
SEQ           = 1024                                # whole documents a window: two villages (512) or one far-facts doc (1024)
TOKENS_PER_STEP = 131_072                           # the optimizer step; BOTH ARMS MUST MATCH (the readout checks it)
RUN           = 2                                   # 1: the first pair (facts in-window, one scene a window); 2: roles,
                                                    # a scene per document, the facts beyond the window (see above)
SHARD_MULT    = 96                                  # ~5% of the tokens each for the two 10M-token shards against the 17.7B
                                                    # cache (size x multiplier draws): ~35M tokens = 3.5 epochs of each at 2 h
DRIVE         = '/content/drive/MyDrive/cubbyllm'
START_CKPT    = f'{DRIVE}/runs/base450m_cont/base450m_cont_stable_end.pt'   # the continued 450M, before its decay
CORPUS_DRIVE  = f'{DRIVE}/token_cache_cont'
CORPUS        = '/content/token_cache_cont'                  # LOCAL SSD (Drive-FUSE reads are ~30x slower)
TOKENIZER     = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
NAME          = f'spatial_{ARM}' + ('' if RUN == 1 else f'_{RUN}')
RUN_DIR       = f'{DRIVE}/runs/{NAME}'
FACTS         = {1: ('facts', 'facts_text'), 2: ('facts_far1k', 'facts_far1k_text')}[RUN]
SHARDS        = {'link': ('spatial_here', FACTS[0]), 'text': ('spatial_text_here', FACTS[1])}[ARM]
MULTS         = {n: (SHARD_MULT if n.startswith('spatial') or RUN == 1 else SHARD_MULT // 2) for n in SHARDS}
                                                    # facts_far1k is 20M tokens, twice the others: half the multiplier
assert ARM in ('link', 'text'), ARM
assert os.path.exists(START_CKPT), START_CKPT
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)

def check_shards():
    names = sorted(json.load(open(f'{CORPUS_DRIVE}/_sizes.json')))
    for n in SHARDS:
        assert n in names, f'{n}.u32 is not in the cache yet: build it and copy it to Drive'
    bad = []
    for n in names:
        for ext in ('.u32', '.pos.i16'):
            p, q = f'{CORPUS_DRIVE}/{n}{ext}', f'{CORPUS}/{n}{ext}'
            if ext == '.pos.i16' and not os.path.exists(p):
                continue                                       # a text shard has no side file
            if not os.path.exists(p):
                bad.append(f'{n} (not on Drive yet: still uploading?)')
            elif not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
                bad.append(n + ext)
    assert not bad, f'missing or short, rerun this cell: {bad}'
    for n in SHARDS:
        a = json.load(open(f'{CORPUS}/_align.json')).get(n)
        assert a and SEQ % a == 0, f'{n} is listed at {a} in _align.json: a window of {SEQ} must hold whole documents'
    return len(names)

assert os.path.exists(f'{CORPUS_DRIVE}/_sizes.json'), f'no cache at {CORPUS_DRIVE}'
!rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
print(check_shards(), 'shards on local disk, all at full size; this arm trains on', SHARDS)
assert os.path.exists(TOKENIZER), TOKENIZER
!du -sh {CORPUS}; df -h /content | tail -1
"""),
    code(r"""
# --- config, and a runner that tees the log to Drive ---
# this arm's two shards at ~5% each; every OTHER hypertoken shard in the cache (the other arm's, the villages
# without addresses, the memory's facts_text_far) at 0 -- size-proportional draws would otherwise give each ~0.1%
OTHERS = sorted(n for n in json.load(open(f'{CORPUS}/_sizes.json'))
                if n.startswith(('spatial', 'facts')) and n not in SHARDS)
MIX_JSON = json.dumps({**{n: 0 for n in OTHERS}, **MULTS})
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR=RUN_DIR, CB_NAME=NAME,
    CB_D='1024', CB_L='32', CB_HEADS='4', CB_WINDOW='512', CB_ATTN_EVERY='3', CB_INIT_FROM=START_CKPT,
    CB_S=str(SEQ), CB_TOKENS_PER_STEP=str(TOKENS_PER_STEP),
    CB_LR='3e-4', CB_BETA2='0.95', CB_WD='0.1',
    CB_WARMUP='100', CB_DECAY_FRAC='0.3', CB_HOURS=str(HOURS_PER_ARM), CB_SEED='3',
    CB_MIX_JSON=MIX_JSON, CB_DECAY_JSON=MIX_JSON,
    CB_EVAL='100', CB_GEN='500', CB_CKPT_MIN='30',
    CB_STABLE_END='1', CB_GRAD_CKPT='0',
)
if ARM == 'link':
    BASE.update(CB_SPATIAL='1', CB_SPATIAL_QUERY='placed', CB_ROUTE_LOSS='1',   # the link, a placed token asks about its
                CB_SPATIAL_GATE='0')                                           # own place, the learned query taught; the
                                                                               # link is ADDED to the 450M (o = 0, gate 0)
# EQUAL TOKENS: the link arm sizes its step count to HOURS_PER_ARM from its own measured speed; the text arm
# (no FFTs, no route loss: faster) takes the link arm's step count instead of its own hours, so both see the
# same tokens -- the comparison is about what the trunk is shown, not how fast the card ran it.
if ARM == 'text':
    LINK_M = f"{DRIVE}/runs/{NAME.replace('text', 'link')}/{NAME.replace('text', 'link')}_metrics.jsonl"
    sized = [json.loads(l) for l in open(LINK_M) if '"sized"' in l] if os.path.exists(LINK_M) else []
    assert sized, 'run the link arm first (its sized step count sets the text arm\'s)'
    BASE['CB_STEPS'] = str(sized[-1]['steps'])
    print(f"text arm: {sized[-1]['steps']:,} steps, the link arm's -- equal tokens")
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 2**30
BASE['CB_MICRO'] = ('16' if RUN == 1 else '32') if gb >= 90 else ('16' if gb >= 70 else '8')   # both arms of a run alike
smem = getattr(props, 'shared_memory_per_block_optin', 0) or (
    160 * 1024 if (props.major, props.minor) in ((8, 0), (9, 0), (10, 0)) else 0)
if smem < 114688:
    BASE['CB_NO_FLEX'] = '1'
print(f"{ARM} (run {RUN}): {props.name}, {gb:.0f} GB -> micro {BASE['CB_MICRO']} at S={SEQ}; {HOURS_PER_ARM} h; shards {MULTS}, "
      f"{OTHERS} x0"
      f"{'; link on, query=placed, route loss 1' if ARM == 'link' else '; no link'}")

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

**Rerun this same cell in every session** (an arm is ~2 h; a stopped cell saves a checkpoint first and the next
session resumes from the newest Drive slot). The first log lines should say `initialised from ...
base450m_cont_stable_end.pt` -- on the `link` arm `...; spatial link ADDED (o = 0: text exact at step 0; gate bias
+0.0)` -- and the epoch table should list this arm's two shards at ~1-4 epochs each and the others at 0.
"""),
    code(r"""
check_shards()                                     # a new runtime starts with an empty local disk
left_h = SESSION_H - 0.4 - (time.time() - SESSION_T0) / 3600
print(f'{left_h:.1f} h left in this session: hard stop after {left_h - 0.3:.1f} h')
run(dict(CB_WALL_H=f'{left_h - 0.3:.2f}'), 'train.log')
"""),
    md(r"""
### Readout: the questions answered (both arms, any session)

`exp_hypertoken_recall.py` scores every held-out question of the arm's two shards on its final checkpoint,
teacher-forced; then the gates. The `link` arm is probed twice: with its places (the read), and with the
questions' `here` tokens stripped of their place (the learned route alone: the trunk's own query, taught by the
route loss, must find the address).
"""),
    code(r"""
import matplotlib.pyplot as plt
def rows_of(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []
PROBE = {}
SUF = '' if RUN == 1 else f'_{RUN}'
for arm, strip in (('link', '0'), ('link', '1'), ('text', '0')):
    final = f'{DRIVE}/runs/spatial_{arm}{SUF}/spatial_{arm}{SUF}_final.pt'
    if not os.path.exists(final):
        print(arm, 'has no final yet'); continue
    shards = {'link': f'spatial_here,{FACTS[0]}', 'text': f'spatial_text_here,{FACTS[1]}'}[arm]
    key = arm + ('_route' if strip == '1' else '')
    out = f'{DRIVE}/runs/spatial_{arm}{SUF}/hypertoken_recall_{key}.json'
    env = dict(os.environ, CB_CKPT=final, CB_CORPUS=CORPUS, CUBBY_SPM=TOKENIZER, CB_SHARDS=shards, CB_DOCS='400',
               CB_BATCH='16', CB_STRIP=strip, CB_OUT=out)
    p = subprocess.Popen([sys.executable, '-u', 'validation/exp_hypertoken_recall.py'], cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:                          # streamed: subprocess.run's output does not show in Colab
        if 'dynamo' not in line:
            print(line, end='')
    assert p.wait() == 0, f'the probe failed on {arm}'
    PROBE[key] = json.load(open(out))['shards']
link, text, route = PROBE.get('link', {}), PROBE.get('text', {}), PROBE.get('link_route', {})
mix = {}
for arm in ('link', 'text'):
    v = [r for r in rows_of(f'{DRIVE}/runs/spatial_{arm}{SUF}/spatial_{arm}{SUF}_metrics.jsonl') if 'val' in r]
    mix[arm] = v[-1]['val'].get('_mix') if v else None
print('\nheld-out text mix:', mix)
checks = {}
if link and text:
    v_link, v_text = link['spatial_here']['exact'], text['spatial_text_here']['exact']
    f_link, f_text = link[FACTS[0]]['exact'], text[FACTS[1]]['exact']
    checks['1. H-S12 gate 1: villages -- link >= 0.90, text < 0.60'] = v_link >= 0.90 and v_text < 0.60
    checks['2. H-K10 gate 4: facts -- link >= 2x text and >= 0.50'] = f_link >= 2 * f_text and f_link >= 0.50
    if mix['link'] is not None and mix['text'] is not None:
        checks['3. H-S12 gate 2: text mix of link within 0.01 of text'] = mix['link'] <= mix['text'] + 0.01
    print(f"villages: link {v_link:.3f} (first token {link['spatial_here']['first_token']:.3f})  text {v_text:.3f}"
          + (f"  route alone {route['spatial_here']['exact']:.3f}" if route else ''))
    print(f"facts:    link {f_link:.3f} (first token {link[FACTS[0]]['first_token']:.3f})  text {f_text:.3f}"
          + (f"  route alone {route[FACTS[0]]['exact']:.3f}" if route else ''))
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'fail'}  {k}")
    print('  kill: gate 1 below 0.75, or gate 3 worse by 0.02')
"""),
    md(r"""
### Profile: where a step's time goes (optional, ~3 min, after both arms)

`validation/profile_base.py` builds this arm's exact model from the same settings, runs a few steps on random
tokens (random places on a third when the link is on) and prints the device time by kind -- matmuls, attention,
the 128k-vocabulary loss, the MinGRU scan, the link's FFTs, the optimizer, copies -- and the top kernels. The link
arm ran at 54.6k tok/s, 29.5% MFU here (2026-10-20); this says what holds it there, so the next runs get the
speed work in the right order.
"""),
    code(r"""
env = dict(os.environ, **BASE, CB_PROF_STEPS='4', CB_PROF_TRACE=f'{RUN_DIR}/profile')
p = subprocess.Popen([sys.executable, '-u', 'validation/profile_base.py'], cwd=REPO, env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open(f'{RUN_DIR}/profile.txt', 'w') as f:
    for line in p.stdout:
        if 'dynamo' in line or 'Graph break' in line:
            continue
        print(line, end=''); f.write(line)
p.wait()
"""),
    md(r"""
### After the run

- The `link` final serves through torch only until grilly2 carries the spatial link (`export_base.py` refuses a
  `spatial` meta).
- Record the result under H-S12 and H-K10 in `CUBBYLLM_HYPOTHESES.md`, with `train.log`, the metrics and the
  two `hypertoken_recall_*.json` copied into `validation/logs/`.
- Next on the same trunk: the clock (H-K4) and the fact map's phases as entity codes (H-K1 → H-K10).
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
