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

**The small-scale reads this rests on (2026-10-20, `validation/exp_k10_hypertoken.py`):** a 3.4M trunk, facts shown
once, questions beyond the window -- 0.94 with the address on the question, 0.82 for the learned route taught by
the route loss, 0.00 for text, 0.03 for attention over the whole document, 0.00 with a mismatched address.

**Before you run**
1. The four shards are on Drive under `token_cache_cont/` (`spatial_here.u32` + `.pos.i16`, `spatial_text_here.u32`,
   `facts.u32` + `.pos.i16`, `facts_text.u32`, and the three index files) -- built 2026-10-20.
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
SEQ           = 1024                                # the shards' documents are 512 tokens: two whole documents a window
TOKENS_PER_STEP = 131_072                           # the optimizer step; BOTH ARMS MUST MATCH (the readout checks it)
SHARD_MULT    = 96                                  # ~5% of the tokens each for the two 10M-token shards against the 17.7B
                                                    # cache (size x multiplier draws): ~35M tokens = 3.5 epochs of each at 2 h
DRIVE         = '/content/drive/MyDrive/cubbyllm'
START_CKPT    = f'{DRIVE}/runs/base450m_cont/base450m_cont_stable_end.pt'   # the continued 450M, before its decay
CORPUS_DRIVE  = f'{DRIVE}/token_cache_cont'
CORPUS        = '/content/token_cache_cont'                  # LOCAL SSD (Drive-FUSE reads are ~30x slower)
TOKENIZER     = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
RUN_DIR       = f'{DRIVE}/runs/spatial_{ARM}'
SHARDS        = {'link': ('spatial_here', 'facts'), 'text': ('spatial_text_here', 'facts_text')}[ARM]
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
        assert json.load(open(f'{CORPUS}/_align.json')).get(n) == 512, f'{n} is not listed at 512 in _align.json'
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
MIX_JSON = json.dumps({**{n: 0 for n in OTHERS}, **{n: SHARD_MULT for n in SHARDS}})
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR=RUN_DIR, CB_NAME=f'spatial_{ARM}',
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
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 2**30
BASE['CB_MICRO'] = '16' if gb >= 70 else '8'
smem = getattr(props, 'shared_memory_per_block_optin', 0) or (
    160 * 1024 if (props.major, props.minor) in ((8, 0), (9, 0), (10, 0)) else 0)
if smem < 114688:
    BASE['CB_NO_FLEX'] = '1'
print(f"{ARM}: {props.name}, {gb:.0f} GB -> micro {BASE['CB_MICRO']} at S={SEQ}; {HOURS_PER_ARM} h; shards {SHARDS} x{SHARD_MULT}, "
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
for arm, strip in (('link', '0'), ('link', '1'), ('text', '0')):
    final = f'{DRIVE}/runs/spatial_{arm}/spatial_{arm}_final.pt'
    if not os.path.exists(final):
        print(arm, 'has no final yet'); continue
    shards = {'link': 'spatial_here,facts', 'text': 'spatial_text_here,facts_text'}[arm]
    key = arm + ('_route' if strip == '1' else '')
    out = f'{DRIVE}/runs/spatial_{arm}/hypertoken_recall_{key}.json'
    env = dict(os.environ, CB_CKPT=final, CB_CORPUS=CORPUS, CUBBY_SPM=TOKENIZER, CB_SHARDS=shards, CB_DOCS='400',
               CB_BATCH='16', CB_STRIP=strip, CB_OUT=out)
    subprocess.run([sys.executable, 'validation/exp_hypertoken_recall.py'], cwd=REPO, env=env, check=True)
    PROBE[key] = json.load(open(out))['shards']
link, text, route = PROBE.get('link', {}), PROBE.get('text', {}), PROBE.get('link_route', {})
mix = {}
for arm in ('link', 'text'):
    v = [r for r in rows_of(f'{DRIVE}/runs/spatial_{arm}/spatial_{arm}_metrics.jsonl') if 'val' in r]
    mix[arm] = v[-1]['val'].get('_mix') if v else None
print('\nheld-out text mix:', mix)
checks = {}
if link and text:
    v_link, v_text = link['spatial_here']['exact'], text['spatial_text_here']['exact']
    f_link, f_text = link['facts']['exact'], text['facts_text']['exact']
    checks['1. H-S12 gate 1: villages -- link >= 0.90, text < 0.60'] = v_link >= 0.90 and v_text < 0.60
    checks['2. H-K10 gate 4: facts -- link >= 2x text and >= 0.50'] = f_link >= 2 * f_text and f_link >= 0.50
    if mix['link'] is not None and mix['text'] is not None:
        checks['3. H-S12 gate 2: text mix of link within 0.01 of text'] = mix['link'] <= mix['text'] + 0.01
    print(f"villages: link {v_link:.3f} (first token {link['spatial_here']['first_token']:.3f})  text {v_text:.3f}"
          + (f"  route alone {route['spatial_here']['exact']:.3f}" if route else ''))
    print(f"facts:    link {f_link:.3f} (first token {link['facts']['first_token']:.3f})  text {f_text:.3f}"
          + (f"  route alone {route['facts']['exact']:.3f}" if route else ''))
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'fail'}  {k}")
    print('  kill: gate 1 below 0.75, or gate 3 worse by 0.02')
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
