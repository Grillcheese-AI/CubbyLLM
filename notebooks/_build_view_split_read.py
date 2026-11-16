"""Generate view_split_read_dev.ipynb and view_split_read_test.ipynb: order 0073, steps 6 and 7, the read on a Colab
card. The answers only (`validation/exp_view_split_torch.py`, one decoder for every arm); they are scored at home on
the shared VM. Colab is given the held-out rows' ids, splits and instructions and nothing else
(`rules/long_rows_0073/read_rows.jsonl`): no target, no reply.

Two notebooks, because test is read once and only after dev has decided point 7:
- **dev**: arm M and arm W, each with and without the rule that holds a description to its line's own text, and
  arm P (round 2's adapter, its own form and grammar);
- **test**: test and test_long, once for each arm, in the variant dev kept. It refuses to run until that choice is
  committed in the code it cloned (`validation/logs/view_split_point7.json`).

The Colab runs are Nick's. Re-run after editing:
    python notebooks/_build_view_split_read.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "cowork/0073"          # the branch the notebooks clone; "master" once the order is merged
ARM_P = "cubby450m_view_reader_v2"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


INTRO = {
    "dev": r"""
# 0073: the reader's read on dev (arms M, W and P)

The answers only. Each instruction of dev (300) is read whole, its answer held token by token to the grammar of its
own instruction. Five passes:

- **arm M** and **arm W**, each twice: free, and with the rule that a description may only use the numbers, strings
  and views its line's own text gives (point 7 of the order). Dev decides which variant each arm keeps.
- **arm P**: round 2's adapter (`cubby450m_view_reader_v2`), with its own form and grammar: the baseline.

Dev is for looking; test is not read here. The answers are far shorter than round 2's programs, so the four passes
of M and W are quick; arm P's pass is round 2's read (about ten minutes for 300 rows on an A100).

It reads, from Drive: the base, the tokenizer, the three adapters and `rules/long_rows_0073/read_rows.jsonl`; it
writes `rules/reads/0073/*_dev.jsonl` (resumable: a rerun keeps the rows already answered).
**Disconnect when it prints `read: dev`.**
""",
    "test": r"""
# 0073: the reader's read on test and test_long (arms M, W and P), once

The answers only. Test (300) and test_long (300) are read **once** for each arm: arm M and arm W in the variant dev
kept, arm P (round 2's adapter) beside them. This notebook refuses to run until that choice is committed in the code
it clones (`validation/logs/view_split_point7.json`), and it does not read again a file that is already whole.

It reads, from Drive: the base, the tokenizer, the three adapters and `rules/long_rows_0073/read_rows.jsonl`; it
writes `rules/reads/0073/*_test.jsonl`. Nobody opens these answers by eye: they are scored at home, counts only.
**Disconnect when it prints `read: test`.**
""",
}

SETUP = r"""
import hashlib, json, os, subprocess, sys
BRANCH, PHASE = '__BRANCH__', '__PHASE__'
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q -b {BRANCH} https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git fetch -q && git checkout -q {BRANCH} && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy regex
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
ROWS      = f'{DRIVE}/rules/long_rows_0073/read_rows.jsonl'
ADAPTERS  = {'m': f'{DRIVE}/rules/cubby450m_view_split_m', 'w': f'{DRIVE}/rules/cubby450m_view_split_w',
             'p': f'{DRIVE}/rules/__ARM_P__'}
READS     = f'{DRIVE}/rules/reads/0073'
for p in [CKPT, TOKENIZER, ROWS] + [f'{d}/talk_lora.{x}' for d in ADAPTERS.values() for x in ('safetensors', 'json')]:
    assert os.path.exists(p), p
for arm in 'mw':
    assert os.path.exists(f"{ADAPTERS[arm]}/DONE"), f'arm {arm} is not trained to the end'
man = json.load(open(f'{REPO}/validation/logs/long_rows_0073_read_rows.manifest.json'))
assert hashlib.sha256(open(ROWS, 'rb').read()).hexdigest() == man['sha256'], 'read_rows.jsonl is not the committed file'
keys = {k for ln in open(ROWS, encoding='utf-8') for k in json.loads(ln)}
assert keys == {'id', 'split', 'instruction'}, f'the rows file holds more than ids and instructions: {keys}'
os.makedirs(READS, exist_ok=True)
print(man['rows'], 'rows', man['by_split'])
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

PASSES = {
    "dev": r"""
SPLITS, N = 'dev', 300
PASSES = [('m', False), ('m', True), ('w', False), ('w', True), ('p', False)]     # (arm, the own-text rule)
def out_of(arm, own):
    return f"{READS}/view_reader_v2_dev.jsonl" if arm == 'p' else f"{READS}/view_split_{arm}_{'own' if own else 'free'}_dev.jsonl"
""",
    "test": r"""
SPLITS, N = 'test,test_long', 300
P7 = f'{REPO}/validation/logs/view_split_point7.json'
assert os.path.exists(P7), 'point 7 is not decided: dev is read and scored first, and the choice committed'
kept = json.load(open(P7))
assert kept['order'] == '0073' and set(kept['kept']) == {'m', 'w'}, kept
PASSES = [('m', bool(kept['kept']['m']['own_text'])), ('w', bool(kept['kept']['w']['own_text'])), ('p', False)]
print('the variant dev kept:', {arm: ('own text' if own else 'free') for arm, own in PASSES[:2]})
def out_of(arm, own):
    return f"{READS}/view_reader_v2_test.jsonl" if arm == 'p' else f"{READS}/view_split_{arm}_test.jsonl"
""",
}

READ = r"""
os.environ['CB_NO_FLEX'] = '1'
want = sum(min(N, man['by_split'][s]) for s in SPLITS.split(','))
for arm, own in PASSES:
    OUT = out_of(arm, own)
    have = [json.loads(ln) for ln in open(OUT, encoding='utf-8')] if os.path.exists(OUT) else []
    assert all(bool(d.get('own_text')) == own for d in have), f'{OUT} was read in another variant: move it aside'
    if len({d['id'] for d in have}) >= want:
        print(f'--- arm {arm}: {OUT} is whole ({len(have)} answers): not read again')
        continue
    print(f"--- arm {arm}{' (own text)' if own else ''}: {SPLITS} -> {OUT}")
    cmd = [sys.executable, '-u', 'validation/exp_view_split_torch.py', '--arm', arm, '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--adapter', ADAPTERS[arm], '--rows', ROWS, '--splits', SPLITS, '--n', str(N), '--batch', '32', '--out', OUT]
    p = subprocess.Popen(cmd + (['--own-text'] if own else []), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'the read failed ({p.returncode})'
    !wc -l "{OUT}"
print('read:', PHASE)
"""

DONE = {
    "dev": r"""
## Done

**Disconnect the runtime.** At home the answers are scored on the VM (`validation/exp_view_split_score.py`; arm P
with round 2's scorer), dev's counts decide point 7 for each arm, and that choice is committed. Only then does
`view_split_read_test.ipynb` read test and test_long.
""",
    "test": r"""
## Done

**Disconnect the runtime.** At home the answers are scored on the VM, counts only, and set against the order's
lines; line 9 says which arm is kept.
""",
}


def cells(phase: str) -> list:
    setup = SETUP.replace("__BRANCH__", BRANCH).replace("__PHASE__", phase).replace("__ARM_P__", ARM_P)
    return [md(INTRO[phase]), code(setup), code(PASSES[phase] + READ), md(DONE[phase])]


for phase in ("dev", "test"):
    nb = {"cells": cells(phase),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"view_split_read_{phase}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
