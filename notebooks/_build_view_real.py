"""Generate view_real_r.ipynb, view_real_read_dev.ipynb and view_real_read_test.ipynb: order 0077, build 4 (from
`_build_view_long.py`, order 0074's).

- **view_real_r**: arm R, 0074's arm L recipe unchanged, trained on `rules/view_real_sft_r.jsonl` (0074's 8,000 train
  records and 2,000 real-prompt records, OASST1 only; 300 held, dev_real). It refuses to train unless the set was
  built from rows whose SHA-256 is committed in the code it cloned (`ROWS_MANIFEST`): the held-out sets are hashed
  before any training.
- **view_real_read_dev / _test**: the answers only, arm R and arm L (0074's adapter), one decoder
  (`validation/exp_view_split_torch.py`, free values). Colab gets the held-out rows' ids, splits and instructions
  and nothing else (`rules/long_rows_0077/read_rows.jsonl`). dev_real first; then test_real and test_dolly for both
  arms, and 0074's test and test_long for arm R (arm L's answers there are 0074's own read, not read again), once,
  only after dev's reports are committed.

The Colab runs are Nick's. Re-run after editing:
    python notebooks/_build_view_real.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "master"                # the branch the notebooks clone
ROWS_MANIFEST = "validation/logs/long_rows_0077_2026-11-18_manifest.json"
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CLONE = r"""
import hashlib, json, os, signal, subprocess, sys, time
BRANCH = '__BRANCH__'
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q -b {BRANCH} https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git fetch -q && git checkout -q {BRANCH} && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy regex
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
""".replace("__BRANCH__", BRANCH)


TRAIN = [
    md(r"""
# 0077: the description reader on real prompts, arm R

0074's reader (arm L: the split, a folded description after each check, free values) learned instructions of 2 to 8
parts, all worded from our templates. Arm R is the same recipe with real prompts in the mix: 2,000 rows whose task is a
person's own prompt (OASST1, English and French), with 2 to 8 checks put before it, after it or after its first
sentence. It is read beside arm L on prompts neither saw (OASST1 and Dolly-15k).

The set is `rules/view_real_sft_r.jsonl`: 10,000 train records (0074's 8,000 and the 2,000 real-prompt ones) and 300
held (dev_real). No Dolly text and no test row is in it. About 1,560 steps.

**Runtime:** an A100 or the G4. It reads, from Drive: the base, the tokenizer and the set; it writes
`rules/cubby450m_view_real_r/` (`talk_lora.*`). Training only: the read is its own notebook.
**Disconnect when it prints `trained:`.**
"""),
    code(CLONE + r"""
DATA = f'{DRIVE}/rules/view_real_sft_r.jsonl'
OUT  = f'{DRIVE}/rules/cubby450m_view_real_r'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], man['by_family'])
assert man['order'] == '0077' and man['arm'] == 'm', "this is not order 0077's set in arm M's target"
assert man['by_split'] == {'train': 10000, 'held': 300}, man['by_split']
assert man['dolly records'] == 0 and man['rows of test_real or test_dolly'] == 0 and man['rows of test or test_long'] == 0
assert man['answers that read back strictly, also with own_text'] == man['records'], 'answers the grammar would refuse'
assert man['records_over_max_len'] == {}, f"records longer than MAX_LEN would be skipped: {man['records_over_max_len']}"
assert {'IFEval', 'IFBench'} <= set(man['benchmarks_checked']), 'the benchmark check did not run'
assert man["0074's train records, the same as 0074's set"] is True, "0074's records were not checked against its set"
rows = json.load(open(f'{REPO}/__ROWS_MANIFEST__'))['sha256']
assert man['rows_sha256']['long_rows_0077'] == rows, "the set's rows are not the rows whose hashes are committed: do not train"
print('rows as committed:', {s: h[:12] for s, h in rows['the lines of rows.jsonl by set'].items()})
!nvidia-smi --query-gpu=name,memory.total --format=csv
""".replace("__ROWS_MANIFEST__", ROWS_MANIFEST)),
    md(r"""
## The settings

0074's, unchanged: rank 16, alpha 32 (every projection of every layer), lr 2e-4 with 50 warmup steps then cosine,
batch 32, max length 1,280, five epochs, seed 0, key tokens weighted 4x. On a card under 70 GB a batch is cut into
chunks of at most 4,096 padded tokens (`--micro-tokens`: the same gradient).
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN, KEY_WEIGHT = 5, 32, 2e-4, 16, 32, 1280, 4.0
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_view_real_r'
import torch
MICRO = 0 if torch.cuda.get_device_properties(0).total_memory >= 70e9 else 4096
print('micro-batch tokens:', MICRO or 'off (the whole batch at once)')
cmd = [sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', '0', '--tag', tag,
       '--micro-tokens', str(MICRO), '--key-weight', str(KEY_WEIGHT)]
p = None
try:
    with open(f'{OUT}/train.log', 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        assert p.wait() == 0, f'the trainer failed ({p.returncode})'
finally:
    if p and p.poll() is None:
        p.send_signal(signal.SIGINT)
subprocess.run(f'cp {REPO}/validation/logs/train_talk_torch{tag}.* {OUT}/ 2>/dev/null', shell=True)
json.dump({'prompts': 'OASST1 (Apache-2.0)', 'templates': "this repo's own (Apache-2.0)"},
          open(f'{OUT}/row_licences.json', 'w'))
open(f'{OUT}/DONE', 'w').write(time.strftime('%F %T') + '\n')
print('trained:', OUT)
"""),
    md(r"""
## Done

**Disconnect the runtime.** Next: `view_real_read_dev.ipynb` (arm R and arm L on dev_real, for looking).
"""),
]

READ_SETUP = CLONE + r"""
PHASE = '__PHASE__'
ROWS     = f'{DRIVE}/rules/long_rows_0077/read_rows.jsonl'
ROWS74   = f'{DRIVE}/rules/long_rows_0074/read_rows.jsonl'
ADAPTERS = {'r': f'{DRIVE}/rules/cubby450m_view_real_r', 'l': f'{DRIVE}/rules/cubby450m_view_long_l'}
READS    = f'{DRIVE}/rules/reads/0077'
for p in [CKPT, TOKENIZER, ROWS, ROWS74] + [f'{d}/{x}' for d in ADAPTERS.values() for x in ('talk_lora.safetensors', 'talk_lora.json', 'DONE')]:
    assert os.path.exists(p), p
mans = {}
for rows_p, name in ((ROWS, '0077'), (ROWS74, '0074')):
    mans[name] = json.load(open(f'{REPO}/validation/logs/long_rows_{name}_read_rows.manifest.json'))
    assert hashlib.sha256(open(rows_p, 'rb').read()).hexdigest() == mans[name]['sha256'], f'{rows_p} is not the committed file'
    keys = {k for ln in open(rows_p, encoding='utf-8') for k in json.loads(ln)}
    assert keys == {'id', 'split', 'instruction'}, f'the rows file holds more than ids and instructions: {keys}'
    print(name, mans[name]['rows'], 'rows', mans[name]['by_split'])
os.makedirs(READS, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

PHASES = {
    "dev": (r"""
# 0077: the read on dev_real (arm R and arm L)

The answers only, free values, one decoder. Each of dev_real's 300 instructions (an OASST1 prompt with 2 to 8 checks
inside it) is read whole, its answer held token by token to the grammar of its own instruction. Dev is for looking;
nothing is decided on it.

It writes `rules/reads/0077/view_real_{r,l}_dev.jsonl` (resumable). **Disconnect when it prints `read: dev`.**
""", r"""
PASSES = [(arm, ROWS, 'dev_real', f'{READS}/view_real_{arm}_dev.jsonl', mans['0077']) for arm in 'rl']
"""),
    "test": (r"""
# 0077: the read on test_real and test_dolly (arms R and L), and 0074's test and test_long (arm R), once

The answers only, free values, one decoder, each set read **once**: test_real (300 OASST1 prompts) and test_dolly
(300 Dolly-15k prompts, a source neither arm saw) for both arms; 0074's test and test_long for arm R (arm L's answers
there are 0074's read, `rules/reads/0074/view_long_l_test.jsonl`). This notebook refuses to run until dev's two
reports are committed in the code it clones, and it does not read again a file that is already whole. Nobody opens
these answers by eye: they are scored at home, counts only.

It writes `rules/reads/0077/view_real_{r,l}_test.jsonl` and `view_real_r_test0074.jsonl`.
**Disconnect when it prints `read: test`.**
""", r"""
for arm in 'rl':
    assert os.path.exists(f'{REPO}/validation/logs/view_real_dev_{arm}.json'), 'dev is read and scored first, and its reports committed'
PASSES = [(arm, ROWS, 'test_real,test_dolly', f'{READS}/view_real_{arm}_test.jsonl', mans['0077']) for arm in 'rl']
PASSES.append(('r', ROWS74, 'test,test_long', f'{READS}/view_real_r_test0074.jsonl', mans['0074']))
"""),
}

READ = r"""
os.environ['CB_NO_FLEX'] = '1'
N = 300
for arm, rows_p, splits, OUT, man in PASSES:
    want = sum(min(N, man['by_split'][s]) for s in splits.split(','))
    have = [json.loads(ln) for ln in open(OUT, encoding='utf-8')] if os.path.exists(OUT) else []
    assert not any(d.get('own_text') for d in have), f'{OUT} was read with the own-text rule: move it aside'
    if len({d['id'] for d in have}) >= want:
        print(f'--- arm {arm}: {OUT} is whole ({len(have)} answers): not read again')
        continue
    print(f'--- arm {arm}: {splits} -> {OUT}')
    cmd = [sys.executable, '-u', 'validation/exp_view_split_torch.py', '--arm', 'm', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--adapter', ADAPTERS[arm], '--rows', rows_p, '--splits', splits, '--n', str(N), '--batch', '32', '--out', OUT]
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'the read failed ({p.returncode})'
    !wc -l "{OUT}"
print('read:', PHASE, '(all passes read)')
"""

DONE = {"dev": "## Done\n\n**Disconnect the runtime** (on `read: dev`). At home dev_real is scored on the VM for both "
               "arms and the two reports are committed; then `view_real_read_test.ipynb` reads the held-out sets once.",
        "test": "## Done\n\n**Disconnect the runtime** (on `read: test`). At home the answers are scored on the VM, counts "
                "only, and set against the order's lines; line 5 says which reader is kept."}


def notebook(cells: list, name: str) -> None:
    out = HERE / name
    out.write_text(json.dumps({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}, indent=1,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)


notebook(TRAIN, "view_real_r.ipynb")
for phase, (intro, passes) in PHASES.items():
    notebook([md(intro), code(READ_SETUP.replace("__PHASE__", phase)), code(passes + READ), md(DONE[phase])],
             f"view_real_read_{phase}.ipynb")
