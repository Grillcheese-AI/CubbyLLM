"""Generate view_long_l.ipynb, view_long_read_dev.ipynb and view_long_read_test.ipynb: order 0074, steps 3 and 4.

- **view_long_l**: arm L, 0073's arm M (the description folded, free values) trained with 0073's recipe unchanged
  on `rules/view_long_sft_l.jsonl` (8,000 train records of 2 to 8 parts, 300 held). It refuses to train unless the
  set was built from rows whose SHA-256 is committed in the code it cloned
  (`validation/logs/long_rows_0074_2026-11-16_manifest.json`): the held-out sets are hashed before any training.
- **view_long_read_dev / _test**: the answers only, arm L and arm K (0073's kept adapter) with one decoder
  (`validation/exp_view_split_torch.py`, free values). Colab gets the held-out rows' ids, splits and instructions
  and nothing else (`rules/long_rows_0074/read_rows.jsonl`). Test, test_long and test_longer are read once, and
  only after dev's reports are committed.

The Colab runs are Nick's. Re-run after editing:
    python notebooks/_build_view_long.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "cowork/0074"          # the branch the notebooks clone; "master" once the order is merged
ROWS_MANIFEST = "validation/logs/long_rows_0074_2026-11-16_manifest.json"
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
# 0074: the description reader learns long instructions, arm L

0073's reader writes the split of an instruction and, after each check, a one-line description; the host compiles
the program. Trained on up to 4 parts, it passed every line on 2 to 4 parts and fell on 5 to 8 (test_long): parts
right at positions 5 to 8 were 559 / 754. This notebook trains the same reader (arm M: folded, free values) with the
same recipe on rows of 2 to 8 parts, built again from a new seed. It is read beside 0073's adapter (arm K) on the same
fresh rows.

The set is `rules/view_long_sft_l.jsonl`: 8,000 train records (3,000 instructions of 2 to 4 parts, 2,000 of 5 to 8,
1,500 minimal pairs) and 300 held (dev). Test, test_long and test_longer are in no set. About 1,250 steps.

**Runtime:** an A100 or the G4. It reads, from Drive: the base, the tokenizer and the set; it writes
`rules/cubby450m_view_long_l/` (`talk_lora.*`). Training only: the read is its own notebook.
**Disconnect when it prints `trained:`.**
"""),
    code(CLONE + r"""
DATA = f'{DRIVE}/rules/view_long_sft_l.jsonl'
OUT  = f'{DRIVE}/rules/cubby450m_view_long_l'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], man['by_family'])
assert man['order'] == '0074' and man['arm'] == 'm', "this is not order 0074's set in arm M's target"
assert man['by_split'] == {'train': 8000, 'held': 300} and man['rows of test or test_long'] == 0, man['by_split']
assert man['answers that read back strictly, also with own_text'] == man['records'], 'answers the grammar would refuse'
assert man['records_over_max_len'] == {}, f"records longer than MAX_LEN would be skipped: {man['records_over_max_len']}"
assert {'IFEval', 'IFBench'} <= set(man['benchmarks_checked']), 'the benchmark check did not run'
rows = json.load(open(f'{REPO}/__ROWS_MANIFEST__'))['sha256']
assert man['rows_sha256'] == rows, "the set's rows are not the rows whose hashes are committed: do not train"
print('rows as committed:', {s: h[:12] for s, h in rows['the lines of rows.jsonl by split'].items()})
!nvidia-smi --query-gpu=name,memory.total --format=csv
""".replace("__ROWS_MANIFEST__", ROWS_MANIFEST)),
    md(r"""
## The settings

0073's, unchanged: rank 16, alpha 32 (every projection of every layer), lr 2e-4 with 50 warmup steps then cosine,
batch 32, max length 1,280, five epochs, seed 0, key tokens weighted 4x. On a card under 70 GB a batch is cut into
chunks of at most 4,096 padded tokens (`--micro-tokens`: the same gradient).
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN, KEY_WEIGHT = 5, 32, 2e-4, 16, 32, 1280, 4.0
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_view_long_l'
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
open(f'{OUT}/DONE', 'w').write(time.strftime('%F %T') + '\n')
print('trained:', OUT)
"""),
    md(r"""
## Done

**Disconnect the runtime.** Next: `view_long_read_dev.ipynb` (arm L and arm K on dev, for looking).
"""),
]

READ_SETUP = CLONE + r"""
PHASE = '__PHASE__'
ROWS     = f'{DRIVE}/rules/long_rows_0074/read_rows.jsonl'
ADAPTERS = {'l': f'{DRIVE}/rules/cubby450m_view_long_l', 'k': f'{DRIVE}/rules/cubby450m_view_split_m'}
READS    = f'{DRIVE}/rules/reads/0074'
for p in [CKPT, TOKENIZER, ROWS] + [f'{d}/{x}' for d in ADAPTERS.values() for x in ('talk_lora.safetensors', 'talk_lora.json', 'DONE')]:
    assert os.path.exists(p), p
man = json.load(open(f'{REPO}/validation/logs/long_rows_0074_read_rows.manifest.json'))
assert hashlib.sha256(open(ROWS, 'rb').read()).hexdigest() == man['sha256'], 'read_rows.jsonl is not the committed file'
keys = {k for ln in open(ROWS, encoding='utf-8') for k in json.loads(ln)}
assert keys == {'id', 'split', 'instruction'}, f'the rows file holds more than ids and instructions: {keys}'
os.makedirs(READS, exist_ok=True)
print(man['rows'], 'rows', man['by_split'])
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

PHASES = {
    "dev": (r"""
# 0074: the read on dev (arm L and arm K)

The answers only, free values, one decoder. Each instruction of dev (300: 150 of 2 to 4 parts, 150 of 5 to 8) is read
whole, its answer held token by token to the grammar of its own instruction. Arm L is this order's adapter; arm K is
0073's kept one (`cubby450m_view_split_m`), read on the same rows. Dev is for looking; nothing is decided on it.

It writes `rules/reads/0074/view_long_{l,k}_dev.jsonl` (resumable). **Disconnect when it prints `read: dev`.**
""", r"""
SPLITS, N = 'dev', 300
def out_of(arm):
    return f'{READS}/view_long_{arm}_dev.jsonl'
"""),
    "test": (r"""
# 0074: the read on test, test_long and test_longer (arm L and arm K), once

The answers only, free values, one decoder: test (300, 2 to 4 parts), test_long (300, 5 to 8) and test_longer (300,
9 to 12) are read **once** for each arm. This notebook refuses to run until dev's two reports are committed in the
code it clones, and it does not read again a file that is already whole. Nobody opens these answers by eye: they
are scored at home, counts only.

It writes `rules/reads/0074/view_long_{l,k}_test.jsonl`. **Disconnect when it prints `read: test`.**
""", r"""
SPLITS, N = 'test,test_long,test_longer', 300
for arm in 'lk':
    assert os.path.exists(f'{REPO}/validation/logs/view_long_dev_{arm}.json'), 'dev is read and scored first, and its reports committed'
def out_of(arm):
    return f'{READS}/view_long_{arm}_test.jsonl'
"""),
}

READ = r"""
os.environ['CB_NO_FLEX'] = '1'
want = sum(min(N, man['by_split'][s]) for s in SPLITS.split(','))
for arm in 'lk':
    OUT = out_of(arm)
    have = [json.loads(ln) for ln in open(OUT, encoding='utf-8')] if os.path.exists(OUT) else []
    assert not any(d.get('own_text') for d in have), f'{OUT} was read with the own-text rule: move it aside'
    if len({d['id'] for d in have}) >= want:
        print(f'--- arm {arm}: {OUT} is whole ({len(have)} answers): not read again')
        continue
    print(f'--- arm {arm}: {SPLITS} -> {OUT}')
    cmd = [sys.executable, '-u', 'validation/exp_view_split_torch.py', '--arm', 'm', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--adapter', ADAPTERS[arm], '--rows', ROWS, '--splits', SPLITS, '--n', str(N), '--batch', '32', '--out', OUT]
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'the read failed ({p.returncode})'
    !wc -l "{OUT}"
print('read:', PHASE)
"""

DONE = {"dev": "## Done\n\n**Disconnect the runtime.** At home dev is scored on the VM for both arms and the two "
               "reports are committed; then `view_long_read_test.ipynb` reads the held-out sets once.",
        "test": "## Done\n\n**Disconnect the runtime.** At home the answers are scored on the VM, counts only, and set "
                "against the order's lines (`validation/exp_view_long_compare.py`); line 9 says which reader is kept."}


def notebook(cells: list, name: str) -> None:
    out = HERE / name
    out.write_text(json.dumps({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}, indent=1,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)


notebook(TRAIN, "view_long_l.ipynb")
for phase, (intro, passes) in PHASES.items():
    notebook([md(intro), code(READ_SETUP.replace("__PHASE__", phase)), code(passes + READ), md(DONE[phase])],
             f"view_long_read_{phase}.ipynb")
