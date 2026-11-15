"""Generate view_split_m.ipynb and view_split_w.ipynb: order 0073, step 5, the reader's two adapters (arm M: the
description folded; arm W: as said). One recipe, round 2's (`view_reader2.ipynb`), and nothing else changed between
the arms: rank 16 / alpha 32, five epochs, lr 2e-4, batch 32, max length 1,280, seed 0, key tokens weighted 4 (here
each check's join tag and description, `validation/key_tokens.py`). The two notebooks differ in the set they read
and the folder they write.

Each refuses to train unless the set was built from rows whose SHA-256 is committed in the code it cloned
(`validation/logs/long_rows_0073_2026-11-15_manifest.json`): the order's test sets are hashed before any training.

The Colab runs are Nick's. Re-run after editing:
    python notebooks/_build_view_split.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "cowork/0073"          # the branch the notebook clones; "master" once the order is merged
ARMS = {"m": "folded: the description has no `not` (\"do not include any\" is `none`)",
        "w": "as said: the description writes `not` where the words negate (\"do not include any\" is `not any`)"}


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


def cells(arm: str) -> list:
    other = "w" if arm == "m" else "m"
    return [
        md(rf"""
# 0073: the instruction reader writes descriptions, arm {arm.upper()}

Round 2's reader wrote a split and then a CubeLang program, and it failed its false-pass line (170 / 1,709 on test).
This reader writes the split alone: one line a sentence or clause, and after each check a one-line description of
what it checks. The host compiles the program. Two adapters are trained on the same rows with the same recipe and
read side by side; they differ in one thing, how a negation in the words is spelled:

- **arm {arm.upper()}** (this notebook): {ARMS[arm]};
- arm {other.upper()} (`view_split_{other}.ipynb`): {ARMS[other]}.

The recipe is round 2's, unchanged: rank 16 / alpha 32, five epochs, lr 2e-4, batch 32, key tokens weighted 4. The
key tokens are now each check's join tag and description. The set is `rules/view_split_sft_{arm}.jsonl`: 6,000 train
records (3,000 long instructions and 1,500 minimal pairs) and 300 held (dev). Test and test_long are in no set.

**Runtime:** an A100 or the G4. The answers are far shorter than round 2's, so it should take less than round 2's
45 minutes. It reads, from Drive: `runs/base450m_cont/base450m_cont_final.pt`, the token cache's tokenizer and the
set; it writes `rules/cubby450m_view_split_{arm}/` (`talk_lora.*`). Training only: the read is its own notebook.
**Disconnect when it prints `trained:`.**
"""),
        code(rf"""
import json, os, signal, subprocess, sys, time
BRANCH = '{BRANCH}'
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q -b {{BRANCH}} https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git fetch -q && git checkout -q {{BRANCH}} && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE, ARM = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm', '{arm}'
CKPT      = f'{{DRIVE}}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{{DRIVE}}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
DATA      = f'{{DRIVE}}/rules/view_split_sft_{{ARM}}.jsonl'
OUT       = f'{{DRIVE}}/rules/cubby450m_view_split_{{ARM}}'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], man['by_family'])
assert man['order'] == '0073' and man['arm'] == ARM, 'this is not arm ' + ARM + "'s set"
assert man['by_split'] == {{'train': 6000, 'held': 300}} and man['rows of test or test_long'] == 0, man['by_split']
assert man['answers that read back strictly, also with own_text'] == man['records'], 'answers the grammar would refuse'
assert man['records_over_max_len'] == {{}}, f"records longer than MAX_LEN would be skipped: {{man['records_over_max_len']}}"
assert {{'IFEval', 'IFBench'}} <= set(man['benchmarks_checked']), 'the benchmark check did not run'
rows = json.load(open(f'{{REPO}}/validation/logs/long_rows_0073_2026-11-15_manifest.json'))['sha256']
assert man['rows_sha256'] == rows, "the set's rows are not the rows whose hashes are committed: do not train"
print('rows as committed; test', rows['the lines of rows.jsonl by split']['test'][:12], 'test_long',
      rows['the lines of rows.jsonl by split']['test_long'][:12])
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
        md(r"""
## The settings

Rank 16, alpha 32 (every projection of every layer, the served adapters' layout), lr 2e-4 with 50 warmup steps then
cosine, batch 32, max length 1,280, five epochs, seed 0, key tokens weighted 4x. On a card under 70 GB a batch is cut
into chunks of at most 4,096 padded tokens (`--micro-tokens`: the same gradient). The same cell in both notebooks.
"""),
        code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN, KEY_WEIGHT = 5, 32, 2e-4, 16, 32, 1280, 4.0
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = f'_view_split_{ARM}'
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
        md(rf"""
## Done

**Disconnect the runtime.** Train the other arm with `view_split_{other}.ipynb` (the same settings). The read comes
after both: dev first, with and without the rule that holds a description to its line's own text (point 7 of the
order), then test and test_long once for each arm, and round 2's adapter beside them (arm P).
"""),
    ]


for arm in ARMS:
    nb = {"cells": cells(arm),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"view_split_{arm}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
