"""Generate view_reader2.ipynb: order 0067, the second reader's adapter, round 2 (Nick, 2026-11-13: "can you create the
notebook so we can enhance training"; his picks: weight the key tokens, polarity pairs, longer and bigger, inverted
phrasing). Round 1 (`view_reader.ipynb`) learned the split (held answer loss 0.050) but not the program's meaning: on
the held-out reads a quarter of parts had a comparison reversed and many named a number or a view the instruction does
not give. Round 2 changes three things, nothing else:
- the data: round 1's set plus `build_pair_rows.py`'s polarity pairs (each long instruction beside its twin with one
  part said the other way; a share of checks said goal first or in the passive), `rules/view_reader_sft_v2.jsonl`;
- the loss: the program's key tokens (operators, compared values, positions, view names, the join's tests) weighted
  4x (`--key-weight 4`, `validation/key_tokens.py`); the held loss stays the plain mean, comparable with round 1;
- five epochs; the adapter stays rank 16 / alpha 32 (Nick, 2026-11-13: "so we can use it with other lora": one LoRA
  layout a trunk, so the reader shares the served seats' trunk).
Cowork runs it; the read is at home on grilly2. Re-run after editing:
    python notebooks/_build_view_reader2.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "view_reader2.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0067: the second reader's adapter, round 2

Round 1 learned to split an instruction (its held answer loss reached 0.050) but not what each check means: on the
held-out reads (dev, with the grammar's rules 1-8) the parts' agreement was 0.56, a quarter of them with a comparison
reversed. Round 2 keeps round 1's recipe and changes three things:

1. **Polarity pairs** (`rules/view_reader_sft_v2.jsonl`): round 1's set plus 3,000 rows in 1,500 minimal pairs, each
   long instruction beside its twin with one part said the other way ("fewer than 48 words" / "at least 48 words"),
   each with its own program; 30% of checks said goal first or in the passive. Built by rule, run in the VM before
   kept; train only, nothing from dev or test.
2. **Key tokens weighted 4x** (`--key-weight 4`): the program's operators, compared values, positions, view names and
   the join's tests, about 14% of an answer's tokens. The held loss stays the plain mean, comparable with round 1.
3. **Five epochs**; the adapter stays rank 16 / alpha 32, the served seats' layout, so it shares their trunk.

**Runtime:** an A100 or the G4 (about 45 minutes). It reads, from Drive: `runs/base450m_cont/base450m_cont_final.pt`,
the token cache's tokenizer and the set; it writes `rules/cubby450m_view_reader_v2/` (`talk_lora.*`). Training only:
the read runs at home. **Disconnect when it prints `trained:`.**
"""),
    code(r"""
import json, os, signal, subprocess, sys, time
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
DATA      = f'{DRIVE}/rules/view_reader_sft_v2.jsonl'
OUT       = f'{DRIVE}/rules/cubby450m_view_reader_v2'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], man['by_family'])
assert man.get('pairs'), 'this set has no polarity pairs: build it with --pairs'
assert not man.get('records_over_max_len'), f"records longer than MAX_LEN would be skipped: {man['records_over_max_len']}"
assert {'IFEval', 'IFBench'} <= set(man['benchmarks_checked']), 'the benchmark check did not run'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md(r"""
## The settings

Rank 16, alpha 32 (every projection of every layer, the served adapters' layout), lr 2e-4 with 50 warmup steps then
cosine, batch 32, max length 1,280, five epochs, seed 0, key tokens weighted 4x. On a card under 70 GB a batch is cut
into chunks of at most 4,096 padded tokens (`--micro-tokens`: the same gradient).

Rank 16 / alpha 32 is the served seats' layout (one LoRA layout a trunk): the reader can share their trunk.
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN, KEY_WEIGHT = 5, 32, 2e-4, 16, 32, 1280, 4.0
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_view_reader_v2'
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

**Disconnect the runtime.** At home, Claude Code reads the adapter on grilly2 with the grammar's rules 1-9: dev first
(beside round 1's dev read and its second chance), then test and test_long (lines 1-4), then 0067's held-out kinds and
IFBench (line 5).
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
