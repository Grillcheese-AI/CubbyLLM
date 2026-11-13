"""Generate view_reader.ipynb: order 0067, the second reader's adapter round (step 5). A LoRA adapter on the frozen
base450m_cont that reads an instruction into its split and its join (one CubeLang program over the reply's views), in
the reader's form (`cubbyllm/speech/view_reader.py`), trained with the talk trainer (`validation/train_talk_torch.py`:
loss on the answer only) on `validation/sft/build_view_reader_sft.py`'s set: step 3's long instructions (train and dev
only), step 2's single checks said in words, step 4's CubeLang curriculum. Cowork runs it; the read is at home on
grilly2. Re-run after editing:
    python notebooks/_build_view_reader.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "view_reader.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0067: the second reader's adapter, round 1

The plan reader fills a fixed form; on kinds it has no form for it writes the nearest one (IFBench, a diagnostic: 36.3%
false passes). This reader writes the check itself in CubeLang over the reply's views, and splits a long instruction
into its parts (check, task, asked, process, not an instruction) before joining the checks into one program.

**The set** (`rules/view_reader_sft_v1.jsonl`, its manifest beside it): step 3's long instructions with 2-4 parts
(train; dev is the held loss), step 2's single checks said in English or French, step 4's compiled CubeLang programs
as continuations. The held-out compositions (2-4 parts) and the long ones (5-8, line 4) are not in it.

**Runtime:** an A100 (records up to 1,280 tokens; three epochs). It reads, from Drive:
`runs/base450m_cont/base450m_cont_final.pt`, the token cache's tokenizer and the set; it writes
`rules/cubby450m_view_reader_v1/` (`talk_lora.*`). Training only: the read runs at home. **Disconnect when it prints
`trained:`.**
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
DATA      = f'{DRIVE}/rules/view_reader_sft_v1.jsonl'
OUT       = f'{DRIVE}/rules/cubby450m_view_reader_v1'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], man['by_family'])
assert not man.get('records_over_max_len'), f"records longer than MAX_LEN would be skipped: {man['records_over_max_len']}"
assert {'IFEval', 'IFBench'} <= set(man['benchmarks_checked']), 'the benchmark check did not run'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md(r"""
## The settings

The served adapters' LoRA layout (rank 16, alpha 32, every projection of every layer), lr 2e-4 with 50 warmup steps
then cosine, batch 32, max length 1,280, three epochs, seed 0. On a card under 70 GB a batch is cut into chunks of at
most 4,096 padded tokens (`--micro-tokens`: the same gradient).
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN = 3, 32, 2e-4, 16, 32, 1280
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_view_reader_v1'
import torch
MICRO = 0 if torch.cuda.get_device_properties(0).total_memory >= 70e9 else 4096
print('micro-batch tokens:', MICRO or 'off (the whole batch at once)')
cmd = [sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', '0', '--tag', tag,
       '--micro-tokens', str(MICRO)]
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

**Disconnect the runtime.** At home, Claude Code reads the adapter on grilly2: the held-out compositions (2-4 parts)
and the long ones (5-8) once each, every program compiled and run on its replies against the generator's verdict
(lines 1-4), then 0067's held-out kinds and IFBench (line 5).
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
