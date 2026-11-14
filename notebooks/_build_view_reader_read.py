"""Generate view_reader_read.ipynb: order 0067, the view reader's read on a Colab card (Nick, 2026-11-13: "why dont we
run the tests on colab might be faster?"). The local read decodes one instruction at a time on grilly2 (about 25 s
each; the grammar is 0.3 s of it); here `validation/exp_view_reader_torch.py` decodes batches in lockstep on the torch
model's O(1) step, each row held to the grammar by the same chooser. It writes the answers only; the VM scores them at
home (`exp_view_reader.py --score`), so nothing here needs cubelang. Re-run after editing:
    python notebooks/_build_view_reader_read.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "view_reader_read.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0067: the view reader's read, on a Colab card

The answers only: each instruction read whole by the adapter, its answer held token by token to the grammar (rules
1-9, `--grammar typed`), the same rows as the read at home (each split's seeded sample). The VM scores the answers at
home. A batch of 32 instructions decodes in lockstep; about ten minutes for 300 rows on an A100 (the local card: two
hours). `N = 300` reads each split whole (900 rows, about half an hour): the counted read; its counted ids are
hashed in `validation/logs/view_reader_round2_counted_ids_2026-11-14.json`. A second pass reads dev again with
rule 9 off (`typed8`, Cowork 2026-11-14: rules 1-8 against 1-9, on dev only; about ten minutes).

**Set `ADAPTER` below** (`cubby450m_view_reader_v1` or `_v2`). It reads, from Drive, the base, the tokenizer, the
adapter and `rules/long_rows_0067/rows.jsonl`; it writes `rules/reads/<adapter>_<grammar>_answers.jsonl` (resumable:
a rerun keeps the rows already answered). **Disconnect when it prints `read:`.**
"""),
    code(r"""
import os, subprocess, sys
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy regex
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
ADAPTER   = 'cubby450m_view_reader_v2'          # or 'cubby450m_view_reader_v1'
N, BATCH = 300, 32                                 # 300 = each split whole (the counted read)
PASSES = [('typed', 'dev,test,test_long'),          # rules 1-9: the counted read
          ('typed8', 'dev')]                         # rule 9 off, dev only: 1-8 against 1-9
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
ROWS      = f'{DRIVE}/rules/long_rows_0067/rows.jsonl'
ADIR      = f'{DRIVE}/rules/{ADAPTER}'
OUTS      = {g: f'{DRIVE}/rules/reads/{ADAPTER}_{g}_answers.jsonl' for g, _ in PASSES}
for p in (CKPT, TOKENIZER, ROWS, f'{ADIR}/talk_lora.safetensors', f'{ADIR}/talk_lora.json'):
    assert os.path.exists(p), p
os.makedirs(f'{DRIVE}/rules/reads', exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code(r"""
os.environ['CB_NO_FLEX'] = '1'
for GRAMMAR, SPLITS in PASSES:
    OUT = OUTS[GRAMMAR]
    print(f'--- {GRAMMAR}: {SPLITS} -> {OUT}')
    cmd = [sys.executable, '-u', 'validation/exp_view_reader_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--adapter', ADIR, '--rows', ROWS, '--splits', SPLITS, '--n', str(N), '--batch', str(BATCH),
           '--grammar', GRAMMAR, '--out', OUT]
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'the read failed ({p.returncode})'
    !wc -l "{OUT}"
"""),
    md(r"""
## Done

**Disconnect the runtime.** At home, Claude Code scores the answers in the VM:
`exp_view_reader.py --score <answers> --splits dev` and `--splits test,test_long` (lines 1-4), then the second
chance on the dev answers; the `typed8` answers (dev only) with `--grammar typed8`.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
