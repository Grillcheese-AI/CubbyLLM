"""Generate emitter_wd.ipynb: order 0048's programs adapter, taught Wikidata's property labels (EN and FR) as HOP1's
targets. It continues the serving adapter (cubby450m_v12e_w_step_cont, on base450m_cont) for one epoch over v12e's
records plus the new ones (validation/emitter_wd_data.py), with that adapter's own settings, and writes
cubby450m_wd_cont beside it (the serving adapter is not touched). Nick runs it; the read is at home, in 0046's R3
harness on grilly2. Re-run after editing:
    python notebooks/_build_emitter_wd.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "emitter_wd.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0048: the programs adapter names Wikidata's properties

**Runtime:** an A100 (any GPU with 40 GB works; the run is about 45 minutes). It reads, from Drive:
`runs/base450m_cont/base450m_cont_final.pt` (the base the serving adapter was trained on),
`emitter/cubby450m_v12e_w_step_cont/` (the serving adapter, continued here, not changed) and
`emitter/emitter_sft_v12e_wd.jsonl` (v12e's records + the Wikidata-label records, with its manifest). It writes
`emitter/cubby450m_wd_cont/`. Training only: the read runs at home. **Disconnect when it prints `trained:`.**
"""),
    code(r"""
import json, math, os, subprocess, sys
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
FROM      = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'     # the serving adapter: continued, never written
DATA      = f'{DRIVE}/emitter/emitter_sft_v12e_wd.jsonl'
OUT       = f'{DRIVE}/emitter/cubby450m_wd_cont'
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json', f'{FROM}/emitter_lora.safetensors'):
    assert os.path.exists(p), p
!nvidia-smi --query-gpu=name,memory.total --format=csv
man = json.load(open(DATA + '.manifest.json'))
print('share:', man['share'], '| wd records:', man['wd_by_set_lang'])
"""),
    md(r"""
## The settings: the serving adapter's own (`emitter_lora.json` of cubby450m_v12e_w_step_cont)

One epoch over the train split, rank 16, alpha 32, lr 2e-4, batch 32, max length 1024, seed 0, continued from the
serving adapter. The `val` split is the held-out loss the trainer reports.
"""),
    code(r"""
meta = json.load(open(f'{FROM}/emitter_lora.json'))
RANK, ALPHA, LR, BATCH, MAX_LEN, SEED = meta['rank'], meta['alpha'], meta['lr'], meta['batch'], meta.get('max_len', 1024), 0
rows = sum(int(r.get('repeat', 1) or 1) for r in map(json.loads, open(DATA, encoding='utf-8')) if r['split'] == 'train')
STEPS = math.ceil(rows / BATCH)
print(rows, 'train rows ->', STEPS, 'steps of', BATCH, '| rank', RANK, 'alpha', ALPHA, 'lr', LR)
assert not os.path.exists(f'{OUT}/emitter_lora.safetensors'), f'{OUT} exists: move it aside first (it would be resumed)'
os.makedirs(OUT, exist_ok=True)
tag = '_wd_cont'
cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--steps', str(STEPS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
       '--resume-adapter', FROM, '--tag', tag, '--no-gen']
with open(f'{OUT}/train.log', 'a', encoding='utf-8', buffering=1) as f:
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end=''); f.write(line)
    assert p.wait() == 0, f'the trainer failed ({p.returncode})'
os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}.* {OUT}/ 2>/dev/null')
print('trained:', OUT)
"""),
    md(r"""
## Done

**Disconnect the runtime.** At home, Claude Code records the base hash (`validation/adapter_base_hash.py`, after one
program on grilly2) and reads the adapter in 0046's R3 harness as `450m_wd`, in both settings (all properties, and the
item's own), against the current adapter, and runs H-E15's benchmark for the adapter's other jobs.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
