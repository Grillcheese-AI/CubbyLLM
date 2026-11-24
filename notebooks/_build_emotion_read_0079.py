"""Generate emotion_read_0079.ipynb: order 0079's first training of the trunk's emotion read (Cowork's addendum,
2026-11-24; Nick: "yes") -- talk_v3a continued on the emotion rows with the pilot's chat rows beside them
(`validation/train_talk_torch.py --init-adapter --by-family`), then the read on the held-out test
(`validation/exp_0079_appraiser_read.py --held --torch-ckpt`, the gate). Re-run after editing:
    python notebooks/_build_emotion_read_0079.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "emotion_read_0079.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# Order 0079: the trunk learns to read the user's emotion (with the talk seat)

**Why.** talk_v3a names the right petal on 11.3% of `emotions.jsonl` and keeps the asked "emotion — petal" form on 9
of 2,914 answers (`exp_0079_appraiser_read`), so the mood serving produces would stay flat.

**What this trains.** talk_v3a continued, one epoch, on `talk/emotion_0079/talk_emotion_0079.jsonl`: 38,134 emotion
rows (emotions.jsonl's train texts x4, GoEmotions train with neutral capped) and the pilot's 868 chat rows beside them
to keep the voice; 105 held rows (chat and emotion) for the held loss by family. Same recipe as talk_v3a (batch 16,
lr 2e-4, rank 16 / alpha 32 from the adapter). **Then the gate:** the read on the held-out test (342 emotions.jsonl
texts, GoEmotions' 5,427 test rows, 300 machine-translated French, flagged).

**Writes** `talk/talk_v3a_emo_0079/` (the adapter, `read_held.json`). `mental_health_svc` is read at home, not here.
"""),
    code(r"""
# --- setup: clone, deps, Drive ---
import os, sys, subprocess, time, json
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy pandas pyarrow "protobuf>=7.35.1"
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code(r"""
# --- train ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
INIT      = f'{DRIVE}/talk/talk_v3a'
DATA      = f'{DRIVE}/talk/emotion_0079/talk_emotion_0079.jsonl'
HELD      = f'{DRIVE}/talk/emotion_0079/emotion_read_0079_held.jsonl'
OUT       = f'{DRIVE}/talk/talk_v3a_emo_0079'
for p in (CKPT, TOKENIZER, f'{INIT}/talk_lora.safetensors', DATA, DATA + '.manifest.json', HELD):
    assert os.path.exists(p), p
os.makedirs(OUT, exist_ok=True)

def stream(cmd, log):
    with open(log, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    assert p.returncode == 0, f'exit {p.returncode}'

if not os.path.exists(f'{OUT}/DONE'):
    stream([sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
            '--data', DATA, '--init-adapter', INIT, '--by-family', '--epochs', '1', '--batch', '16', '--lr', '2e-4',
            '--out', OUT, '--tag', '_v3a_emo_0079'], f'{OUT}/train.log')
    !cp {REPO}/validation/logs/train_talk_torch_v3a_emo_0079.jsonl {OUT}/
    open(f'{OUT}/DONE', 'w').write(time.strftime('%F %T'))
print('trained ->', OUT)
"""),
    code(r"""
# --- the gate: the read on the held-out test (strict parse = as served) ---
stream([sys.executable, '-u', 'validation/exp_0079_appraiser_read.py', '--torch-ckpt', CKPT, '--cubby', OUT,
        '--tokenizer', TOKENIZER, '--held', HELD, '--tag', '_v3a_emo_0079_held'], f'{OUT}/read.log')
!cp {REPO}/validation/logs/exp_0079_appraiser_read_v3a_emo_0079_held.json {OUT}/read_held.json
r = json.load(open(f'{OUT}/read_held.json'))['held']
for s, v in r.items():
    print(s, v)
print('read done ->', OUT)
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
