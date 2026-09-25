"""Generate talk_sft.ipynb: the talk adapter (H-E6) on Colab, LoRA over the frozen 450M base.
The notebook is thin; the logic is validation/train_talk_torch.py (data and adapter layout shared
with the grilly2 trainer through validation/talk_data.py). Re-run after editing:
    python notebooks/_build_talk.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "talk_sft.ipynb"


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
# The talk adapter on the 450M base

**What this trains.** H-E6 in `CUBBYLLM_HYPOTHESES.md`: a LoRA over the frozen 450M base that says what the VM
returned — the asked entity's value, the facts over the weights, and `The facts don't say.` when they hold no
answer. H-E5 measured the base alone binding a value to the right entity at chance (52%) and answering from memory
22/24 times against a contradicting fact.

**The data** is `standin/data/build_ground_sft.py`'s: facts blocks in the talk format

```
Facts:
- <entity> — <relation>: <value>
Question: <question>
Answer: <answer></s>
```

five families (profile, relation, bind, counter, absent), 14.8k records, split by entity hash. **talk_v2** adds, through
`standin/data/build_talk_mix.py` (counts in its manifest): ChatQA passages and web passages with their "don't say"
twins, who-said-it quotes, and instruction-following chats that pass the host's guard -- a passage, a quote block or
the earlier turns take the facts block's place. Up to 1,024 tokens. Loss on the answer tokens only.
**talk_v2.1** (`build_talk_mix.py --mix v2.1`) puts the facts-block form back to about half the mix: ground_sft
twice, the same families from the hypernet scaling-law set (`build_hdc_sft.py`, with 2-3 hop chains) and dated
historical events (`build_events_sft.py`) and the history graph (`build_history_sft.py`: causes, effects, chains,
what would change); fewer passages and chats. **talk_v2.2** (`--mix v2.2`) is v2.1 with more facts-block "the
facts don't say" records (hist_absent 15k, every hdc fact_absent), for H-E6's absent bar. ~216k train records.

**The adapter** is LoRA r16/α32 on all 181 projections (7.86M trainable); the base is never merged. The file is keyed
by grilly2's module names, so what this writes loads on the local card for the gate (`validation/exp_e6_talk_gate.py`)
and for serving — the same file the local grilly2 trainer (`validation/train_talk_grilly.py`) writes.

**Before you run**
1. `master` on GitHub has `validation/train_talk_torch.py` and `validation/talk_data.py`; the setup cell checks.
2. On Drive: the base (`cubbyllm/runs/base450m/base450m_final.pt`, from the pretraining run), the tokenizer
   (`cubbyllm/token_cache_base/tokenizer/`), and the data (`cubbyllm/talk/talk_v2_2.jsonl`).
3. A large GPU. talk_v2.1 (203k train records, 6,346 steps) took 26.8 min at 0.25 s/step; talk_v2.2 is ~6,750
   steps.
"""),
    code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, subprocess, time, signal
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
for f in ('validation/train_talk_torch.py', 'validation/talk_data.py'):
    assert os.path.exists(f'{REPO}/{f}'), f'{f} is not on master yet: push it first'
"""),
    code(r"""
# --- EDIT to your layout and the run you want ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m/base450m_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
DATA      = f'{DRIVE}/talk/talk_v2_2.jsonl'
EPOCHS, BATCH, LR, RANK, ALPHA = 1, 32, 2e-4, 16, 32
MAX_LEN   = 1024
TAG       = '_v2_2'
OUT       = f'{DRIVE}/talk/talk{TAG}'          # the adapter: talk_lora.safetensors + talk_lora.json
for p in (CKPT, TOKENIZER, DATA):
    assert os.path.exists(p), p
os.makedirs(OUT, exist_ok=True)
"""),
    md(r"""
### The run

The log is teed to Drive beside the adapter; the adapter is saved every 200 steps and at the end. Held answer loss is
printed every 100 steps on held entities (never trained on).
"""),
    code(r"""
cmd = [sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--tag', TAG]
p = None
try:
    with open(f'{OUT}/train.log', 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
finally:
    if p and p.poll() is None:
        p.send_signal(signal.SIGINT)
!cp {REPO}/validation/logs/train_talk_torch{TAG}.* {OUT}/ 2>/dev/null; ls -la {OUT}
"""),
    md(r"""
### After the run

Copy `cubbyllm/talk/talk_v2_2/` off Drive to the adapter folder beside the grilly2 export, then run the gates on
the local card (held entities, greedy):

```
python validation/exp_e6_talk_gate.py --export <export dir> --adapter <that folder> \
    --data standin/data/out/ground_sft.jsonl --tag _v2_2
python validation/exp_e6_host_view.py --data standin/data/out/ground_sft.jsonl --run _v2_2
python validation/exp_e8_history_retrieval.py --export <export dir> --adapter <that folder> --tag _v2_2
```

Record the verdicts under H-E6 and H-E8 against their pre-registered gates.
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
