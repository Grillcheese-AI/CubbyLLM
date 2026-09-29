"""Generate emitter_450m.ipynb: the 450M program emitter (H-E15) on Colab, LoRA over the frozen base,
trained on SLOTTED programs. The notebook is thin; the logic is validation/train_emitter_torch.py
(data in validation/emitter_data.py, the adapter layout shared with grilly2). Re-run after editing:
    python notebooks/_build_emitter.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "emitter_450m.ipynb"


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
# The 450M program emitter (H-E15)

**What this trains.** The owner's seat decision of 2026-09-29: the 450M is the *program* emitter and the
LFM2.5-2.6B stand-in keeps the talk seat. A LoRA over the frozen 450M base learns to write CubeLang programs
from a question — and it writes them in **slot form**: the host marks the question's names and numbers
(`[E1: iridomyrmex bigi]`, `[N1: 48]`), the emitter writes `bind frame, H1_PARENT_TAXON, "$E2"` and
`assign s0 = $N1;`, never the name or the number itself. Panel round 3's finding behind it: a VM-verified chain
about the wrong entity carries real facts and clears every hop; a 450M with copy loss 1.2 binding a copied value
to the asked entity at chance would produce exactly those. With slots the copy is the host's, deterministic and
ledgered, and a program that names a slot the host does not have is refused before it runs.

**The data** is v12e — the stand-in's program set (arithmetic 6,148, kernel 1,769, role-binding 1,500, chain
1,214, plan 1,214; every program verified through the VM) — converted by `validation/emitter_data.py`:

| family | slots | note |
|---|---|---|
| chain | hop objects → `$E` | the index is every name in the fact lines + the seed (what the host knows) |
| plan | the seed → `$E1` | the HOP words stay words (a closed vocabulary the emitter learns) |
| arithmetic, kernel | the prompt's numbers → `$N` by position | in the step comments too; a derived constant (`div s0, 2` for "half") stays |
| role-binding | none | the copy control: free spans of the sentence, no index holds them |

The verbatim `# <question>` comment is dropped from every program; 14 records whose literal is not in the prompt
are dropped (the coverage gate). Loss on the program tokens and `</s>`; up to 1,024 tokens.

**The read.** After training, the val split is generated (greedy, no repetition guards, stops at `</s>`), each
program is checked against its slot table (unknown slot / copied literal = a refusal), filled, and written to
`val_generations.json`. The VM read runs on the local card:

```
python standin/eval_emitter_vm.py --val-generations <OUT>/val_generations<TAG>.json --data <slots jsonl> --tag <TAG>
```

**The gate** (H-E15, against v12e on the same val records: chain gold_match 1.0, arithmetic 0.675; forge probe
1.00/1.00/1.00): the 450M slot emitter is compared on executes / gold_match per family, plus what slots add —
the copied-literal rate (must be ~0), the unknown-slot rate, and the wrong-slot rate (a slot the host had, the
wrong one: the binding failure in another costume, H-E15's kill).

**Before you run**
1. `master` has `validation/train_emitter_torch.py`, `validation/emitter_data.py` and the `train_talk_torch.py`
   changes (LoRALinear exposes `out_features`); the setup cell checks.
2. On Drive: the base to adapt (`runs/base450m/base450m_final.pt`, or the continued one once its readout passed),
   the tokenizer, and the slot-form data (`emitter/emitter_sft_v12e_slots.jsonl`, from
   `python validation/emitter_data.py standin/data/out/emitter_sft_v12e.jsonl <that path>`).
3. Any GPU: 11.8k records × 2 epochs ≈ 1,500 steps of 16 — minutes on an A100; the generation pass is
   ~64 records per family at up to 640 tokens each, 10–20 minutes.
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
for f in ('validation/train_emitter_torch.py', 'validation/emitter_data.py', 'validation/train_talk_torch.py'):
    assert os.path.exists(f'{REPO}/{f}'), f'{f} is not on master yet: push it first'
assert 'out_features' in open(f'{REPO}/validation/train_talk_torch.py').read(), 'push the LoRALinear change first'
"""),
    code(r"""
# --- EDIT to your layout and the run you want ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m/base450m_final.pt'        # or runs/base450m_cont/base450m_cont_final.pt once it passed
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
DATA      = f'{DRIVE}/emitter/emitter_sft_v12e_slots.jsonl'   # validation/emitter_data.py <v12e jsonl> <this>
EPOCHS, BATCH, LR, RANK, ALPHA = 2, 16, 2e-4, 16, 32
MAX_LEN   = 1024
GEN_PER_TASK, GEN_MAX_NEW = 64, 640
TAG       = '_v12e_slots'
OUT       = f'{DRIVE}/emitter/cubby450m{TAG}'                 # the adapter: emitter_lora.safetensors + emitter_lora.json
for p in (CKPT, TOKENIZER, DATA):
    assert os.path.exists(p), p
os.makedirs(OUT, exist_ok=True)
"""),
    md(r"""
### The run

The log is teed to Drive beside the adapter; the adapter is saved every 500 steps and at the end; held program
loss every 200 steps. Then the generation pass writes `val_generations<TAG>.json` beside the adapter with the
slot verdicts per family (slots_ok / copied / unknown_slot / text_exact).
"""),
    code(r"""
cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN),
       '--gen-per-task', str(GEN_PER_TASK), '--gen-max-new', str(GEN_MAX_NEW), '--tag', TAG]
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
!cp {REPO}/validation/logs/train_emitter_torch{TAG}.* {OUT}/ 2>/dev/null; ls -la {OUT}
"""),
    md(r"""
### After the run

On the local card, with the repo's cubelang build:

```
python standin/eval_emitter_vm.py --val-generations "<Drive>/cubbyllm/emitter/cubby450m_v12e_slots/val_generations_v12e_slots.json" \
    --data standin/data/out/emitter_sft_v12e_slots.jsonl --tag _cubby450m_v12e_slots
```

gives executes / gold_match per family for the filled programs — the numbers to set against v12e's on the
same records. Live, through grilly2 (the base export beside the adapter):

```
python standin/eval_emitter_vm.py --cubby <adapter dir> --export <export dir> --tokenizer <bbpe128k.json> \
    --data standin/data/out/emitter_sft_v12e_slots.jsonl --per-task 40
```

Record the verdict under H-E15 against its pre-registered gate and kill.
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
