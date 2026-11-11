"""Generate talk_sft.ipynb: the talk adapter (H-E6, order 0010) on Colab, LoRA over the frozen 450M base.
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
**talk_v3a** (order 0010, `validation/sft/build_talk_voice.py --mix-out`) is talk_v2.2 unchanged plus Cubby's voice:
38,196 train records in English and French -- a hello by name, what Cubby can do, thanks and goodbyes, a joke when
asked, a fact behind a frame or as a list, a wrong assumption answered. A template may be model-written, a fact
never is: every fact in a voice record is a line of its facts block or of Cubby's own record.

**The two runs here, both on `base450m_cont`** (talk_v2.2 itself sits on the earlier `base450m_final.pt`):
- `_v2_2_cont`, the control: talk_v2.2's data on the new base (216,035 records, 13,503 steps at batch 16);
- `_v3a`, the voice: talk_v3a (254,231 records, 15,890 steps at batch 16).
The base, the batch and the data all change against talk_v2.2; the control carries the first two, so the gates
read the voice records alone.

**The adapter** is LoRA r16/α32 on all 181 projections (7.86M trainable); the base is never merged. The file is keyed
by grilly2's module names, so what this writes loads on the local card for the gate (`validation/exp_e6_talk_gate.py`)
and for serving — the same file the local grilly2 trainer (`validation/train_talk_grilly.py`) writes.

**Before you run**
1. `master` on GitHub has `validation/train_talk_torch.py` and `validation/talk_data.py`; the setup cell checks.
2. On Drive: the base (`cubbyllm/runs/base450m_cont/base450m_cont_final.pt`), the tokenizer
   (`cubbyllm/token_cache_cont/tokenizer/`), and the data (`cubbyllm/talk/talk_v2_2.jsonl`, `cubbyllm/talk/talk_v3a.jsonl`).
3. A large GPU (A100 80 GB). Run on 2026-11-11 at batch 16 with `CB_NO_FLEX=1`: 68.5 and 79.8 minutes at
   0.30 s/step. Batch 32, talk_v2.2's setting (26.8 min at 0.25 s/step in September), filled the 80 GB card that
   day on either attention path and ran at 0.55 s/step; why is not measured yet.

**Run all** does everything: both runs, then the last cell releases the GPU. A run already finished (a `DONE` file
beside its adapter on Drive) is not run again, so a session that ends early is simply run again.
"""),
    code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, subprocess, time, signal, json
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
# --- EDIT to your layout and the runs you want ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_cont/tokenizer/grillcheese_bbpe128k.json'
RUNS      = [('_v2_2_cont', f'{DRIVE}/talk/talk_v2_2.jsonl'),   # the control: talk_v2.2's data on the new base
             ('_v3a',       f'{DRIVE}/talk/talk_v3a.jsonl')]    # the voice: the same mix plus the voice records
EPOCHS, BATCH, LR, RANK, ALPHA = 1, 16, 2e-4, 16, 32   # batch 16: 32 filled the 80 GB card (2026-11-11)
MAX_LEN   = 1024
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
for p in (CKPT, TOKENIZER, *[d for _, d in RUNS]):
    assert os.path.exists(p), p
"""),
    md(r"""
### The runs

Each run's adapter goes to `cubbyllm/talk/talk<tag>/` (`talk_lora.safetensors` + `talk_lora.json`), saved every 200
steps and at the end; its log is teed to Drive beside it. Held answer loss is printed every 100 steps on held
entities (never trained on).
"""),
    code(r"""
def run(tag, data):
    out = f'{DRIVE}/talk/talk{tag}'          # the adapter: talk_lora.safetensors + talk_lora.json
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/DONE'):
        print(f'{tag}: already trained ({open(f"{out}/DONE").read().strip()}), not run again')
        return
    cmd = [sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--data', data, '--out', out, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
           '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--tag', tag]
    p, t0 = None, time.time()
    try:
        with open(f'{out}/train.log', 'a', encoding='utf-8', buffering=1) as f:
            f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end=''); f.write(line)
            p.wait()
    finally:
        if p and p.poll() is None:
            p.send_signal(signal.SIGINT)
    subprocess.run(f'cp {REPO}/validation/logs/train_talk_torch{tag}.* {out}/ 2>/dev/null', shell=True)
    if p and p.returncode == 0 and os.path.exists(f'{out}/talk_lora.safetensors'):
        step = json.load(open(f'{out}/talk_lora.json'))['step']
        with open(f'{out}/DONE', 'w') as f:
            f.write(f"{time.strftime('%F %T')}, {step} steps, {(time.time() - t0) / 60:.1f} min")
        print(f'{tag}: done, {step} steps in {(time.time() - t0) / 60:.1f} min -> {out}')
    else:
        print(f'{tag}: NOT finished (exit code {p.returncode if p else None}); its log is {out}/train.log')

for tag, data in RUNS:
    run(tag, data)
"""),
    code(r"""
# --- release the GPU: nothing is billed once the adapters and logs are on Drive ---
from google.colab import runtime
for tag, _ in RUNS:
    d = f'{DRIVE}/talk/talk{tag}'
    print(tag, '->', open(f'{d}/DONE').read().strip() if os.path.exists(f'{d}/DONE') else 'NOT finished, see its train.log')
drive.flush_and_unmount()                     # the last writes reach Drive before the runtime goes
runtime.unassign()
"""),
    md(r"""
### After the runs

Copy `cubbyllm/talk/talk_v2_2_cont/` and `cubbyllm/talk/talk_v3a/` off Drive to adapter folders beside the grilly2
export of `base450m_cont`, then run the gates on the local card for both (held entities, greedy):

```
python validation/exp_e6_talk_gate.py --export <export dir> --adapter <that folder> \
    --data standin/data/out/ground_sft.jsonl --tag _v3a
python validation/exp_e6_host_view.py --data standin/data/out/ground_sft.jsonl --run _v3a
python validation/exp_e8_history_retrieval.py --export <export dir> --adapter <that folder> --tag _v3a
```

Record the verdicts under H-E6 and H-E8 against their pre-registered gates, the voice run against the control
(order 0010: 0 wrong spoken; spoken-and-correct at least the control's minus 0.02, the host reading replies through
the opener view for both).
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
