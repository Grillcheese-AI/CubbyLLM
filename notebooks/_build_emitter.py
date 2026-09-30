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

**The A/B (H-E18, program-first data).** One session trains two adapters on the same base, at the same number of
steps, seed and settings, and they differ only in the data:

| arm | data | train rows (v12e's repeats applied) |
|---|---|---|
| `v12e` (control) | `emitter_sft_v12e_slots.jsonl` | 14,744 |
| `v12e_pf` | `emitter_sft_v12e_pf_slots.jsonl` = v12e + 1,341 arithmetic + 1,079 plan + 1,101 chain written program-first | 18,265 |

Both arms run `STEPS` = the control's two epochs (1,843 steps of 16), so the program-first arm sees each record
fewer times: the comparison is at equal compute, as pre-registered.

Each arm then generates twice: the v12e val split (the same records for both arms; the gate's "no v12e family
drops > 2 pts") and the **program-first held-out eval** (`pf_heldout_eval_slots.jsonl`: 236 arithmetic, 285
plan, 294 chain — shapes and seeds never in training, worded by the *other* writer). Pre-registered gate: on the
held-out eval, VM-correct +10 pts for `v12e_pf`; no v12e val family down > 2 pts; 0 copied literals and 0
unknown slots. Kill: under +3 pts → binding, not wording, is the bottleneck. Read accuracy **by step count**: the
program-first arithmetic kept 95% of 2-step questions but 28% of 6-step ones, so the new data leans short.

**The base** is `base450m_cont_final.pt`: the continuation passed its readout 2026-09-29 (mix 2.2265 vs 2.3615,
copy 1.155).

**Before you run**
1. `master` has `validation/train_emitter_torch.py` (with `--gen-extra`), `validation/emitter_data.py` and the
   `train_talk_torch.py` changes (LoRALinear exposes `out_features`); the setup cell checks.
2. On Drive, under `cubbyllm/`: `runs/base450m_cont/base450m_cont_final.pt`, the tokenizer, and in `emitter/` the
   three slot files (`emitter_sft_v12e_slots.jsonl`, `emitter_sft_v12e_pf_slots.jsonl`, `pf_heldout_eval_slots.jsonl`).
3. An A100: 1,843 steps of 16 per arm (minutes), then the generations — 320 v12e val records and 815 held-out
   records per arm at up to 640 tokens, roughly 45–60 min per arm. `GEN_HELD_PER_TASK` trims the second pass.
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
assert '--gen-extra' in open(f'{REPO}/validation/train_emitter_torch.py').read(), 'push the --gen-extra change first'
"""),
    code(r"""
# --- EDIT to your layout and the run you want ---
import json, math
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'   # passed its readout 2026-09-29
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
ARMS = {                                                        # H-E18: the same run, two data sets
    'v12e':    f'{DRIVE}/emitter/emitter_sft_v12e_slots.jsonl',     # the control
    'v12e_pf': f'{DRIVE}/emitter/emitter_sft_v12e_pf_slots.jsonl',  # + the program-first records
}
HELD      = f'{DRIVE}/emitter/pf_heldout_eval_slots.jsonl'      # program-first held-out eval (the other writer)
RUN_ARMS  = ['v12e', 'v12e_pf']                                 # one arm per session also works; STEPS is the same
EPOCHS, BATCH, LR, RANK, ALPHA, SEED = 2, 16, 2e-4, 16, 32, 0
MAX_LEN   = 1024
GEN_PER_TASK, GEN_MAX_NEW = 64, 640
GEN_HELD_PER_TASK = 0                                           # 0 = all 815; e.g. 120 to cut the second pass
for p in (CKPT, TOKENIZER, HELD, *ARMS.values()):
    assert os.path.exists(p), p

def train_rows(path):                                           # what the trainer counts: train split x repeat
    n = 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r['split'] == 'train':
            n += int(r.get('repeat', 1) or 1)
    return n
STEPS = math.ceil(EPOCHS * train_rows(ARMS['v12e']) / BATCH)    # the control's 2 epochs, for BOTH arms
print({arm: train_rows(p) for arm, p in ARMS.items()}, '-> STEPS', STEPS)
"""),
    md(r"""
### The runs

Per arm: the log teed to Drive beside the adapter (`emitter/cubby450m_<arm>_cont/`), the adapter saved every
500 steps and at the end, held program loss every 200 steps; then `val_generations_<arm>_cont.json` (v12e val)
and `val_generations_<arm>_cont_pfheld.json` (the program-first held-out eval), with the slot verdicts per family.
An arm whose adapter and both generation files already exist is skipped, so a restarted session picks up
where it stopped.
"""),
    code(r"""
def run_arm(arm):
    tag = f'_{arm}_cont'
    out = f'{DRIVE}/emitter/cubby450m{tag}'
    done = [f'{out}/emitter_lora.safetensors', f'{out}/val_generations{tag}.json', f'{out}/val_generations{tag}_pfheld.json']
    if all(os.path.exists(p) for p in done):
        print(f'{arm}: done already -> {out}'); return
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--data', ARMS[arm], '--out', out, '--steps', str(STEPS), '--epochs', str(EPOCHS), '--batch', str(BATCH),
           '--lr', str(LR), '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
           '--gen-per-task', str(GEN_PER_TASK), '--gen-max-new', str(GEN_MAX_NEW), '--tag', tag,
           '--gen-extra', HELD, '--gen-extra-per-task', str(GEN_HELD_PER_TASK), '--gen-extra-tag', '_pfheld']
    p = None
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
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}.* {out}/ 2>/dev/null')
    print(f'{arm}: exit {p.returncode if p else "?"} -> {out}'); os.system(f'ls -la {out}')

for arm in RUN_ARMS:
    run_arm(arm)
"""),
    md(r"""
### After the runs

On the local machine, with the repo's cubelang build — one command reads all four generation files through the
real VM and prints the gate:

```
python validation/exp_he18_ab.py --drive "<Drive>/cubbyllm/emitter"
```

It reports, per arm: executes / correct on the v12e val families (the "no family down > 2 pts" line), correct on
the held-out eval by task and by arithmetic step count, and the slot rates (copied literals, unknown slots); the
verdict against H-E18's gate and kill goes to `validation/logs/exp_he18_ab.json`. Record it under H-E18. The
adapters are also H-E15's first 450M emitter on the continued base: set the control arm's numbers against v12e
(the stand-in) on the same val records.
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
