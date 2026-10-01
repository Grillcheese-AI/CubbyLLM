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
### Regenerate the v12e val (the decode fix, 2026-09-30)

The first control run decoded its programs with the tokenizer's default, which skips special tokens -- and the
role names `ACTION`, `AGENT`, `OBJECT` are special tokens in bbpe128k. Every role-binding program came back as
`bind evt, , "Name";` and none ran (role-binding executes 0.000). The model wrote the roles; the text lost them.
`train_emitter_torch.py` now keeps every special token but the control ones. This cell regenerates only the 64
role-binding records of an arm's v12e val file whose role-binding programs show the empty role (every other
family decodes identically -- checked on the references -- and the held-out eval has no role-binding), from the
saved adapter, without training, and merges them into the file; the old file is kept as `.prev.json`. A few
minutes per arm; an arm generated after the fix is skipped.
"""),
    code(r"""
import re
!cd /content/CubbyLLM && git pull -q --ff-only
assert '--gen-only' in open(f'{REPO}/validation/train_emitter_torch.py').read(), 'push the decode fix first'

def stale(path):                         # a role-binding program with an empty role: decoded before the fix
    outs = json.load(open(path, encoding='utf-8'))['outputs']
    return any(re.search(r'bind\s+\w+\s*,\s*,', o['generated_slotted']) for o in outs if o['task'] == 'role_binding')

def regen_val(arm):
    tag = f'_{arm}_cont'
    out = f'{DRIVE}/emitter/cubby450m{tag}'
    val = f'{out}/val_generations{tag}.json'
    if not os.path.exists(val) or not stale(val):
        print(f'{arm}: nothing to regenerate'); return
    cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--gen-only', '--resume-adapter', out,
           '--ckpt', CKPT, '--tokenizer', TOKENIZER, '--data', ARMS[arm], '--out', out, '--seed', str(SEED),
           '--gen-per-task', str(GEN_PER_TASK), '--gen-max-new', str(GEN_MAX_NEW), '--tag', tag,
           '--gen-tasks', 'role_binding']
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'{arm}: exit {p.returncode}'
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}_regen.log {out}/ 2>/dev/null')
    print(f'{arm}: regenerated -> {val}', '(still stale!)' if stale(val) else '')

for arm in RUN_ARMS:
    regen_val(arm)
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
    md(r"""
## H-E19 — arithmetic as a world and a school

**Gate A, data and format.** One more arm, `v12e_w_tg`: v12e re-slotted with every number the host can place
(numbers in words, the arithmetic world's conversions as `$K`, no `= value` in the step comments) plus
TinyGSM shard 0 converted through the VM (`standin/data/import_tinygsm.py`). It trains longer than the A/B arms
(`TG_EPOCHS` over its own rows — this gate is absolute, not an equal-steps comparison), then generates all v12e
val records and the program-first held-out eval, both re-slotted the same way. The read, locally:
`python validation/exp_he18_ab.py --files <ctrl val> <ctrl held> <this val> <this held>` puts it beside the control;
Gate A is arithmetic VM-correct at one try ≥ 20% (the control: 3.1%).
"""),
    code(r"""
TG_ARM   = 'v12e_w_tg'
TG_DATA  = f'{DRIVE}/emitter/emitter_sft_v12e_w_tg_slots.jsonl'     # v12e_w + TinyGSM shard 0, slot form
TG_VAL   = f'{DRIVE}/emitter/emitter_sft_v12e_w_slots.jsonl'        # its val split: the v12e val, re-slotted
TG_HELD  = f'{DRIVE}/emitter/pf_heldout_eval_w_slots.jsonl'         # the program-first held-out, re-slotted
TG_EPOCHS = 1
for p in (TG_DATA, TG_HELD):
    assert os.path.exists(p), p
TG_STEPS = math.ceil(TG_EPOCHS * train_rows(TG_DATA) / BATCH)
print(TG_ARM, train_rows(TG_DATA), 'train rows ->', TG_STEPS, 'steps')

def run_tg():
    tag = f'_{TG_ARM}_cont'
    out = f'{DRIVE}/emitter/cubby450m{tag}'
    if all(os.path.exists(p) for p in (f'{out}/emitter_lora.safetensors', f'{out}/val_generations{tag}.json',
                                      f'{out}/val_generations{tag}_pfheld.json')):
        print('done already ->', out); return out
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--data', TG_DATA, '--out', out, '--steps', str(TG_STEPS), '--batch', str(BATCH), '--lr', str(LR),
           '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
           '--gen-per-task', '0', '--gen-max-new', str(GEN_MAX_NEW), '--tag', tag,
           '--gen-extra', TG_HELD, '--gen-extra-per-task', '0', '--gen-extra-tag', '_pfheld',
           '--gen-tasks', 'arithmetic']            # gate A reads arithmetic: 329 + 236 records, not all 1,429
    with open(f'{out}/train.log', 'a', encoding='utf-8', buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}.* {out}/ 2>/dev/null')
    return out

TG_OUT = run_tg()
"""),
    md(r"""
**The write-back arm (`v12e_w_step`, 2026-09-30).** Gate A was killed (val 14.6%, held-out 5.5%) and the
teacher-forced read (`validation/exp_he19_next_step.py`) said why: with the gold program in front of it the
450M picks the first step at 80% and every later one at ~22%, starting 65% of later steps from a fresh question
number instead of the register it should chain from — a register name is a pointer with no content it can bind
to. So the state leaves the program: one emission per step, the VM runs it, and the value comes back as a slot
in the question (`So far: [S1: 154 = N1 / N2]`; `cubbyllm/reasoning/step_loop.py`). This cell continues the
Gate A adapter on the re-cut data (v12e_w's arithmetic as step rows + 30K TinyGSM records as step rows + the
other families as they were; the register-digit slot bug of the earlier files fixed) and generates the held-out
step rows. Training only (`--no-gen`, ~20–30 min, ~3–5 credits): disconnect when it prints `trained:`. The read
runs locally for free, the loop around the adapter and the real VM:
`python validation/exp_he19_step_loop.py --export ... --adapter <this out> --tokenizer ... --data pf_heldout_eval_w_slots.jsonl`.
Gate (pre-registered): arithmetic correct through the loop ≥ 20% on the held-out (Gate A's line, now for the
loop); the register-name failure is what this arm removes, so the per-step read at k ≥ 1 should rise well above 22%.
"""),
    code(r"""
ST_ARM   = 'v12e_w_step'
ST_DATA  = f'{DRIVE}/emitter/emitter_sft_v12e_w_tg30_step.jsonl'    # v12e_w (arithmetic re-cut) + 30K TinyGSM step rows
ST_HELD  = f'{DRIVE}/emitter/pf_heldout_eval_w_step.jsonl'         # the held-out, re-cut: 1,161 next-step rows
ST_FROM  = f'{DRIVE}/emitter/cubby450m_v12e_w_tg_cont'             # continue from the Gate A adapter (same dialect)
ST_EPOCHS = 1
for p in (ST_DATA, ST_HELD, f'{ST_FROM}/emitter_lora.safetensors'):
    assert os.path.exists(p), p
ST_STEPS = math.ceil(ST_EPOCHS * train_rows(ST_DATA) / BATCH)
print(ST_ARM, train_rows(ST_DATA), 'train rows ->', ST_STEPS, 'steps')

def run_step():
    tag = f'_{ST_ARM}_cont'
    out = f'{DRIVE}/emitter/cubby450m{tag}'
    if os.path.exists(f'{out}/emitter_lora.safetensors'):
        print('done already ->', out); return out
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--data', ST_DATA, '--out', out, '--steps', str(ST_STEPS), '--batch', str(BATCH), '--lr', str(LR),
           '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
           '--resume-adapter', ST_FROM, '--tag', tag,
           '--no-gen']            # train only: the read is the loop + the VM, run locally on the card for free
    with open(f'{out}/train.log', 'a', encoding='utf-8', buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}.* {out}/ 2>/dev/null')
    return out

ST_OUT = run_step()
"""),
    md(r"""
**The school** (`validation/train_school_torch.py`, the curriculum in `cubbyllm/reasoning/school.py`): the
Gate-A adapter is the pupil. Levels open one at a time (one operation → two steps → three-four → words and units
→ distractors → five-six → seven-ten); each problem is tried up to its level's budget, which halves as the level is
mastered (16 → 1); every try runs in the **VM** against the gold; a solve gives dopamine (reward − expected), a
wrong try is paired with the right program at its first wrong step; each round's adapter is kept only if the
held-out slice does not drop. The VM on Colab: cubelang is built here from its repo (Rust, a few minutes, once per
session) — or put a Linux `cubelang` binary at `cubbyllm/bin/cubelang` on Drive and it is used instead.
"""),
    code(r"""
BIN = f'{DRIVE}/bin/cubelang'
if os.path.exists(BIN):
    os.environ['CUBELANG_EXE'] = BIN; os.chmod(BIN, 0o755)
else:
    if not os.path.exists('/content/cubelang'):
        !git clone -q https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
    if not os.path.exists(os.path.expanduser('~/.cargo/bin/cargo')):
        !curl -sSf https://sh.rustup.rs | sh -s -- -y -q
    !cd /content/cubelang && ~/.cargo/bin/cargo build --release -q
    os.environ['CUBELANG_EXE'] = '/content/cubelang/target/release/cubelang'
!$CUBELANG_EXE --version || echo "no VM: the school needs one"

SCHOOL_CONTROL = False       # True: the same loop with every level open at once (the ordering control, gate B')
SCHOOL_ROUNDS, SCHOOL_PROBLEMS, SCHOOL_GATE_N = 8, 128, 64   # ~15 min a round on an A100; a round is resumable
SCHOOL_OUT = f'{DRIVE}/emitter/school_{TG_ARM}' + ('_shuffled' if SCHOOL_CONTROL else '')
cmd = [sys.executable, '-u', 'validation/train_school_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--adapter', TG_OUT, '--pool', TG_DATA, '--heldout', TG_HELD, TG_VAL, '--out', SCHOOL_OUT,
       '--rounds', str(SCHOOL_ROUNDS), '--problems', str(SCHOOL_PROBLEMS), '--max-tries', '16',
       '--temperature', '0.8', '--gate-n', str(SCHOOL_GATE_N), '--regress', TG_VAL, '--regress-n', '8'] \
      + (['--no-curriculum'] if SCHOOL_CONTROL else [])
p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end='')
p.wait()
"""),
    md(r"""
**The step school (2026-10-01, gate B on the write-back pupil).** The write-back adapter (`v12e_w_step`) passed
its gate: the loop alone solves 22.9% of the held-out, and with the gold state given it picks a later step right
55% of the time. That is inside the school's range, so the school now runs on STEPS
(`train_school_torch.py --step`): one gold step row is one problem, tried up to its level's budget, each candidate
run in the VM and judged by the value it computes (a value another step of the same plan computes is neutral);
dopamine, mistake pairs and shown solutions are per step, so the reward lands on the step that went wrong. The
gate each round is the loop itself on 64 held-out questions; a round that drops it, or forgets the other
families, is reverted. Self-contained: needs the setup and config cells only, plus the VM (built below, or a
Linux `cubelang` at `cubbyllm/bin/cubelang` on Drive). Gate B (pre-registered): +10 pts over 22.9% through the
loop on the full held-out (read locally afterwards, free); B′ is `STEP_SCHOOL_CONTROL = True`, the same rounds
with every level open at once — the curriculum must beat it by +3.
"""),
    code(r"""
BIN = f'{DRIVE}/bin/cubelang'
if os.path.exists(BIN):
    os.environ['CUBELANG_EXE'] = BIN; os.chmod(BIN, 0o755)
elif not os.path.exists('/content/cubelang/target/release/cubelang'):
    if not os.path.exists('/content/cubelang'):
        !git clone -q https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
    if not os.path.exists(os.path.expanduser('~/.cargo/bin/cargo')):
        !curl -sSf https://sh.rustup.rs | sh -s -- -y -q
    !cd /content/cubelang && ~/.cargo/bin/cargo build --release -q
    os.environ['CUBELANG_EXE'] = '/content/cubelang/target/release/cubelang'
else:
    os.environ['CUBELANG_EXE'] = '/content/cubelang/target/release/cubelang'
!$CUBELANG_EXE --version || echo "no VM: the school needs one"

STEP_POOL    = f'{DRIVE}/emitter/emitter_sft_v12e_w_tg30_step.jsonl'   # step rows (train split)
STEP_HELD    = f'{DRIVE}/emitter/pf_heldout_eval_w_slots.jsonl'        # whole held-out questions: the loop's gate
STEP_REGRESS = f'{DRIVE}/emitter/emitter_sft_v12e_w_slots.jsonl'       # the other families: nothing forgotten
STEP_FROM    = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'           # the pupil: the write-back adapter
for p_ in (STEP_POOL, STEP_HELD, STEP_REGRESS, f'{STEP_FROM}/emitter_lora.safetensors'):
    assert os.path.exists(p_), p_
STEP_SCHOOL_CONTROL = False   # True: gate B' (every level open at once, same budgets)
STEP_ROUNDS, STEP_PROBLEMS, STEP_GATE_N = 8, 192, 64
STEP_OUT = f'{DRIVE}/emitter/school_step' + ('_shuffled' if STEP_SCHOOL_CONTROL else '')
cmd = [sys.executable, '-u', 'validation/train_school_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--adapter', STEP_FROM, '--step', '--pool', STEP_POOL, '--heldout', STEP_HELD, '--out', STEP_OUT,
       '--rounds', str(STEP_ROUNDS), '--problems', str(STEP_PROBLEMS), '--max-tries', '16',
       '--temperature', '0.8', '--gate-n', str(STEP_GATE_N), '--regress', STEP_REGRESS, '--regress-n', '8'] \
      + (['--no-curriculum'] if STEP_SCHOOL_CONTROL else [])
p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end='')
p.wait()
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
