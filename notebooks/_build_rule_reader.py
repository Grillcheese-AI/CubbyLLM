"""Generate rule_reader.ipynb: order 0067, step 3's adapter round (Nick, 2026-11-12: "yes"). A LoRA adapter on the
frozen base450m_cont that reads an instruction into a rule plan in the reader's form (`cubbyllm/speech/rule_reader.py`),
trained with the talk trainer (`validation/train_talk_torch.py`: loss on the answer only) on
`validation/sft/build_rule_reader_sft.py`'s set (seen kinds only; the new-wording rows are not in it). Cowork runs it;
the read is at home on grilly2 (`validation/exp_rule_reader.py`, then `exp_rule_agree.py`). Re-run after editing:
    python notebooks/_build_rule_reader.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "rule_reader.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0067: the rule reader's adapter, round 6

Agreement with IFEval's checker on new wordings: round 1 82.3%, round 2 92.7%, round 3 91.7%; rounds 2-3's misses are
mostly "less than", kept out of training, read as ">=". Round 4 (Nick: everyday comparisons allowed) trains with "at
least" / "less than" and is read on a fresh new-wording split whose comparisons training never had. Nick folded in his
annotated instruction file: natural length rules from real prompts (the host's grammar reads the rule) and rule-free
prompts, which teach the reader to answer `none`. Round 5 (round 4 held out: on unseen kinds it never abstained)
adds rows whose answer is `asked` (tone, style, properties no measurement covers). Round 6 (round 5 on IFBench: on a
whole prompt the reader says `none` or invents a count) adds real people's prompts (OASST1, Dolly-15k), alone, with a
rule put inside them, or with a phrase no measurement covers; and longer records (`MAX_LEN` 640).

**Runtime:** an A100 (22,458 training records, most short, the longest 546 tokens of prompt; one epoch is about 702
steps of 32, minutes rather than hours). It reads, from Drive: `runs/base450m_cont/base450m_cont_final.pt` (the base
every served adapter is on), the token cache's tokenizer, and `rules/rule_reader_sft_v6.jsonl` with its manifest. It
writes `rules/cubby450m_rules_v6/` (`talk_lora.safetensors` + `talk_lora.json`, the talk trainer's file names; the trunk
loads it as the `rules` role). Training only: the read runs at home. **Disconnect when it prints `trained:`.**
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
DATA      = f'{DRIVE}/rules/rule_reader_sft_v6.jsonl'      # round 6 (5: _v5; 4: _v4; ...; 1: rule_reader_sft.jsonl)
OUT       = f'{DRIVE}/rules/cubby450m_rules_v6'          # earlier rounds' adapters are not touched
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], '| new-wording rows left out:', man['new_wording_rows_left_out'],
      '| slot boundaries tokenized otherwise:', man['slot_boundaries_tokenized_otherwise'])
assert man['slot_boundaries_tokenized_otherwise'] == 0
assert not man.get('records_over_max_len'), f"records longer than MAX_LEN would be skipped: {man['records_over_max_len']}"
assert 'IFEval' in man['benchmarks_checked'] and man['records_sharing_5_words_with_a_benchmark'] == 0, 'a benchmark prompt in the set'
print('benchmarks checked:', man['benchmarks_checked'], '| records sharing 5 words with one:', man['records_sharing_5_words_with_a_benchmark'])
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md(r"""
## The settings

The served adapters' LoRA layout (rank 16, alpha 32, every projection of every layer: the trunk refuses an adapter
with another), lr 2e-4 with 50 warmup steps then cosine, batch 32, max length 640 (round 6's long prompts; round 5: 160), one
epoch, seed 0. On a card under 70 GB a batch is cut into chunks of at most 4,096 padded tokens (`--micro-tokens`: the
same gradient; round 6's first try ran a 22 GB L4 out of memory on a batch of 32 long prompts). The `held`
split (the dev rows: the same wordings as train, other rules) is the held-out answer loss printed every 100 steps.
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN = 1, 32, 2e-4, 16, 32, 640   # round 6: long prompts (5: 160)
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_rules_v6'
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

**Disconnect the runtime.** At home, Claude Code reads the adapter as the `rules` role on grilly2: `exp_rule_reader.py`
on the dev rows (the instruction alone, no examples), then the new-wording rows once, then `exp_rule_agree.py` on its
plans against IFEval's checker (line 3: agreement at least 98% on new wordings). The held-out kinds are read once,
only if that line is met.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
