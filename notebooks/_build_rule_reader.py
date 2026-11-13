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
# 0067: the rule reader's adapter, round 3

Round 1 (3 wordings a kind): 82.3% agreement with IFEval's checker on new wordings. Round 2 (10 a kind): 92.7%, its
misses in all capitals and in comparisons worded as training never had. Round 3 adds wordings for those (capitals and
lowercase said without the word, more comparison phrasings) and is read on a fresh new-wording split.

**Runtime:** an A100 (the set is 9,977 short training records, at most 89 tokens each; one epoch is about 312 steps
of 32, minutes rather than hours). It reads, from Drive: `runs/base450m_cont/base450m_cont_final.pt` (the base every
served adapter is on), the token cache's tokenizer, and `rules/rule_reader_sft_v3.jsonl` with its manifest. It writes
`rules/cubby450m_rules_v3/` (`talk_lora.safetensors` + `talk_lora.json`, the talk trainer's file names; the trunk
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
DATA      = f'{DRIVE}/rules/rule_reader_sft_v3.jsonl'      # round 3 (2: _v2; 1: rule_reader_sft.jsonl)
OUT       = f'{DRIVE}/rules/cubby450m_rules_v3'          # rounds 1 and 2's adapters are not touched
for p in (CKPT, TOKENIZER, DATA, DATA + '.manifest.json'):
    assert os.path.exists(p), p
man = json.load(open(DATA + '.manifest.json'))
print(man['records'], 'records', man['by_split'], '| new-wording rows left out:', man['new_wording_rows_left_out'],
      '| slot boundaries tokenized otherwise:', man['slot_boundaries_tokenized_otherwise'])
assert man['slot_boundaries_tokenized_otherwise'] == 0
assert 'IFEval' in man['benchmarks_checked'] and man['records_sharing_5_words_with_a_benchmark'] == 0, 'a benchmark prompt in the set'
print('benchmarks checked:', man['benchmarks_checked'], '| records sharing 5 words with one:', man['records_sharing_5_words_with_a_benchmark'])
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md(r"""
## The settings

The served adapters' LoRA layout (rank 16, alpha 32, every projection of every layer: the trunk refuses an adapter
with another), lr 2e-4 with 50 warmup steps then cosine, batch 32, max length 160, one epoch, seed 0. The `held`
split (the dev rows: the same wordings as train, other rules) is the held-out answer loss printed every 100 steps.
"""),
    code(r"""
EPOCHS, BATCH, LR, RANK, ALPHA, MAX_LEN = 1, 32, 2e-4, 16, 32, 160
os.environ['CB_NO_FLEX'] = '1'                # the reference attention path; the trainer does not compile flex
assert not os.path.exists(f'{OUT}/DONE'), f'{OUT} is already trained: move it aside to train again'
os.makedirs(OUT, exist_ok=True)
tag = '_rules_v3'
cmd = [sys.executable, '-u', 'validation/train_talk_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--data', DATA, '--out', OUT, '--epochs', str(EPOCHS), '--batch', str(BATCH), '--lr', str(LR),
       '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', '0', '--tag', tag]
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
