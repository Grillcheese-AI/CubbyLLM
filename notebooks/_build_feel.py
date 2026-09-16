"""_build_feel.py — generate notebooks/standin_feel_sft.ipynb.

The notebook is the artefact; this writes it, the way `_build_gen3.py` and
`_build_v14e.py` write theirs. Run:  python notebooks/_build_feel.py
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "standin_feel_sft.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


md(r"""# [stand-in] the **feel** family — say it, don't name it

**What this is.** A talk-adapter SFT family for one narrow skill: turning a *body state* into a
sentence that sounds like somebody in that state, without naming the state. It exists because of a
specific, repeated failure in live runs — the model is handed a percept record and reads the field
names back:

> *"I am thinking about my body and what it can take, so I will try to survive this maze at all cost."*

That is a model with no idea what speaking from a body looks like, describing the fields instead.
Nothing at runtime fixes it: the record is already correct, the guard already passes it. It is a
**form** problem, and form is the one thing a corpus is for. (WO-2.13's framing: the corpus carries
the form, the map carries the content — so this is not a capability retrain and the no-retraining
principle holds.)

## Why GoEmotions is here, and what it is NOT used for

`google-research-datasets/go_emotions` — Apache 2.0, 58k curated Reddit comments, 27 emotions +
neutral. The repo already uses it for **recognition** (`build_chat_sft.py`, task `emotion`: given a
message, name the emotion and its Plutchik petal). This family is the **inverse** — expression — and
reuses that file and that taxonomy rather than fetching a second copy of either.

It is deliberately **not** used as the training targets, and the reason is measured rather than
assumed. Filtering the simplified train split to single-label, first-person, no second-person, no
`[NAME]`, under 90 characters, across the 12 emotions a solitary agent can even have:

| emotion | single-label | usable |
|---|---:|---:|
| joy | 853 | 251 |
| sadness | 817 | 232 |
| annoyance | 1451 | 224 |
| confusion | 858 | 215 |
| disappointment | 709 | 173 |
| fear | 430 | 110 |
| nervousness | 85 | 37 |
| relief | 88 | 31 |
| pride | 51 | 25 |
| **total (12 labels)** | **7,827** | **1,755** |

Two things make those the wrong targets:

1. **The register is Reddit reacting to people.** *"WHY THE FUCK IS BAYLESS ISOING"* (anger),
   *"Dirty Southern Wankers"* (annoyance), *"Fucking coward."* (anger). Cubby-man is alone in a maze
   reporting his own situation; trained on this he would swear at nobody, against his own voice rules.
2. **The usable ones mostly NAME the emotion** — *"I'm so proud to be british"*, *"makes me
   nervous"*, *"I was worried about that too"* — which is the exact failure being fixed.

Roughly half of the 27 labels are social (admiration, gratitude, remorse, embarrassment, love,
caring, approval) with no referent for a solitary agent. That is the same conclusion `pacman.py`
reached independently with `_SOCIAL_CORNERS`.

**So GoEmotions supplies the taxonomy and a style reference, and the targets are written in his own
register from his own runs.** `GOEMOTIONS_PETAL` (already in `build_chat_sft.py`) maps all 28 labels
onto the Plutchik petals the compass already draws, which is the bridge between the two.

## Where the inputs come from

- **The situations** are harvested from a real headless `CubbyGhost` run — actual percept records,
  actual hormone states, his vocabulary, his maze. Not invented scenes.
- **The targets** are written by an LLM at dataset-build time only (`validation/.env`, never printed,
  never committed) — *"the llm is only to build the dataset"*. Nothing here runs at inference.
- **Every target is filtered** by the same `grounded_ok` guard the live loop uses, plus an
  anti-naming check and the copy check, so a pair that would be refused in production never becomes
  a training example.

**Tag every number this notebook prints as `[stand-in]`.** None of it is a CubbyLLM result.""")

code(r"""# --- setup (run once per session) ---
import os, json, time, random, re, sys
!pip -q install unsloth trl datasets pandas pyarrow
from google.colab import drive; drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/cubbyllm/standin'

# the repo checkout (loud: a failed clone/pull used to hide behind -q and surface much later
# as a missing import, which is exactly the far-from-the-cause failure PATH.md S6.10 is about)
if not os.path.exists('/content/CubbyLLM/.git'):
    !rm -rf /content/CubbyLLM && git clone https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull --ff-only
assert os.path.exists('/content/CubbyLLM/standin/pacman.py'), 'checkout failed - stop here, do not carry on'

REPO = '/content/CubbyLLM'
for p in (REPO, f'{REPO}/standin', f'{REPO}/standin/data'):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED = 7
rng = random.Random(SEED)
DATA = f'{DRIVE}/data/feel_sft_v1.jsonl'
os.makedirs(os.path.dirname(DATA), exist_ok=True)
print('[stand-in] repo at', REPO, '| dataset ->', DATA)""")

code(r"""# --- GoEmotions: fetch once, filter as measured, keep as a STYLE REFERENCE ---
# Reuses the parquet build_chat_sft.py already expects (HF_DIR/goemotions_train.parquet) and
# downloads it only if absent. Same file, same taxonomy, opposite direction.
import pandas as pd

GOE_URL = ('https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/'
           'refs%2Fconvert%2Fparquet/simplified/train/0000.parquet')
GOE_LOCAL = f'{DRIVE}/data/hf/goemotions_train.parquet'
os.makedirs(os.path.dirname(GOE_LOCAL), exist_ok=True)
if not os.path.exists(GOE_LOCAL):
    print('[stand-in] fetching GoEmotions (apache-2.0) ...')
    pd.read_parquet(GOE_URL).to_parquet(GOE_LOCAL)
goe = pd.read_parquet(GOE_LOCAL)

from build_chat_sft import GOEMOTIONS                    # the 28 labels, already in the repo
from goemotions_cube import style_bank_key, UNPLACED, validate
# the label -> corner mapping is DERIVED from GOEMOTIONS_PETAL and _PETAL and
# checked against the agent's own nearest-corner classifier. A hand-written
# coordinate table is a third opinion about one space, and this project has
# been bitten twice by two opinions disagreeing in the band between them.
assert [x for x in validate() if not x.startswith('note:')] == [], 'mapping does not survive its own classifier'
print('[stand-in] labels with no corner (anticipation has no Lovheim vertex):', UNPLACED)

# the emotions a SOLITARY agent can have. The rest of the 27 are social -- admiration, gratitude,
# remorse, embarrassment, love, caring, approval -- and have no referent in an empty maze, which is
# the same call pacman.py's _SOCIAL_CORNERS makes.
SOLO = ('fear', 'nervousness', 'excitement', 'relief', 'pride', 'disappointment',
        'sadness', 'annoyance', 'confusion', 'realization', 'curiosity', 'joy')

_FIRST  = re.compile(r"\b(i|i'm|im|my|me|myself|i've|i'll)\b", re.I)
_SECOND = re.compile(r"\b(you|your|you're|u|ur)\b", re.I)

def _usable(t: str) -> bool:
    return bool(_FIRST.search(t)) and not _SECOND.search(t) and '[NAME]' not in t and len(t) < 90

STYLE = {}          # plutchik petal -> [sentences], the reference for HOW feeling shows in words
rows = []
for name in SOLO:
    i = GOEMOTIONS.index(name)
    sub = goe[goe.labels.apply(lambda L: list(L) == [i])]
    use = sub[sub.text.apply(_usable)]
    rows.append((name, len(sub), len(use)))
    key = style_bank_key(name)
    if key:
        STYLE.setdefault(key, []).extend(use.text.tolist())

print(f"[stand-in] {'emotion':<16}{'single-label':>13}{'usable':>9}")
for n, s, u in rows:
    print(f"[stand-in] {n:<16}{s:>13}{u:>9}")
print(f"[stand-in] {'TOTAL':<16}{sum(r[1] for r in rows):>13}{sum(r[2] for r in rows):>9}")
print('[stand-in] style banks by CORNER:', {k: len(v) for k, v in sorted(STYLE.items())})""")

code(r"""# --- the situations: harvested from a REAL headless run, not invented ---
# His register, his maze, his vocabulary. `blank=True` so nothing is inherited from disk.
#
# The record is captured by wrapping `percepts`, which is the exact object the live loop hands the
# model, built at the exact moment it is built. Reading `man._last` after the step instead looked
# equivalent and was not: `_last` is the event dict (eaten/caught/failed), it has no `about` key, so
# every situation fell through to 'idle' and 600 near-identical "moving on" records came out. A
# corpus of one repeated scene is the thing exp_r37 measured models reciting.
from collections import Counter
from neurochem import Neurochemistry
from pacman import CubbyGhost, GhostVerse

def harvest(steps=700, seed=0, ghost_free=1, fallers_from=1):
    env = GhostVerse(level=1, ghost_free_levels=ghost_free, fallers_from_level=fallers_from)
    man = CubbyGhost(env, seed=seed, probe=0.0, memory=None, blank=True)
    man.chem = Neurochemistry()
    man.verbalize = False            # we want the RECORD, not a sentence from the stand-in model

    grabbed, _percepts = [], man.percepts
    def spy(kind, **d):
        rec = _percepts(kind, **d)
        body, names = man.chem.body(2), man.chem.could_be()
        if body or names:            # nothing to feel is not a training case
            grabbed.append({
                'record': man.render_percepts(rec),       # the guard's referent
                'facts': man.say_flatly(rec),             # what he may actually say about
                'body': ', '.join(body),
                'could_be': names,
                'petal': man.emotion()['name'],
                'corner': man.chem.corner_position()['corner'],
                'about': rec.get('about'),
                'manner': man.chem.manner(),
                'state': {k: round(v, 3) for k, v in man.chem.to_dict().items()
                          if isinstance(v, (int, float))},
            })
        return rec
    man.percepts = spy
    for _ in range(steps):
        man.step()
    return grabbed

SITUATIONS = []
for s in range(4):
    SITUATIONS += harvest(steps=700, seed=s)
# de-duplicate on the flat facts: a maze repeats itself, and a repeated scene is the one a model recites
_seen = set()
SITUATIONS = [x for x in SITUATIONS
              if not (x['facts'] in _seen or _seen.add(x['facts']))]
rng.shuffle(SITUATIONS)
print('[stand-in] situations harvested:', len(SITUATIONS))
print('[stand-in] by compass petal:', Counter(x['petal'] for x in SITUATIONS).most_common())
print('[stand-in] by event:', Counter(x['about'] for x in SITUATIONS).most_common(10))
# If one event or one petal is most of the set, STOP and widen the run before generating targets --
# a lopsided corpus teaches the majority scene and nothing else.
print('\n[stand-in] one of them:\n ', json.dumps(SITUATIONS[0], indent=2)[:700])""")

code(r'''# --- the targets: written at BUILD TIME by an external model, never at inference ---
# Owner's standing rule: "the llm is only to build the dataset". The key lives in the gitignored
# validation/.env and is never printed, logged, or written into a record.
import urllib.request

def _api_key():
    for line in open(f'{REPO}/validation/.env', encoding='utf-8'):
        if line.startswith('OPENROUTER_API_KEY='):
            return line.split('=', 1)[1].strip()
    raise SystemExit('no OPENROUTER_API_KEY in validation/.env')

BUILDER_MODEL = 'anthropic/claude-3.5-sonnet'      # dataset construction only
KEY = _api_key()

RULES = """Write ONE short sentence, first person, as this character speaking to himself.

HARD RULES
- Never use the words: body, record, situation, percept, mission, "could be".
- Do NOT state the emotion word itself. Someone rattled does not say "I am rattled" - they say
  what they notice and it comes out rattled. At most 1 in 5 sentences may name a feeling.
- Only mention things listed under FACTS. Invent nothing - no places, numbers or objects that
  are not there.
- Never address anyone. He is alone. No "you", no questions to a reader.
- Under 18 words. Plain words. No metaphor stacking, no exclamation marks."""

def _ask(prompt: str, temperature=0.9) -> str:
    body = json.dumps({'model': BUILDER_MODEL, 'temperature': temperature,
                       'max_tokens': 60,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=body,
                                 headers={'Authorization': f'Bearer {KEY}',
                                          'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['choices'][0]['message']['content'].strip()

def build_prompt(sit: dict) -> str:
    # the style bank is a REFERENCE for how feeling shows in words -- shown, never copied, and
    # explicitly marked as the wrong register so the model takes the manner and not the content
    bank = STYLE.get(sit['corner']) or []
    ref = '\n'.join(f'  - {t}' % () for t in rng.sample(bank, min(4, len(bank)))) if bank else '  (none)'
    names = ' or '.join(sit['could_be']) if sit['could_be'] else '(nothing strong)'
    return f"""A character is alone in a maze. This is how his body feels and what is around him.

HOW HIS BODY FEELS: {sit['body'] or '(nothing notable)'}
WHAT HE MIGHT CALL IT (pick at most one, and prefer not to say it): {names}
FACTS HE CAN SEE RIGHT NOW: {sit['facts']}
DELIVERY: {sit['manner'] or 'plain'}

For REFERENCE ONLY - how people let a feeling show without announcing it. Different setting,
different voice; take the manner, never the content or the topic:
{ref}

{RULES}"""

N_TARGETS = 600          # start here; the guards below throw a chunk of it away
raw_pairs = []
t0 = time.time()
for i, sit in enumerate(SITUATIONS[:N_TARGETS]):
    try:
        raw_pairs.append((sit, _ask(build_prompt(sit))))
    except Exception as e:
        print(f'[stand-in] {i}: {type(e).__name__} {str(e)[:80]}')
    if (i + 1) % 50 == 0:
        print(f'[stand-in] {i+1}/{N_TARGETS}  {time.time()-t0:.0f}s')
print('[stand-in] generated:', len(raw_pairs))''')

code(r"""# --- the guards: a pair that production would refuse must never become a training example ---
from pacman import CubbyGhost, GhostVerse, grounded_ok, longest_run
_probe = CubbyGhost(GhostVerse(), probe=0.0, memory=None, blank=True)
FACTS = {}          # grounded_ok's fact dict; empty is the strict setting

_SCAFFOLD = re.compile(r'\b(body|record|situation|percept|mission|could be)\b', re.I)
_SECOND_P = re.compile(r'\b(you|your|you\'re)\b', re.I)

def named_feeling(text: str, names: list[str]) -> bool:
    return any(re.search(rf'\b{re.escape(n.split()[-1])}\b', text, re.I) for n in names)

kept, why = [], Counter()
for sit, text in raw_pairs:
    t = ' '.join(text.replace('"', '').split())
    if not t or len(t.split()) > 20:                 why['length'] += 1;    continue
    if _SCAFFOLD.search(t):                          why['scaffold'] += 1;  continue
    if _SECOND_P.search(t) or '?' in t:              why['addressed'] += 1; continue
    if longest_run(sit['facts'], t) > 6:             why['copied'] += 1;    continue
    if not grounded_ok(sit['record'], t, FACTS, ''): why['ungrounded'] += 1; continue
    kept.append((sit, t, named_feeling(t, sit['could_be'])))

named = sum(1 for _, _, n in kept if n)
print('[stand-in] kept', len(kept), 'of', len(raw_pairs), f'({100*len(kept)/max(1,len(raw_pairs)):.0f}%)')
print('[stand-in] refused:', why.most_common())
print(f'[stand-in] names the feeling: {named}/{len(kept)} ({100*named/max(1,len(kept)):.0f}%) - target is under 20%')

# Cap the naming rate rather than trusting the instruction. A corpus that names the feeling most of
# the time teaches exactly the habit this family exists to remove.
CAP = 0.20
names_allowed = int(CAP * len(kept))
final, used = [], 0
for sit, t, is_named in kept:
    if is_named:
        if used >= names_allowed:
            continue
        used += 1
    final.append((sit, t))
print('[stand-in] after capping the naming rate:', len(final))
print('\n[stand-in] samples:')
for sit, t in final[:8]:
    print(f"  [{sit['petal']:<12}] body: {sit['body'][:42]:<42} -> {t}")""")

code(r"""# --- write the JSONL in the shape the talk builder/notebook already reads ---
from pacman import CubbyGhost
SPEAK_RULES = ('Answer in one short sentence, first person, without repeating the list back. '
               'Do not invent anything you do not perceive.')

def to_record(i, sit, target):
    prompt = (f"Situation: {sit['record']}.\n"
              f"What are you thinking? {SPEAK_RULES}"
              + (f" Your delivery: {sit['manner']}." if sit['manner'] else ''))
    return {'id': f"feel:{sit['petal']}:{i}", 'task': 'feel', 'subtype': sit['petal'],
            'source': 'cubbyman_run+goemotions_style', 'prompt': prompt, 'program': target,
            'gold': None, 'system': None, 'state': sit['state'], 'lang': 'en',
            'vm_ok': None, 'gold_match': None}

recs = [to_record(i, s, t) for i, (s, t) in enumerate(final)]
rng.shuffle(recs)
with open(DATA, 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('[stand-in] wrote', len(recs), '->', DATA)
print('[stand-in] by petal:', Counter(r['subtype'] for r in recs).most_common())
print('\n[stand-in] one record:\n', json.dumps(recs[0], indent=2)[:800])

# A HELD-OUT SPLIT BY SITUATION, not by row. Two records from the same maze scene in different
# splits is the leak that makes an eval look better than the model is (PATH.md S6.8).
holdout = {r['prompt'] for r in recs[:max(1, len(recs)//10)]}
train = [r for r in recs if r['prompt'] not in holdout]
val   = [r for r in recs if r['prompt'] in holdout]
print(f'[stand-in] train {len(train)} | val {len(val)}')""")

code(r"""# --- LoRA + SFT: the v8t talk recipe, unchanged (alpha 64, LR 1e-4, 10% warmup, 8-bit AdamW) ---
from unsloth import FastLanguageModel
BASE = 'LiquidAI/LFM2.5-1.2B-Instruct'      # whatever the current talk base is
MAX_SEQ = 2048
model, PROC = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_SEQ,
                                                load_in_4bit=True, dtype=None)

def chat(r):
    msgs = ([{'role': 'system', 'content': r['system']}] if r.get('system') else []) + \
           [{'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': r['program']}]
    return {'text': PROC.apply_chat_template(msgs, tokenize=False)}

from datasets import Dataset
ds_train = Dataset.from_list([chat(r) for r in train])
ds_val   = Dataset.from_list([chat(r) for r in val])

from trl import SFTTrainer, SFTConfig
model = FastLanguageModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.0, bias='none',
                                         use_gradient_checkpointing='unsloth', random_state=SEED)
trainer = SFTTrainer(model=model, tokenizer=PROC, train_dataset=ds_train, eval_dataset=ds_val,
                     args=SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8,
                                    warmup_ratio=0.10, num_train_epochs=2, learning_rate=1e-4,
                                    optim='adamw_8bit', lr_scheduler_type='cosine', logging_steps=10,
                                    max_seq_length=MAX_SEQ, seed=SEED, output_dir='/content/out',
                                    report_to='none'))
trainer.train()""")

code(r"""# --- export FIRST, then evaluate (a lost adapter costs the whole run) ---
OUT = f'{DRIVE}/models/feel_v1'
model.save_pretrained_merged(f'{OUT}/merged', PROC, save_method='merged_16bit')
model.save_pretrained_gguf(f'{OUT}/gguf', PROC, quantization_method='q4_k_m')
print('[stand-in] exported ->', OUT)""")

code(r"""# --- the read: does it SAY it instead of NAMING it, on held-out situations? ---
# The bar is not a loss number. Three things, each of which the live loop actually cares about:
#   GROUNDED   the production guard accepts the sentence (it would have reached the page)
#   UNNAMED    it does not just say the candidate word back
#   NOT-A-COPY it is not the record with the punctuation moved (longest_run <= 6)
FastLanguageModel.for_inference(model)

def say(prompt):
    msgs = [{'role': 'user', 'content': prompt}]
    ids = PROC.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors='pt').to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=40, temperature=0.85, do_sample=True)
    return PROC.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

by_sit = {r['prompt']: r for r in val}
ok = Counter()
print('[stand-in] held-out samples:\n')
for i, r in enumerate(val[:40]):
    t = ' '.join(say(r['prompt']).replace('"', '').split())
    sit_rec = r['prompt'].split('Situation: ', 1)[1].rsplit('.\n', 1)[0]
    g = grounded_ok(sit_rec, t, FACTS, '')
    c = longest_run(sit_rec, t) <= 6
    n = not _SCAFFOLD.search(t)
    ok['grounded'] += g; ok['not_a_copy'] += c; ok['no_scaffold'] += n; ok['n'] += 1
    if i < 12:
        print(f"  {'OK ' if (g and c and n) else 'no '} {t}")
n = max(1, ok['n'])
print(f"\n[stand-in] grounded {ok['grounded']}/{n} | not-a-copy {ok['not_a_copy']}/{n} | "
      f"no-scaffold {ok['no_scaffold']}/{n}")""")

md(r"""### How to read this, and what it does not prove

**The comparison that matters is against the incumbent talk adapter on the SAME held-out
situations**, not against nothing. The live keep rate before this family is roughly 45% (exp_r37),
and most refusals are the model reciting the record. A `feel` adapter is worth keeping only if it
raises the grounded rate *and* lowers the naming rate — either alone is easy to get by cheating.
Raising the grounded rate by saying less is not an improvement, so watch sentence length too.

**What this family cannot fix.** It teaches form: how a body-state turns into words. It does not
make the state correct. Every affect bug found on 2026-09-15 — the saturated ODE, the craving
clipped to zero, the compass pinned to "warm" — was a *content* bug, and a better-spoken model on
top of a wrong state is a more convincing wrong answer, which is worse than an obvious one. Run
`validation/exp_r39_stakes.py` and the isolated escalation curve first; if the hormones are wrong,
fix them before training on their output.

**The targets are synthetic and that is a real limitation.** They are written by an external model
at build time from real situations and real states, filtered by the production guard — but they are
still one model's idea of how a maze-dweller talks. The honest upgrade path is already collecting
itself: `CubbyGhost.experiences` logs every live attempt with the state, the modulation it was said
under, and `kept` — whether the guard accepted it. A few thousand live steps gives targets that are
his own, with a measured outcome attached rather than an assumed one. That is the half GrillCheese
hardcoded to `quality=0.7` and lost; do not repeat it here.

**GoEmotions' role, restated so it does not drift.** Taxonomy and style reference only. Its text is
never a target, because its register is wrong and its usable rows mostly name the emotion. If a
later pass wants more from it, the strong use is as an *instrument* — a classifier that scores
whether a produced sentence reads as the emotion the state implied. That is a measurement, it needs
no training run of ours, and measurement is what every real finding on this project has come from.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
