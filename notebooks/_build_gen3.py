"""Generate standin_gen3_masked_sft.ipynb from the gen-2 notebook: the same v8e/v12e recipe on
gen 3 (`emitter_sft_v13e.jsonl`, gen 2 unchanged + the reverse-built free-text records), with ONE
switch -- `STANDIN_ARM=masked` trains on the assistant response only (Unsloth's
`train_on_responses_only`), `STANDIN_ARM=full` on the whole chat text as v8e/v12e did. Same data,
same seed, same everything else, so the A/B is attributable. Re-run after editing:
    python notebooks/_build_gen3.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
SRC = HERE / "standin_gen2_emitter_sft.ipynb"
OUT = HERE / "standin_gen3_masked_sft.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def set_cell(nb, i, text):
    nb["cells"][i]["source"] = _src(text)


def replace_in_cell(nb, i, old, new, count=1):
    src = "".join(nb["cells"][i]["source"])
    assert src.count(old) >= 1, f"cell {i}: {old!r} not found"
    nb["cells"][i]["source"] = _src(src.replace(old, new, count))


INTRO = r"""# [stand-in] CubeLang emitter — **gen 3** (v13e): the reverse-built free-text set, and the masking A/B

**What this is.** The v8e/v12e Unsloth LoRA SFT (LFM2.5-2.6B, r=32, 2 epochs, batch 32, lr 2e-4 cosine, MAX_SEQ
4096, `<think>` trained empty) on `emitter_sft_v13e.jsonl` = **gen 2 unchanged + the gen-3 records** that
`standin/data/build_gen3.py` built the reverse way (2026-09-13): a chain the store certifies first (the
encyclopedia's frame-read facts with their sentences, a seeded slice of the wiki world, two-hop chains off
those), a free-text wording of it second (`When was X born?`, `On what day, month, and year was X born?`,
`Where was X born?`, `Who is X married to?`, `When was the father of X born?`), and the plan the emitter must
learn is the **host's** -- the seed and the question's own relation words -- admitted only when the serving
gate itself certified it (coverage, typed answer class, resident VM, the VM's answer equal to the chain's
object). No LLM wrote any of it. A `plan` record (question alone → CotPlan) and a `chain` record (question +
Facts → the CotChain the walk built) per certified question, provenance on each.

**The A/B — one switch, `STANDIN_ARM`:**

- `masked` (default): the loss is taken on the **assistant response only** (Unsloth `train_on_responses_only`);
  the system prompt and the question are context, not targets. Nick's ask (2026-09-12): SFT with masking for gen 3.
- `full`: the v8e/v12e recipe as it was -- loss on the whole chat text.

Same data, same seed, same LoRA, same schedule; the adapter, merged model and GGUF land in an arm-named folder.
The read is not this notebook's exact-match smoke test; it is the gate on your machine: exp_r11 on the same 600
SimpleQA the earlier emitters ran (gen 2: 6 verified / 5 correct / 1 near / 0 wrong), and exp_r17 on the gen-3
**held** split (entities the training records never saw), scored verified / correct / **wrong** -- the last must
stay at zero, or the masking bought confidence without truth.
"""

HOW_TO_READ = r"""### How to read

- **This notebook proves nothing about the gate.** The smoke eval is a format read. The gate is on your machine:
  `python validation/exp_r11_search_learn.py --wikidata --n 600 --seed 7 --lexicon --gguf standin/models/emitter_v13e_<arm>.Q4_K_M.gguf --tag _gen3_<arm>`
  (the same 600 SimpleQA as gen 2's `_local` run: 6 verified, 5 correct, 1 near, 0 wrong, 0 API calls), and
  `python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v13e_<arm>.Q4_K_M.gguf --tag _<arm>`
  (the gen-3 **held** split: free-text wordings over entities no training record used).
- **What to compare, masked vs full, on the same questions:** plans the disposer accepts; VM-verified and
  correct; and the honest one, **wrong**, which must not rise. Then gen 2 → gen 3 on SimpleQA: more verified at
  0 wrong is the free-text hole closing; more verified with wrong rising is the loop teaching confident wrong
  plans, and the disposer -- not the emitter -- is what to look at next.
- **Everything here is `[stand-in]`.** It goes in `standin/README.md` and the research doc, never in
  `CUBBYLLM_HYPOTHESES.md` except as a pointer.
"""


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    set_cell(nb, 0, INTRO)
    # --- setup: version, arm, output folder, manifest check ---
    replace_in_cell(nb, 1, "VERSION = os.environ.get('STANDIN_VERSION', 'v12e')    # gen 2 of the program emitter; v8e was gen 1",
                    "VERSION = os.environ.get('STANDIN_VERSION', 'v13e')    # gen 3 of the program emitter: gen 2 + the reverse-built free-text set\n"
                    "ARM = os.environ.get('STANDIN_ARM', 'masked')          # 'masked': loss on the assistant response only | 'full': the v8e/v12e recipe\n"
                    "assert ARM in ('masked', 'full'), ARM")
    replace_in_cell(nb, 1, "OUT = OUT + MODEL_TAG", "OUT = OUT + MODEL_TAG + '_' + ARM\nprint('arm:', ARM, '| out:', OUT)")
    replace_in_cell(nb, 1, "assert 'r7_chains' in m and 'plan_records' in m, 'this is not a gen-2 manifest (build_gen2_partition.py writes r7_chains / plan_records)'\n"
                    "print('manifest', m['version'], ':', m['by_task'], '| records', m['n_records'], '| train rows after repeat', m['train_rows_after_repeat'])\n"
                    "print('gen 1 =', os.path.basename(m['gen1']), 'sha', m['gen1_sha256'][:12], '| r7 chains', m['r7_chains'], f\"(x{m['r7_mult']})\",\n"
                    "      '| r7 gold-correct', m['r7_gold_correct'], 'gold-wrong', m['r7_gold_wrong'], '(reported, not filtered)',\n"
                    "      '| plan records', m['plan_records'], '| excluded from the gate', m['n_excluded_questions'])",
                    "assert m.get('version') == 'v13e' and 'gen3' in m, 'this is not a gen-3 manifest (build_gen3.py --merge writes emitter_sft_v13e.manifest.json)'\n"
                    "print('manifest', m['version'], ':', m['by_task'], '| records', m['n_records'], '| train rows after repeat', m['train_rows_after_repeat'])\n"
                    "print('by source:', m['by_source'])")
    # --- data: the held split is the val split; gen-3 records enter on the gate's word ---
    replace_in_cell(nb, 2, "R7 = 'cubbyllm/cot_harvest_r7'\n"
                    "recs = [r for r in recs if r.get('vm_ok') in (True, None) and (r.get('gold_match') is not False or r.get('source') == R7)]\n"
                    "train = [r for r in recs if r['split'] == 'train']; val = [r for r in recs if r['split'] == 'val']",
                    "R7 = 'cubbyllm/cot_harvest_r7'\n"
                    "recs = [r for r in recs if r.get('vm_ok') in (True, None) and (r.get('gold_match') is not False or r.get('source') == R7)]\n"
                    "# gen 3's records carry the gate's verdict (vm_ok True on the chain record, gold_match True: the VM's answer IS the chain's\n"
                    "# object); their 'held' split is entities no training record uses -- the val split here, and exp_r17's held-out set\n"
                    "train = [r for r in recs if r['split'] == 'train']; val = [r for r in recs if r['split'] in ('val', 'held')]\n"
                    "print('by source (train):', Counter(r.get('source') for r in train).most_common())")
    replace_in_cell(nb, 2, "from collections import Counter\nprint('train'", "print('train'")
    replace_in_cell(nb, 2, "SYSTEM = EMITTER_SYSTEM\n", "SYSTEM = EMITTER_SYSTEM\nfrom collections import Counter\n")
    # --- the switch: masked vs full ---
    replace_in_cell(nb, 3, "    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds_train, args=cfg)\n",
                    "    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds_train, args=cfg)\n"
                    "    if ARM == 'masked':\n"
                    "        # loss on the assistant response only: the chat template's user / assistant headers, read off a\n"
                    "        # rendered probe rather than assumed (LFM2.5 is ChatML-shaped: <|im_start|>user ... <|im_start|>assistant)\n"
                    "        from unsloth.chat_templates import train_on_responses_only\n"
                    "        probe = tokenizer.apply_chat_template([{'role': 'user', 'content': 'XQX'}, {'role': 'assistant', 'content': 'YQY'}], tokenize=False)\n"
                    "        i, j = probe.find('XQX'), probe.find('YQY'); k = probe.rfind('<|im_start|>', 0, i)\n"
                    "        INSTR = probe[k:i]; between = probe[i + 3:j]; RESP = between[between.rfind('<|im_start|>'):] if '<|im_start|>' in between else between\n"
                    "        assert INSTR and RESP and INSTR != RESP, (INSTR, RESP)\n"
                    "        print('masking: instruction_part', repr(INSTR), '| response_part', repr(RESP))\n"
                    "        trainer = train_on_responses_only(trainer, instruction_part=INSTR, response_part=RESP)\n"
                    "        ex = trainer.train_dataset[0]; n_lab = sum(1 for x in ex['labels'] if x != -100); n_tok = len(ex['labels'])\n"
                    "        assert 0 < n_lab < n_tok, (n_lab, n_tok)\n"
                    "        print(f'masking check on row 0: {n_lab}/{n_tok} tokens carry loss ({n_lab / n_tok:.0%}); the rest is context')\n"
                    "        print('  loss tokens decode to:', repr(tokenizer.decode([t for t in ex['labels'] if t != -100])[:200]))\n"
                    "    else:\n"
                    "        print('full: loss on the whole chat text, as v8e / v12e')\n")
    replace_in_cell(nb, 3, "    json.dump({'version': VERSION, 'base': MODEL, 'data_sha256': m['output_sha256'], 'gen1_sha256': m['gen1_sha256'],\n"
                    "               'r7_sha256': m.get('r7_sha256'), 'train_rows': len(ds_train), 'final_loss': stats.training_loss,",
                    "    json.dump({'version': VERSION, 'arm': ARM, 'base': MODEL, 'gen2': m.get('gen2'), 'gen3': m.get('gen3'),\n"
                    "               'train_rows': len(ds_train), 'final_loss': stats.training_loss,")
    # --- export: the local gate commands for gen 3 ---
    replace_in_cell(nb, 4, "    print('\\nlocally, as standin/models/emitter_v12e.Q4_K_M.gguf, the gate:')\n"
                    "    print('  python validation/exp_r3_emitter_floor.py --gguf standin/models/emitter_v12e.Q4_K_M.gguf --exclude standin/data/out/gen2_exclusions.json --tag gen2')\n"
                    "    print('  python validation/exp_r7_emitter_planned_walk.py --r3 validation/logs/exp_r3_emitter_floor_gen2.json --exclude standin/data/out/gen2_exclusions.json --resident --tag _gen2')",
                    "    print(f'\\nlocally, as standin/models/emitter_v13e_{ARM}.Q4_K_M.gguf, the gate:')\n"
                    "    print(f'  python validation/exp_r11_search_learn.py --wikidata --n 600 --seed 7 --lexicon --gguf standin/models/emitter_v13e_{ARM}.Q4_K_M.gguf --tag _gen3_{ARM}')\n"
                    "    print(f'  python validation/exp_r17_gen3_heldout.py --gguf standin/models/emitter_v13e_{ARM}.Q4_K_M.gguf --tag _{ARM}')")
    set_cell(nb, 6, HOW_TO_READ)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
