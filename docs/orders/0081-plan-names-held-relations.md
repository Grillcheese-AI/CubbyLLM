# 0081 - The programs seat names only relations the store holds: a decode-time constraint, no training
status: in progress (Code B, 2026-11-24)
written: 2026-11-24 by Cowork, after an outside review of the hypothesis file · approved: 2026-11-24 by Nick ("yes") · hypothesis: H-E15 (the emitter never copies a name or a number)

## Why
H-E15's live run (2026-10-17: eight questions, the 450M against the stand-in on the same loop) spoke 0 wrong on both
arms, and the 450M's refusals came from the words of its plans:
- "Where was Marie Curie born?" and "Where was Ernest Hemingway born?": the plan said `birth`. The store says
  `birthplace` and `birth date`, so coverage refused the plan and lever 6 found no label for it.
- "who is the prime minister of canada?": plan `prime minister | sue`.
- the two-hop capital of citizenship: plan `country | citizens | E2`, a chain shape the store does not hold.

The host already knows every relation the store holds (`KnownRelations`, the set the gate checks a plan against), and
grilly2's decode takes a callback that says which tokens may come next (order 0071; `cubbyllm/speech/rule_reader.py`
uses it to keep the seat inside a rule plan's shape). So the seat can be kept from writing a relation the store does
not hold, with no training:
- inside a plan's relation literal (`bind frame, HOP1, "...";`), only tokens that continue a held label, then the
  closing quote; the seat still chooses which label;
- inside the seed literal (`bind frame, SEED, "...";`), only the slot ids the host put in the prompt (`$E1`, `$E2`,
  ...), so an unknown slot cannot be written.

The outside review (2026-11-24) named this as the cheapest lever left on the knowledge side. It is cheap because
nothing trains and the gate (coverage, the walk, the VM) does not change.

- **What it brings CubbyLLM:** the 450M's knowledge questions stop failing on a word the store spells differently;
  the questions it refused for that reason get answered, still checked by the VM.
- **Over a conventional model:** a conventional decoder can write any word, and its answer rests on whatever it
  wrote. Here the decoder can only name relations that exist in Cubby's store, so a misnamed relation cannot happen,
  and every answer is still verified before it is said.
- **What it does for you:** ask Cubby where someone was born and it answers, instead of refusing because its small
  model said "birth" where its notes say "birthplace".

## Build
1. **`cubbyllm/emit/plan_trie.py`** (new):
   - `PlanTrie(tokenizer, relations, slots)`: the options as a trie, the way `rule_reader._Trie` holds them. For a
     relation literal: every label `known` holds, normalized as `emitted_plan` reads them (`planner.normalize`);
     the wiki world's share is H-G4b's 163-entry relation table, and the other worlds add theirs. For the seed
     literal: the slot ids of the prompt's `SlotTable`.
   - `allowed(new_ids) -> list | None`: reads the text written so far. Outside a literal it returns None (no
     constraint). Inside `bind frame, HOPk, "` it returns the trie's continuations; inside `bind frame, SEED, "`, the
     slot ids' continuations.
   - The CotChain form (`H1_<RELATION>` role names) is handled the same way if the adapter writes it. First check
     which form the served programs adapter writes on the live eight, and say so in the report.
   - **The trap is the token boundary.** bbpe128k can merge the opening quote with a label's first letters, and a
     label's last piece with the closing quote. Build each option from the tokens the tokenizer writes in context
     (`encode(prefix + label + '"')` minus `encode(prefix)`, checked to split cleanly), not from the label alone. One
     test pins each end.
2. **The seat:** `Cubby450mEmitter.emit` takes `allowed=None` and passes it to `trunk.generate` as
   `prefix_allowed_tokens_fn` when given (the hook's call as in `validation/exp_rule_reader.py`). `SlotEmitter.emit`
   builds the seed options from its `SlotTable`; the ask loop (`cubbyllm/reasoning/ask.py`, `_ask`) hands it
   `self.known`.
3. **The flag:** `CB_EMIT_TRIE` = `off` (default), `rel` or `rel+slot`. `off` decodes exactly as today, pinned by a
   test.
4. **The benches get a 450M arm** (`--cubby450m <export> --emitter-adapter <dir>`). The seat is built the way
   `cubby_serve.build` builds it: `SlotEmitter(Cubby450mEmitter(trunk))` over the worlds' names.
   - `validation/exp_r17_gen3_heldout.py`: the gen-3 held split (entities no training record used), 600 questions,
     seeds 7, 8 and 9. It uses the same gate and store as the stand-in's runs (v13e: 599, 600 and 600 of 600; v12e on
     seed 7: 531).
   - H-E15's eight live questions, through the served loop with H-E15's flags (`--ask --web off`).
   - H-E18's unseen-writer eval (`pf_heldout_eval_w_slots.jsonl`, the slot form the served adapter reads: 285
     plan and 294 chain records), generated per arm with the seat as served and scored by `validation/exp_he18_ab.py`'s reader. Reported, with no gate of
     its own: arm A here is the floor the next emitter adapter (H-E18's plan and chain records in its mix) is
     read against (order 0082).
   - Arms: A `off`, B `rel`, C `rel+slot`, on the same rows, greedy.

## Acceptance (pre-registered)
- **Tests:**
  - the trie on a stand-in tokenizer: a held label written whole, a prefix two labels share, the quote merged at each
    end, and text outside a plan's literals left free;
  - `off` decodes exactly as before;
  - the seed literal admits only the table's ids;
  - `emitted_plan` reads every program arm B writes on the live eight.
- **Numbers to report, per arm and per seed:**
  - planned, verified, correct and WRONG;
  - the refusals by reason: coverage, unknown relation, ambiguous relation, retrieval exhausted, unknown slot, no plan;
  - the relations arm A wrote that the store does not hold, and what arm B wrote on the same rows;
  - median and p95 time per emission.
- **Gate:**
  - **The held split:** on seed 7, arm B or C is right on at least 5 points more than arm A (30 of 600), and the lead
    holds on seeds 8 and 9. If arm A has fewer than 60 refusals whose plan names a relation the store does not hold,
    the gate is instead half of those refusals turned right.
  - **The live eight:** the two "born" questions are verified, and the four questions the 450M answered stay answered.
  - **Wrong answers:** 0 on every arm, set and seed.
  - **Time:** the median emission takes at most 1.2x arm A's.
- **Kill:** under +2 points on seed 7 with under a quarter of those relation refusals turned right, or any wrong
  answer spoken by B or C. The flag then stays off, and the 450M's knowledge gap is the plan's shape (the next
  emitter adapter's job, on H-E18's chain data), not its spelling.
- **The read order:** seed 7 first, all three arms (about 1,800 emissions on the home card). Seeds 8 and 9 run only if
  seed 7 passes.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The host narrows the choice to what exists; the seat chooses. Nothing in the gate changes: `plan_verify`, coverage,
  the walk and the VM dispose exactly as today.
- No training and no new adapter: the base and the served programs adapter (`cubby450m_v12e_w_step_cont` in H-E15;
  name the one `cubby_serve` serves today in the report) stay as they are.
- **The fetch tiers:** with the store fixed (`--web off`, and r17's "no source to fetch from"), a relation the store
  does not hold cannot be learned anyway. With a source on, lever 4 can learn such a relation, and the constraint
  closes that path. On the live eight, report any question arm A answered through lever 4 that B or C loses.
- **The decode:** the arms decode as the served seat does: greedy, no repetition guards, the full head
  (`CB_HEAD_SHORTLIST` off). `core/decoding.DecodeConfig`'s defaults (repetition penalty 1.3, no repeated
  3-grams) are not on this path; if a bench you use decodes through it, set 1.0 and 0 and say so.
- **Held-out discipline** (the hypothesis file, section 0): nothing here is tuned, and each set is read once
  per arm.
- The home card is shared with Code A: check the inbox before a long run, and don't start one while Code A has one
  going.
- Logs go to `validation/logs/exp_0081_*_<date>.json`, with no local paths in committed files.

## Out of scope
- The plan's shape (the two-hop capital of citizenship): the next emitter adapter, on H-E18's plan and chain data,
  is its own order.
- The stand-in's GGUF seat (llama.cpp's grammars could do the same; not here).
- The talk seat, and the arithmetic step format (0078's).
- Turning the flag on where a person talks to Cubby: Nick's call, after the report.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
