# 0081 - The programs seat names only relations the store holds: a decode-time constraint, no training
status: done (2026-11-25, Cowork: the kill as pre-registered; `CB_EMIT_TRIE` stays off)
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
**Verdict: the kill line fires on seed 7.** Arm B is 1.7 points below arm A. It turned 57 of arm A's 362
unheld-relation refusals right (16%), and it lost 67 questions A had right. 0 wrong on every arm, set and seed. The
flag stays off. The argument fails in a way the kill line did not name: these refusals do not come from the
spelling. Coverage reads the question's own word ("born", "year", "when") against the plan, and lever 6 aliases that
word to a held label. The trie takes the word away, so coverage refuses the held label, and the alias path closes.
Arm B gains only where the question itself names the held label ("What is the birth date of X?": 13 → 44 of 46).
On the live eight, the two "born" questions stay refused (the seat writes `birthplace`). Two of the four answered
are lost, because the trie also closes lever 4: the seat can no longer write a relation the source would teach.

- **Commits:** `90eda82` (the trie, the seat, the flag, the benches' 450M arm), `86af508` (the unseen-writer
  runner), `ee64f91` (every bench log names its card), `8d496ae` (the runs and the runner fixes), the report commit.
- **Files:**
  - new: `cubbyllm/emit/plan_trie.py`, `tests/emit/test_plan_trie.py`, `validation/exp_0081_seat.py`,
    `validation/exp_0081_live.py`, `validation/exp_0081_pfheld.py`;
  - changed: `cubbyllm/ops/grilly_seats.py` (`allowed_processor`, `device_name`), `cubbyllm/emit/seats_450m.py`
    (`allowed=`), `cubbyllm/reasoning/slots.py` (`relations`, the seed trie; the file was over 300 lines and grew by
    6, reason in `90eda82`), `cubbyllm/reasoning/ask.py` (hands `known` to the seat when the flag is on),
    `validation/exp_r17_gen3_heldout.py` (the 450M arm, three arms on the same rows, resumable),
    `validation/retrieval_bench/r9_vectors.py` (0046's test-order bug, below);
  - logs: `validation/logs/exp_0081_r17_s7_2026-11-24.{json,log}`, `exp_0081_live_2026-11-24.json`,
    `exp_0081_pfheld_2026-11-24.json`.
- **Tests:** `tests/emit/test_plan_trie.py`, 9 tests:
  - a held label written whole, and a prefix two labels share;
  - the quote merged with a label's first letters (`, "bi`) and with its last piece (`rth";`);
  - bbpe128k's own literals;
  - text outside a literal left free;
  - the seed literal admits only the table's ids;
  - off passes no hook (decodes as before), and an unknown flag value raises;
  - `emitted_plan` reads the program arm B writes;
  - the processor's mask.

  Full suite after merging master (`fdb4344`): 2,055 passed, 44 skipped.
- **Measured** (the gen-3 held split, seed 7, 600 questions, the home card, AMD Radeon RX 6750 XT;
  `exp_0081_r17_s7_2026-11-24.json`):

  | arm | planned | verified = correct | WRONG | coverage | unknown rel. | ambiguous | retrieval exh. | unknown slot | no plan |
  |---|---|---|---|---|---|---|---|---|---|
  | A `off` | 594 | 198 | 0 | 265 | 1 | 39 | 91 | 0 | 6 |
  | B `rel` | 594 | 188 | 0 | 395 | 0 | 7 | 4 | 0 | 6 |
  | C `rel+slot` | 594 | 188 | 0 | 395 | 0 | 7 | 4 | 0 | 6 |

  - **Gate, held split: fails; the kill line fires.**
    - B − A = −10 (−1.7 points), against the +30 asked.
    - A has 362 refusals whose plan names a relation the store does not hold (more than 60, so the 5-point gate is
      the one that applies). B turned 57 of them right (16%), under the kill's quarter.
    - B gained 57 and lost 67:
      - 51 of the losses: A wrote `born`, lever 6 aliased it to `date of birth`, and it verified. B wrote
        `birth date` (held), and coverage refused it.
      - 13 more: A's `administrative territorial entity`, aliased to `located in the administrative territorial
        entity`. B wrote `political position | state | region served`.
      - The gains: A wrote `current birth date` (28) or `current birth` (19); B wrote `birth date` (51 of 57).
    - Seeds 8 and 9 do not run (the read order).
  - **By wording, A → B (verified of asked):**

    | wording | A | B |
    |---|---|---|
    | "What is the birth date of X?" | 13 | 44 of 46 |
    | "What is the date of birth of X?" | 0 | 20 of 46 |
    | "In what year was X born?" | 24 | 0 of 43 |
    | "When was X born?" | 10 | 0 of 38 |
    | "On what day, month, and year was X born?" | 17 | 0 of 38 |
    | "Which administrative territorial entity is X located in?" | 13 | 0 of 13 |
  - **Relations arm A wrote that the store does not hold, and what B wrote in the same hop:**
    - 431 rows, 522 mentions.
    - `year` 62 → `first` 25, `in order to` 12, `date` 11
    - `birth` 57 → `birth date` 35, `birthplace` 22
    - `born` 56 → `birth date` 46
    - `current birth date` 48 → `birth date` 37
    - `current birth` 44 → `first` 24, `birth date` 19
    - `city` 39 → `birthplace` 15, `birth date` 8, `population` 7
    - `when` 27 → `first` 20
    - `E1` 22, `on` 19, `current_year` 16, `E2` 12
    - B and C wrote 0 unheld relations, as built.
  - **Plan length:** A wrote 500 one-hop plans and 3 five-hop; B wrote 390 and 67. When the seat meant a word that
    is not a relation (`year`, `when`, `current`), the trie forced the nearest held label, and greedy decoding then
    ran on through more held labels: `first | weaker than ×4` 29 times, `first | time of setting ×4` 13.
  - **C equals B on all 600 rows** (plans and outcomes). The seed trie entered 594 times and changed nothing: the
    seat writes a table slot unprompted, and there are 0 unknown-slot refusals under A as well.
  - **Time:**
    - median per emission: A 2,413 ms, B 2,479 (1.03×), C 2,487 (1.03×). The gate line (at most 1.2×) holds.
    - p95: A 3,096, B 4,247, C 4,258 (1.37×), from the longer plans.
    - The first emission of the run took about 9 s (warm-up); without it, the medians are unchanged.
  - **Wrong answers:** 0 on all three arms.
  - **The live eight: the gate fails** (`exp_0081_live_2026-11-24.json`). One loop per (everyday, arm), booted
    as the server boots it with `--ask --web off`: the Wikidata source is on, `known` holds 2,793 relations, and
    the history is off.
    - **Everyday table off** (H-E15's condition, 2026-10-17: the seat plans every question):

      | question | A `off` | B `rel` (C the same) |
      |---|---|---|
      | mother of Justin Trudeau | verified (`mother`, learned by lever 4) | refused: `father` |
      | country of Oak Park | verified (`country`, learned by lever 4) | refused: `selected by \| position \| location` |
      | father of Bill Haslam | verified | verified |
      | who is marie curie? | the profile | the profile |
      | Marie Curie born | refused: `birth` | refused: `birthplace` |
      | Hemingway born | refused: `birth` | refused: `birthplace` |
      | capital of citizenship | refused: `country \| citizens \| E2` | refused: `country of citizenship` ×3 |
      | prime minister of Canada | refused: `prime minister \| sue` | refused: `country of citizenship` ×2 |

      Every refusal above is a coverage refusal.
    - **The gate's lines:**
      - neither "born" question is verified on B or C;
      - of the four the 450M answered, two stay answered (Haslam, the profile);
      - 0 wrong.
    - **Lever 4 (the order's fetch-tier line):** A answered the Trudeau and Oak Park questions by learning `mother`
      and `country` from Wikidata; the store held neither at boot. B and C lose both: the trie keeps those labels
      from being written, so the loop never fetches.
    - **Everyday table on** (today's default): the three arms are identical. The host restates six of the eight
      (both "born" questions verified as `place of birth`), the profile answers one, and the seat plans only the
      capital of citizenship and the prime minister, which are refused on every arm.
  - **The unseen-writer set** (no gate; `exp_0081_pfheld_2026-11-24.json`). H-E18's eval: the served seat,
    greedy, max_new 640, the full head; each record's own slot table checks and fills the program, and
    `exp_he18_ab.score` runs every program in the VM. `known` is the wiki world's 2,793 relations.

    | arm | plan correct | chain correct | plan ms, median / p95 | chain ms, median / p95 |
    |---|---|---|---|---|
    | A `off` (0082's floor) | 77 / 285 (27.0%) | 267 / 294 (90.8%; 98.3% execute) | 2,351 / 2,986 | 6,656 / 12,718 |
    | B `rel` | 17 / 285 (6.0%) | 10 / 10 | 2,583 / 3,349 | (10 records) |
    | C `rel+slot` | 17 / 285 (6.0%) | 10 / 10 | 2,598 / 3,359 | (10 records) |

    0 copied and 0 unknown slot on every arm.
    - **Chains:** the trie never fires on the CotChain form (0 entries). The 10 chain records timed under all three
      arms were byte-identical across them, so B and C read only those 10 and the rest ran under A only (said in
      the inbox before the run). `--tasks chain --arms rel,rel+slot --resume` adds them, about 70 min.
    - **Why the plans drop:** 216 of the 285 references name a relation the wiki store does not hold. The top ones
      are `country` 44, `administrative territorial entity` 27, `country of citizenship` 25, `given name` 24 and
      `country of origin` 20. A wrote them and was right on 63 of those records; B cannot write them.
      - On the 69 records whose relations are all held, B has 17 right and A 14 (B gains 3 and loses 0).
      - The same pattern as r17: the trie helps only where the store and the question already agree.
    - C equals B on all 285 plan records, and B writes A's program on 45 of 285.
- **Not verified here:**
  - seeds 8 and 9 (not run: the read order);
  - Colab (all runs are on the home card);
  - the stand-in's arm (out of scope).
- **Found on the way:**
  - **The hook.** grilly2's `prefix_allowed_tokens_fn` must return a non-empty list at every step, so "no
    constraint outside a literal" cannot be said through it. The seat uses a one-row `logits_processor` instead
    (`ops.allowed_processor`): the scores pass unchanged outside a literal, and a `-inf` mask with the options at
    0 is applied inside one. Same effect, and only `cubbyllm/ops/` touches grilly.
  - **Relations:** the store holds 2,797 relations in r17's store (2,793 in the served loop), not H-G4b's 163. 0
    were left out of the trie (every option split cleanly), and the trie builds in 0.02 s.
  - **Smoke run:** 3 rows of seed 7 ran before the run, to check the card path. Their log is scratch
    (`_exp_0081_r17_s7_smoke_*`), and nothing was changed after it. It already showed the coverage refusals.
  - **The plan form:** the served adapter (`cubby450m_v12e_w_step_cont`, `cubby_serve`'s `cubby` preset today)
    writes CotPlan (SEED + HOPk). Under B, each of its 1,143 hops on r17 went through a `bind frame, HOPk` literal.
    On the live eight, every program the seat wrote is CotPlan, and `emitted_plan` read every one B wrote. It writes CotChain (`Hk_<REL>`) when a Facts block is given (the unseen-writer chain records), and
    the trie leaves that form free.
  - **The live eight's wording:** H-E15's log did not keep the exact questions; `exp_0081_live.py` restates them
    from its entry.
  - **One loop learns across arms.**
    - My first live run asked the three arms in one loop. What lever 4 learned on arm A (`mother`, `country`) was
      then in the store for B and C, and the trie let them write it. That hid the lever-4 loss.
    - The runner now boots one loop per (everyday, arm); `--shared` keeps the old way. The shared run's log is
      scratch (`_exp_0081_live_shared_*`).
    - The same run booted with the server's default `--history`, so it appended 9 bench asks to the sleep cycle's
      input. That file is in Code A's checkout; I left it as it is and told Cowork. The runner now passes
      `--history ""`.
  - **The seed on "Where was Marie Curie born?"** (everyday off) is `marie` on every arm, so even a covered plan
    would look up the wrong name. Under C the seat wrote `$E1`, which the table fills with "marie". The slot trie
    admits every slot of the table, so it cannot choose between them.
  - **A test-order bug in 0046's `r9_vectors.block_codes`:** it blocked grilly with `setdefault`, which does
    nothing once an earlier test has imported grilly, so three R9 tests failed in a full run. It now blocks
    `grilly.*` for cubemind's import and restores it afterwards (`8d496ae`).
- **Questions:**
  - Lever 6's alias table already maps a question word to a held label (`born` → `date of birth`). Would a next
    step admit a held label **or** the question's own words inside the literal, so coverage and lever 6 still see
    the word? Or have coverage read lever 6's table in reverse (a plan's `birth date` covers "born")? The second
    changes the gate, so it is not mine to take. Either is its own order.

## Cowork's check (2026-11-25)
- **The report holds against its logs.** `exp_0081_r17_s7_2026-11-24.json`: verified = correct 198 (A), 188 (B),
  188 (C) of 600, coverage refusals 265 / 395 / 395, 0 wrong on every arm. The live eight and the unseen-writer set
  read as the report says. The kill fires as pre-registered; the order is done and the flag stays off.
- **Where the order was wrong** (Cowork's, not the build's):
  - **Its argument.** The refusals were not the spelling. Coverage, lever 6 and lever 4 all read the question's
    own word ("born", "year", "mother"), and a constraint on the plan's words removes exactly that evidence. The
    trie helps only where the question already names a held label ("What is the birth date of X?": 13 -> 44 of
    46).
  - **The 163-entry table** in the build was the wrong number for this store: the served loop holds 2,793
    relations, and every one went into the trie.
  - **The unseen-writer set** was read with the wiki world's relations, while 216 of its 285 plan references name
    relations of H-E18's own fact store; the fair read there is the 69 records whose relations are held (B 17,
    A 14).
- **Kept from it:** arm A's unseen-writer numbers are order 0082's floor (plan 77 of 285, chain 267 of 294, 0
  copied, 0 unknown slot); every bench log now names its card; one loop per arm in the live runner.
- **Code B's question** (admit the question's own words inside the literal, or have coverage read lever 6's table
  in reverse) is Nick's to decide; neither is ordered here.
- **The 9 bench asks** appended to `ask_history.jsonl` by the first live run are set aside, not deleted
  (`ask_history.set_aside.jsonl`, beside it, with the reason).
