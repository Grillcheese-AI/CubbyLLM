# 0036 — Namesakes, the rest: inverse questions, a chosen item kept as itself, the kind-to-relations table, and an audit of the relation tier
status: review (items 1-4 built and measured; the audit, by my reading, misses its line: question 1)
written: 2026-10-23 by Cowork, from 0033's review (fixes toward the first rule; Nick told) · approved: — · hypothesis: none

## Why
Order 0033 stopped two items sharing a label from being merged in an answer (74 of 100 mixed before, 0 after). Three
paths can still mix them, and one decision rule needs an audit:
- inverse questions ("what has X as its capital?") still read every namesake of a shared object label;
- a chosen item's fetched facts go into the wiki world under the bare label, so a later bare question reads that one
  item as if the name were one thing ("What is Ottawa?" already does this);
- the relation tier decides by scanning up to 2,000 members of a kind, which can miss a rare relation;
- the tier's choices (France's capital, Japan's currency) were never audited one by one.
In plain words: when names are shared, Cubby never mixes the things up, whichever way the question is asked.

## Build
1. **Inverse questions:** the same rule as 0033 for objects: a shared object label is decided by the question's own
   context or asked back, never read through every namesake.
2. **A chosen item kept as itself:** facts fetched for a clarify choice are stored under the item (its id), with the
   label as an alias; a later bare question still asks which.
3. **The kind-to-relations table:** built once with the hdc world (which relations each kind's members carry), in place
   of the scan: complete and instant.
4. **The audit:** 50 of the relation tier's decisions, drawn by seed, each with its label, the items and the one
   chosen, read by hand (Cowork reads the list; it is written to the log for that).

## Acceptance (pre-registered)
- An inverse probe like 0033's (50 shared object labels): 0 mixed answers.
- A chosen item, then the bare label asked again: asks which (test).
- The table gives the same decisions as the scan on 0033's probe, and the tier's latency is reported.
- The audit: at least 49 of 50 right; one wrong and the tier goes strict (the asker or a qualifier only).
- 0014's bench: 179 of 179, 0 wrong; the real stack's 20 answers as after 0033.

## Constraints
- Rules in `CLAUDE.md` hold.

## Report
- **Commits:** `728d246` (in progress), `eac843e` (items 1-4, the tests and the logs), this report.
- **Files:** `cubbyllm/worlds/hdc_kinds.py` (new: the table, `build`, `Table`, `for_world`; 107 lines),
  `cubbyllm/worlds/hdc_namesakes.py` (`object_items`; `applies` reads the table), `cubbyllm/worlds/hdc_store.py` (the
  object index under the rule; opens the table), `cubbyllm/reasoning/ask_relation.py` (a chosen item's facts stored under
  its id), `cubbyllm/reasoning/ask_profile.py` (the profile reads them there); tests `tests/reasoning/test_chosen_item.py`
  (new), `tests/worlds/test_hdc_namesakes.py` (+2); measures `validation/namesake_inverse_probe.py`,
  `validation/hdc_kinds_check.py`, `validation/audit_namesake_tier.py` (new).
- **Tests:** 1,238 passed, 6 skipped, 1 failed: 0002's known guard (`cubbyllm.spatial`, `cubbyllm.reasoning.step_loop`).
  Against 0034's run, every file the same but 6 new tests (2 namesakes, 1 chosen item, 3 of 0035 A's)
  ([log](../../validation/logs/0036_tests_2026-10-23.json)).
- **Measured:**
  1. **Inverse questions.** The object index now reads a label several items share (as objects) as none of them, one
     item's rows otherwise ("Springfield", pointed at by two items, gives nothing; a Q-id gives that item). The probe, 50
     shared object labels drawn by seed 0: **0 mixed**; the index reads a shared label 0 times. But all 50 questions
     ("What has X as its R?") got **no plan**: see Found on the way 1
     ([log](../../validation/logs/namesake_inverse_probe_2026-10-23.json)).
  2. **A chosen item kept as itself.** The facts fetched for the asker's choice go in under the item's id (`kept_as` in
     the record); the bare name asked again still asks which, and the other choice gives its own facts only (test).
  3. **The kinds table.** Built once beside the world: 2,399,127 items, 20,113 (kind, relation) pairs, 7.0 s, 1.7 MB.
     On 0033's 11 hops: **the same decisions** as the scan; 34.5 ms in all against 616 ms (Japan's currency 4.6 ms
     against 521; Brazil's language 4.5 against 61). The world reads the table only when its size stamp matches the
     world's file (a rebuilt world scans until the table is rebuilt; pinned)
     ([log](../../validation/logs/hdc_kinds_check_2026-10-23.json)).
  4. **The audit.** 50 of the tier's decisions, seed 0 (370 shared labels tried to find 50), each with its items, their
     kinds and values and the one chosen, for Cowork's reading
     ([list](../../validation/logs/audit_namesake_tier_2026-10-23.md), [log](../../validation/logs/audit_namesake_tier_2026-10-23.json)).
     **My reading, for yours: the line (49 of 50) is not met.** At least 9 decisions cannot be settled by the question:
     several people share the name and one of them was chosen (7 Steve Mills, 11 Paul Rudolph, 17 Bill Carter, 23 David
     Hirst, 24 Jack McGill, 40 Jim Henry), or several films share the title (6 The Substitute, 25 Wetlands: a German and a
     Canadian film, 38 Ek Hi Bhool: two Indian films). The cause is one guard: the tier counts an item with no kind
     (`instance of`) as one that cannot carry the relation, and 46 of the 50 decisions have such a namesake; in these
     nine the untyped items are plainly of the chosen item's kind by their own facts (a citizenship, an occupation, a
     cast). Decision 1 (Huey Long's "country of origin": the film, not the politician) is a judgment call. By the
     pre-registered rule the tier goes strict (the asker or a qualifier only); I have not switched it (your reading
     decides). Strict costs the stack's France's capital and Japan's currency, which the tier decides today (0033's report).
  5. **0014's bench:** 179 of 179, 0 wrong, the rows identical to 0033's
     ([log](../../validation/logs/exp_hdc_bench_2026-10-23_0036.json)). **The real stack** (the 450M seats on grilly2,
     wiki + hdc, the ask loop offline, plugins, the board; `python -m cubby_serve`): the same 20 records as after 0033,
     ids, times and timings aside ([log](../../validation/logs/0036_stack_2026-10-23.json)).
- **Not verified here:** the online Wikidata source with a real clarify choice (the stack runs offline; item 2 is pinned
  with a stand-in source).
- **Found on the way:**
  1. **No serving path reads the object index.** The ask loop has no inverse route: "What has X as its R?" is no plan
     for all 50. Item 1's guard holds in the index (for whatever reads it later), but the probe's 0 is not a decision the
     loop made. An inverse route would be its own order.
  2. **The board's counts in a stack check move with a 5-second cache**, not with the code: `/board/home` is cached for 5
     s (`cubby_serve/board/api.py`), so it shows the history from before the last ask or two (this run's board saw 19 of
     the 20 records, so its "last question" is the 19th's 3 steps; 0033's saw 18). The history held all 20 (dumped from
     the running server). 0033's log gave the Globe Theatre's refusal as the reason; it is the cache's timing.
- **Questions:**
  1. Your reading of the audit, and with it: strict now (pre-registered), or a guard that reads an untyped namesake's
     kind from its own relations (the kinds table makes that a lookup)? A naive version would cost Italy's capital: a
     town in New York named Italy carries "country" and "located in" like a state does.
