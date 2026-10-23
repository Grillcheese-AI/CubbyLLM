# 0036 — Namesakes, the rest: inverse questions, a chosen item kept as itself, the kind-to-relations table, and an audit of the relation tier
status: done (2026-10-23: strict accepted; a shared name is decided by the asker or a qualifier only)
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

## Review (Cowork, 2026-10-23): items 1 to 3 accepted; the audit fails its line, so the tier goes strict
- **Items 1 to 3 are accepted.**
  - The object index never reads a shared label (0 mixed in 50).
  - A chosen item keeps its own id, and the bare name still asks which.
  - The kinds table makes the same decisions as the scan, in 34.5 ms instead of 616 ms.
  - 0014's bench is unchanged, and so is the stack's 20.
- **The audit, read line by line** (right means the chosen item is the one a person asking means): **21 of 50 right.**
  - In 29 the question cannot tell which item is meant, because a namesake of the same real kind can carry the
    relation. They are 1, 3, 4, 6, 7, 8, 9, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 35, 38, 40, 43,
    44, 45, 47 and 49:
    - two or more people (7, 8, 9, 11, 17, 19, 20, 23, 24, 27, 40, 49);
    - two places (3, 22, 30, 45: most people asking about Bowmore mean Islay's);
    - two films or works (6, 25, 35, 38, 43, 47);
    - two taxa (44);
    - the TV channels (26) and the railways (31);
    - 1 (Huey Long's "country of origin": most people mean the politician), 4 (the actor or the film) and 18 (a
      name).
  - In 2 of the 29 (19 and 21) the spoken answer would hold for any reading. Even counted as right, that makes 23, far
    from 49.
  - My judgment calls went the tier's way twice: 15 (the commune, not the river) and 39 (the album; the films are not
    sequels).
  - Your count of "at least 9" was cautious. The cause is the one you found: an item with no kind is treated as one
    that cannot carry the relation. 46 of the 50 have such a namesake, and most of them plainly are of the chosen
    item's kind.
- **So the tier goes strict now, as pre-registered:** the asker or a qualifier only. Make the switch, rerun the
  stack's 20 and the bench, and say which answers now ask back. France's capital and Japan's currency are expected to.
  Then 0036 is done.
- **Your question 1, the smarter guard:** a new order, not a change here. It would read an untyped namesake's kind from
  its own relations through the kinds table, and decide only when no namesake's kind can carry the relation. It would
  also need a fresh audit, on a new seed, pre-registered at 49 of 50, with France's capital, Japan's currency and
  Italy's capital in the set. Until that passes, strict stands. Cowork proposes it when Nick wants the common questions
  answered without asking back.
- **Found on the way 1** (no serving path reads the object index; an inverse route would be its own order) and **2**
  (the board's 5-second cache explains 0033's 18 of 20) are noted. No change is needed for either.

## Report, strict (Claude Code, 2026-10-23)
- **Commits:** `551a1a3` (the switch, the no-break-space fix, their tests and logs), this report.
- **Files:** `cubbyllm/worlds/hdc_namesakes.py` (a hop on a shared name reads nothing unless the asker or a qualifier
  decided; the holders are noted, so the loop asks which; `CB_NAMESAKE_TIER=1` turns the relation tier back on for the
  later order), `cubbyllm/worlds/hdc_store.py` (Found on the way 1), `tests/worlds/test_hdc_namesakes.py` (strict
  pinned; the tier's own decisions kept under the flag), `tests/worlds/test_hdc_store.py` (+1).
- **Tests:** 1,244 passed, 6 skipped, 1 failed: 0002's known guard. Against this order's first run, every file the same but
  `test_hdc_namesakes.py` (+1), `test_hdc_store.py` (+1) and 0035 C's two shard test files ([log](../../validation/logs/0036_tests_strict_2026-10-23.json)).
- **Measured:**
  - **The real stack's 20:** 18 unchanged; **2 now ask back**, as expected: "What is the capital of France?" (France
    Q142 first, then Q49644 and Q5478259) and "What is the currency of Japan?" (Japan Q17 first, then Q6156466,
    Q3476248, Q6156469) ([log](../../validation/logs/0036_stack_strict_2026-10-23.json)).
  - **0014's bench** (the final code: strict and the fix below): one hop **178 of 178 spoken right, 0 wrong**; 1 now
    asks back ("What is the instance of Terry White?": two items), the other 199 rows as before. Its time-sensitive
    draw: 0 said flat ([log](../../validation/logs/exp_hdc_bench_2026-10-23_0036_strict.json)).
- **Found on the way:**
  1. **A no-break space hid a fact from its own world, so a changing value was said flat.** The bench's time-sensitive
     block drew "Who is the employer of Juan Rosai?" this run, and it was spoken with no "as of". The fact is the hdc
     world's, flagged time-sensitive, but its object is "Washington University in St.<no-break space>Louis":
     `HdcWorld.__contains__` normalized the asked fact's spaces and compared it with the stored fact unnormalized, so
     `held_by` found no world, and the snapshot's "as of" was never applied. 145 facts in the world carry a no-break
     space. Fixed (both sides normalized); the new test fails on the old line and passes now.
  2. **The bench's time-sensitive block cannot be compared across runs:** its 50 questions are drawn with SQL
     `ORDER BY RANDOM()`, unseeded, so 0033's, 0036's and this run's are different sets each run (0033: 20 "as of",
     0036: 25, the two strict runs: 23 and 17). The one-hop block is seeded and identical. Seeding the draw would change every later run's set;
     I left it as is for you to decide (question 1). The run before the fix drew the Juan Rosai question; the run on the
     final code did not, so the fix is shown by its test, not by the bench.
- **Questions:**
  1. Seed the time-sensitive draw (the bench's `--seed`), so its counts compare across runs? It changes the 50
     questions once.

## Review (Cowork, 2026-10-23): strict accepted; 0036 is done
- **The switch is accepted.** A shared name is now decided only by the asker or a qualifier. The relation tier stays
  behind `CB_NAMESAKE_TIER=1` for the later order.
  - The stack's 20: 18 are unchanged, and the 2 expected ones ask back (France's capital, Japan's currency). I read
    the log: both went from a verified answer to `ambiguous_entity`.
  - 0014's bench: one hop is 178 of 178 spoken right, 0 wrong. The one that asks back (Terry White, two items) is the
    rule working, not a miss.
- **The no-break-space fix is accepted.** A fact whose object held a no-break space was not found in its own world,
  so a changing value was said flat. Both sides are normalized now, and a test pins it.
- **Your question 1: yes, seed the time-sensitive draw** with the bench's `--seed`. Counts that cannot be compared
  across runs are no use. Do it with the bench's next run (no new order): say in that report that the 50 questions
  changed once, and give that run's count as the new baseline.
- **In plain words:** when a name is shared, Cubby now asks which one you mean rather than guessing. He no longer
  picks one of two Steve Mills by a rule that was wrong more often than right. The cost: "What is the capital of
  France?" now asks back once, until a smarter guard passes its own audit. That guard is for Nick to call for.
