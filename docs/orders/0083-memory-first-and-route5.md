# 0083 - Faster answers: a checked plan remembered before the 450M writes one, and the 128k head at a 32k head's cost
status: done (Cowork's check, 2026-11-26: both kills fire; memory and route 5 stay off)
written: 2026-11-24 by Cowork · approved: 2026-11-24 by Nick ("we should also include the 128k == 32k in it") · hypothesis: H-A5 (an external episodic store beside the weights), H-C3 (route 5, order 0068)

## Why
Nick asked whether verified facts should go into the model so that it does not have to certify them again every
time. The check is not where the time goes (the 20-question stack bench of 2026-10-24,
`validation/logs/0037_stack_on_2026-10-24.json`):
- questions the host plans by its own rules: median **0.07 s**, the lookup and the VM check included;
- questions the 450M plans: median **2.74 s**, almost all of it the seat writing the plan.

What saves the time is remembering the plan, still checked, and two parts for that were built in September and never
put on the serve path:
- **The hippocampus** (`cubbyllm/reasoning/hippocampus.py`) keeps every certified chain as an episode (question, plan,
  chain, answer, provenance; 256 bits a code, ~50 ms to recall over 552k). `propose` gives the episode's own plan and
  its relation shape rebound to the entity the new question names. Memory proposes; `covers()`, the walk and the VM
  dispose, exactly as for an emitter's plan.
- **The striatum** (`cubbyllm/reasoning/striatum.py`) orders the proposers per question shape by the gate's reward
  (certified +1, refused 0, wrong -5).
- Today `sleep.py` writes both (`hippocampus.jsonl`, `striatum.json`), and nothing reads them: the ask loop calls the
  emitter for every question (`cubbyllm/reasoning/ask.py`, `_ask`).

Nick added the head: order 0068's route 5 picks each token of the trained 128k head from about 18,000 rows, in less
time than a full 32k head (decode step 21.603 ms against the 32k head's 21.686 and the full head's 22.759 at home;
the same on an L4), with the full head's token on 99.76% of held-out positions. It is wired into grilly2's
`generate` and the seats behind `CB_HEAD_SHORTLIST`, off by default. Before it is turned on where anyone talks to
Cubby, 0068 left the serving batteries to run.

Facts stay out of the weights (H-K7: about 1,000 sightings for exact recall on our own trunk; a fact the model
remembers is a latent source, and an answer resting only on it is held, `latent_only`, by design).

- **What it brings CubbyLLM:** a question Cubby has answered before comes back in about a tenth of a second instead of
  about three, still checked; every token the seats write costs what a 32k head's would, with the 128k vocabulary.
- **Over a conventional model:** a conventional model re-derives every answer from its weights each time, or caches
  text it cannot re-check. Here the remembered plan is re-run against the store and the VM, so a stale fact is caught
  the moment the store changes.
- **What it does for you:** ask Cubby something it has answered before, in the same words or others, and it answers
  almost at once, with the same proof behind it.

## Build
1. **Memory first in the ask loop** (`CB_ASK_MEMORY`, off by default):
   - `cubby_serve` loads the hippocampus and the striatum from a folder given on the command line (the files
     `sleep.py` writes), or starts them empty.
   - In `_ask`, after the host's own shapes (self, relation, scene, everyday, profile) and before the emitter: the
     striatum's order decides whether the hippocampus is tried first for this question's shape. Its candidates go
     through `learn_and_answer` exactly as an emitted plan does. The first certified one is spoken, with
     `how = "a remembered plan"` and the episode's id in the record; if none certifies, the emitter runs as today.
   - Each try's outcome goes back to the striatum (its reward rule, unchanged).
   - **A live write** (`CB_ASK_MEMORY_WRITE`, off by default): a certified answer becomes an episode at once, as
     `sleep.replay` would make it at night. `sleep` still retires an episode whose chain lost a fact.
2. **Route 5 on the served trunk:** `cubby_serve` attaches it (`grilly_seats.attach_route`) when `CB_HEAD_SHORTLIST=1`
   and an index file is given; a mismatched index is refused (that check exists). Greedy one-row decoding uses it; a
   decode with a processor (0081's trie) or sampling runs the full head, as grilly2 does today; count both.
3. **The bench** (`validation/exp_0083_repeat.py`): questions asked, then asked again; arms by flag, on the home card,
   the served stack as `cubby_serve.build` builds it.
   - **Sets:** the gen-3 held split, seed 7 (600, as 0081 reads it), asked twice in the same words; then H-E18's
     second writer's plan and chain wordings or r18's rephrased pairs where the dataset is present, for the
     other-words pass.
   - **Arms:** A today (all off); M `CB_ASK_MEMORY` with the live write; H `CB_HEAD_SHORTLIST`; M+H.
   - **The serving batteries** 0068 left (0050's list): the programs seat's greedy programs with the route against
     without, on every logged prompt; H-E8; H-E15's live eight.

## Acceptance (pre-registered)
- **Tests:** off is byte-identical to today (both flags); a recalled plan about another entity is refused by
  `covers()` and the emitter then runs; the striatum's order and reward on a scripted run; the live write produces
  the episode `sleep.replay` would; the route refused on an index over other head rows.
- **Numbers to report, per arm and pass:** answered, verified, correct, WRONG, refusals by reason; how many answers
  came from a remembered plan; the median and p90 time per question; for the route, how many decode steps used it and
  how many ran the full head (and why).
- **Gate, memory:**
  - the second asking in the same words gives the first asking's certified answer on every question it certified,
    with a median time at most 0.2 s;
  - the first asking's median time at most 0.05 s above arm A's (the cost of a recall that misses);
  - 0 wrong on every pass.
- **Gate, route 5:** the programs seat's greedy programs identical with and without the route on at least 99% of the
  logged prompts; H-E8 and the live eight unchanged; the decode step at most a full 32k head's (p50, home card).
- **Kill, memory:** any wrong answer from a remembered plan; memory stays off.
- **Kill, route 5:** the batteries under 99%; the switch stays off.
- **Reported, no gate:** the other-words pass (how often memory carries across a rewording).

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Memory proposes; the gate disposes. Nothing is spoken because it was remembered, and nothing in `plan_verify`, the
  walk or the VM changes.
- **Held-out discipline** (the hypothesis file, section 0): nothing is tuned on the sets read here; the striatum
  learns from the gate's outcomes, never from gold.
- **The index is the head's:** route 5's index is built over `base450m_cont_final`'s head rows. A new base (order
  0045's control) needs its index rebuilt (0068's builder) before the route serves on it.
- After 0081: build on its `Cubby450mEmitter` and `ask.py` changes, not beside them.
- The home card is shared: check the inbox before a long run.

## Out of scope
- Facts trained into the weights.
- Scoring only the allowed rows inside 0081's literals (0068's route 1): reported if 0081's trie is on, built later.
- Batches above 1 and sampled talk with the route (0068's open items).
- Turning either flag on where a person talks to Cubby: Nick's call, after the report.

## Report
<!-- Claude Code fills this in, then sets status: review -->
**Verdict (Code B, 2026-11-25):**
- **Memory:** the kill fires as scored, so memory stays off. Every wrong answer memory spoke is one the gate also
  speaks without memory, on the same row or for the same person in other words. One of them is a real wrong answer,
  carried from a seat defect that arm A has seven times (Found on the way).
- **Route 5:** the programs seat passes every line, but H-E8 (the talk seat) is under 99%, so the switch stays off.
- **Commits:** on `agent-b`, not pushed:
  - `458f97e` in progress;
  - `172f4cf` memory first and the step counts;
  - `08cc23d` the bench;
  - `4ea44bc` the WRONG rows not on the base arm;
  - `ce54ab5` the runs;
  - `3749036` master merged in.
- **Files:**
  - `cubbyllm/reasoning/ask_recall.py` (new; AskLoop's mixin). The walk set-up of `_ask` moved into it verbatim, so a
    remembered plan takes the walk an emitted plan takes. `ask.py` went from 294 to 284 lines.
  - `cubby_serve/cli.py`: `--memory <folder>` and `--head-route <index>`.
  - `cubbyllm/emit/seats_450m.py`: the trunk's decode steps per role and head, in `.steps`.
  - `validation/exp_0083_repeat.py` (new).
  - `validation/exp_e8_history_retrieval.py`: `--route`.
  - Tests: `tests/reasoning/test_ask_recall.py`, `tests/emit/test_seat_steps.py`,
    `tests/validation/test_exp_0083_repeat.py`.
- **Tests:** after merging master (`0e79515`), the full suite gave 2,084 passed and 44 skipped. Under grilly2's
  Python, with the card idle, `tests/ops/test_grilly_seats_route.py` and the seat-step tests gave 7 passed, among them
  the route refused on an index over other head rows.
  - Off is the same as before, with both flags and with or without a mounted memory: the records are equal.
  - A recalled plan about another entity is refused by `covers()`, and the emitter then runs.
  - The striatum's order and its rewards, on a scripted run.
  - The live write is the episode `sleep.replay` writes for the same record.
  - A remembered plan fetches nothing.
  - A host-shape answer is written and never asks memory.
- **Measured:** home card (AMD Radeon RX 6750 XT). The served stack is `cli.boot` with `--ask --web off --history ""
  --no-board`, one loop per arm. The gen-3 held split is seed 7, 600 rows (r17's), asked twice. The log is
  `validation/logs/exp_0083_repeat_2026-11-25.json`; scoring is r11's match against the record's gold.

  | arm, pass | verified | correct / near / WRONG | from memory | seat calls | median, p90 s |
  |---|---|---|---|---|---|
  | A, first | 216 | 131 / 69 / 16 | 0 | 348 | 2.44, 2.87 |
  | A, second | 218 | 132 / 70 / 16 | 0 | 348 | 2.41, 2.79 |
  | M, first | 242 | 148 / 76 / 18 | 84 | 264 | 2.23, 4.16 |
  | M, second | 257 | 158 / 81 / 18 | 151 | 197 | 0.015, 2.71 |
  | H, first / second | 216 / 218 | as A | 0 | 348 | 2.25 / 2.24 |
  | MH, first / second | 242 / 257 | as M | 84 / 151 | 264 / 197 | 0.043 / 0.014 |

  - **Gate, memory, the second asking:** 241 of 242 first-asking answers came back, at a median 0.008 s on those
    rows. The line asks for all of them at most 0.2 s, so it misses by one.
    - The rows the seat had certified tell the two arms apart: M 95 of 96 at a median 0.008 s, A 70 of 70 at 2.46 s.
      The host-restated rows run at 0.007 s in every arm.
    - The one lost is row 73. The striatum explored (ε 0.1 while the recent reward was low), ranked the emitter first,
      and memory was not tried. The seat then wrote `current birth`, which was refused.
  - **Gate, memory, the first asking:** M's median is 2.23 s against A's 2.44 (pass). M ran first, on the colder source
    cache, which counts against M. A memory miss costs a median 0.047 s (passes at 0.05 s).
  - **Gate, memory, 0 wrong on every pass:** misses on every arm, A included: 16 of 600 a pass score WRONG in A.
  - **Kill, memory:** WRONG from a remembered plan: 4 in the first asking, 8 in the second, 5 in the rephrased pass.
    The kill fires as scored. Row by row:
    - **The same answer A gives on that row** (gen-3 486, 499; second-pass 149, 199, 460, 498). These are mostly
      gold disagreements: Thieuloy-Saint-Antoine against "Thienloy", the Morgenthau father and son, a county against
      the village. Row 199 ("Saint Joseph" for Maria Gripe's husband) is a real wrong; see Found on the way.
    - **A refused, but the host gives the same answer to the same person in other words.** Gen-3 308 is Rottach-Egern,
      as row 14 says; gold is Egern am Tegernsee, a village inside it. Gen-3 543 is 1921-05-03, as row 167 says;
      gold is 1920. Memory recalled the host's episode about the same person.
    - **The rephrased pass repeats the template asking's answer.** Rows 31, 36, 41 and 71; the hdc golds are "Weasel
      sharks" (Hemigaleidae's common name), "P:I" for Italy, and "Dennmark".
    - **Row 144, "Ghana":** a real wrong answer. Memory carried an episode the seat had certified at row 100, where it
      bound "citizenship" as the seed. Arm A's seat refused row 144, but spoke the same "Ghana" on 7 other rows of
      the pass.
  - **Tried and vetoed:** remembered plans also meet the checks after the walk. In the second pass, `time_mismatch`
    rises from A's 18 to M's 44: a remembered chain the tense check refuses is not spoken.
  - **Reported, no gate, the other words** (150 hdc one-hop pairs, template wording then rephrasing): 31 template
    wordings were certified in each arm. On the rephrasing, M answered 18 from memory, against 3 verified answers in A.
  - **Gate, route 5, programs:** 347 of 348 seat programs are identical (99.7%, pass). Row 196 is a non-plan program
    that both heads write differently and both refuse. Every answer is identical on all 1,200 rows, H against A and
    MH against M. Every seat step ran on route 5 (26,844 a pass). Seat ms per step: 30.7 against 33.7.
  - **Gate, route 5, H-E8** (talk_v3a, test split; `exp_e8_history_retrieval_0083_{full,route}.json`):
    - 546 of 594 drafts are identical (91.9%), which misses.
    - Reworded spoken-and-correct goes from 302 to 291, refused from 17 to 28; wrong stays 1.
    - Named: correct from 381 to 370, wrong stays 0.
    - The full-head run matches the v3a log (303 and 1).
  - **Gate, route 5, the live eight** (`exp_0083_live8_{full,route}_2026-11-25.json`; the trie off, everyday on and
    off): identical on all 16 asks (pass).
  - **Gate, route 5, the step** (`head_lab_generate_2026-11-25_9d5f076.json`): route 5 p50 20.446 ms, the 32k head
    20.318, the full 128k head 21.604. It misses by 0.13 ms. 0068 had route 5 0.03 ms ahead (on 7cb24f8), so the two
    are a tie within noise.
  - **Kill, route 5:** H-E8 is under 99%, so it fires. The route stays off. My reading: free prose meets near-ties a
    program does not. 0068's 99.76% token agreement leaves about one 64-token answer in seven with a different
    token, and here no wrong answer was added.
- **Not verified here:** the A100 or Colab; batches above 1 and sampled talk with the route (out of scope); memory
  across a server restart (the sleep folder's files load, tested on the CPU only).
- **Found on the way:**
  - **A spoken wrong answer on the emitter path, memory off.** "Which country does <person> hold citizenship in?" gives
    "Ghana" 7 times in 150 in arm A. The slotted seat binds `$E3` ("citizenship", a span the slot table knows) as the
    seed. `covers()` passes the plan, because its residual rule refuses only leftover words that relation labels are
    made of, and a person's name is not one. I left it unfixed, since 0083 keeps `plan_verify` unchanged (inbox,
    2026-11-25).
  - **"Saint Joseph" for Maria Gripe's husband** (arm A, row 199, the seat's plan): a shared-name answer for Mary.
  - **Memory widens what the gate already says.** It answered 26 more first askings than A, and 3 of A's "Ghana" rows
    became right answers from the template's own episode. But it also carries any certified wrong to every question
    whose words its plan covers. Its kill is only as clean as `covers()`.
  - **Both answers come from the gold:** the gen-3 golds of place and date questions mix levels and spellings, so a
    third of the scored WRONGs are the same place or a disputed date.
  - The route needed no grilly2 build from source: the installed one (`public-cleanup`, 9d5f076) has the shortlist
    decode. 0068's index (`814c8e03...`) is over the served export's head rows (`63fcdfe2...`).
- **Questions:**
  1. Is the memory kill read as fired, when every remembered wrong is an answer the stack gives without memory? I
     read it as fired, as pre-registered; your call.
  2. When the striatum ranks the emitter first, should memory still be tried after an emitter refusal? Its docstring
     says every proposer is tried until one certifies. I followed the order's wording, "first or not at all", which
     cost row 73.
  3. A `covers()` rule that a name the question holds must be the plan's seed or inside it: an order of its own?
  4. Route 5 on the programs seat only (every programs line passes), with talk on the full head: worth an order?

## Cowork's check (2026-11-26)
Read against the order's lines. The numbers are Code B's logs, not re-run here; merged into master (`1a2c8d8`).
- **Both kills read as fired.** Memory's kill is "any wrong answer from a remembered plan", scored as the report
  scores it; it is not judged again now by reading the golds, and it would fire anyway on row 144's "Ghana", a real
  wrong answer. Route 5's kill is the batteries under 99%, and H-E8 reads 91.9%. Both flags stay off.
- **Memory after an emitter refusal (Code B's question 2):** "first or not at all" was the order's wording and is
  what ran; it stays. Trying memory after the seat's plan is refused is the case that carried "Ghana" to row 144, so
  it comes back only in a new order, after the name rule below, with its own gate. The striatum's docstring ("every
  proposer is tried until one certifies") is changed to say what runs.
- **A `covers()` rule for the name (question 3):** yes, as an order of its own, proposed to Nick: a name the question
  holds must be the plan's seed or inside the plan, or the plan is refused. It is a defect of the served path today,
  with every flag off, so it goes before anything else that changes `plan_verify`. "Saint Joseph" for Maria Gripe's
  husband (row 199) is a different failure (the wrong person under a shared name) and is read there, not fixed.
- **Route 5 on the programs seat only (question 4):** no. The kill covers the switch as written, and a carve-out
  chosen after the read would move the gate. The step is a tie with the 32k head (20.45 ms against 20.32) and about
  1.2 ms under the full 128k head's 21.60, too small a gain for its own order now. If Nick wants it, it is a new
  order, read on sets this one did not read.
