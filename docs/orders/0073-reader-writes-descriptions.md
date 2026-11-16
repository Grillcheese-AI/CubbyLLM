# 0073 - The instruction reader writes descriptions and the host compiles them (round 3: two spellings of a negation, read side by side)
status: open (0067's round 2 was closed on 2026-11-14, a negative result on its line 3, so the build can start; builder: Code A, or Cowork on a branch while Code A is signed out, Nick 2026-11-14: "we will continue here then"; steps 1 to 5 built on branch `cowork/0073` by 2026-11-15: the rows exist and the hashes of test and test_long are committed, the grammar is checked on every row, the two SFT sets are on Drive and the notebooks are written; nothing is trained (the Colab runs are Nick's; the branch is on GitHub since 2026-11-16, on Nick's yes); of step 6 the decoder and the scorer are written, the read notebook, line 9's comparison and step 7 are open)
written: 2026-11-14 by Cowork, on Nick's "yes" to writing round 3 as one order on the panel's synthesis - approved: 2026-11-14 by Nick ("I approve 0073", with the arms as defined here, narrower than in the synthesis) - hypothesis: none (0067's second reader)

## Why
0067's second reader writes a split and then a CubeLang program of about 1,500 characters. Round 2 took parts right
as written from 59 / 196 to 141 / 198 on the same 64 dev rows, and it is still over the kill line: false passes
46 / 492 = 0.0935. What is left is one thing: 29 of its 30 reversals are a negation said the wrong way. The join is
written hundreds of tokens after its cue, and a fifth check was seen numbered "check 1" again.

The panel (`docs/research/2026-11-13-instruction-reader-synthesis.md`; eight models, opinions) and Cowork's sealed
answer agree on the next step: the reader writes what the generator itself starts from, a short description of each
check on its own split line, and the host compiles the program with the generator's compiler. The program is still
what the VM runs and what the user is shown.

What the code shows, and the panel could not see: `view_wording.say_not` already words `not (count > N)` as "at most
N". So the wording follows one form and the stored part another (4,150 parts in train and dev are written two ways).
The target of this order is **the part as it is said**. One question stays open, and the two arms read it: where the
words themselves carry a negation ("that do not begin with", "should not end with", "do not include any"), does the
reader write a `not` and leave the logic to the host, or fold it into the operator itself?

## The target (both arms)
The reader's answer is the split alone. No `---`, no program, no part numbers.

    <kind>[ <join>][ <description>] | <the text, copied exactly>

- **kind:** check, task, asked, process, none, as today.
- **description** (check lines only): the part as its wording says it, in the slots of `view_programs`: `n`, `count`,
  `any`, `none`, `all`, `at`; the view; a filter of one or two column comparisons; the comparison and its number or
  value; for `at`, the position and a new slot `absent: pass` where the words say "if there is one". The notation is
  Code A's choice (short, one line, every slot from a closed set); the slots are not.
- **join** (on the line of the check that carries the connective): `either`, `unless` or `if`, with `prev` or `next`
  for the neighbouring check it ties to. Every group in train and dev joins two adjacent parts (1,617 of 1,617), so a
  neighbour is enough. Checks not tied are joined by "and"; the reader never writes "and".
- **The host** numbers the checks in the order of their lines, builds the groups from the tags, and compiles.

**The two arms differ in one thing.** The instructions, rows and replies are the same; only the target's spelling
of a negation that is in the words differs.
- **Arm M (folded):** the description has no `not`. "that do not begin with X" is `first_word != X`; "do not include
  any" is `none`; "should not end with" under "if there is one" is `!=` with `absent: pass`.
- **Arm W (as said):** the description writes `not` where the words carry a negation word, before the thing it
  negates, and keeps the operator and the quantifier of the plain wording (`not first_word == X`, `not any`). The
  host folds it. Each template branch states whether its words negate: for a number the templates already say the
  complement ("at most" for `not >`), so both arms write the comparison said; for a value they say "not".

## Build
1. **One compiler, in the core.** Move the spec language, its evaluator, the compiler and the join
   (`validation/builders/view_programs.py`, `view_compile.py`, `view_join.py`) under `cubbyllm/speech/`, so the host
   and the builders use the same code (the core never imports `validation/`). First with no change of behaviour:
   every stored row's program rebuilt byte for byte (3,900 rows and the 3,000 pair rows). Then add `absent: pass` for
   `at`, and `fold(part)`: arm W's form to arm M's. Pinned: a part and its folded form give the evaluator's verdict
   on every reply of the pool; a disagreement stops the build.
2. **The wording returns what it says.** `say` / `say_not` return, with the sentence, the part as said in both
   spellings (M and W). The stored part may stay as drawn; the target is the said form. Pinned on every template,
   English and French: `fold(W) == M`, and M compiles to a program that agrees with the stored part on the pool.
3. **The rows, built again from a new seed** (`long_rows_0073`, off-repo; manifest and SHA-256 in the repo): train
   3,000, 1,500 minimal pairs (3,000 rows), dev 300, test 300 (2 to 4 parts, compositions not in train), test_long 300 (5 to
   8). No instruction text of 0067's rows. Changes to the draw, the same for both arms:
   - **bounds balanced:** among said comparisons with a number other than 0, "at least / more than" within 45 to 55%;
   - **groups evenly placed:** the tied pair on parts 1-2, 2-3 and 3-4 in equal shares where the row has them (today
     52% sit on the first pair), each connective said before and after its partner where a template exists;
   - **replies that can tell:** where the pool has them, each row's replies include, for every part with a number N,
     one reply whose measured value is N - 1, one at N and one at N + 1, and for every group the four combinations of
     its two parts' verdicts. Each row records the share of its one-reversal mutants that its replies tell apart from
     the gold (computed with the evaluator);
   - the words that carry a sign come from the templates only (no model-drafted wording in this set).
4. **The reader's form and its grammar** (`cubbyllm/speech/`): `prompt`, `target(row, arm)`, `parse`; a grammar for
   the split with descriptions: slots from closed sets, a column only of the line's view, numbers and strings only
   from the line's own text, the text copied in order from the instruction, a join tag only on a check line. The
   host's accounting: every sentence of the instruction is a line or is flagged; a join tag that does not point at a
   check, or a description that does not compile, makes that part "asked". Nothing is dropped silently. The grammar is
   built as a state machine: see "The grammar is a state machine" below.
5. **Two SFT sets** (same records, the target in arm M or arm W) and two notebooks, built on round 2's recipe and
   changing nothing else: rank 16, alpha 32, five epochs, key tokens (the slots) weighted 4. The Colab runs are
   Nick's.
6. **The read and its scoring:** answers decoded on Colab in batches as in round 2, one decoder for every arm;
   scored at home on the shared VM. `exp_view_reader.py --score` compiles the descriptions and reports what
   "Acceptance" lists. `exp_view_second_chance.py` works on descriptions (one slot reversed at a time), on dev only.
7. **A baseline arm P, no training:** round 2's adapter (the program target, rules 1 to 9), read once on the same
   test and test_long rows with the same decoder.

## Acceptance (pre-registered; test and test_long of `long_rows_0073` read once per arm; their SHA-256 committed before any adapter of this order is trained; dev is for looking, test rows are never opened by eye)
1. **Accounted:** every sentence a line or flagged; silent drops 0 (**kill**).
2. **The split:** span and kind for at least 90% of parts.
3. **The join:** agreement at least 95% of the replies the VM could check; false passes at most 2% of them (**kill**
   above 5%). Reported also over the replies built to fail, which is the rate that counts for a user.
4. **Longer than seen:** trained on up to 4 parts, read on 5 to 8: agreement at least 90% of the in-length rate,
   counted only if line 3 passes in-length; false passes on test_long at most 2% (**kill** above 5%); parts right at
   positions 5 to 8 at least 90% of the rate at positions 1 to 4.
5. **Parts:** a part's description, compiled and run alone, gives the generator's verdict on every reply of its row
   for at least 95% of parts; parts fixed by one reversal at most 3% of parts (**kill** above 8%).
6. **Groups:** the connective and both its members right for at least 97% of groups (**kill** below 90%).
7. **Coverage:** replies the VM could check at least 98% on test and 95% on test_long; parts "asked" at most 10%
   (these rows are all in the description language).
8. **Time, reported with its line:** seconds per instruction at home, median and 95th percentile, by number of
   parts. Pass: a median of at most 8 s for 2 to 4 parts. Above today's 25 s the target is not served.
9. **Which arm is kept (written now):** the arm with fewer false passes over replies built to fail on test; if the
   two are within a 95% interval taken over rows, the arm with more parts right; if still within, arm M. An arm that
   hits a kill is out. If both hit a kill, the order ends as a negative result and nothing is served.
10. **Reported for every arm,** raw counts with each rate: by direction of the bound (at least or more than; at most
    or fewer than; exactly; other than) with the unweighted mean of the four; by negation in the words (none, one,
    two); by place (size, count compared, filter, position value); by number of parts (2 to 8) and by the part's
    position; by connective; by language; both twins of a pair right; the share of mutants the replies could tell;
    arm P beside M and W.
- **Tests:** the compiler's move changes no stored program; `fold` and the said forms pinned on every template; the
  grammar accepts every gold target of both arms and never leaves the reader without a way on (checked on prefixes);
  the host's accounting pinned on a split with a missing sentence, a join tag with no partner, a description that
  does not compile.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim; the core opens no connection and
  imports nothing from `validation/`).
- No model writes a rule, a description, a program or a training target: rows are built in reverse and run in the VM
  before kept. No benchmark sentence in a training row (the 5-word test). CubeLang's capabilities read from a build;
  a cubelang change waits for Nick's yes in the cubelang repo.
- No host code for one kind of instruction: the compiler works on slots, and a slot is added only if it is a view,
  a column, a quantifier or a comparison usable across kinds.
- One change per comparison: the arms share rows, recipe, decoder and grammar; arm P shares the rows and the decoder.
- A fix designed after a test read is read on rows no one has seen. Rows stay off-repo; ids, counts and hashes only.
- Nothing is wired into serving (`CB_RULES` stays off). The Colab runs and any paid compute start on Nick's go.

## Out of scope
- **A free program for what the descriptions cannot say,** and the set that measures what is lost without it (checks
  CubeLang can run and the slots cannot say; a census by a person of the held-out kinds and the benchmark's
  constraint types): its own order, proposed to Nick. Until it is read, such an instruction is "asked".
- Abstaining when the operator's score is close to its complement's; a second reading of the complement; an
  isolated-part task; reading each part in its own pass; a margin loss; training on 5 or more parts (line 4).
- 0067's lines 5 and 6 (unseen kinds, the plan reader side by side): after an arm is kept.
- A dev-only record of the reader's confidence in each slot (the R-U-SURE idea Cowork raised on 2026-11-14): Nick,
  2026-11-15: "r-u-sure might be too heavy so no".

## Settled before the build (Cowork, 2026-11-14, on Code A's reading of the order)
Code A read the order against the code and raised six points (inbox, 2026-11-14). Each is settled here; where this
section and the text above differ, this section holds.
1. **The said form is the sentence's own form, not the stored part with a flag.** `amount` already says a count
   `> 0` or `!= 0` as "at least one" and `< 1` or `<= 0` as "no": the target carries what is said. Quantifier words
   go to the quantifier slot ("at least one" is `any`, "no" / "aucun" is `none`, "every" / "chaque" is `all`); a
   count said with a number is `count` with that comparison and number. **Pinned over the whole set: one sentence
   never has two targets** (two rows whose check text is the same have the same description, in each arm).
2. **The words that make a `not` in arm W, and no others:** "not", "do not", "ne ... pas", "n'... pas" on a verb.
   "no" / "aucun" is the quantifier `none` in both arms. "without" / "sans" and "other than" / "autre que" are the
   comparison said, in both arms. So the arms differ only where a verb is negated.
3. **`not all` with two conditions** (said as "at least one ... that does not ... or does not ...": 90 of 10,064
   parts) is not drawn in the new rows. The slots keep an and-filter only. It is listed for the order on what the
   descriptions cannot say. `not all` with one condition is said "at least one ... that" with the complement, so its
   target is `any`.
4. **A column with two values has one folded spelling:** arm M writes `==` with the value that holds ("is not part
   of a list" is `item == "no"`); arm W writes `not item == "yes"`. Where the words carry no negation ("ends with
   closing punctuation"), both arms write the comparison as stored.
5. **`view_wording_inv`** (goal first and passive) is held to the same list and the same pins.
6. **Replies that can tell are a target, not a gate, and for the rows that are read:** dev, test and test_long keep
   the 8 replies plus, where the pool has them, N - 1, N and N + 1 for each numbered part and a group's four
   combinations; train rows keep 8 (their replies only prove the row). The pool's size is Code A's call within about
   2.5 times today's build time. No reply is edited to reach a value in this order. Each read row records what it
   reached, and the report gives the shares (today: all three values for 14% of numbered parts, all four
   combinations for 10% of groups).
7. **"Numbers and strings only from the line's own text" (round 2's rule 9) is decided on this order's dev, not
   assumed:** each arm's dev is decoded with and without it (no training); the variant with fewer false passes is
   kept, on a tie the one with higher agreement, and it is frozen before test is read. Round 2's own 1-8 against 1-9
   dev read is reported beside it.
8. **The move:** only the spec language, `evaluate`, `atoms`, the compiler and the join go under `cubbyllm/speech/`;
   the drawing code (`sample`, `sample_atom`, `_cond`, `_mixed`) stays in `validation/`. The old names stay as shims
   for one cycle.

## The grammar is a state machine (Cowork, 2026-11-14; Nick: "yes add the idea")
Step 4's grammar is built this way. Where this section and step 4 differ, this section holds.

**Why.** Round 2's decoder asks the grammar, piece after piece, whether the answer with that piece still looks right
(`viable`). Seen on 0067's dev answers decoded on Colab (2026-11-14), and in the code:
- an allowed prefix can have no way on: 5 of the 21 unfinished dev answers stop at one (round 1 leaves 19 of 300
  dev answers unfinished, round 2 leaves 2 of its first 100);
- a piece that prints nothing is always allowed, because the text does not change (67 ids decode to the empty
  string; `chooser` judges `text + piece`);
- when none of the 512 best pieces fits, every id is judged one by one (13 s a scan before `ec7044c`).

This order's answer is one short line a part, so its grammar can be exact.

**What is built** (plain Python under `cubbyllm/speech/`, no new dependency):
1. **A machine per instruction.** States and moves for the line form of "The target": the kind, the join tag, the
   description's slots from their closed sets, `|`, the text copied in order from the instruction, then the next
   line or the end. It is built from the instruction itself: its text, the columns of each view and, in point 7's
   variant, the numbers and strings of the line's own text. The rules step 4 lists are the machine's moves, not
   checks made afterwards. Characters or bytes is Code A's choice.
2. **No dead end, counted in ids.** Every state the decoder can reach has at least one allowed id, and an accepting
   state can be reached from it. By construction, and pinned (tests below).
3. **Allowed ids are looked up.** The vocabulary is held once as a prefix tree of its pieces; a state's allowed ids
   come from walking the tree along the state's moves, and are kept per state. The whole vocabulary is never judged
   piece by piece.
4. **A piece that prints nothing is never allowed.** The end token is allowed in an accepting state and nowhere
   else.
5. **How an answer ends is recorded.** Each answer keeps `end` (`finished`, `error` or `limit`) and the number of
   ids generated. A row with no allowed id ends at once with its error, never at the token limit; with point 2 it
   cannot happen for arms M and W, and if it does the read is not scored until the cause is found. The read prints,
   per split and arm: finished, ended by error, ended at the limit.
6. **Arm P** keeps round 2's grammar (`TypedGrammar`, rules 1 to 9) under the same decoder: points 4 and 5 hold for
   it, points 1 to 3 do not. Its answers that end by error or at the limit are reported as such, beside "asked".

**Tests, added to the order's:**
- every gold target of both arms is accepted, and every prefix of it is allowed;
- the machines of at least 50 dev instructions are walked (whole, or to a stated bound where one is too large):
  every state reached has an allowed id and reaches an accepting state;
- refused, one case each: a slot outside its set, a column of another view, a number or string not in the line's
  text (in that variant), text out of order, a join tag on a line that is not a check, a piece that prints nothing;
- two implementations agree: on the gold targets and on their one-slot mutants, the machine accepts a complete
  answer if and only if `parse` with the host's checks accepts it.

**Not a dependency.** `google-research/opengrm` (Pynini and Thrax, on OpenFst) is the same idea as a library; its
builds are for Linux and macOS, so nothing under `cubbyllm/` imports it. Line 8 (time) is unchanged.

## Report
<!-- Claude Code fills this in, then sets status: review -->

### Step 1, first half, built: one compiler in the core, no change of behaviour (Cowork, 2026-11-14, branch `cowork/0073`)
Built by Cowork on a branch of its own while Code A was signed out (Nick, 2026-11-14: "we will continue here then").
Nothing here is on master; the review is Code A's or Nick's.
- **Moved:** `cubbyllm/speech/view_spec.py` (the spec language: `FILTERS`, `NUM_OPS`, `STR_OPS`, `cmp`, `match`,
  `evaluate`, `atoms`), `cubbyllm/speech/view_compile.py` (`body`, `views_of`, `program`) and
  `cubbyllm/speech/view_join.py` (`HEAD`, `spec`, `join`, `program`). The code is the builders', unchanged; the core
  imports nothing from `validation/`.
- **Stays in `validation/builders/view_programs.py`:** what draws a spec (`VIEW_WEIGHTS`, `held_out`, `_values`,
  `_cond`, `_mixed`, `sample_atom`, `sample`). It re-exports the core's names; `view_compile.py` and `view_join.py`
  there are shims. Every importer (the row builders, the SFT builder, `rerun_vm_sets`, `vm_key_scan`, the tests)
  is untouched.
- **Byte for byte** (`validation/exp_view_core_rebuild.py`, `validation/logs/view_core_rebuild_2026-11-14.json`),
  with the core's code only: long rows 3,900 of 3,900 programs identical and 31,087 of 31,087 stored verdicts; pair
  rows 3,000 of 3,000 and 23,993 of 23,993; step 2's program rows 2,000 of 2,000 and 15,956 of 15,956. No row
  differs.
- **Tests** (`tests/speech/test_view_core.py`, 7 cases): the old names are the core's objects, not copies; the three
  core modules name nothing under `validation/`; the evaluator, the compiler's text and the join's text pinned on a
  small spec. The existing tests of the builders, the reader and the grammar pass unchanged (50 in the run with
  these 7).
- **Not yet:** `absent: pass` for `at`, and `fold` (the second half of step 1).

### Step 1, second half, built: `absent: pass` and `fold` (Cowork, 2026-11-14, branch `cowork/0073`)
- **Said forms** (`cubbyllm/speech/view_spec.py`): a condition or an atom may carry `"not": True` (on a condition:
  its comparison; on an atom: its quantifier or comparison; on `at`: the comparison only), a part may be
  `{"not": atom}` as before, and `at` may carry `"absent": "pass"` ("if there is one": no such row, it holds). The
  evaluator reads all of them; a spec with none of them is read exactly as before.
- **`fold(part)`** writes the same part as one atom with no `not`: a comparison turned, `any` and `none` exchanged,
  `not all` of one condition as `any` of its complement, `not at` as the turned comparison with `absent: pass` (and
  back). A two-valued column (`line.item`, the only one) gets `==` with the value that holds, whatever was said.
  `not all` of two conditions raises: it has no folded form (point 3 above).
- **The compiler** (`view_compile.py`) writes `absent: pass` (the flag starts at 1 and is set again once the row is
  found) and refuses a said form: a part is folded first, so there is one program per meaning.
- **Pinned on every stored part and reply** (`validation/exp_view_fold_pin.py`,
  `validation/logs/view_fold_pin_2026-11-14.json`): 22,083 parts of the 3,900 long rows and 3,000 pair rows.
  - The fold keeps the evaluator's truth on every reply: 21,859 parts folded, none differs. 224 parts are `not all`
    of two conditions and have no folded form.
  - 58,705 other said forms of the same parts (the `not` moved onto the atom, onto each condition, around the part
    twice, onto the other quantifier) have the part's truth and fold to the same atom: none differs.
  - Where the fold changed the part (12,023 parts, 3,178 of them `absent: pass`), the folded atom compiled and run in
    the VM (`b3b3360`) gives the evaluator's verdict on every reply: 96,006 runs, none differs.
- **Nothing stored moved:** `exp_view_core_rebuild.py` again after the change: 3,900, 3,000 and 2,000 programs byte for
  byte, 31,087, 23,993 and 15,956 verdicts.
- **Tests** (`tests/speech/test_view_fold.py`, 6 cases): each fold on replies with and without the row, the
  two-valued column's one spelling, `not all` of two conditions refused, the compiler's refusal and the text of
  `absent: pass`.
- **Next:** step 2, the wording returns the part as said in both spellings.

### Step 2, built: the wording's said form in both spellings (Cowork, 2026-11-14, branch `cowork/0073`)
- **Where it lives:** `validation/builders/view_said.py`. `view_wording.py` is at 265 of its 300 lines, so the said
  forms sit beside it and follow its branches one by one; `view_said.say(part, lang, rng, how)` returns the sentence
  with arm W's form and arm M's (its fold) from one call. `how` is the imperative (`say` / `say_not`), the condition
  of an "if" or "unless" (`say_cond`), or the goal-first wording (`view_wording_inv.say_inverted`, point 5). The
  wording functions are untouched, so every stored sentence is what it was.
- **The rules, as settled above:** amounts as `amount` says them ("at least one" is `any`, "no" is `none`, a number
  keeps its comparison; a view's size and a numeric column never carry a `not`); a `not` only where a verb is negated;
  "if there is one" is `absent: pass`; a list line is said of `item == "yes"`.
- **What that gives, to read before step 3:**
  - English and French can differ on the same part in arm W, because only one of them negates a verb: "Do not
    include any ..." is `not any`, "Ne mets aucun ..." is `none`; "No sentence should begin with ..." is `none` of
    the plain condition, "Chaque phrase ne commence pas par ..." is `all` of `not` the condition. Arm M differs
    between the languages only in the second case (`none` against `all` of the turned comparison): both are true to
    their sentence.
  - A view's size has no quantifier slot: "Write at least one sentence" is `n >= 1`, "Write no sentences" `n == 0`.
  - "Use the word X at least once" is `any`, like "Include the word X".
  - `not all` of two conditions is worded with "or" and has no description (224 stored parts, 448 sentences).
- **Pinned on every stored part, English and French** (`validation/exp_view_said_pin.py`,
  `validation/logs/view_said_pin_2026-11-14.json`): 22,083 parts, 93,407 sentences (the imperative 22,083 in each
  language; the condition 9,950 in each; goal first 14,826 in English and 14,515 in French), 92,959 with a
  description, 19,303 of them carrying a `not` in arm W.
  - A `not` exactly where a verb is negated: each form carries as many `not` as its sentence has negation words
    ("not"; "pas"), quoted values aside: none differs. Arm M's forms carry none.
  - The truth is the stored part's: arm W's form and arm M's have the stored part's truth on every reply of its
    row: none differs.
  - One sentence, one target: 28,582 distinct sentences with a description, none with two forms in either arm.
  - The VM agrees: arm M's form compiled and run on the row's replies (the imperative in the row's language, where
    the form is not the stored part itself): 104,602 runs on `b3b3360`, none differs.
- **One fault found and fixed on the way,** in the pin's own counter: a value that is the quote mark itself
  (`"""`) hid a "not" from the count. The said form was right; the counter now reads such a value.
- **Tests** (`tests/validation/test_view_said.py`, 7 cases): each rule on a sentence of its own, in both languages
  where they differ.
- **Next:** step 3, the rows built again (`long_rows_0073`).

### Step 3, the design before the build (Cowork, 2026-11-14, branch `cowork/0073`): to be read before any row is trained on
Where the order leaves a choice, this is the one taken. Nothing below is built yet.
- **Today's draw, measured** (0067's train rows, their said forms): 6,159 numeric comparisons with a number other
  than 0: 1,554 lower bounds ("at least", "more than"), 2,682 upper bounds, 400 "exactly", 1,523 "other than". The
  lower share of the bounds is 0.367: a view's size 0.326, a count 0.106 (94 against 794), a position's value 0.489,
  a filter 0.481. Tied pairs: 772 start at part 1, 457 at part 2, 241 at part 3 (0.525 at the first).
- **Bounds balanced:** "within 45 to 55%" is read as the lower bounds' share of the bounds said with a number other
  than 0, in each split; "exactly" and "other than" are counted and reported beside it. The share is also reported
  by place (size, count, position, filter). How: the draw keeps a running count by place and, among the candidates
  it already tries for a part (the atom and its negation, six tries), takes the best one on the side that is behind
  when there is one. A count's lower bound passes fewer replies, so more draws fail; that is the cost.
- **`not all` of two conditions is not drawn** (point 3).
- **Groups evenly placed.** Two to four parts, exactly: two parts, a tied pair with probability 0.3; three parts, a
  pair at 1-2 or at 2-3 with 0.255 each; four parts, one pair at 1-2, 2-3 or 3-4 with 0.157, 0.247 and 0.157, two
  pairs (1-2 and 3-4) with 0.09, so each of the three places is tied in 0.247 of the rows. Five to eight parts: the
  places tried in a random order, each tied with 0.3 when both its parts are free.
- **Each connective before and after its partner:** "If C, A" and "A if C"; "A, unless B" and "Unless B, A"; "Si C,
  A" and "A si C"; "A, sauf si B" and "Sauf si B, A"; half each. "Either A, or B" has one order.
- **What a row carries besides today's fields:** for each check its said forms (arm W and arm M) and how it was
  worded; on the check that carries the connective, its tag (`if`, `unless` or `either`, with `prev` or `next`;
  "Either" sits on the first of its two checks: `either next`).
- **Replies that can tell** (dev, test, test_long; train keeps 8): the 8 as today, then from the row's pool, for each
  part whose own comparison has a number N (a view's size, a count, a position's numeric value) one reply measuring
  N - 1, one N and one N + 1 where the pool has one, and for each group one reply for each of the four combinations
  of its two parts' truths. Each added reply is run in the VM like the eight; no reply is edited. The row records
  which it reached.
- **Mutants** (every row, the evaluator only): each part's folded form with one slot reversed (a comparison negated,
  a bound mirrored, `any` and `none` exchanged, `absent` toggled), and each group with its join changed (untied; `if`
  for `or` and back; the `if` turned round). The row records how many there are and how many its replies tell apart
  from the gold.
- **Held out:** no row's instruction is an instruction of 0067's rows; no dev, test or test_long row has a train
  row's parts and groups (folded forms, values included). The share of held-out rows whose shape without its values
  occurs in train is reported: it will be high for two parts and is not a gate.
- **The pool** stays at 80 replies (160 for five to eight parts), refreshed every five rows, unless the reached
  shares are low; then it is raised within 2.5 times today's build time and the report says so.
- **1,500 minimal pairs** as in round 2 (one lone part flipped, the twin drawing its wording as the first did, 0.3
  goal first), under the same draw.

### Step 3, built: the rows (`long_rows_0073`) (Cowork, 2026-11-15, branch `cowork/0073`)
Built off-repo with `validation/builders/build_rows_0073.py` (seed 73): `rows.jsonl` (train 3,000, dev 300, test
300, test_long 300) and `pair_rows.jsonl` (1,500 pairs, 3,000 train rows). Counts and SHA-256 in
`validation/logs/long_rows_0073_2026-11-15_manifest.json`; the set read back from its files and checked in
`validation/logs/long_rows_0073_2026-11-15_check.json`. No adapter of this order exists yet, so the hashes of test
and test_long are committed before any training. Test and test_long rows were not opened by eye: only counts.
- **Run before kept:** 59,152 VM runs, 0 disagreements with the evaluator. A row's program is the join of its
  folded said forms; on every usable reply of each pool the evaluator's verdict on the folded forms was its verdict
  on the parts as drawn (a difference stops the build).
- **Read back** (`validation/exp_rows_0073_check.py`, the evaluator only): 6,900 rows, 21,749 checks, 59,152
  replies, 1,500 pairs. Each of its 18 counts that must be 0 is 0: an instruction of 0067's rows (6,900 looked
  up), an instruction or id twice, a train or dev instruction sharing 5 words with a benchmark, a part of a kind
  kept for the test, a held-out row with a train row's folded parts and groups, segments that do not tile, parts
  not numbered in the text's order, a join tag off its pair, a said form that is not the wording's, an arm M form
  that is not the fold of arm W's, an arm W form whose `not`s are not the negation words of the check's text, a
  program that is not the join of the folded forms, a stored verdict that is not the evaluator's (on the folded
  forms; on the parts as drawn), a train row with replies added, twins that differ in more than the flipped
  part's sentence, a SHA-256 that is not the manifest's.
- **Bounds** (lower bounds' share of the bounds said with a number other than 0; 0067's train rows: 0.367): train
  with its pairs 6,478 lower and 6,476 upper (0.5001), with 776 "exactly" and 2,227 "other than"; dev 335 and 335
  (0.5); test 318 and 319 (0.4992); test_long 745 and 745 (0.5). By place in train with its pairs: a view's size
  1,642 and 1,642; a count 1,323 and 1,321 (0067: 0.106); a position's value 615 and 614; a filter 2,898 and 2,899.
- **Tied pairs by place** (train's 3,000 rows): two parts, tied in 267 of 888 (0.301); three parts, 269 at 1-2 and
  269 at 2-3 of 1,054; four parts, 262, 261 and 262 of 1,058 (0067: 772 at part 1, 457 at part 2, 241 at part 3).
  Pair rows: three parts 278 and 278 of 1,092; four parts 184, 186 and 186 of 854; two parts never tied (below).
  Dev: 27 of 90; 25 and 25 of 98; 28, 27 and 28 of 112. Test: 29 of 95; 28 and 27 of 107; 25, 24 and 24 of 98.
- **Connectives** (train with its pairs): "if" said first 342 and after 340; "unless" first 347 and after 331;
  "either" 1,342. Dev: if 18 and 22, unless 22 and 11. Test: if 22 and 26, unless 16 and 14. Test_long: if 48 and
  55, unless 51 and 61.
- **Where the arms differ:** arm W's form carries a `not` in 4,032 of train's 17,970 checks (one 3,848, two 183,
  three 1), 230 of dev's 922, 222 of test's 903, 466 of test_long's 1,954. Goal-first wording is in the pairs only
  (1,727 checks), as in round 2.
- **Replies that can tell** (dev, test, test_long; the 8 as before, then those found):
  | set | replies a row | built to fail | numbered parts with N - 1, N and N + 1 | on deciding replies | tied pairs with their four combinations | on deciding replies |
  |---|---|---|---|---|---|---|
  | dev | 11.0 (3,300; 907 added) | 1,625 (0.492) | 371 of 449 (0.826) | 284 (0.633) | 126 of 160 (0.788) | 113 (0.706) |
  | test | 11.0 (3,291; 899 added) | 1,639 (0.498) | 365 of 457 (0.799) | 266 (0.582) | 116 of 157 (0.739) | 105 (0.669) |
  | test_long | 15.4 (4,621; 2,221 added) | 2,618 (0.567) | 866 of 1,033 (0.838) | 466 (0.451) | 296 of 394 (0.751) | 199 (0.505) |
  | train, 8 replies | 8.0 | 0.504 | 931 of 9,103 (0.102) | 367 (0.040) | 348 of 2,702 (0.129) | 132 (0.049) |

  A deciding reply is one where the row's verdict turns on that part or pair alone (see 2 below).
- **One-reversal mutants the replies tell apart from the gold:**
  | set | part mutants | told by the row's verdict | told by the part run alone | group mutants | told by the row's verdict | told by the changed groups alone |
  |---|---|---|---|---|---|---|
  | dev | 2,305 | 2,051 (0.890) | 2,219 (0.963) | 620 | 599 (0.966) | 617 (0.995) |
  | test | 2,270 | 1,993 (0.878) | 2,155 (0.949) | 605 | 579 (0.957) | 595 (0.983) |
  | test_long | 5,009 | 4,167 (0.832) | 4,817 (0.962) | 1,662 | 1,461 (0.879) | 1,636 (0.984) |
  | train, 8 replies | 45,747 | 35,004 (0.765) | 42,709 (0.934) | 10,796 | 7,586 (0.703) | 9,243 (0.856) |

  So on test about 12% of the wrong readings of one part pass every reply of their row, and 4% of the wrong joins:
  a false pass counted on these rows is a floor, not the whole rate.
- **Held out:** 1 test candidate had a train row's parts and groups and was dropped; none is left (read back). A
  held-out row's composition without its values is a train row's for 5 of dev's 90 two-part rows and 2 of test's
  95, and for no row of three parts or more.
- **Dropped while building:** too few usable replies on one side 215 (train), 272 (pairs, 209 of them the twin),
  13, 29 and 31 (dev, test, test_long); 5 words shared with a benchmark 35 (train), 15 (pairs), 1 (dev); a twin
  with no description or of a kind kept for the test 25.
- **Time:** 1,105 s (train 492, pairs 281, dev 70, test 79, test_long 184), so 0.16 s a train row, 0.23 s a dev
  row and 0.61 s a test_long row. 0067's builder, timed the same day on 220 rows while this build ran: 0.17, 0.2
  and 0.53 s. About the same on train and 1.2 times on the held-out sets, inside the 2.5 times point 6 allows.
- **What a row carries beyond 0067's fields:** `said` (for each part `how`, `w` arm W's form, `m` arm M's);
  `how` on each check segment and `join` (`[kind, "prev" | "next"]`) on the check that carries a connective;
  `join` (the program) is the join of the `m` forms; `replies[].extra` on a reply added; `reach` (`numbers`: part
  and 0, 1 or 2 for N - 1, N, N + 1; `groups`: group and the four combinations; 2 is a deciding reply); `mutants`
  (the six counts above). Ids: `d` and six digits (the first is the set), `dp` five digits and `a` or `b` for twins.
- **Tests:** `tests/validation/test_view_draw.py` (7), `test_view_compose.py` (8), `test_view_tell.py` (5),
  `test_rows_0073.py` (8: a small build run, read back with the checker, and a spoiled copy caught).

**Where the build differs from the design above** (each decided before any row is trained on):
1. **640 more replies are searched for the replies that can tell.** The parts are still drawn on a pool of 80 (160
   for five to eight parts), so the draw is the same in every set; the search alone also sees 640 replies drawn
   beside it. Measured on 40 dev rows a setting (not the same rows): all three values for 0.56, 0.66, 0.82, 0.85
   and 0.89 of numbered parts with 0, 160, 320, 640 and 1,280 more.
2. **A deciding reply is taken first.** A reply at N - 1 that another part fails changes nothing in the row's
   verdict. So the search first takes a reply on which the row's other groups hold and, for a tied part, its
   partner leaves the pair's verdict to it; then any reply. `reach` says which was found (2 or 1).
3. **Mutants:** a tied pair tied to its other neighbour is counted too (the design listed untied, `or` for `if`
   and back, the `if` turned round). Each mutant is counted two ways: told by the row's verdict (what line 3 can
   see) and told by what was changed run alone (what lines 5 and 6 can see).
4. **The placement is taken from the count of rows kept.** Drawn at random, the first build came out at 0.202 tied
   for two parts (0.3 intended) and 0.243, 0.260, 0.208 for four, because a row with a pair is dropped more often
   for too few replies on one side. That build was overwritten; only its counts were read. Now the placement furthest behind
   its share is taken. A minimal pair flips a lone part, so pair rows of two parts are never tied and of four have
   one pair at most (each place 0.217).
5. **A kind kept for the test is judged on the folded said form.** 0067's draw looked only at the atom a part
   equals, so "every word is other than X" (a word absent, said another way) went through: 41 parts in 0067's rows
   (37 train rows, 4 test_long rows). None here.
6. **Twins differ in one sentence.** The connective's order is now drawn, so a twin drawing "its wording as the
   first did" from one stream could drift after the flipped part. The layout and each part's wording draw from
   their own streams; read back, every pair differs in the flipped part's sentence only.
7. **The pool** is redrawn every five rows kept, or after ten misses running (0067 redrew it on every miss at a
   multiple of five).

### Step 4, the design before the build (Cowork, 2026-11-15, branch `cowork/0073`)
Where the order leaves a choice, this is the one taken. Nothing below is built yet. Counted on the rows of step 3
(both spellings of all 21,749 checks): a filter has one or two conditions, never none; a position is 1, 2, 3 or the
last; the largest number is 905 and the longest string 9 characters; arm W writes a `not` on `any`, on a condition
and on a position's comparison, and nowhere else.
- **The answer:** a newline, then one line a segment, in the text's order; nothing after the last line.
  `<kind>[ <join>][ <description>] | <text>`, as "The target" has it.
- **The description, one line, every slot from a closed set:**
  - `n <view> <op> <number>`
  - `count <view> <op> <number> where <condition>[ and <condition>]`
  - `any | none | all <view> where <condition>[ and <condition>]`
  - `at <view> <position> <column> <op> <value>[ absent pass]`
  - a condition is `<column> <op> <value>`; a position is `1`, `2`, ... or `last`; a number is digits; a string is
    in double quotes, written as JSON writes it; `<op>` is one of the six for a number and `==` or `!=` for a
    string; a column is one of the view's (`view_spec.FILTERS`).
  - **Arm W** may write `not ` before the quantifier, before a condition and before a position's column
    (`not any word where lower == "so"`; `any sentence where not first_word == "so"`; `at sentence 1 not first_word
    == "so" absent pass`). **Arm M** has no `not`. That is the only difference between the two grammars.
- **The join tag,** after `check`: `either next`, `if next`, `if prev`, `unless next`, `unless prev`. The tagged
  check is the condition ("if", "unless") or the first of the two ("either"). The grammar allows a tag on any check
  line and does not look for its partner: a tag the host cannot resolve makes the part "asked", which is the safe
  outcome; forbidding it would turn the pair into two plain checks without a word.
- **The text is copied with no gap.** A line's text starts at the first character after the last line's text that
  is not white space, runs forward character by character, never crosses a line break of the instruction, and
  does not end on white space. The answer can end only when the instruction is used up. Two things follow by
  construction: no sentence can be left out (line 1's silent drops), and the reader cannot stop early (round 2
  left 189 of 224 long answers unfinished). The reader still chooses where each line ends and what kind it is.
  This is stricter than "copied in order"; it is the one choice here that goes past the order's words.
- **Targets carry the text without its outer white space** (a segment such as "If C, " is written "If C,"); spans
  are compared the same way in the scoring.
- **Point 7's variant ("numbers and strings only from the line's own text"), made exact.** The description comes
  before its text, so the rule is held in two moves: a value must be in the instruction between the line's start
  and the next line break (a number as digits or as a number word, 0 and 1 always; a position 1, an ordinal or a
  number of the text, `last` always; a string one of its column's own values or quoted in the text; a view named
  in the text, any view when none is named: round 2's rule 9, `view_words`), and the line's text must then run on
  until it holds every value its description used. The other variant leaves numbers (up to 4 digits) and strings
  (up to 40 characters) free. Both are built; dev decides (point 7).
- **The machine** (characters; plain Python): a state is where the line is (which slot, what was chosen that the
  next slots depend on: the view, the column's type, how far the text must run) or a place in the instruction.
  Every slot's choices are written with the delimiter that follows them, so none is the start of another. From
  every state a line can be finished and the instruction used up, so there is no dead end in characters; in ids
  it is pinned on the real vocabulary. Allowed ids come from walking a prefix tree of the vocabulary's pieces
  along the machine's moves, kept per state; inside a free string, the pieces with no quote or backslash are kept
  by length once for the vocabulary. A piece that prints nothing is never allowed; the end token only in the
  accepting state.
- **`parse(answer, instruction, arm)`** is written apart from the machine (a line pattern, then the description
  read slot by slot against the same closed sets, then the copy rule), so the two can be compared: on the gold
  targets and their one-slot mutants, the machine accepts if and only if `parse` does. With `strict=False` it
  finds each text going forward and allows gaps (for answers that did not come through the machine, arm P's
  split among them).
- **The host's accounting** (`account`): checks are numbered in line order; each description is folded (arm W)
  and compiled; a description that does not compile makes its part "asked"; a tag whose neighbour on that side is
  not a check, or is already in a pair, makes its part "asked"; **a pair with an "asked" member is "asked"
  whole** (one half of "A unless B" checked alone would fail replies the instruction allows). Every stretch of
  the instruction that is not in a line is flagged (none can be, through the machine). It returns the lines, the
  parts and groups it could check, what is asked or flagged and why, and the program (the join of the parts it
  could check). How a row with an "asked" part counts in lines 3 and 7 is step 6's to state before any read.
- **Files** (each under 300 lines, in `cubbyllm/speech/`): `view_desc.py` (the notation: write, read),
  `view_split.py` (`prompt`, `target(row, arm)`, `parse`), `view_own.py` (what a line's text allows, with where
  each value is), `view_machine.py` (the machine), `view_tokens.py` (the prefix tree, allowed ids, a decoder's
  step with `end` recorded), `view_account.py`. 0067's `view_reader.py`, `view_grammar.py`, `view_types.py` and
  `view_words.py` stay as they are: arm P reads with them.
- **Tests,** the order's and the machine section's: every gold target of both arms accepted and each of its
  prefixes allowed (all 6,900 rows, characters; ids on a sample); 50 dev machines walked; one refusal each; the
  two implementations agree; the accounting on a missing sentence, a tag with no partner, a description that
  does not compile.

### Step 4, built: the reader's form, its grammar as a state machine, the host's accounting (Cowork, 2026-11-15, branch `cowork/0073`)
In `cubbyllm/speech/` (each file under 300 lines; the core imports nothing from `validation/`): `view_desc.py` (the
description's notation: write, read), `view_split.py` (`prompt`, `target(row, arm)`, `parse`), `view_own.py` (what a
line's own text lets its description say), `view_machine.py` (the machine, in characters), `view_tokens.py` (the
prefix tree, the ids allowed from a state, `Reader` with `end`), `view_account.py` (the host's accounting). 0067's
reader and grammar are untouched: arm P reads with them. Checked on every row of step 3
(`validation/exp_view_split_gold.py`, log `validation/logs/view_split_gold_2026-11-15.json`; counts only):
- **The form, on all 6,900 rows and both arms:** each target parses back strictly, and with the own-text rule, to
  the row's kinds, tags, said forms and spans (6,900 of 6,900, four times). A target is about 430 characters at
  the median and 1,127 at most; round 2's program alone was about 1,500.
- **The accounting:** from those lines it gives the row's groups and, byte for byte, the program step 3 ran in the
  VM, for 6,900 of 6,900 in each arm; nothing asked, nothing flagged.
- **The machine:** it takes every target whole in both arms and both variants (27,600 runs), so each prefix of each
  is allowed. Arm M's machine refuses all 3,587 arm W targets that differ from arm M's.
- **Two implementations agree:** on 48,000 one-slot changes of the gold answers (dev and 300 train rows, both
  arms: a comparison, a view, a column, a number, a string, a position, a `not`, `absent`, the quantifier, a
  condition more or fewer, a join tag, a kind, a line cut in two, a line left out or added, the last character
  left out) the machine takes an answer whole if and only if `parse` reads it: 28,655 taken by both and 19,345
  refused by both with values free; 22,156 and 25,844 with the own-text rule; the two never differ.
- **In ids, with the base's tokenizer** (127,996 ids; 67 print nothing and are never allowed; every character of
  the instructions has a piece of its own): 400 gold targets (100 dev rows, both arms, both variants), encoded as
  the trainer encodes them, walk to the end token and end `finished`. The grammar costs 1.7 ms an id at the median
  (2.3 ms on the slowest target), about 0.2 s for an answer of 110 ids, and no id is ever judged one by one. A
  state allows 5 ids at the median and 127,259 inside a free string.
- **No dead end:** the machines of 50 dev instructions (200 machines) walked at random in ids reached 417,940
  states; none was left without an allowed id unless the answer was whole, and from each of 8,000 of them the
  plainest way on reaches a whole answer.
- **The own-text rule holds on the gold:** all 43,498 gold descriptions (21,749 checks, both spellings) say only
  what their line's text says, and the text holds every value by the line's end.
- **Tests:** `tests/speech/test_view_desc.py` (26), `test_view_split.py` (5), `test_view_own.py` (7),
  `test_view_machine.py` (22: one refusal each among them), `test_view_tokens.py` (4), `test_view_account.py` (5: a
  missing sentence, a tag with no partner, a description that does not compile, a tag on a check already in a
  pair), `tests/validation/test_view_split_gold.py` (3, on a small build).

**Where the build differs from the design above:**
1. **One space may come before the answer's first newline.** The trainer encodes a space and then the answer
   (`talk_data.encode_records`), so that is what an adapter is taught to write first. The machine and `parse` take
   it. Round 2's grammar refused it at the first token.
2. **A line may hold several sentences,** up to the instruction's next line break. The design did not say; the
   reader chooses where a line ends, and the scoring compares its spans with the gold.
3. **A column the view compares lowercased** takes a quoted string as quoted or in lower case. Round 2's rule 9
   lowercased both sides; 59 gold checks compare a character in upper case, as the text quotes it.
4. **A free string stays free while its line quotes nothing.** With the own-text rule a string of a column with no
   values of its own must be one the line's text quotes, unless that text quotes nothing (rule 9's own wording). So
   the machine lets such a string be anything, and then the line must end before the first quote; if the string is
   one the text quotes, the line may also run past it. The same for a view: any view, if the line ends before a
   view is named.
5. **With the own-text rule a check's line does not end inside a quoted string** (what its text quotes would then
   not be what the description was held to). Both the machine and `parse`.
6. **The accounting also asks the check an unresolved tag points at,** not only the tagged one, and both members
   of a pair when one is asked: a check that was a condition's other half is not checked alone.
7. **The machine does not look for a tag's partner** (as designed), and it does not hold a line to a sentence's
   end: it knows line breaks, not sentences.

### Step 5, built: the two SFT sets and their notebooks; step 6's decoder (Cowork, 2026-11-15, branch `cowork/0073`)
Nothing is trained: the Colab runs are Nick's.
- **The sets** (`validation/sft/build_view_split_sft.py`; off-repo, on Drive under `rules/`; manifests in
  `validation/logs/view_split_sft_m.manifest.json` and `..._w.manifest.json`): `view_split_sft_m.jsonl` and
  `view_split_sft_w.jsonl`, 6,300 records each: 6,000 train (step 3's 3,000 train rows and its 1,500 pairs) and 300
  held (dev). Test and test_long are in neither. The two sets have the same ids, prompts and splits in the same
  order; their answers differ on 3,200 records, the ones where the words negate.
  - Every answer reads back strictly, also with the own-text rule (6,300 of 6,300 in each arm): no answer is taught
    that the grammar would refuse. No instruction shares 5 words with IFEval or IFBench.
  - Tokens a record, as the trainer counts them: 216 at the median (217 in arm W), 389 at most; none is over the
    trainer's 1,280.
  - **The key tokens** (`validation/key_tokens.py`, weighted 4 in training) are each check's join tag and
    description: 166,214 of arm M's 703,542 answer tokens (0.236) and 170,676 of arm W's 708,004 (0.241). Round
    2's were about 14% of an answer. With no `---` in an answer the spans are the slots; with one, round 2's.
  - 0067's set also taught one-part instructions and CubeLang programs to continue. Neither is here: this reader
    writes no program, and "same records, the target in arm M or arm W" is kept to step 3's rows.
- **The notebooks** (`notebooks/view_split_m.ipynb`, `view_split_w.ipynb`, from `_build_view_split.py`): round 2's
  recipe and cell, unchanged (rank 16, alpha 32, five epochs, lr 2e-4, batch 32, max length 1,280, seed 0, key
  weight 4). They differ in the set read and the folder written (`rules/cubby450m_view_split_m`, `..._w`). Each
  refuses to train unless the set is its arm's, has 6,000 train and 300 held records, and was built from the rows
  whose SHA-256 is in the code it cloned (`long_rows_0073_2026-11-15_manifest.json`).
- **They clone branch `cowork/0073`,** which is on this PC only. Colab needs it on GitHub: a push of the branch is
  Nick's yes. Once the order is merged, `BRANCH` in `_build_view_split.py` goes to `master` and the notebooks are
  made again.
- **Step 6's decoder,** written with it so the read needs no second push (`validation/exp_view_split_torch.py`):
  one decoder for every arm. Arms M and W: each row's answer is held by its instruction's machine and the next
  token is the best-scored of the ids its state allows; `--own-text` for point 7's variant. Arm P: round 2's
  chooser (`exp_view_reader_torch.decode`, which now takes `skip_empty` and records `end` and `ids`; off, it reads
  as it did). Every answer records `end` and `ids`.
- **Tests:** `tests/validation/test_key_tokens.py` (5: the new form's spans among them), `test_view_split_sft.py`
  (3, on a small build), `test_view_split_torch.py` (6: a model that scores the gold token highest decodes the
  gold answers; one that always wants the end token still writes a whole answer inside the grammar; an answer
  that outruns the limit ends `limit`).
- **Left of step 6:** the read notebook (dev first, both variants; the variant frozen in the repo before test is
  read), the rows the read needs on Drive (ids and instructions only), `exp_view_reader.py --score` on
  descriptions with what "Acceptance" lists, the second chance on dev. Step 7 is arm P's read, with the same
  decoder.

### Step 6, the scoring stated before any read (Cowork, 2026-11-15, branch `cowork/0073`)
No adapter of this order exists and no row has been read. This is how each line of "Acceptance" is counted; it is
written now so that no count is chosen after seeing answers. Nothing below is built yet.
- **An answer** comes through the machine, so it is read strictly. One that did not end `finished` (the limit, or an
  error) is read loosely on its whole lines; what they do not hold is flagged. Such a row is **not whole**.
- **A row is checked** when its answer is whole and the host's accounting has nothing asked and nothing flagged.
  Its program is then run on each of its replies; a reply whose views are too long for the VM is not checked.
  **Every reply of a row that is not checked counts as not checked** (line 7): a program that leaves a part out
  gives no verdict on the instruction, and the host says so.
- **Line 1 (accounted):** the characters of the instruction that are in no line and are not flagged. The
  accounting flags every stretch outside the lines, so this is 0 by construction and is reported as counted; with
  it, the rows that are not whole and the rows with something flagged.
- **Line 2 (the split):** a gold check counts when the answer has a check line with its span (outer white space
  aside). Over all gold checks. Reported with it: every gold segment's span and kind.
- **Line 3 (the join):** over the replies of checked rows: agreement (the VM's verdict is the generator's) and
  false passes (the VM passes a reply the generator fails); false passes also over the replies built to fail. A
  gold check the reader wrote as `none`, `task` or `process` leaves its row checked, so a reply that fails only
  that check is a false pass here: the reader's silence is counted where it costs.
- **Line 4 (longer than seen):** line 3's two rates on test_long; and line 5's rate for the gold checks at
  positions 5 to 8 against positions 1 to 4, both on test_long.
- **Line 5 (parts):** a gold check is right when the answer has its check line (line 2), the description compiles,
  and run alone it gives the generator's verdict for that check on every reply of the row. A description equal to
  the gold folded form is right without a run (the same program); any other is run in the VM. Over all gold
  checks. **Fixed by one reversal:** a check that is not right, whose line and description are there, and for which
  one reversal of one slot of its folded form (`view_tell.part_mutants`: a comparison negated or mirrored, `any`
  and `none` exchanged, `absent` toggled) gives the generator's verdict on every reply (the evaluator); over all
  gold checks.
- **Line 6 (groups):** a gold pair is right when the answer ties the same two checks (their spans) with the same
  connective in the same roles (which one is the condition). Over all gold pairs. Reported with it: pairs right
  when "unless" and "either" are taken as the same (they compile the same), and the pairs the reader made that
  are not gold pairs.
- **Line 7 (coverage):** checked replies over all replies; and, over all gold checks, the ones asked: written as
  `asked` by the reader, or asked by the host (a description that does not compile, a tag it cannot resolve, the
  other half of such a pair).
- **Line 8 (time)** is measured at home on grilly2, one instruction at a time, on dev rows, after an adapter
  exists; it needs a local read of the new form and is the last piece built.
- **Line 9 (which arm):** as written. The 95% interval is over rows: 2,000 resamples of the test rows with
  replacement, the same resamples for both arms, on the difference of the two rates.
- **Line 10** is counted on line 5's "right" for each gold check by: the direction of its bound (from its gold
  folded form, at its own comparison: lower, upper, exactly, other than; no number: apart), with the unweighted
  mean of the four; the `not`s in arm W's gold form (none, one, two or more); the place of its number (size, count,
  position; a filter's number is counted by whether the check has one); the row's number of parts; the check's
  position; the language. Groups by connective and by place in the row (line 6). Rows by number of parts and
  language (line 3). "Both twins of a pair right" cannot be read on held-out rows: the pairs are train rows. It is
  reported as not measured. The share of mutants the replies could tell is step 3's, per split.
- **Arm P** is scored with round 2's scorer on the same rows (`exp_view_reader.py --score`, lines 1 to 4 as round
  2 counted them) and set beside M and W on lines 2, 3 and 4. Its answers that end by error or at the limit are
  reported as such.
- **Point 7 (the own-text rule), on dev only:** each arm's dev is read with and without it. Kept for an arm: the
  variant with fewer false passes over checked replies; on a tie, the higher agreement; on a tie again, the rule
  on. The choice is committed (`validation/logs/view_split_point7.json`, with dev's counts) before the notebook
  will read test or test_long.
- **What Colab gets:** a file of ids, splits and instructions only (`rules/long_rows_0073/read_rows.jsonl`). No
  target and no reply leaves the PC.

### Step 6, the scorer built (Cowork, 2026-11-16, branch `cowork/0073`)
No adapter exists yet and no answer has been read. The scorer is checked on a small build: its gold answers, and
answers changed by hand.
- `validation/view_split_rowscore.py` scores one row as stated above (`score`).
  `validation/exp_view_split_score.py` adds the rows up into each line's counts (`tally`, `rates`); every rate is
  given with its two counts. `--keep` writes one line a row with its id and its counts (no text), which line 9's
  resamples read.
- **One thing the statement above left open, settled now, before any answer exists.** A line the reader writes
  `asked` is a sentence the host asks about, and the gold rows have about one each (dev: 310 in 300 rows). So an
  `asked` line cannot make a row "not checked". Where the gold has a check there, the row stays checked, as for
  `none`, `task` and `process`, and a reply that fails only that check is a false pass at line 3. Those false
  passes are also counted apart ("on rows where the reader wrote a gold check as asked"), and the check counts at
  line 7 as asked by the reader.
- A row that was not read counts in every denominator: all of it is flagged and none of its replies is checked.
- Tests (`tests/validation/test_view_split_score.py`, 7): the gold answer of every row scores whole, checked and
  right on every count, in both arms (arm W under the own-text rule); a reversed comparison is not right and is
  fixed by one reversal (6 cases; the row's verdict changes on some reply); a gold check written `none` or `asked`
  leaves the row checked, and its false passes are exactly the replies the rest of the row passes and the
  generator fails; an answer cut at the limit is not whole, its rest is flagged and no reply is checked; "unless"
  written as "either" is right only where the two are taken as one, a dropped tag and an "if" turned round are
  not right; a description that does not compile is asked by the host and its row is not checked; the script
  scores a file of answers and keeps the rows' counts.
- Still open in step 6: the rows file for Colab (ids and instructions only), the read notebook, the point 7
  file, line 9's comparison, line 8's timing at home.

### Step 6, where each line is judged, and point 7's rule in numbers (Cowork, 2026-11-16, before any answer exists)
No adapter of this order is trained yet. Two things "Acceptance" leaves unsaid are fixed here, so that neither is
chosen after seeing counts.
- **Which set a line is judged on.** Lines 2, 3, 5 and 6 name no set. They are judged on **test** (2 to 4 parts,
  the lengths the reader was trained on), and their kills are test's, as round 2's line 3 was (its kill was hit on
  test). The same counts on test_long are reported beside them and decide nothing there. **Line 4 judges
  test_long:** its agreement against test's, its own false passes with their own kill, and parts right at
  positions 5 to 8 against 1 to 4. Line 7 names both sets. Line 1's kill holds on both: a silent drop anywhere.
- **Point 7's rule in numbers** (`validation/exp_view_split_point7.py`, 4 tests). The two variants of an arm do
  not check the same number of replies, so "fewer false passes over checked replies" compares false passes /
  checked replies as exact fractions, and "the higher agreement" compares agree / checked replies the same way. A
  variant that checked no reply loses to one that checked some. A report is taken only if it is that arm's dev
  alone, read whole, in that variant, and all four were scored on the same rows. The file it writes
  (`validation/logs/view_split_point7.json`) holds each arm's choice with dev's counts; the test notebook reads
  the choice from it.
