# 0073 - The instruction reader writes descriptions and the host compiles them (round 3: two spellings of a negation, read side by side)
status: open (Code A; starts when 0067's round 2 is closed: its dev scoring and its counted read)
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
