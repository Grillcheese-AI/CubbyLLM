# Brief: the instruction reader, and what it should write (2026-11-13)

Written by Cowork for an OpenRouter panel (Nick: "would it be worth it to open a panel for our instruction problem?",
then "yep launch the panel"). Every fact below comes from GRL's repositories, orders and logs. Where something is not
stated here, say you don't know; don't fill it in.

## 1. The task
A user gives Cubby a request with instructions about the reply ("at most 3 sentences, none starting with a number,
unless the reply has a list"). A reader turns the instruction into a check that the VM runs on the reply. The reader
is an adapter on Cubby's own model. Today it writes two things: a split of the instruction into parts, and one
CubeLang program with a function per part and a main that joins them. Measured (section 4), it now splits well and
reads the meaning of a part less well; what is left is mostly a negation said the wrong way.

Deliver, in this order:
1. **Diagnosis.** For each error class of section 4.5, say which cause the numbers point to: the form of the target
   (about 1,500 characters of program), the gold's two spellings (4.6), the imbalance between bounds (4.6), the
   model, greedy decoding under a grammar, or the training. Say which are structural to "a model of this size writes
   a program this long" and which the data can fix.
2. **What the reader should write.** Compare three targets:
   - (A) the program, as today, held to a grammar that grows a rule at a time (3.5);
   - (B) the generator's own description of each part and of the groups (3.3), which the host compiles into the
     program;
   - (C) a mix: descriptions where one fits, a free program only where none does.
   For each: which error classes of 4.5 it removes by construction and which it leaves; what it costs in generality
   (an instruction whose check is not a composition the description language has); what happens to an unseen kind of
   instruction (line 5 of section 2); the decode time; and a set we could build to measure the generality lost.
3. **Negation and direction.** At most three proposals beyond those of section 5, ranked by expected gain per cost.
   For each: what changes, which set reads it, a pass line and a kill line written now, and the cost.
4. **The join and long instructions.** How the logic between parts (and, either-or, if, unless) and instructions
   longer than any seen in training should be read. Proposals with lines.
5. **The measure.** Are the lines of section 2 the right bar? What else should be reported (by direction of a bound,
   by depth of negation, by number of parts)?
6. **What you would not do,** and your open questions.

Keep it under about 4,000 words. Plain, exact prose. Cite the numbers you rely on by section.

## 2. Hard rules and lines (each comes from a written decision; don't trade any away)
- **Only the VM decides.** The reader proposes a check; the VM runs it on the reply. A part whose program does not
  compile, or that the reader is not held to, is "asked": shown to the user as not enforced. Nothing is dropped
  silently.
- **A false pass is the worst outcome:** the reader's program passes a reply that the true check fails. "Asked" is
  better than a wrong check.
- **No model writes a rule, a program or a training target.** Training rows are built in reverse: a generator draws
  the check, words it from templates (English and French), and each program is run in the VM on replies built to
  pass and to fail before the row is kept. A model may draft wording variations only. No external model at serve time.
- **No benchmark sentence in a training row** (public instruction benchmarks are tests only).
- **No host code for one kind of instruction.** The point of this reader is that an unseen instruction is a new
  composition of a few views of the reply, not a new hand rule.
- **Pre-registration.** Lines are written before a run; each held-out set is read once; a fix designed after seeing
  test rows is read on rows no one has seen.
- **The lines of this reader (order 0067, written before any read):**
  1. every sentence accounted for (a part, "asked" or "not an instruction"); dropped silently: 0 (kill);
  2. the split: each part's span and kind recovered for at least 90% of parts, on held-out compositions;
  3. the join: the reader's program agrees with the generator's on at least 95% of replies; false passes at most 2%
     (kill above 5%);
  4. longer than seen: trained on up to 4 parts, read on 5 to 8; agreement at least 90% of the in-length rate;
  5. unseen kinds (held-out kinds and a public benchmark's out-of-domain constraints): a compiling program for at
     least half; where written, agreement at least 95% and false passes at most 2% (kill above 5%).
- **The aim for the whole system:** a frontier model's quality on its tasks at one hundredth of the cost or less. No
  time line is set for reading an instruction; say what line you would set.

## 3. The system
### 3.1 The model
Cubby is GRL's own model: 32 layers of width 1,024, a recurrent mixer with windowed attention (a 512-token window, 4
heads, rotary positions) every third layer, and a 128,000-token vocabulary; about 576M weights in all, 445M read per
token. One LoRA adapter (rank 16) per job on a shared base. Served on GRL's own Vulkan compute framework on a home
GPU.

### 3.2 The reply as views
The host hands the VM the reply as lists: characters, words, sentences, lines, paragraphs, list items, JSON fields.
Each has a size and columns (a sentence's number of words, first word and end mark; a list item's marker; a word's
length, case and last letter; a JSON field's key, type and value; and so on: 17 columns in use).

### 3.3 The generator's parts and groups
A part is one check, described by a few slots:
- `n`: a view's size compared to a number;
- `count`: how many elements pass a filter (one or two column comparisons), compared to a number;
- `any`, `none`, `all`: whether some, no or every element passes a filter;
- `at`: the element at a position (first, second, last), one column compared to a value. Said with "if there is
  one", the part passes when there is no such element; the generator writes that as a `not` around the opposite test;
- any part may be wrapped in a `not`.

Groups join parts: a part alone (`one`), `either` (or), `unless`, `if`. Parts and groups not otherwise joined are
joined by "and". Over train and dev (3,300 rows, 10,064 parts): `count` 2,405, `at` 1,789, `n` 1,748, `any` 1,434,
`none` 1,390, `all` 1,298; groups: `one` 6,830, `either` 838, `if` 396, `unless` 383.

The set: 3,900 rows. Train 3,000 and dev 300 (2 to 4 parts); test 300 (2 to 4 parts, compositions not in train);
test_long 300 (5 to 8 parts). Round 2 adds 3,000 rows in minimal pairs (4.3).

One training row (shortened):

    Instruction: "The second sentence, if there is one, should not end with a question mark. Include at least one
      paragraph that has at most 2 sentences, unless your reply has at most 2 paragraphs that have more than 6
      words." (plus a task sentence and three other sentences)
    Parts:  [{"not": {"q": "at", "view": "sentence", "pos": 2, "col": "end_mark", "op": "==", "val": "?"}},
             {"not": {"q": "all", "view": "paragraph", "where": [{"col": "n_sentence", "op": ">", "v": 2}]}},
             {"q": "count", "view": "paragraph", "where": [{"col": "n_word", "op": ">", "v": 6}], "op": "<=", "v": 2}]
    Groups: [["one", 0], ["unless", 1, 2]]
    Program, part 2 and the main:
      function part2(n_paragraph: number, paragraph_n_sentence: number): number {
          create k1: number; assign k1 = 0;
          create i2: number; assign i2 = 0;
          while (i2 < n_paragraph) { let x3 = paragraph_n_sentence[i2]; if (x3 > 2) { add k1, 1; } add i2, 1; }
          create c4: number; assign c4 = 0;
          if (k1 == n_paragraph) { assign c4 = 1; }
          create c5: number; assign c5 = 0;
          if (c4 == 0) { assign c5 = 1; }
          return c5;
      }
      public function solve(mention: str): number {
          let p1 = part1(n_sentence, sentence_end_mark);
          let p2 = part2(n_paragraph, paragraph_n_sentence);
          let p3 = part3(n_paragraph, paragraph_n_word);
          create g1: number; assign g1 = 0;
          if (p2 == 1) { assign g1 = 1; } if (p3 == 1) { assign g1 = 1; }
          create g2: number; assign g2 = 0;
          if (p1 == 1) { if (g1 == 1) { assign g2 = 1; } }
          return g2;
      }

### 3.4 What the reader writes today
First the split, one line per sentence or clause, then the program:

    check 1 | The last JSON field, if there is one, should have the value "null".
    check 2 | Your reply should have fewer than 9 lines
    check 3 condition | , unless your reply has a number of lines other than 3.
    process | Read the question carefully first.
    none | This is for our team newsletter.
    check 4 | The first list item, if there is one, should not be marked with "-".
    ---
    program Instruction implements ISolve { ... }

### 3.5 Decoding
Greedy, each token held to a grammar: CubeLang's syntax, plus nine rules added on 2026-11-13 as errors were found
(types; a part's own names; calls in order; "and" by default, an or / if group only when the instruction's words say
so; the join in the generator's form; no parameter declared twice; one view per loop; parts numbered and called in
order; a part says only what its own words say: numbers, positions, views). Each rule removes one free choice from
the program's text.

### 3.6 Training
Round 1: 9,018 records, 788 steps, held-out answer loss 0.050. Round 2: 12,018 records (round 1's, plus the 3,000
pair rows, 30% of whose checks are said goal first or in the passive), the program's key tokens weighted 4 times in the loss,
five epochs, 1,782 steps, held-out answer loss 0.034.

## 4. What is measured
Agreement and false passes are counted over the replies the VM could check (each row has replies built to pass and
to fail). "Span and kind" is line 2. A part "fixed by one reversal" becomes the generator's part on every reply when
one comparison is negated or mirrored, or the part's `not` is toggled.

### 4.1 Round 1 on test rows no read had seen (50 rows a split; grammar rules 1 to 5; read at home)
| | test (2 to 4 parts) | test_long (5 to 8 parts) |
|---|---|---|
| span and kind | 99 / 151 = 0.656 | 188 / 319 = 0.589 |
| replies the VM could check | 384 / 400 = 0.960 | 312 / 360 = 0.867 |
| agreement | 195 / 384 = 0.508 | 171 / 312 = 0.548 |
| false passes | 44 / 384 = 0.115 | 35 / 312 = 0.112 |

Lines 2 and 3 missed, the kill line of 3 hit, on both splits. Length was not the trouble in round 1: the long split
agrees as often as the short one.

### 4.2 Round 1 on 100 dev rows (grammar rules 1 to 8)
Agreement 433 / 774 = 0.559; false passes 97 / 774 = 0.125. Of 312 parts: right as written 101; fixed by one
reversal 70; wrong with no single reversal 105; not matched to a part's span 36. Programs right as written: 6 of 100.
The 70 reversals: 35 `==` / `!=`, 29 a comparison for its complement (`>` for `<=`), 5 mirrored (`>` for `<`), 1 a
missing `not`.

### 4.3 Round 1 against round 2 on the same 64 dev rows
The shorter 64 of the 100. Round 1 was read at home with rules 1 to 8, round 2 on a rented card in batches with rules
1 to 9, so decoder and grammar differ as well as training. Scored by Cowork with the repo's tools on the dev rows read
so far; the scoring of all 100 follows.

| | round 1 | round 2 |
|---|---|---|
| parts right as written | 59 / 196 | 141 / 198 |
| parts fixed by one reversal | 49 | 30 |
| parts wrong with no single reversal | 66 | 14 |
| parts not matched to a span | 22 | 13 |
| programs right as written | 5 / 64 | 25 / 64 |
| span and kind | 0.762 | 0.889 |
| agreement | 289 / 502 = 0.576 | 337 / 492 = 0.685 |
| false passes | 63 / 502 = 0.1255 | 46 / 492 = 0.0935 |

### 4.4 Round 2's counted read (test and test_long, rows no one has read)
**Not read yet.** Round 2's numbers in 4.3 are dev numbers. Its read on test and test_long rows that no one has seen
comes after this brief. Treat round 2 on unseen compositions, and on 5 to 8 parts, as unknown.

### 4.5 What is left, on dev
- **Negation.** Round 2's 30 reversals on the 64 rows: 29 negations (a comparison for its complement, or `==` for
  `!=`), 1 mirrored. By place: 15 in a count's filter or a position's value, 10 where a count is compared, 5 at a
  view's size.
- **The join.** Seen by eye on three dev rows, not yet counted: where the instruction ties checks 2 and 3 ("check 2,
  unless check 3"), the reader's or-group joined checks 1 and 2, every part being right in one of them.
- **Long instructions.** Not counted (4.4). Seen by eye on two rows of 5 checks in round 2: the fifth check was
  labelled "check 1" again and no fifth part was written. Round 1 showed no loss on long rows (4.1).

### 4.6 The gold says one meaning two ways (train and dev, counted by Cowork)
- Every comparison of a count or of a size is written two ways in near-equal shares: itself, or a `not` around its
  opposite. Count at most N: 244 as `<=`, 260 as `not >`. Count fewer than N: 210 and 194. Count other than N: 329
  and 361. Size at most N: 196 and 189. All 18 such meanings, about 4,150 parts. Likewise "none" (996) against
  `not any` (1,051), and "any" (383) against `not none` (394).
- The wording does not follow the spelling. In the row of 3.3 the words say "at least one paragraph that has at most
  2 sentences" and the part says `not all` paragraphs have more than 2.
- After `>` or `>=` the gold goes on with a `not` in 64 to 68% of size parts and 77 to 90% of count parts; after `<`
  or `<=`, in 9 to 32%.
- Bounds are not balanced. Among comparisons with a number other than 0, 548 parts mean "at least" or "more than";
  1,648 mean "at most" or "fewer than".

### 4.7 Length and time
A row's program is 1,546 characters on average (the longest 2,474). The same row's parts and groups, written as in
3.3, are 316 characters (the longest 567): 4.9 times shorter. Reading one instruction takes about 25 s at home (the
grammar is 0.3 s of it); on a rented A100 in batches of 32, about 2 s for the shortest instructions and 9.9 s on
average over the 64 dev rows.

## 5. Already decided or proposed (so you need not repeat it)
- **In round 2:** minimal pairs (an instruction and its twin with one part said the other way, both run in the VM),
  key tokens weighted, goal-first and passive wordings.
- **Proposed for round 3, not built:** one spelling per meaning (a negation pushed into the comparison; `not` kept
  only for positions); a grammar rule that refuses a `not` around a count or size comparison; "at least" and "at
  most" balanced; an or / if group tied to the two checks the split joins.
- **Advice from two outside models, read and not adopted as such:** hold out longer instructions (done: test_long);
  rotary positions (already); a margin loss on the operator token (kept for a round where it is the only change);
  pairs that swap the conditions of two parts.
