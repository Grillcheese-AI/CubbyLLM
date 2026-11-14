# Cowork's own answer to the instruction-reader brief, sealed before the panel runs (2026-11-13)

Written and committed before any panel answer is read, so the synthesis can say what the panel added and what it
only repeated. The brief is `2026-11-13-instruction-reader-brief.md`; section numbers below are its.

## 1. Diagnosis
- **Negation (29 of round 2's 30 reversals; 64 of round 1's 70): the data, not the model.** The gold writes each
  count or size meaning two ways, and the wording does not follow the spelling (4.6). The reader must get two tokens
  right that have to agree, the operator and the wrapper, and after `>` or `>=` the gold adds a `not` two times in
  three, so "at least" drifts to "fewer than". The imbalance of bounds pushes the same way. Round 2's pairs flip a
  whole part with a `not`: they teach direction and keep both spellings alive. Fixable in data.
- **Wrong with no single reversal (66 to 14 on the same rows): the form.** Mostly shape errors in 1,500 characters
  written token by token. Training and grammar rules reduce them one at a time; a shorter target removes the class.
- **The join: the form again.** The main is written hundreds of tokens after the cue ("unless") was read; the reader
  falls back on the commonest shape. Where the group is written beside the split, the distance is gone.
- **Spans not matched (13 of 198): the split, whatever the target.** Training.
- **Long instructions: unknown.** If the two rows seen hold, it is the numbering: "check 5" was never written in
  training. A target with no part numbers has nothing to extrapolate.
- **Not the model's size.** At the same size, round 2 took parts right as written from 30% to 71% (4.3).
- **The grammar can cost as well as save:** rule 9 gave up 45 right answers for 4 fewer false passes on 75 dev rows.
  A rule that removes a choice can force a legal wrong answer.

## 2. What the reader should write: (C), built as (B) first
- **The description is the default, written beside the split:** each check line carries its kind and its
  description; a group is written on the line of the check it hangs on. The host compiles with the generator's own
  emitter. The program is still what runs and what the user is shown.
- **Removed by construction:** operator and wrapper that disagree (one slot left), shape errors, dead ends, the
  join's placement. **Left:** the direction of a bound, a negation inside a filter, the wrong view or column, spans.
- **Generality.** (B) is exactly as general as the generator. Every row of train, dev, test and test_long is a
  composition the description language has, so today's measurements cannot tell (A) from (B). What (A) claims
  beyond that has never been measured.
- **A set that measures it:** checks CubeLang can run and the language cannot say (one view compared to another,
  "twice as many", a filter inside a filter, each element compared to the one before), each written as a program and
  run on replies built to pass and to fail, never trained on. Under (B) every one must come out "asked" with no false
  pass. A free program (the "C" part) earns its place only if it compiles and agrees on at least half of them with
  false passes at most 2%; otherwise those instructions stay "asked" and a new view or column is added to the
  language, which is typed and general, not a rule for one instruction.
- **Time line I would set:** 2 s an instruction at home for up to 4 parts.

## 3. Negation and direction, beyond section 5
1. **Bounds as words, not operators:** the slot holds one of a closed set (at least, at most, more than, fewer than,
   exactly, other than; none, any, every) and the host maps it to the comparison. Closer to what the instruction
   says. Read on dev by direction of bound. Pass: parts fixed by one reversal at most 5% of parts. Kill: no gain
   over the one-spelling form alone.
2. **A say-back check at serve time, no gold:** the host words the description back with the generator's templates
   and compares its cue (at least / at most / not) with the part's own span; a cue that disagrees makes the part
   "asked". It turns a flip into "asked" instead of a false pass. Pass: at least half of dev's reversed parts caught,
   under 2% of right parts caught. Kill: more right parts caught than reversed ones.
3. **Replies at the boundary:** for every number N in a part, the row's replies include N - 1, N and N + 1, so a
   strict comparison read as a loose one cannot pass unseen. A change to the measure as much as to the training.

## 4. The join and long instructions
- The group on the check's line, tied by the grammar to the segment that carries the cue.
- No part numbers in the target: a check is followed by its own description.
- **Split first, then read each part alone** from its own span: a long instruction becomes a loop over short reads,
  not an extrapolation, and the per-part reads batch.
- If training on longer rows is wanted, line 4 is written again first (train on 2 to 6, read on 7 to 10).

## 5. The measure
Keep the lines. Add: false passes by direction of bound and by depth of negation; each part's agreement when run
alone; "asked" by reason; the generality set of section 2; seconds per instruction. Check how the failing replies are
built: if few sit at a boundary, a 2% line can hide a class of flips.

## 6. What I would not do, and what I do not know
- Not now: a larger model; a margin loss before the form is fixed; more grammar rules on program text; swap pairs
  before the wrong-view errors are counted under the new form.
- Unknown: how often real instructions fall outside the description language (line 5's read will say); whether the
  split's own errors rise once it also carries the groups.
