# Synthesis: the instruction-reader panel (2026-11-13)

By Cowork. The brief: `2026-11-13-instruction-reader-brief.md`. The answers: `2026-11-13-instruction-reader-panel.md`
(8 of 8, $2.52 as reported by OpenRouter). Cowork's own answer, sealed before the run:
`2026-11-13-instruction-reader-cowork-sealed.md`.

**Model opinions, not findings.** Nothing below is adopted until an order pre-registers it and its line is read.

## 1. Where the panel agrees
- **The reader should stop writing the whole program (7 of 8).** No member argues for growing the grammar on the
  program as the main line. Seven make the generator's part and group descriptions the normal target, compiled by the
  host; DeepSeek ranks that second among its fixes and does not choose a target. They differ on the free program for
  what the descriptions cannot say: four (OpenAI, Anthropic, Qwen, xAI) would run descriptions with "asked" as the
  only exit until a free path passes a read of its own; three (Google, Moonshot, Z.ai) would keep a bounded free path,
  Z.ai only after a set of out-of-language checks is read. The sealed answer said "(C), built as (B) first": the same.
- **Negation is the data before it is the model (8 of 8):** the gold's two spellings, with the imbalance of bounds
  behind them. No member sees evidence for a larger model.
- **The join is read where the connective is read (8 of 8):** on the split's line; the host builds the main; the
  reader never copies part indices a second time.
- **The host owns the numbering (7 of 8).** Do not train on 5 to 8 parts: line 4 forbids it (DeepSeek alone would).
- **The lines are a minimum, not enough (8 of 8):** see section 2, point 3.
- **Not now:** a larger model, a margin loss on the operator, more grammar rules on program text.

## 2. What the panel added to the sealed answer
1. **One spelling per wording, and the host does the logic** (Anthropic, xAI; close to the sealed "bounds as
   words"). The reader writes the relation the words say ("not more than 3" as `not >`, "at most 3" as `<=`); the
   host pushes the `not` into the comparison, exactly. Round 3's plan (one spelling per meaning) makes the reader do
   that step. Run the two as arms of one round, same rows, decoder and grammar.
2. **"If there is one" as its own slot** (Anthropic, Google). The sealed answer kept `not` for positions; the panel
   is right that this keeps the one place where the gold's operator is the opposite of the words ("should have the
   value null" written as `!=` inside a `not`). A slot "absent: pass", and the operator the words say.
3. **Four holes in the measure:**
   - false passes are counted over all checkable replies; a false pass can only happen on a reply built to fail, so
     the rate over those is about double today's 0.0935 if half the replies are built to fail (an estimate; count it);
   - reply agreement hides broken parts: under "and", one part that rightly fails a reply covers its wrong siblings
     (round 2: agreement 0.685 with 25 of 64 programs right). A part-level bar is needed;
   - no floor on what the VM could check, and no ceiling on "asked": a reader can pass by not compiling hard rows;
   - a group's replies must cover its truth table, or a misbound join passes unseen; line 4 is a ratio, and round 1
     met it while missing line 3.
4. **Fixes that need no training, to try on dev first:** a grammar on the split with the check's number owned by the
   host (Z.ai, Google, xAI); a second version of a part compiled when the operator's score is close to its
   complement's, the part "asked" on replies where the two disagree (Google; Moonshot's closed-slot scoring).
5. **A census by a person** of the held-out kinds and the benchmark's constraint types (Anthropic, Google,
   Moonshot): in the description language, in CubeLang only, or not checkable from the views. It decides whether the
   free path is needed, with no training run.
6. **Count before fixing** (Anthropic, Z.ai): where the grouped pairs sit in train; the reversals by direction.
   Done: section 4.
7. **Keep the words that carry the sign in templates** (Anthropic): a model's wording variation must not touch "no
   fewer than", "other than", "unless". The VM checks a program against replies, not a wording against a program.
8. **Later, each as its own round:** reading one isolated part as a training task and as a cheap gate (Moonshot,
   Qwen); the spans of a comparison and of its negation as an extra target (OpenAI); "passes when" and "fails when"
   read separately, "asked" where they are not complements (OpenAI, Anthropic).
9. **A rule to reword** (OpenAI): "no model writes a program" reads as forbidding the reader's own program. Meant:
   no model writes a training target; the reader proposes and the VM decides.
10. **A caution on reading each part alone** (OpenAI, Anthropic): a group can tie parts read apart, and a part can
    point at another sentence. The split and the groups are read over the whole instruction first.

## 3. Where the panel is wrong, or could not know
- **"test and test_long are spent"** (Anthropic): round 2 read a 100-row sample of each split; 200 a split are
  unread.
- **Running a check and its complement on the reply, "asked" where they disagree** (Z.ai): a comparison and its
  complement disagree on every reply, so every such part would be asked. Usable only behind a score threshold, as in
  point 4.
- **The 512-token window as the cause of the join error** (Google): the brief gives characters, not tokens. Not shown.
- **"Five epochs sharpened the numbering"** (Google): a hypothesis; round 1 was never read for it.
- **Operators licensed from a list of relation words in the host** (xAI's second): a word list per relation is near
  the hand rule the reader exists to avoid. Not taken.

## 4. Two counts made after the panel (train and dev only)
- **Where the grouped pairs sit:** 1,617 groups, every one on adjacent parts; 846 on parts 1 and 2 (52%), 502 on
  parts 2 and 3, 269 on parts 3 and 4. The first pair is the commonest, which fits the join error seen on dev (the
  reader ties the first two parts). A redraw over positions is cheap.
- **Round 2's 30 reversals on the 64 dev rows, by the comparison that would have been right:** upper bound 11, "other
  than" 8, lower bound 6, "equal" 5. So the flips are spread; 13 of 30 are equal / other-than. By the gold's shares
  (548 lower against 1,648 upper) a lower bound flips about 1.6 times as often as an upper one (an estimate). The
  imbalance contributes; it is not the main story, and "at least read as fewer than" was overstated from a few rows
  read by eye.

## 5. What Cowork proposes (for Nick; each step an order with its lines written first)
1. **Now, no training:**
   - round 2's counted read on the 200 unread rows a split, reported also over replies built to fail, by direction,
     by number of parts and by part index, with parts right and the share "asked";
   - the split's numbering owned by the host, read on dev;
   - the join error counted on dev (rows whose parts are all right and whose program is not).
2. **Round 3, one order, two arms, same rows, decoder and grammar:** the description target (each check's
   description on its split line, a group as a pointer to the neighbouring check, no main, no part numbers), with an
   "absent: pass" slot for positions, bounds balanced, groups drawn evenly over positions, sign words from templates
   only. Arm M: one spelling per meaning. Arm W: the spelling follows the wording and the host lowers it. Lines to
   write then: reversals at most 3% of parts; false passes at most 2% (kill above 5%), also over replies built to
   fail; "asked" at most 10% of parts on in-language rows; the group and both its members right for at least 97% of
   groups; a time line at home once the first read gives a number.
3. **The generality set and the census,** before any free program is served: checks CubeLang can run and the
   descriptions cannot say, built in reverse, never trained on. The free path opens only if it reads false passes at
   most 2% there.
4. **Only if 2 leaves line 3 unmet:** the score-threshold "asked", the second reading of the complement, the
   isolated-part task, each alone.
