# 0033 — Follow-ups from the 0030 and 0031 reviews: a merged namesake spoke a wrong place, and six small fixes
status: open
written: 2026-10-22 by Cowork, from the reviews of 0030 and 0031 · approved: — (fixes from a review; Nick told) · hypothesis: none

## Why
0031's baselines found "Where is the Globe Theatre?" answered "located in: London Borough of Southwark, Boston": two
items sharing a label were merged, and the wrong one's place was spoken. The project's first rule is that a wrong
answer is never spoken; a refusal or a question back is always allowed. The rest are small things the reviews found.
In plain words: when two things share a name, Cubby asks which one you mean instead of mixing them up.

## Build
1. **Namesakes never merge (first).** Find where the Globe Theatre's two items meet (the hdc world's label lookup, the
   wiki world, or the ask loop's entity resolution) and fix it there: a label that names several items either is
   decided by the question's own context (another name in it, a type) or answers `ambiguous_entity` with the items
   named, as "what is Ottawa?" already does. Probe: 50 labels the hdc world shares between two or more items, asked
   "where is X?" / "what is X?": 0 answers mixing two items.
2. **The flaky Brave timing test** (5 in 20 before 0031, 3 in 20 after): make it deterministic (a fake clock), or
   mark it with its reason.
3. **Two local-path leftovers** (0030's review): the `.stdout` log and the doubly escaped JSON; the guard learns both
   forms (`.stdout` files; `\\\\`-escaped drive paths) and fails on a planted one.
4. **The data root** (0031's answer 6): every data place the package names under `standin/data/out` resolves under
   `$CUBBY_DATA_ROOT` (the default is today's place); the caches move, nothing is fetched again.
5. **The loci reasons' routes** (0031's answer 1) and **the fetch's fixed name** (answer 2).
6. **mowm's `FastWordEncoder` ported** into `cubbyllm/` (answer 3), pinned against mowm's output.
7. **The shim counter** (answer 7): a test that reports in-repo imports of shimmed `standin/` names.

## Acceptance (pre-registered)
- The namesake probe: 0 mixed answers in 50; "Where is the Globe Theatre?" says one place or asks which; 0014's bench
  unchanged (177 of 177, 0 wrong).
- The Brave test passes 20 of 20 runs; the path guard fails on both planted forms and finds 0 in `git ls-files`.
- The real stack's 20 fixed answers unchanged, except the Globe Theatre's if it is among them (report it).

## Constraints
- Rules in `CLAUDE.md` hold. Nothing new in `standin/`.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
