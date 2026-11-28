# 0087 - A plan must use the names the question holds: the "Ghana" defect, and "who are you?"
status: in progress (Code B, 2026-11-27; before 0092 and CubeLang P7 and P8; rewritten 2026-11-27 before any build, after the audit)
written: 2026-11-26 by Cowork · approved: 2026-11-26 by Nick ("yes"); the rewrite approved 2026-11-27 ("yes") · hypothesis: H-E15 (the host places the names; the gate checks the plan) · found in: 0083 and the served brain

## Why
Code B found the first defect on 0083's other-words pass (hdc's rephrased one-hop pairs, r18's file), in arm A with
every flag off, which is the served stack as it is today:
- "Which country does <person> hold citizenship in?": the slotted seat writes `SEED "$E3"`, `HOP1 "country"`, and
  `$E3` is "citizenship", a span the name index holds as a name. The person is not in the slot table.
- The walk finds "ghana is the country of citizenship", the VM certifies it, and Cubby says "Ghana": 7 times in 150
  (rows 67, 100, 125, 127, 132, 142, 149), where the golds are Indonesia, the US, Poland, the UK, Canada, Liberia and
  the US.
- `covers()` passes the plan. Its split tier (v6) claims "country" and "citizenship" for `country of citizenship`,
  and its residual rule refuses only leftover words that relation labels are made of; a person's name is not one.
  Checked on the CPU: `covers("Which country does Jane Nossette Jarvis hold citizenship in?", country of
  citizenship)` is True.

Cowork found the second in the served brain's history (2026-11-27): "who are you?" was answered with the Wikidata
profile of "You", a Japanese television personality born in 1964. The seed was the pronoun, because the name index
holds an item by that name.

The audit of 2026-11-27 (`docs/AUDIT_FOR_CLAUDE_2026-11-27.md`, sections 4.1 and 7.1) asked that the fix read the
slot table's spans rather than add another surface rule to `covers()`. This version does; the audit's larger fix
(plans that cite the question's span for each relation) is a design order of its own, later.

- **What it brings CubbyLLM:** it closes two ways the served stack speaks a certified answer about the wrong subject.
- **Over a conventional model:** a model that answers about the wrong person sounds just as sure; here the gate
  refuses a plan whose subject is not one the question names.
- **What it does for you:** Cubby won't answer a question about someone, or about itself, with a fact about
  something else.

## Build (behind `CB_COVER_NAMES`, off by default)
The checks live in a new module (`cubbyllm/reasoning/cover_names.py`), called wherever a plan is disposed of, after
`covers()`, for every proposer: the emitter, the grammar and a remembered plan. They read the question's slot table
(H-E15's `SlotTable`: the names the host knows, plus the question's own capitalised runs, `NameIndex`), not a new
scan of the text. Nothing is added to `plan_verify.py`.
1. **The seed is a name, not a relation word** (`seed_is_relation`): a plan whose seed, normalized, is made only of
   words of the relations it walks is refused ("citizenship" for `country of citizenship`). This alone refuses the
   seven "Ghana" rows, whatever the question's casing.
2. **No name of the question is left unused** (`name_left_over`): an entity span of the slot table that the plan
   neither seeds nor binds is refused. A span that lies inside a claimed relation's words is the relation, not a
   name; a span inside an aside about the seed (covers() v4's parenthesis) is not counted.
3. **A pronoun is never a seed** (`seed_is_pronoun`): a closed list (I, me, my, mine, you, your, yours, he, him, his,
   she, her, hers, it, its, we, us, our, ours, they, them, their, theirs, this, that, these, those, someone,
   anyone, everyone, something, who, what) never seeds a plan, and with the flag on the slot table offers none of
   them as an entity span, even when the name index holds an item by that name. A question whose only subject is a
   pronoun ("who are you?", "what are you?") is refused by the ask loop with this reason, and the report says what
   the served chat then answers.
4. **Counted:** refusals by the three reasons, on every read.
5. **Tests:** the seven "Ghana" rows, as written and lowercased, refused with the flag and spoken without it (as
   today); "who are you?" and "what are you?" refused with `seed_is_pronoun`, with no "You" span in the slot table;
   planted cases: an unused name span; an aside about the seed (accepted); a name that carries its own parenthesis
   ("Tombo (album)", accepted); a plan that uses every span (accepted).

## Acceptance (pre-registered; the rule is fixed above, before any read)
- **Read, flag off and on, arm A as 0083 ran it (home card):** the gen-3 held split (seed 7, 600 rows) and H-E15's
  live eight, each once per setting; the other-words pass (150), where the first defect was found, reported apart.
- **Gate:**
  - the other-words pass: no spoken answer whose seed is a relation word ("Ghana" on 0 rows);
  - "who are you?" and "what are you?" refused with `seed_is_pronoun`;
  - gen-3: no WRONG added (every WRONG with the flag on is one arm A speaks without it), and correct at most 6 rows
    (1 point) below the flag off;
  - the live eight unchanged.
- **Kill:** a WRONG the flag adds, or correct more than 12 rows (2 points) down. The flag stays off, and the report
  lists the spans or seeds that refused right answers.
- **Between the kill and the gate:** reported; turning the flag on where a person talks to Cubby is Nick's call.
- **Reported:** refusals by the three reasons; which of arm A's 16 gen-3 WRONG rows the rule refuses; how many gen-3
  questions hold a pronoun span today; the time it adds per question.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Held-out discipline** (the hypothesis file, section 0): the rule is fixed in this order before any read; nothing
  is tuned on gen-3 or the live eight. The other-words pass is where the first defect was found, so it shows the fix,
  not how far it carries.
- Nothing else in `plan_verify`, the walk or the VM changes, and nothing is added to `plan_verify.py` (950 lines).
  Memory stays off (0083).
- 0084 (Code A, refused plans tried again) reads with `CB_COVER_NAMES` as it stands when it runs, and says which.

## Out of scope
- Plans that cite the question's span for each relation, and splitting `plan_verify.py` (the audit's item 10): a
  design order of its own.
- "Saint Joseph" for Maria Gripe's husband (0083's row 199): the wrong person under a shared name. Counted in the
  read, not fixed here.
- 0081's follow-up (the literal admitting the question's own words, or lever 6's table read in reverse): Nick's open
  decision.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
