# 0087 - A plan must use the names the question holds: the "Ghana" defect in `covers()`
status: open (Code B, before CubeLang P7 and P8)
written: 2026-11-26 by Cowork · approved: 2026-11-26 by Nick ("yes") · hypothesis: H-E15 (the host places the names; the gate checks the plan) · found in: 0083

## Why
Code B found it on 0083's other-words pass (hdc's rephrased one-hop pairs, r18's file), in arm A with every flag off,
which is the served stack as it is today:
- "Which country does <person> hold citizenship in?": the slotted seat writes `SEED "$E3"`, `HOP1 "country"`, and
  `$E3` is "citizenship", a span the name index holds as a name. The person is not in the slot table.
- The walk finds "ghana is the country of citizenship", the VM certifies it, and Cubby says "Ghana": 7 times in 150
  (rows 67, 100, 125, 127, 132, 142, 149), where the golds are Indonesia, the US, Poland, the UK, Canada, Liberia and
  the US.
- `covers()` passes the plan. Its split tier (v6) claims "country" and "citizenship" for `country of citizenship`,
  and its residual rule refuses only leftover words that relation labels are made of; a person's name is not one.
  Checked on the CPU: `covers("Which country does Jane Nossette Jarvis hold citizenship in?", country of
  citizenship)` is True.

- **What it brings CubbyLLM:** it closes a way the served stack speaks a certified answer about the wrong subject.
- **Over a conventional model:** a model that answers about the wrong person sounds just as sure; here the gate
  refuses a plan that leaves the person out.
- **What it does for you:** Cubby won't answer a question about someone with a fact about something else.

## Build (behind `CB_COVER_NAMES`, off by default)
The two checks live in a new module (`cubbyllm/reasoning/cover_names.py`; `plan_verify.py` is already 950 lines),
called wherever `covers()` decides, for every proposer: the emitter, the grammar and a remembered plan.
1. **The seed is a name, not a relation word:** a plan whose seed, normalized, is made only of words of the relations
   it walks ("citizenship" for `country of citizenship`) is refused, reason `seed_is_relation`.
2. **No name is left over:** after the relations and the seed are claimed, what is left of the question (the text
   the claim succeeded on, asides removed if that try removed them) holds no name. A name here is a span the slot
   table holds as an entity that the plan does not use, or a word written with a capital in the question that is not
   its first word, not a frame word and not inside a claimed span. Refused, reason `name_left_over`.
3. **Counted:** refusals by the two reasons, on every read.
4. **Tests:** the seven "Ghana" rows of the pass, refused with the flag and spoken without it (as today); planted
   cases: a plan whose seed is a relation word; a capitalized name left over; an aside about the entity (accepted,
   covers() v4's case); a capital at the start of the question ("In which country ..."); a name that carries its own
   parenthesis ("Tombo (album)"), accepted.

## Acceptance (pre-registered; the rule is fixed above, before any read)
- **Read, flag off and on, arm A as 0083 ran it (home card):** the gen-3 held split (seed 7, 600 rows) and H-E15's
  live eight, each once per setting; the other-words pass (150), where the defect was found, reported apart.
- **Gate:**
  - the other-words pass: no spoken answer whose seed is a relation word ("Ghana" on 0 rows);
  - gen-3: no WRONG added (every WRONG with the flag on is one arm A speaks without it), and correct at most 6 rows
    (1 point) below the flag off;
  - the live eight unchanged.
- **Kill:** a WRONG the flag adds, or correct more than 12 rows (2 points) down. The flag stays off, and the report
  lists the leftover words that refused right answers.
- **Between the kill and the gate:** reported; turning the flag on where a person talks to Cubby is Nick's call.
- **Reported:** refusals by the two reasons; which of arm A's 16 gen-3 WRONG rows the rule refuses; the time it adds
  per question.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Held-out discipline** (the hypothesis file, section 0): the rule is fixed in this order before any read; nothing
  is tuned on gen-3 or the live eight. The other-words pass is where the defect was found, so it shows the fix, not
  how far it carries.
- Nothing else in `plan_verify`, the walk or the VM changes. Memory stays off (0083).
- 0084 (Code A, refused plans tried again) reads with `CB_COVER_NAMES` as it stands when it runs, and says which.

## Out of scope
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
