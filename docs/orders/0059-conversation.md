# 0059 — Conversation: follow-ups, clarifying questions and refusals said plainly, every fact still certified
status: open (queue: after 0055 and 0010)
written: 2026-11-06 by Cowork · approved: 2026-11-06 by Nick ("fluency and conversation, we have plenty of chat datasets"; queue "right after 0055"; non-commercial sets "research runs only") · hypothesis: H-E10 step 2 (multi-turn), H-E15

## Why
0010 makes a single reply sound like a person. A conversation needs more than that:
- **Follow-ups:** "and when was she born?" only means something with the turn before it.
- **Clarifying questions:** "which Ottawa?" when a name is ambiguous. Today that is an `ambiguous_entity` refusal
  that lists the candidates.
- **Refusals said plainly:** "I don't have that in my records" instead of a reason code.

**What it brings, in plain words:** you can talk to Cubby the way you talk to a person, with follow-ups, "the other
one" and "why not?", and it still only tells you what it can prove.
- **Over a conventional model:** a chat model guesses who "she" is and answers anyway. Cubby works out who "she" is
  from the conversation, asks when it can't tell, and keeps every answer tied to a record.

## Build
0. **Licences first.** The rule is 0010's addendum 3: every licence is recorded in a manifest, non-commercial sets go
   to research runs only, and no model-written set is used, as input or as target.
1. **Conversation state, kept by the host:**
   - the entities named so far, by id, with the turn that named them;
   - the last question, its program and its certified answer;
   - any open clarifying question.

   The model never holds the state; the host renders it.
2. **Follow-up rewriting.** A follow-up is rewritten into a standalone question, and then the usual pipeline answers
   it with no change.
   - **The rewriter:** a LoRA adapter on base450m, trained on CANARD's human-written rewrites
     (`gaussalgo/Canard_Wiki-augmented`, on disk; CC BY-SA expected, to check). Disfl-QA (`google-research-datasets/
     disfl_qa`, on disk) teaches it to clean up hesitations and self-corrections ("in, no, after 1990").
   - **The guard:** every entity in a rewrite must come from the current turn or the conversation state, by id. A
     rewrite that names anything else is refused. Similarity only leads to a key, as everywhere else.
   - **Two candidates:** when two remembered entities of the same type fit a pronoun, Cubby asks instead of choosing
     ("Do you mean Marie Curie or Pierre Curie?").
3. **Clarifying questions.** `ambiguous_entity` becomes a question that names the candidates in plain words. The
   user's next turn ("the city", "the second one") is resolved against those candidates only, by id. No answer is
   given until one candidate remains.
4. **Refusals said plainly.** Each refusal reason becomes a short reply in Cubby's voice, through the 0010 talk
   adapter:
   - the reasons: no plan, relation missing, fact missing, walk failed, not certified, ambiguous, the kinds of
     conflict, dates too coarse;
   - a reply may name only what the reason names (the entity, the relation), never a fact that was not certified;
   - where it helps, it says what would work ("I know where she was born, not when").
5. **Small talk** (greetings, thanks, "how are you?") goes through 0010's casual family with no facts, under the
   existing voice and identity rules.
6. **French:** the rewriter is trained in English first. French follow-ups are measured and reported apart; a French
   rewriter is a later step.

## Acceptance (pre-registered)
- **A multi-turn battery:** 200 conversations of 3 to 6 turns, written by rule from held-out store facts, EN and FR.
  They include follow-ups by pronoun, by "the other one", by ellipsis ("and his wife?"), two-candidate cases,
  ambiguous names and unanswerable turns. A seeded sample of 30 is read by hand.
- **CANARD's own test split,** with the rewriter alone: exact match and token F1 against the human rewrites, as
  reported in the CANARD paper.
- **Hard line: 0 wrong spoken.** A wrong spoken answer caused by a bad rewrite counts as wrong.
- **Reported:**
  - right, wrong and refused per turn type;
  - rewrites refused by the guard;
  - clarifying questions asked, and how often the next turn resolved them;
  - the share of refusals said with a reason a person can act on.
- **Nick's blind A/B:** 30 whole conversations against the current stack, judged for flow.
- **Kill:** any wrong answer from a rewrite. Rewrites then go behind a confirmation ("You mean Marie Curie?") until
  the cause is found and fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **No external model** on the serving path or in the data. Model-written sets are excluded from training,
  including as inputs.
- **Non-commercial sets** are for research runs only, never in a served adapter.
- Training notebooks are Nick's to run on Colab. Claude Code never starts a paid run.
- No local path in a committed file.

## Out of scope
- Emotional support. That is 0010's addenda 1 and 2, with the crisis route as their prerequisite.
- Memory across sessions, and a user profile.
- Voice: speech in or out.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
