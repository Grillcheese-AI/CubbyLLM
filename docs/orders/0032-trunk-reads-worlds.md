# 0032 — The trunk reads the worlds: world facts given to the 450M as loci tokens, answered through its copy path, and the store decoding the trunk's vector
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("both, in that order") · hypothesis: H-K10 (gate 4), H-K1 · **after 0031**

## Why
Today the worlds answer through Python around the model: the ask loop looks a fact up in a store, the VM verifies it,
and the 450M only plans and words it. The trunk's own fact machinery (loci tokens, the copy path, copy or refuse:
run 5's 0.946 on facts shown once 800+ tokens back, 0022's 1.7% wrong) reads only facts written in its own document.
Nick wants the model to use the knowledge, not a lookup beside it. This order measures how far the trunk can be the
reader: first with the world's candidate facts placed in its input, then with no facts in the input at all.

What it brings: the model itself picks the fact, refuses when none fits, and gets better at it by training, while the
store and the VM keep it honest. A conventional model either memorizes facts in its weights (and drifts) or reads
them as text in a prompt (and pays per token); here a fact is one placed unit read by algebra.
In plain words: Cubby reads its own library instead of having someone read it out to it.

## Build
- **Arm A, a scene from the world.** For a question, the host gathers the candidate facts for its entity from the
  worlds (at most 8, the time rule applied), writes them into the trunk's input as loci tokens at their addresses (the
  `make_fact_shard` layout: subject and relation give the address), and the question asks at the address. The answer
  comes from the copy path under copy or refuse (0022's contract, `end_id`), and the VM still checks it against the
  world before it is spoken. Run 5's trunk and copy, nothing retrained first; then, if needed, the copy alone trained
  on world-built documents (`CB_TRAIN_ONLY=copy`).
- **Arm B, the store decodes the trunk (H-K10 gate 4, H-K1).** No facts in the input: at the asking position the
  trunk's read vector is decoded against the hdc world (nearest entity, then the fact map's unbinding for the relation),
  against the text head's answer on the same questions.
- Benches: 0014's (177 one-hop questions the hdc world answers), the 0012 books bench, and 200 questions on entities
  the worlds do not hold (absent). Fixed before any run.

## Acceptance (pre-registered)
- **A:** exact on 0014's bench at least 0.90 of the host lookup's correct answers (177), 0 wrong spoken (the VM's
  check counts), absent questions refused at least 95%; latency per question reported beside the host's.
- **B:** H-K10 gate 4 as written: the store-decoded answer at least the text head's exact on the same questions.
- **Kill:** A below 0.70 of the host's correct answers: the worlds stay read by the host and the trunk keeps the
  wording; B is then reported but not pursued.
- Tests: off = bit-identical (the arms are flags); the scene built from a world places exactly the facts it lists.

## Constraints
- Rules in `CLAUDE.md` hold. Lives in `cubbyllm/` (after 0031): the arms in `cubbyllm/reasoning/` and
  `cubbyllm/model/`, never in `standin/`.
- The VM verifies before anything is spoken, in every arm.

## Out of scope
- Retraining the trunk; Wikipedia's tranche (0015's next); news facts (they follow arm A if it passes).

## Note (Cowork, 2026-10-24, from 0035 C)
The copy works only in the shape it was taught in:
- the space form ("Answer:", its ":" asking, each name written and answered as " name");
- each piece of a name at its own address.

In 0034's own form it never fires (things 0.006). Whatever this order hands the trunk as loci tokens should be
written in that form, or the report should say why not.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
