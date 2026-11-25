# 0085 - An arithmetic answer carries its certificate: each step's program, the VM's value, and where every number came from
status: open (Code A, after 0078's addendum part 1, before or with the GSM8K test read)
written: 2026-11-25 by Cowork · approved: 2026-11-25 by Nick ("yes") · hypothesis: H-E19 (arithmetic is a world and a school); records: 0064, 0076

## Why
Order 0076 gave every knowledge answer a certificate anyone can check: the statements it read, the store's snapshot
and the name statement, written as CheezML records (`cubbyllm/records/cheez_writer.py`), signed, and replayed by
`validation/system_eval/replay_cert.py` (an answer re-asked over a world holding only what its certificate names must
come out the same). An arithmetic answer from 0078 carries nothing like it today, although the step loop already
holds everything a certificate needs: the slot table (every number the host placed, and the conversion facts it
offered), each step's program, the VM's value after each step, and the vote over the tries.

- **What it brings CubbyLLM:** a GSM8K answer anyone can re-run step by step, without trusting Cubby or its model.
- **Over a conventional model:** a model's worked solution is text; nobody can tell whether a number in it was
  computed or guessed. Here each step is a program with its hash, run by the VM, and each number in it points at
  the place in the question it was read from.
- **What it does for you:** an answer to a word problem comes with its working, and anyone you send it to can check
  that working on their own machine.

## Build
1. **The record** (`cubbyllm/records/cheez_arith.py`, beside `cheez_writer.py`), for a spoken arithmetic answer:
   - the question's hash;
   - the slot table: each `$N` with its value and its span in the question (start, end, the text read, number words
     included); each `$K` by its fact in the arithmetic world's closed list; each `$S` by the step that made it;
   - each step: its rendered program, the program's SHA-256, the VM's outcome (return or error) and value;
   - the stop, and the answer;
   - the vote: k, m, every try's final value, and how many tries reached the answer;
   - the answer-kind checks of 0078's addendum, each with its outcome;
   - the VM build (CubeLang's commit), and the adapter's name and SHA-256.
   A refusal gets a Maasdam record with its reason (no value reached m votes; a rule refused the value; a slot was
   copied or unknown).
2. **CheezML:** written at the pin (`vendor/cheezml`, v0.2), with the classes v0.2 already has for a program run, a
   check and a certificate. Where v0.2 has no term for something here (a slot's span, a vote), the report lists it;
   adding a term to CheezML is Nick's, as its vocabulary is his.
3. **The replay** (`validation/replay_arith_cert.py`), from the certificate and the question text alone:
   - re-read each `$N` from its span, and each `$K` from the closed list;
   - re-run each step's program in a fresh VM, in order, and compare every value;
   - recount the vote from the recorded tries' values;
   - outcomes: `same`, or the first field that differs.
4. **Where it runs:** in the GSM8K test read, so every test answer is written with its record. Before that, on dev
   80:200's spoken answers: built from the dev read's saved rows if they keep each step's operations; if they keep
   only values, 20 of those questions are read again with the writer on, and that read is said in the report.

## Acceptance (pre-registered)
- **Tests:** a planted change caught each time: a slot's span moved by one character, a step's program edited, a
  value changed, a try's value changed so the vote no longer reaches m, a `$K` that is not in the closed list.
- **Line 1:** every spoken answer gets a certificate, every refusal a Maasdam record, and every record conforms to
  CheezML's shapes at the pin. The shape check is tested first on the pin's own examples (`docs/v0.2/examples/`): it
  accepts all 8 that conform and rejects all 15 broken ones in `negative/`, or it is fixed before it rules on a
  record (from Gemini's review, 2026-11-25, on Nick's yes).
- **Line 2:** every certificate's replay is `same`.
- **Line 3, the freeze:** the test read writes, at its start and at its end, the SHA-256 of the adapter's weights, of
  the speak rule's code (`cubbyllm/reasoning/speak_vote.py` and the answer-kind rules beside it), of the settings (k,
  m, temperature, top-k) and of the VM build. The two sets must match, and every certificate carries the start's; a
  mismatch voids the read, and the report says so (same review).
- **Kill:** one replay that is not `same` stops the writer until the cause is fixed and recorded.
- **Reported:** records written, the time the writer adds per question, the terms v0.2 lacks.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The record is written from what the loop did, never from what the model says it did: no text from the model goes
  into a certificate except its emitted step program, which is what the VM ran.
- The records stay off the repository (`--records`, JSON lines), as 0076's do; the report holds counts only.
- Nothing about the answer changes: the writer reads the loop's state after the fact.

## Out of scope
- Signing and the envelope (0076's, reused as they are).
- New CheezML terms (Nick's).
- Certificates for the knowledge loop (0076 has them).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
