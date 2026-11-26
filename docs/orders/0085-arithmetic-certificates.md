# 0085 - An arithmetic answer carries its certificate: each step's program, the VM's value, and where every number came from
status: in progress (Code A, 2026-11-25; the CheezML mapping waits on Nick's ruling, see the Report)
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
### Code A, step 1 (2026-11-25): the record, the replay, the freeze and the shape check built; the certificate's terms proposed
**Stopped on the vocabulary, as the order says to: v0.2 cannot say an arithmetic certificate.** Its certificate
shapes need the certifying run to have read a stored statement (a `cheez:FactVersion` with its source's revision),
a run reading one to name a store snapshot and an as-of time, and a satisfied check of a kind in `cheez:CheckKinds`.
None of the 20 kinds is the slot check, the vote or an answer-kind rule. A certificate drafted with v0.2 terms as
defined fails the pin. A second draft stretched three terms (each number a fact version sourced from the question,
the slot check as `NumberInRegionCheck`) and still failed on the store snapshot. Refusals fit v0.2 as they are.
Nick chose to have the terms drafted (2026-11-25): `docs/research/2026-11-25-cheezml-arith/PROPOSAL.md` (5 check
kinds, a threshold parameter for m, 2 artifact kinds, the W3C Web Annotation namespace for offsets, and
CertifyingRunShape relaxed to "a statement version or a number of the question, through the runs it was informed
by"). Until the pin moves, refusals are written as v0.2 Maasdam records and certificates with the proposed terms.

- **The trace** (`cubbyllm/records/arith_trace.py`), from the loop's state after the fact (the dev read's chains
  keep a log of each step only when recording; off, the rows are unchanged, pinned by a test). It holds the
  question's SHA-256 and the slots:
  - each `$N` with its span (start, end, the text exactly as the question has it, number words included) and the
    value filled in;
  - each `$K` by its fact;
  - each `$S` by its step.

  For a spoken answer it holds the chosen chain's steps: each as the VM got it with its slots, the SHA-256 of the
  whole filled program the VM ran, the outcome and the value. It also holds the stop and the answer, the vote (k,
  m, every try's final value, the votes), the answer-kind checks (satisfied / violated / does not apply), and the
  freeze. A refusal gets its why: "no try reached a value" with the tries' reasons counted, "rule: <name>", or "too
  few agree" / "tie".
- **The CheezML record** (`cubbyllm/records/cheez_arith.py`): refusals in v0.2 terms (`cubby:NoPlan` when no try
  reached a value; `cubby:NotCertified` for the vote short or a rule). Certificates follow the proposal's mapping:
  - the numbers as `cubby:FieldSuggestion`s read from `cubby:DocumentRegion`s of the hashed question;
  - each step a `cube:ProgramRun` of its own `cube:Program`, informed by the runs whose `$S` it used;
  - the vote's m as a `cheez:Threshold` calibrated on S1's dev rows (frozen in `68a8706`).
- **The replay** (`validation/replay_arith_cert.py`), from the record and the question alone. It re-reads each
  `$N` from its span and finds each `$K` in the closed list. It recounts the vote, then re-fills, re-hashes and
  re-runs each step in a fresh VM. It returns `same` or the first field that differs.
- **The freeze** (`validation/arith_freeze.py`, `arith_records.py`): the SHA-256s of the adapter's weights, of
  `speak_vote.py` + `answer_kind.py` (read with LF line endings), of the settings and of the VM binary, and the VM's
  commit. They are taken at the read's start (carried on every record) and at its end, written to
  `<records>.freeze.json`; a difference voids the read.
- **The shape check** (`validation/cheez_shapes.py`, needs rdflib and pyshacl: `uv run --no-project --with rdflib
  --with pyshacl`). It checks the pin's shapes, the pin's allowed namespaces, and that every CheezML term used is
  defined at the pin. **Tested first on the pin's examples: all 8 accepted, all 15 negative ones rejected.**
- **Wired:** `exp_0078_dev_read.py --records <file> --speak-k --speak-m [--rules] --calibration <dev rows>
  --calibration-at --run-name`, and `--ids` (read only those ids, keeping the dev order's index, so the seeds are
  the read's).

**The dev rows keep only values** (each try's final value and its steps' values, not its operations), so line 4's
fallback applies. The re-read covers 20 of dev 80:200's spoken answers: the first 20 by dev order of S1's 32 at
k 8, m 7 with the rules (`validation/logs/gsm8k_dev_cert_ids_0085_2026-11-25.json`; all 32 right). The cell is in
`notebooks/gsm8k_dev_read.ipynb` (Nick's L4 run). The replay and the shape checks run at home on what it writes.

**Measured so far** (not the acceptance read):
- **Tests:** a planted change is caught each time: a span moved by one character (`slots.$N1.span`), a step's
  program edited (`steps.0.program`), a step's value (`steps.1.value`), a try's value so the vote misses m
  (`vote`), a `$K` not in the closed list (`slots.$K1.fact`).
- **A smoke run on the real VM** (5 of the 20 questions, each fed its own gold steps; no model): 5 certificates,
  every replay `same`. At the pin 0 of 5 conform; under the proposal 5 of 5. The writer adds 0.5 to 0.9 ms a question.
- **The proposal keeps every verdict:** under it the pin's 8 examples still conform and its 15 negative ones still
  fail. A certificate whose runs used no number of the question still fails.

**Left:**
- Nick's ruling on the terms and the pin move.
- The 20-question re-read (Nick).
- Lines 1 and 2 read on its records.
- The test read, which writes its records with the frozen arm. It is 0078's line 5, after S2 and 0082.

**Questions:**
- Nick: the proposal (which terms, which names).
- Cowork: is "the first 20 of the 32 spoken, by dev order" the right 20?