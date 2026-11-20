# 0076 - Comté certificates anyone can check, and Maasdam refusals that show disagreeing facts (CheezML on 0055's store)
status: in progress (Code A, 2026-11-19, while 0077's test read waits on Colab; approved by Nick on 2026-11-17: "I approve both"; built on CheezML and order 0064)
written: 2026-11-17 by Cowork, on Nick's "yes" after the field watch's finds · approved: 2026-11-17 by Nick · hypothesis: none (0054's certificate, 0055's store, 0064's records)

## Why
Every answer Cubby speaks carries a certificate:
- the statements it read (the record's `trace`), each with its source;
- the store and its snapshot;
- the name statement (`reasoning/name_statement.py`), which says how the question's name reached the item.

`validation/system_eval/replay_cert.py` checks that a certificate carries what its answer rests on. CheezML
(order 0064; vocabulary in Grillcheese-AI/cheezml, v0.2 at `https://w3id.org/cheezml/core/v0.2`) already defines
these records:
- the certificate (`cheez:Certificate`, the Comté protocol);
- the refusal (`cubby:Refusal`, the Maasdam protocol), with a reason that can be `cubby:Conflict` and a conflict kind
  such as `cubby:ValueConflict`;
- the fact version with its revision (`cheez:FactVersion`) and the store snapshot (`cheez:storeSnapshot`).

0064's addendum of 2026-11-10 fixes how a record is kept private and still checkable: a signed public envelope, an
encrypted body, and salted commitments for selective proof. Its writers are queued after 0055. This order builds
the certificate and refusal writer and its envelope now, on 0054's runs and 0055's store, because two things are
still missing.

**Nobody outside can check a certificate.** A reader has to trust that the statements are the ones the store held at
that snapshot, and that the record was not changed after the answer. Several projects found by the field watch
(2026-11-15 and 16) show how others close this gap:
- [answerproof](https://github.com/royalpinto007/answerproof): source hashes, a Merkle root and an offline verifier;
- [jip](https://github.com/larsw/jip): a signed provenance certificate per SPARQL row, with the data version and an
  RDF canonical digest;
- the [Hyphae provenance paper](https://celiums-ai-22l.pages.dev/papers/hyphae-provenance.pdf): a hash-chained
  journal with a signed head.

0064's envelope is the same idea, in CheezML's terms and with the answers kept private.

**When the store's facts disagree, the refusal does not show both sides.** Wikidata often holds several best-rank
values for one property of one item, such as two dates of birth, both of normal rank. The walk refuses an ambiguous
hop (`ambiguous_hop`). We have not measured:
- how often a final answer rests on such a property;
- what the host does then;
- whether the refusal names every value and its references.

CheezML already has the record for it: a refusal for `cubby:Conflict`, of kind `cubby:ValueConflict`.
[Eleutheria](https://github.com/SteveVitali/Eleutheria), [Positorium](https://github.com/Roenbaeck/positorium) and
[temporal-rag](https://github.com/pavan139/temporal-rag) keep a disagreement as an object, and refuse on it with
both sides shown.

- **What it brings CubbyLLM:** every answer leaves a CheezML receipt whose signature anyone can check. People allowed
  to read the record can re-derive every statement from the same store snapshot, offline. When the sources disagree,
  the refusal says so and shows both.
- **Over a conventional model:** a model's citation cannot be checked against what it read, and chat logs sit in
  clear and can be edited. Here each statement is pinned to a revision, the record to a signature, and the body to
  its readers.
- **What it does for you:** you, or anyone you give an answer to, can confirm it is genuine and unchanged. Only the
  people allowed can see what it says. When the sources disagree, Cubby says so and shows both.

## Build
1. **The vocabulary at a pinned commit** (0064 step 0): CubbyLLM takes Grillcheese-AI/cheezml at a pinned commit,
   as a submodule or a vendored copy (the builder's call), and the tests run its `validate.py`. The envelope's terms
   (envelope, key id, commitment, signature) are added in the cheezml repository, through its checks, as 0064's
   addendum says, and in nothing else.
2. **The certificate and refusal writer** (0064 step 3, its first two record kinds), written by the host from the run
   record and never by the program or a model. It is behind a flag and changes no answer:
   - **an answer:** a `cubby:Answer` with its `cheez:Certificate`, giving:
     - each fact read as a `cheez:FactVersion`, with its entity, property, value, rank, the entity's revision, and
       the statement's id where the store keeps one;
     - the run's `cheez:storeSnapshot` (the store manifest's hash) and its `cheez:asOf`;
     - the name statement, in CheezML's terms;
   - **a refusal:** a `cubby:Refusal` with its `cubby:refusalReason`, and for an ambiguous hop the values the walk
     found;
   - a run that reads a fact version and lacks its snapshot or as-of date gets no record, and the report counts
     those (v0.2 rule).
3. **The envelope** (0064 addendum, sections 2 and 3, as written there):
   - the plain record is canonicalised (RDFC-1.0 over its graph: key order and spacing change nothing) and signed
     with JWS and Ed25519 (sign, then encrypt);
   - the body is encrypted with JWE and AES-256-GCM, in two parts: the conversation part for the tenant's members,
     and the provenance part for the audit view only;
   - the envelope holds the record's `urn:uuid`, type, recorded time, key ids, the signature and an HMAC-SHA-256
     commitment with a random salt for the answer value and for each fact read. Plain hashes of values are not used,
     as a short answer could be guessed from its hash;
   - maintained libraries only (`cryptography` or `joserfc`, and `pyld` for RDFC-1.0), no home-made cryptography;
   - keys come from the OS key store or a key service, made and kept by Nick. No key is in a file, a log, an
     argument or a commit. Until Nick makes the keys, the tests use keys made in memory for the test, never written.
4. **A verifier** (`validation/verify_cheez.py`; offline, reading only the home store):
   - **anyone:** checks the envelope's signature with the public key. It says `signed` or `changed`;
   - **with a reader's key:** it decrypts its part and says:
     - `same`: the signed record recomputes, and every fact version is in the store at its revision;
     - `missing`: a fact version is not in the store at that revision, named;
     - `other snapshot`: the record names a snapshot the store is not;
   - **an owner** can reveal one value and its salt. The verifier says whether it matches its commitment.
5. **Disagreement, measured first.** Over 0054's stored records of arm C (the five benchmarks), count:
   - the questions whose trace, or whose refusal, touches an item and property holding two or more best-rank values
     that differ;
   - for each, what the host did: answered with one, listed all, or refused (and with which reason);
   - whether the record names every value and its references.

   Counts only, by benchmark (`validation/exp_disagreements.py`, log in `validation/logs/`).
6. **The design for disagreement, written after step 5 and before any change** (in this order's Report, for Nick's
   yes). A proposal to be decided on the counts:
   - a Maasdam refusal with `cubby:refusalReason cubby:Conflict` and `cubby:conflictKind cubby:ValueConflict`,
     carrying each value as a `cheez:FactVersion` with its rank, references and revision, where the question asks
     for one value;
   - an unchanged answer where the question asks for all of them.

## Acceptance (pre-registered; the record lines are 0064's, met here for certificates and refusals)
1. **Records from real runs:** replaying 0054's latest arm C run writes one certificate or one refusal per question.
   No question is left without a record, except those counted under step 2's rule, and 100% conform to CheezML's
   shapes.
2. **Records agree with what ran:** each record is re-derived from the run record and compared field by field, with
   0 differences. **Kill:** a record that conforms but disagrees with what ran; the writer stops until the cause is
   fixed.
3. **The verifier says `signed` and `same`** on every untouched record, against the snapshot the run used.
4. **Spoiled copies are caught:** the verifier must not say `signed` or `same` on any of them (**kill**). They are:
   - a value changed in one fact version;
   - one fact version dropped;
   - one fact version added;
   - the snapshot hash changed;
   - the name statement's item changed;
   - one byte of the envelope changed;
   - one byte of the body changed.
5. **Privacy holds, as 0064's tests say:**
   - a body encrypted for tenant A does not open with tenant B's key;
   - a member's key does not open the provenance part;
   - a revealed value with its salt matches its commitment, and any other value fails;
   - after a run over planted secret strings, none of them appears in any log or in the envelope;
   - no key material is in any tracked file.
6. **No answer changes:** `replay_cert.py` gives the same outcomes as before on the same records.
7. **Offline and cheap:** the verifier opens no connection (pinned by a test). Its time per record is reported,
   median and 95th percentile, at home.
8. **Disagreement counted:** step 5's counts, by benchmark and by what the host did, raw counts with each share.
   The design of step 6 is written in the Report, and nothing of it is built in this order.
- **Tests:**
  - canonicalisation: the same graph in another key order gives the same signature input;
  - the writer on an answer and on a refusal;
  - each spoiled copy;
  - the privacy tests above;
  - no network.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). The core opens no connection
  and imports nothing from `validation/`.
- CheezML changes are made in the cheezml repository, through its checks. A record says only what ran: no truth
  score, no status declared by hand, no prose rationale (0064's hard line).
- No secret in a file, log, argument or commit. Keys are made and kept by Nick. Never print a key. Destroying a key
  is never done without Nick's yes.
- Real records are never published: only the vocabulary, the shapes and made-up examples are public.
- The Rust store is read, not changed. If a statement's id is not stored, the fact version uses entity, property,
  value, rank and revision, and says so.
- Nothing is served differently until step 6's design is approved. Nothing paid starts here.

## Out of scope
- 0064's other record kinds: routing and split, helper rejection, field review, conversation turn.
- Publishing records anywhere (a public log, a transparency server).
- Changing which statements are truthy, or how ranks are read.
- 0075's grader audit.

## Report
<!-- Claude Code fills this in, then sets status: review -->
### Step 1 (Claude Code, Code A, 2026-11-19): the vocabulary at a pinned commit
- **CheezML** (Grillcheese-AI/cheezml `a99a774`, on main, pushed on Nick's "yes merge" of 2026-11-19): a fact version
  read from a statement store states its statement with published terms only, nothing minted (Cowork's ruling,
  the README's reuse rule): `rdf:subject` / `rdf:predicate` / `rdf:object` (Wikidata IRIs), `wikibase:rank`,
  `schema:identifier` for the statement id, and the entity's revision as `schema:version` on its source
  (`shapes:StoredStatementShape`). A valid-time beginning is optional where the source states none (Nick's yes,
  2026-11-19); on a stored statement a beginning that names no part of the source (`prov:wasDerivedFrom`) fails.
  Example 08 conforms; n08 (no rank) and n08b (a beginning written by hand) fail with their messages. Its
  `validate.py`: 8 conform, 15 fail, all checks pass; `check_site.py`: all pass.
- **The pin:** `vendor/cheezml`, a git submodule at `a99a774`; `tests/test_cheezml_pin.py` checks the pin and runs its
  `validate.py` (skips when the submodule or rdflib / pyshacl are missing).
- **What the store gives** (one read-only `wdstore serve` `get`): the entity's revision and modified time, and per
  statement its guid, rank, best flag, references and qualifiers: every field a fact version needs.
- **Still to come in cheezml:** the envelope's terms (step 3). The name statement uses v0.2's `cheez:Decision` over
  `cheez:Option`s with the rule as `dcterms:type` (Cowork, 2026-11-19); new terms only if a check shows that cannot
  say it.

### Steps 2 to 5 and the acceptance read (Claude Code, Code A, 2026-11-19)
- **What the run records** (`CB_CERT_FACTS=1`, off by default; records only, nothing reads them to decide):
  `cubbyllm/worlds/wd_store_statements.py` (the statement behind each fact the store hands out: item, property,
  guid, rank, the item's revision and modified time, its time qualifiers); `ask_memory.ran` (the host's five checks
  on a walked answer, with their outcome); `validation/system_eval/cert_facts.py` (per question: the trace's facts and
  their statements, an ambiguous hop's values and theirs, each program's SHA-256, the as-of year, the time).
- **The writer** (`cubbyllm/records/`: `cheez_writer.py`, `cheez_facts.py`, `refusals.py`): from the run's record
  only. A certificate when every trace fact was handed by the store with one statement, the program is one hash and
  the question names no year; a refusal when its reason word has a code. Ids are UUIDv5 of run, question and role.
  The store snapshot by its BLAKE3 hash (wdstore's own); a valid-time beginning only from a start qualifier at day
  precision, derived from it. CheezML: a99a774 (stored statements, Nick's yes on the optional beginning) and bc85026
  (the host's check kinds, confirmed against `ask_memory._checked`; Nick's yes), both merged and pushed; pinned.
- **The envelope** (`cubby_serve/records/envelope.py`) and **the verifier** (`validation/verify_cheez.py`), as step 3
  and 4 say; tests in `tests/records/test_envelope.py`, keys made in memory.
- **The run:** `run_0076_cert` = `run_0054_names3`'s flags and programs with `CB_CERT_FACTS=1` (master 425c2bb).
  10,561 of 10,568 records identical in spoken, answer, reason and trace; the 7 others are refusals in both runs (6
  "self" questions whose text is the loop's counters; nq:83 refused "no plan" for "plan does not cover", one code).

**Acceptance:**
1. **Records from real runs** (`validation/logs/cheez_records_0076_2026-11-19.json`): 9,931 records for 10,568
   questions, **9,931 conform** (56 certificates, each with its name statement; 9,875 refusals, with Cowork's codes of
   2026-11-19 for `no_path`, `no_events_for_date`, and `time_mismatch` / `class_mismatch` as `cubby:NotCertified` with
   the failed check violated). No record for 637, each counted with its reason (v0.2's rule): 607 refusals whose word
   has no code yet (`entity_unresolved` 572, `profile` 27, `self` 8) and 30 of the 86 spoken answers (11 rest on the
   live source, which keeps no revision; 10 have a name statement that names no item, because it names the wiki
   world while the fact came from the store; 4 ran more than one program; 2 rest on Wikidata5M; 2 sentences held by
   two or three statements; 1 names a year). Records off-repo.
2. **Records agree with what ran:** each record is re-derived from the run record and compared field by field, with
   0 differences. **Kill:** a record that conforms but disagrees with what ran; the writer stops until the cause is
   fixed.
3. **The verifier says `signed` and `same`** on every untouched record, against the snapshot the run used.
4. **Spoiled copies are caught:** the verifier must not say `signed` or `same` on any of them (**kill**). They are:
   - a value changed in one fact version;
   - one fact version dropped;
   - one fact version added;
   - the snapshot hash changed;
   - the name statement's item changed;
   - one byte of the envelope changed;
   - one byte of the body changed.
5. **Privacy holds, as 0064's tests say:**
   - a body encrypted for tenant A does not open with tenant B's key;
   - a member's key does not open the provenance part;
   - a revealed value with its salt matches its commitment, and any other value fails;
   - after a run over planted secret strings, none of them appears in any log or in the envelope;
   - no key material is in any tracked file.
6. **No answer changes:** `replay_cert.py` gives the same outcomes as before on the same records.
7. **Offline and cheap:** the verifier opens no connection (pinned by a test). Its time per record is reported,
   median and 95th percentile, at home.
8. **Disagreement counted:** step 5's counts, by benchmark and by what the host did, raw counts with each share.
   The design of step 6 is written in the Report, and nothing of it is built in this order.
- **Tests:**
  - canonicalisation: the same graph in another key order gives the same signature input;
  - the writer on an answer and on a refusal;
  - each spoiled copy;
  - the privacy tests above;
  - no network.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). The core opens no connection
  and imports nothing from `validation/`.
- CheezML changes are made in the cheezml repository, through its checks. A record says only what ran: no truth
  score, no status declared by hand, no prose rationale (0064's hard line).
- No secret in a file, log, argument or commit. Keys are made and kept by Nick. Never print a key. Destroying a key
  is never done without Nick's yes.
- Real records are never published: only the vocabulary, the shapes and made-up examples are public.
- The Rust store is read, not changed. If a statement's id is not stored, the fact version uses entity, property,
  value, rank and revision, and says so.
- Nothing is served differently until step 6's design is approved. Nothing paid starts here.

## Out of scope
- 0064's other record kinds: routing and split, helper rejection, field review, conversation turn.
- Publishing records anywhere (a public log, a transparency server).
- Changing which statements are truthy, or how ranks are read.
- 0075's grader audit.

## Report
<!-- Claude Code fills this in, then sets status: review -->
### Step 1 (Claude Code, Code A, 2026-11-19): the vocabulary at a pinned commit
- **CheezML** (Grillcheese-AI/cheezml `a99a774`, on main, pushed on Nick's "yes merge" of 2026-11-19): a fact version
  read from a statement store states its statement with published terms only, nothing minted (Cowork's ruling,
  the README's reuse rule): `rdf:subject` / `rdf:predicate` / `rdf:object` (Wikidata IRIs), `wikibase:rank`,
  `schema:identifier` for the statement id, and the entity's revision as `schema:version` on its source
  (`shapes:StoredStatementShape`). A valid-time beginning is optional where the source states none (Nick's yes,
  2026-11-19); on a stored statement a beginning that names no part of the source (`prov:wasDerivedFrom`) fails.
  Example 08 conforms; n08 (no rank) and n08b (a beginning written by hand) fail with their messages. Its
  `validate.py`: 8 conform, 15 fail, all checks pass; `check_site.py`: all pass.
- **The pin:** `vendor/cheezml`, a git submodule at `a99a774`; `tests/test_cheezml_pin.py` checks the pin and runs its
  `validate.py` (skips when the submodule or rdflib / pyshacl are missing).
- **What the store gives** (one read-only `wdstore serve` `get`): the entity's revision and modified time, and per
  statement its guid, rank, best flag, references and qualifiers: every field a fact version needs.
- **Still to come in cheezml:** the envelope's terms (step 3). The name statement uses v0.2's `cheez:Decision` over
  `cheez:Option`s with the rule as `dcterms:type` (Cowork, 2026-11-19); new terms only if a check shows that cannot
  say it.

### Steps 2 to 5 and the acceptance read (Claude Code, Code A, 2026-11-19)
- **What the run records** (`CB_CERT_FACTS=1`, off by default; records only, nothing reads them to decide):
  `cubbyllm/worlds/wd_store_statements.py` (the statement behind each fact the store hands out: item, property,
  guid, rank, the item's revision and modified time, its time qualifiers); `ask_memory.ran` (the host's five checks
  on a walked answer, with their outcome); `validation/system_eval/cert_facts.py` (per question: the trace's facts and
  their statements, an ambiguous hop's values and theirs, each program's SHA-256, the as-of year, the time).
- **The writer** (`cubbyllm/records/`: `cheez_writer.py`, `cheez_facts.py`, `refusals.py`): from the run's record
  only. A certificate when every trace fact was handed by the store with one statement, the program is one hash and
  the question names no year; a refusal when its reason word has a code. Ids are UUIDv5 of run, question and role.
  The store snapshot by its BLAKE3 hash (wdstore's own); a valid-time beginning only from a start qualifier at day
  precision, derived from it. CheezML: a99a774 (stored statements, Nick's yes on the optional beginning) and bc85026
  (the host's check kinds, confirmed against `ask_memory._checked`; Nick's yes), both merged and pushed; pinned.
- **The envelope** (`cubby_serve/records/envelope.py`) and **the verifier** (`validation/verify_cheez.py`), as step 3
  and 4 say; tests in `tests/records/test_envelope.py`, keys made in memory.
- **The run:** `run_0076_cert` = `run_0054_names3`'s flags and programs with `CB_CERT_FACTS=1` (master 425c2bb).
  10,561 of 10,568 records identical in spoken, answer, reason and trace; the 7 others are refusals in both runs (6
  "self" questions whose text is the loop's counters; nq:83 refused "no plan" for "plan does not cover", one code).

**Acceptance:**
1. **Records from real runs** (`validation/logs/cheez_records_0076_2026-11-19.json`): 9,941 records for 10,568
   questions, **9,941 conform** (66 certificates, 9,875 refusals; with Cowork's codes of 2026-11-19 for `no_path`,
   `no_events_for_date`, and `time_mismatch` / `class_mismatch` as `cubby:NotCertified` with the failed check violated).
   No record for 627, each counted with its reason (v0.2's rule): 607 refusals whose word has no code yet
   (`entity_unresolved` 572, `profile` 27, `self` 8) and 20 of the 86 spoken answers (11
   rest on the live source, which keeps no revision; 4 ran more than one program; 2 rest on Wikidata5M; 2 sentences
   held by two or three statements; 1 names a year). Records off-repo.
2. **Records agree with what ran:** read back field by field against the run's record: **0 differences**; no kill.
3. **"signed" and "same"** (`validation/logs/cheez_seal_0076_2026-11-19.json`): **9,931 of 9,931**, against the store
   at the run's snapshot (keys made in memory for the check).
4. **Spoiled copies:** caught, never "signed" and "same": a fact's value changed, a fact dropped, a fact added, the
   snapshot changed, the answer changed, the name statement's item changed, one byte of the envelope, of either part;
   a wrong fact or snapshot signed by the host itself is "missing" / "other snapshot".
5. **Privacy:** tenant B's key opens nothing of tenant A's; a member's key opens the conversation part only; a reveal
   matches its commitment and another value does not; planted strings are not in the envelope; no key body in a
   tracked file.
6. **No answer changes:** `replay_cert.py --names` on the new run gives the same outcomes as on `run_0054_names3`,
   bench for bench (86 of 86 spoken "same"; `validation/logs/cert_replay_2026-11-19-0076.json`).
7. **Offline and cheap:** the verifier opens no connection (a test). Per record at home: the signature 0.133 ms
   median, 0.206 ms 95th percentile; a reader's full check (decrypt, recompute, the store) 0.983 / 1.492 ms.
8. **Disagreement** (`validation/logs/disagreements_0076_2026-11-19.json`): of 10,568 questions, 134 touch a stored
   statement and 41 of them (31%) touch an item and property holding two or more differing best-rank values (webq
   38, simpleqa 2, nq 1). The host: answered with one value 2 (webq; e.g. Utah's time zone, three best-rank values);
   refused 39: an ambiguous hop 25, `profile` 12, a time mismatch 2. What the record names, per pair touched: every
   value 32, one value 7, some of them 9 (an ambiguous hop the walk narrowed). 145 of those values carry references,
   and no record carries references yet.

### Step 6: the design for disagreement (Nick's answer, 2026-11-19; nothing of it is built in this order)
**A disagreement is two or more differing best-rank values on a property meant to hold one.** Which properties are
meant to hold one is the store's own fact: Wikidata's single-value or single-best-value constraint on the property
(P2302 = Q19474404 / Q52060874), read from the store at the run's snapshot. Of the 41 questions above, most touch a
property meant to hold several (official languages, time zones, instruments, notable works, citizenship): several
best-rank values there are several true values, not a disagreement.
- **Several values on a property meant to hold several:** the answer gives every value, its certificate naming
  each fact version. This is also Nick's case of one thing written two ways ("if its written 2 way I would return
  both"; Utah's "Mountain Time Zone", "UTC−07:00", "America/Denver"). Where the store links the values (P460 "said to
  be the same as"), the certificate shows that statement; "part of" / "has part" links are shown as such, never as
  sameness. No string compared, no model.
- **Differing best-rank values on a property meant to hold one**, the question asking for one value: a Maasdam
  refusal, `cubby:Conflict` / `cubby:ValueConflict`, each value as a `cheez:FactVersion` with its rank, revision and
  statement id (its references once CheezML has a term for them). A value whose qualifiers date it apart from the
  others (a population at different points in time) is a time question first, read by the host's existing time
  rules, before it is a conflict.
- **Plural questions** list every value; **a preferred value alone** among normal ones is not a disagreement
  (Wikidata's best rank); the 12 `profile` refusals are left as they are.

**The count on this run** (`validation/logs/disagreements_0076_2026-11-19.json`): of the 41 questions touching
differing best-rank values, **26 would answer with every value** (2 spoken today with one value, e.g. webq:37, Utah;
24 refused today on an ambiguous hop: Haiti's official languages, Louis Armstrong's instruments, ...; 2 of them the
store also links as one thing), **1 would be a conflict refusal** (nq:472, Prince Edward Island's population: two
values on a single-value property, refused today on an ambiguous hop) and **14 keep their reason** (12 `profile`, 2
a time mismatch). The build is a later order.

**The name statement** (`cubbyllm/records/cheez_name.py`, Cowork's ruling): each step from the asked name to the
item is a `cheez:Question` over the items it kept (`cheez:Option`, `schema:about` the item) and its `cheez:Decision`,
the rule as `dcterms:type` (`cubby:NameSteps`: fold, label, alias, namesake, first by id, place qualifier; cheezml
branch `name-steps`, `d4a2911`, all its checks pass, not yet merged: a push needs Nick's yes); the steps chained,
the certifying run using the last one. The read-back checks its item and its number of steps.

**Still open in this order:** the `name-steps` merge (Nick); `cubby:NameNotFound` for `entity_unresolved` (Cowork's
go, then Nick's); `profile` and `self` (Cowork rules: proposed as replies of their own kind, no refusal record);
Nick's keys from the OS key store (the code takes keys handed in; nothing reads a file).
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
