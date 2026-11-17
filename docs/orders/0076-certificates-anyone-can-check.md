# 0076 - Comté certificates anyone can check, and Maasdam refusals that show disagreeing facts (CheezML on 0055's store)
status: open (approved by Nick on 2026-11-17: "I approve both"; revised the same day on his "lets not forget cheezeml" to be built on CheezML and order 0064, not beside them; builder: Code A, or Cowork on a branch while Code A is signed out)
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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
