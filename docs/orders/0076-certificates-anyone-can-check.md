# 0076 - Certificates anyone can check, and facts that disagree said as such (on 0055's store)
status: proposed
written: 2026-11-17 by Cowork, on Nick's "yes" to writing it after the field watch's finds · approved: — · hypothesis: none (0054's certificate, 0055's store)

## Why
Every answer Cubby speaks carries a certificate:
- the statements it read (the record's `trace`), each with its source;
- the store and its snapshot;
- the name statement (`reasoning/name_statement.py`), which says how the question's name reached the item.

`validation/system_eval/replay_cert.py` checks that a certificate carries what its answer rests on. The same answer
must come out of a world that holds only those statements.

Two things are still missing.

**Nobody outside can check a certificate.** A reader has to trust that the statements are the ones the store held at
that snapshot, and that the certificate was not changed after the answer. Several projects found by the field watch
(2026-11-15 and 16) close this gap with hashes:
- [answerproof](https://github.com/royalpinto007/answerproof): source hashes, a Merkle root and an offline verifier;
- [jip](https://github.com/larsw/jip): a signed provenance certificate per SPARQL row, with the data version and a
  canonical digest;
- the [Hyphae provenance paper](https://celiums-ai-22l.pages.dev/papers/hyphae-provenance.pdf): a hash-chained
  journal with a signed head.

0055's store already gives what is needed. Each entity's revision is recorded, and the manifest carries the hashes of
the inputs, the chunks and the code.

**When the store's facts disagree, the refusal does not show both sides.** Wikidata often holds several best-rank
values for one property of one item, such as two dates of birth, both of normal rank. The walk refuses an ambiguous
hop (`ambiguous_hop`). We have not measured:
- how often a final answer rests on such a property;
- what the host does then;
- whether the refusal names both statements and their references.

[Eleutheria](https://github.com/SteveVitali/Eleutheria), [Positorium](https://github.com/Roenbaeck/positorium) and
[temporal-rag](https://github.com/pavan139/temporal-rag) keep a disagreement as an object, and refuse on it with
both sides shown.

- **What it brings CubbyLLM:** a certificate becomes evidence a third party can check offline against the same
  snapshot, without trusting Cubby. A disagreement in the sources is shown, not settled in silence.
- **Over a conventional model:** a model's citation cannot be checked against what it read. Here every statement
  is pinned to a revision, and the whole certificate is pinned to one digest.
- **What it does for you:** you, or anyone you give an answer to, can confirm it came from exactly these facts. When
  the sources disagree, Cubby says so and shows both.

## Build
1. **A digest on every certificate.** It is a SHA-256 over a canonical JSON (sorted keys, no spaces, UTF-8) of:
   - the store's snapshot hash;
   - each statement of the trace: entity, property, value, rank, the entity's revision, and the statement's id where
     the store keeps one;
   - the name statement.

   It is written into the record beside the certificate, by the same code that builds the certificate. It changes
   no answer. It goes in `cubbyllm/reasoning/` (or beside `name_statement.py`); the core opens no connection.
2. **A verifier** (`validation/verify_cert.py`; offline, and it reads only the home store). It takes a record and
   says:
   - `same`: the digest recomputes, and every statement is in the store at its revision;
   - `changed`: the digest does not recompute;
   - `missing`: a statement is not in the store at that revision, named;
   - `other snapshot`: the record names a snapshot the store is not.
3. **Signing, behind a flag.** An Ed25519 signature over the digest, with a key Nick makes and keeps. No key is in
   the repository, a log or an argument; the key's path comes from the environment. The verifier checks the
   signature with the public key, when one is given. Without a key the digest alone is built, and the order is
   complete.
4. **Disagreement, measured first.** Over 0054's stored records of arm C (the five benchmarks), count:
   - the questions whose trace, or whose refusal, touches an item and property holding two or more best-rank values
     that differ;
   - for each, what the host did: answered with one, listed all, or refused (and with which reason);
   - whether the record names every value and its references.

   Counts only, by benchmark (`validation/exp_disagreements.py`, log in `validation/logs/`).
5. **The design for disagreement, written after step 4 and before any change** (in this order's Report, for Nick's
   yes). A proposal to be decided on the counts:
   - a refusal `sources_disagree` that carries each value with its rank, references and revision, where the
     question asks for one value;
   - an unchanged answer where the question asks for all of them.

## Acceptance (pre-registered)
1. **Digest:** every certificate of arm C's stored records gets one, and `verify_cert.py` says `same` on every one
   against the snapshot the run used. **Kill:** one `changed` or `missing` on an untouched record.
2. **Spoiled copies are caught:** the verifier must not say `same` on any of them (**kill**). They are:
   - a value changed in one statement;
   - one statement dropped;
   - one statement added;
   - the snapshot hash changed;
   - the name statement's item changed.
3. **No answer changes:** `replay_cert.py` gives the same outcomes as before on the same records.
4. **Offline and cheap:** the verifier opens no connection (pinned by a test). Its time per certificate is reported,
   median and 95th percentile, at home.
5. **Signing,** if built: a signed digest verifies with the public key, and fails with another key or with one byte
   of the digest changed.
6. **Disagreement counted:** step 4's counts, by benchmark and by what the host did, raw counts with each share.
   The design of step 5 is written in the Report, and nothing of it is built in this order.
- **Tests:** the canonical JSON (key order and spacing do not change the digest); the verifier on each spoiled copy;
  no network; signing with the right key, a wrong key and a changed byte.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). The core opens no connection
  and imports nothing from `validation/`.
- No secret in a file, log, argument or commit. Keys are made and kept by Nick. Never print a key.
- The Rust store is read, not changed. If a statement's id is not stored, the digest uses entity, property, value,
  rank and revision, and says so.
- Nothing is served differently until step 5's design is approved. Nothing paid starts here.

## Out of scope
- Publishing certificates anywhere (a public log, a transparency server).
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
