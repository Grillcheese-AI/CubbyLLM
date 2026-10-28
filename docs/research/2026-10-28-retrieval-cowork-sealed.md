# Retrieval, replanned: Cowork's sealed plan (2026-10-28)

Written before CubbyConsul's board saw the brief, and kept out of the repository until the board has
answered, so the two can be compared. Its SHA-256 is given to Nick before the board runs.

## What retrieval is in Cubby
Every step between a question and a fact the VM can check: reading the question (which names, which
relation), finding the candidate items for each name, choosing the one meant, fetching its facts, choosing
among several values and times, and recalling what this asker said before. Text (books, encyclopedias, news)
enters as facts through the readers and the gate; finding the passage to read is retrieval too.

## What is wrong now
- **Rules written per case.** Which name wins (label over alias, cue words, KATA guard, the initials tier
  proposed in 0043), the offer line, place qualifiers: each tier was written for rows of the scoreboard and is
  scored on the same rows (0043's audit drew all 50 names from them). Nothing measures them on unseen names.
- **Entry is the product** (`retriever.py`, exp_r4): 61% of the failures never produce a hop-0 seed. On the
  everyday scoreboard 39 of 187 are `retrieval_exhausted`, 15 `ambiguous_entity`, 44 `ambiguous_hop`.
- **Candidates come from Wikidata's search API,** which misses the president for "Obama" at 5, 10 and 20 hits.
- **The scoreboard is a parsed subset:** 187 questions the hand grammar reads (`parsed_only`), English only.
- **Three similarity codes that do not meet:** the qFHRR word table (semantic, weak on entity names: 0.081
  nDCG@10 on dbpedia-entity), the capsules' letter-trigram codes (good on misspelt names), the hippocampus'
  SimHash; and the VM's `HippocampalMemory` keys hypervectors by a hash of the key string, so "Georgia" and
  "Georgie" share nothing there.

## Principles (kept from the invariants)
1. **Facts stay exact, finding them may be similar.** A fact is an exact key with provenance (inbox
   2026-10-24); everything that leads to the key (which item, which relation, which world, which passage) may
   be found by similarity. Only (symbol, similarity) crosses into the VM.
2. **Every stage reports its candidates and why** (the existing "a miss must explain itself"), so each stage
   has its own recall number.
3. **Knowledge from data, rules only for guarantees.** What a name usually means, what an item is like, what
   this asker meant: stored from data. Rules: 0 wrong (refuse when not sure), values tied to their own item,
   never fame within a kind (the James Young rule), latent until corroborated.
4. **Abstain by margin, calibrated per deployment** (the paper: tau does not travel). Set on a dev split, read
   once on test.
5. **English and French from the start;** a stage that works only in English does not pass.

## The shape I would test
- **A name index** (lexical): every label and alias, EN and FR, to items; exact (an fst segment, as the VM's
  knowledge store already builds) plus near matches by letter-trigram codes (the capsules' code: misspellings,
  OCR). Source: the local Wikidata dump (TODO: 74.5M entities, 959 GB streamed to 40-80 GB, waits on Nick's go),
  Wikidata5M (local now) for the first tests.
- **An item memory** (semantic): one prototype per item, qFHRR, bundling its kind, description, the labels of
  its facts' values, and its own text (abstract / article opening, EN and FR). One normalized prototype per
  item, never a sum over mentions. Compact form (80 B) for a shortlist, dense re-rank (the paper's two-stage).
- **A usage table** (between kinds only): how often a name means each item in text, from anchors if a corpus
  keeps links, else from mentions the memory linked with a clear margin. Never used between two of a kind.
- **The choice:** context (question + thread, qFHRR) against each candidate's prototype; a pick needs a
  calibrated margin; within it, answer each or ask back (as now).
- **An asker memory:** this asker's corrections and choices, keyed by a context code, recalled for similar
  later questions; per asker, never shared. Explicit codes, not a superposed memory, unless SDM wins R4.
- **A relation index:** the question's predicate to a property, by similarity over property labels and
  aliases EN+FR (the local table exists), checked by the property's own constraints.
- **Facts:** the local dump's claims with rank and qualifiers (offline, fast); the live source for
  time-sensitive relations (Nick: live wins); Wikidata5M and the worlds as now.
- **Passages:** hybrid lexical (FTS5/BM25) + qFHRR shortlist for the readers.

## Experiments, before anything is built (each offline, held out, with a pass line)
- **R0 Sets and baselines.** Held out from everything tuned so far:
  - Mintaka (20k, Wikidata ids, EN + FR translation);
  - WebQSP_EL and GraphQ_EL (ELQ's, Wikipedia titles to ids by sitelinks; ELQ's F1 87.4 / 70.3 as a published
    reference);
  - AmbER (names shared by several Wikidata entities, head vs tail; retrievers fail twice as often on the tail)
    for the fame check;
  - the hdc set's QA (Wikidata5M ids).

  Run today's pipeline on them per stage. The 187 and the stack's 20 become regression only.
- **R1 Candidates:** the name index (exact; + normalization; + trigram codes; + FR) against the search API.
  Recall@1/5/20 of the gold item, latency, size. Pass: recall@20 above the API's on every set.
- **R2 The choice:** prototypes × context against today's rules, on the same candidates. Risk-coverage curve.
  Pass: at a wrong-pick rate no higher than the rules' (and at most 1 in 50), more questions decided. The AmbER
  tail at least as well as the head, and 0 picks between two of a kind without a telling word.
- **R3 Usage between kinds:** with and without. Pass: a gain with no tail loss on AmbER.
- **R4 Asker memory:** explicit codes vs SDM (D = 10,240), with a capacity curve; recall of a correction for a
  reworded question, false recall for unrelated ones, isolation between askers. Pass: at least 90% recall, 0
  cross-asker.
- **R5 Relations:** question to property against today's grammar, on WebQSP (its SPARQL gives the property),
  EN + FR.
- **R6 Facts, local vs live:** coverage, agreement, staleness of time-sensitive values; decides the freshness
  rule by measurement.
- **R7 Scale:** the shortlist at 1M, 10M and 75M items (grilly2 Hamming, the VM's LSH, Qdrant binary): recall
  vs latency vs RAM.
- **R8 End to end:** the best of each against today's pipeline, EN and FR. Kill: any wrong spoken value.

## Order of work
R0 first (it costs only downloads of small eval sets and runs). Then R1, R2 and R3 on Wikidata5M (local).
Ask Nick for the dump's download only if R1 or R2 pass on Wikidata5M. Then R4 to R7, then R8. Build only
what passed.

## Decisions for Nick
1. The usage table between kinds: allowed? It is a frequency, but of what a name means in text, never between
   two of a kind.
2. The local dump's download (a day, 959 GB streamed, 40-80 GB kept).
3. French parity as a pass line from the start.
