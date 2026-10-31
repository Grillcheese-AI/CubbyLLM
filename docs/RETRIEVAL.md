# Retrieval, replanned (2026-10-28)

Nick, 2026-10-28: "the whole retrieval side needs to be rethought and replanned, and tested before
implementing". This is the plan. Order 0046 runs its experiments; nothing reaches the serving path
until an experiment has passed. Orders 0039, 0043 and 0044 stay `blocked` until this plan says which
of their items survive (the end of this file).

**In plain words:** Cubby finds what a question is about by looking it up in indexes built from the
data it already holds, not by rules written one question at a time, and every choice is measured on
questions nobody tuned it on, in English and French.
- **Over a conventional model:** a model guesses the famous meaning of a name and the latest value
  of a fact, silently. Here each step says what it found and why, a name shared by two of a kind is
  never settled by fame, and nothing is spoken that the VM has not checked.

## How the plan was made
Three plans were written apart, on one brief (`docs/research/2026-10-28-retrieval-brief.md`):
- **Cowork's**, written and sealed before the others ran
  (`docs/research/2026-10-28-retrieval-cowork-sealed.md`, SHA-256 `9e5ec1f3…3e60a1`);
- **CubbyConsul's board:** two proposers, a prover, two skeptics, a librarian and a scribe, 5 rounds,
  $1.51;
- **Gemini 4 Argon,** asked directly by Nick.

What all three agree on is taken as the design. Where they disagree, an experiment decides, or Nick
does (the decisions at the end).

| | Brought | Got wrong |
|---|---|---|
| **Cowork** | a usage table between kinds ("uk"), the asker memory tested as a small-capacity problem, French parity as a pass line, a scale test at 75M items | underplayed the 44 `ambiguous_hop` rows, the largest refusal class |
| **Board** | evaluation integrity: freeze the lexicon before held-out data; build the fame set by name collision and publish it before any design; candidates cross unordered; memory may supply a referent, never a value; an erroneous admission to the store disqualifies; misses labelled by route before any dump decision | said no public French set with Wikidata ids exists (Mintaka and QALD-9-plus do); proposed growing the hand grammar; its seats could not read the code |
| **Gemini** | the most complete design: statement qualifiers and ranks (the 44 rows), CronQuestions, ShadowLink, a coverage audit before the dump, checking whether the local Wikipedia text kept its links, live Wikidata fetched by id, unbinding kept to one item's own relations | read `hippocampus.py` as hashing whole keys (that is the VM's memory; the hippocampus bundles word codes and already rebinds a recalled plan to a new entity); gave Mewsli-9 for French (it has none) |

## What all three agree on
1. **Facts stay exact keys with provenance;** similarity may only lead to a key.
2. **No superposed memory for facts, items or episodes.** SDM, Hebbian, NLMS and DG-sparse traces
   lose recall between 512 and 1,024 associations (`hippocampus.py`); the worlds hold millions.
3. **Candidates come from a local name index, not Wikidata's search:**
   - exact and normalized labels and aliases, English and French, in a finite-state transducer;
   - redirects and link anchors when a corpus keeps them;
   - letter-trigram binary codes for near misses (`capsules.py`'s code);
   - not the qFHRR word table, which is weakest on entity names (0.081 nDCG@10 against BM25's ~0.31).
4. **The James Young rule holds by construction.** Candidates cross into the choice unordered. Two of
   a kind are separated only by words the question itself carries (a place, a date, an occupation).
   Otherwise answer each (at most three) or ask.
5. **Rules only for guarantees.** Every other choice comes from data pushed into the stores.
6. **Held-out public sets with Wikidata ids;** the 187 and the stack's 20 become regression only.
7. **Measure before the 959 GB dump:** label every miss by its route first.

## The shape to test
| Stage | Stores | From | Decides | Crosses into the VM |
|---|---|---|---|---|
| **Read** | relation lexicon: property labels and aliases, EN+FR, frozen and checksummed; mention spans by longest match over the name index | the property table that exists; the name index | which spans are names, which properties the rest asks | spans; (property, score) |
| **Find** | name index (exact + trigram); BM25 over aliases and abstracts as a supplement only | Wikidata5M, DBpedia's abstracts, EN/FR Wikipedia (titles, redirects, anchors) | the candidate set, unordered | (item) set |
| **Choose** | per item: kind (P31/P279), its own facts, its text; kind × relation counts | Wikidata5M; abstracts | a kind that cannot carry the relation drops out; within a kind, only the question's own words separate | (item, score, reason) |
| **Fetch** | statements with rank and qualifiers (start, end, point in time) | the local dump's claims for these items; live Wikidata by id for time-sensitive relations | which values still hold; the source's own order (rank, dates) | exact facts |
| **Recall** | the asker's corrections and choices, per asker | the conversation | a referent the asker named ("him", "the first one") | (item) |
| **Read text** | FTS5 passages + a qFHRR re-rank around the mention | books, encyclopedia, Wikipedia, news | which passage a reader reads | facts, latent until corroborated |

**Where each stage lives** (Nick, 2026-10-31: "shouldn't we use the VM for retrieval?" -- yes, for
everything exact):
- **In the VM's layered store** (CubeLang's knowledge segments and history, `docs/STORE.md` in the
  CubeLang repo):
  - the name index: names and aliases to items, as segment keys, with an alias costing one key rather
    than a copy;
  - typo-tolerant lookup through the segment's own index, one or two edits away, as the history search
    already does;
  - the statements with their rank and dates;
  - the asker's sealed history.

  CubeLang programs query them, and the VM checks what they return in the same place.
- **Outside the VM:** the similarity work (the qFHRR context for choosing between namesakes, the
  relation codebook). It runs on grilly and hands the VM (item, score) only, as the boundary rule says.
  The VM's own hippocampal memory hashes key strings, so it cannot measure resemblance.
- **In plain words:** one store finds and checks every lookup, so what Cubby finds and what it verifies
  cannot drift apart.
- **Over a conventional model:** its retrieval is a separate search engine beside the model, and nothing
  checks what it hands over.
- **Serving it** depends on order 0042 (the VM as a service), still proposed.

**Branch, check, decide** (Nick, 2026-10-31). Where a name gives several items, or a fact several values:
1. One branch per reading runs its program in the VM (`QUERY`, verified hops).
2. Each branch's reading goes to the possibility worlds (H-E17): an impossible branch drops, an unknown one
   waits latent.
3. The graph over the branches that verify decides: answer, say once, answer each, or `ASK`.

The forked trunk (H-P1, H-P5) writes each branch's program from one reading of the question. It is tested
in 0046 as arms B1 to B3, before anything is built.

## Experiments (order 0046 runs them)
Each one is offline, on held-out data, with its pass line and kill line written before it runs.
- **R0, the denominator.** Run today's pipeline on the held-out sets. Label every refusal by its stage
  and route: question reading, no candidate, wrong or ambiguous candidate, relation, several values,
  absent from every local store. Report English and French apart. Check whether the local Wikipedia
  text kept its links. The dump line is fixed in 0046 before R0 counts anything.
- **R1, finding.** The name index variants against the search API: recall@K for K in {5, 20, 100},
  English and French, plus a typo-perturbed set. Entities whose label is absent from the index's own
  build split are scored apart.
- **R2, choosing.** Two ways to separate a kind's namesakes, on the same candidates:
  - lexical cue overlap (BM25) between the question and each item's facts and text;
  - qFHRR context against one prototype per item.

  Both are scored against today's tiers on the fame set (bare names: 0 single picks allowed), ShadowLink
  (Top, Shadow, Neutral) and AmbER (head, tail). A third arm adds the usage table between kinds, which
  is never allowed within a kind.
- **R3, the relation.** The frozen exact lexicon, the qFHRR codebook behind a rank-normalized margin,
  and today's grammar, on SimpleQuestions-Wikidata, QALD-9-plus (EN, FR) and Mintaka (EN, FR).
- **R4, several values and time.** The source's own order (preferred rank, then dates) and an
  interval filter for an explicit time, on CronQuestions and the 44 + 2 scoreboard rows. A present
  tense counts as "now" only in the arm Nick approves (decision 2).
- **R5, the asker's memory:** referents from a conversation, corrections recalled for a reworded
  question, never across askers.
- **R6, passages:** a reader's facts from retrieved passages, with any erroneous admission to the
  canonical store disqualifying.
- **R7, scale:** the name index and the shortlist at 5M, 20M and 75M names: recall, latency, memory,
  build time, stated with the machine.
- **R8, end to end:** the arms that passed, together, against today's pipeline on held-out sets,
  English and French. Kill: any wrong spoken value.

**Order:** R0 → R1 → (R2, R3, R4 in parallel) → R5, R6 → R7 → R8. Build only what passed. The dump
waits on R0 and R1.

## Sets
| Set | Gives | Language | Licence (to confirm in 0046) |
|---|---|---|---|
| Mintaka (Sen et al., 2022) | 20k questions, Wikidata ids | EN + FR | to confirm |
| QALD-9-plus (Perevalov et al., 2022) | 507 questions with Wikidata SPARQL | EN, FR | CC BY 4.0 |
| WebQSP_EL, GraphQ_EL (Li et al., 2020) | mentions in questions, Wikipedia ids | EN | released with BLINK |
| SimpleQuestions-Wikidata (Diefenbach et al., 2017) | one-hop questions, Wikidata | EN | to confirm |
| ShadowLink (Provatorova et al., 2021) | 16K snippets: Top, Shadow, Neutral, Tail | EN | to confirm |
| AmbER (Chen et al., 2021) | shared names in Wikidata, head and tail | EN | to confirm |
| CronQuestions (Saxena et al., 2021) | 410k temporal questions over 328k Wikidata facts | EN | to confirm |
| the fame set (built in 0046) | bare-name questions on names shared by two of a kind, intended item hand-checked | EN + FR | ours, published before R2 |

## What not to build
- A superposed memory as a store of facts, items or episodes.
- The qFHRR word table, or co-occurrence vectors, as the entity retriever.
- More hand-written tiers, or any ranking by sitelinks, links, degree or description length within a
  kind.
- Raw cosine thresholds as gates (they do not travel); rank-normalized margins only, set on a dev split.
- Wikidata's search API as the candidate source once the local index passes R1.
- A larger hand grammar. Whether reading needs a grammar at all is R3's question.

## Decisions for Nick
1. **The usage table between kinds:** how often a name means each item in text, used only between
   different kinds ("uk": the United Kingdom over Uttarakhand). Allowed?
2. **Present tense as "now":** may "who is the mayor of X" read only the values with no end date? The
   board's skeptics say no unless the question says "now" or "current". Recommended: test both arms
   in R4 and decide on the measured wrong count.
3. **The asker's past choice:** may it settle a tie ("last time you meant Georgia the state"), or
   only be offered as a question? The board says offer only; the ledger decides today.
4. **The dump line:** recommended, the full dump only if, after R1, at least 20% of the route-labelled
   seed misses on the held-out sets are items absent from every local store (a sample of at least
   1,000 mentions).
5. **The links:** if the local Wikipedia text lost its links, fetch the redirect tables (a few GB)
   before any dump. R0 (2026-10-28): it kept no links and no redirects, so this applies.
6. **The names extract (R1b in 0046):** EN and FR labels, aliases and Wikipedia titles for every item
   of the current dump, without its claims. R1 found that the missing names, not missing items, are what
   the local index lacks.

**Nick's answers (2026-10-31), on Cowork's recommendations:**
1. **The usage table:** allowed between different kinds only, never within one.
2. **Present tense as "now":** tested both ways in R4. The measured wrong count decides.
3. **The asker's past choice:** offered as a question ("last time you meant Georgia the state: that
   one?"), never used to settle a tie. Today's ledger stays until the stage is rebuilt from R5.
4. **The dump line:** 20% of the route-labelled seed misses absent from every local store.

Still open: 5 and 6 (the redirect tables and the names extract).

R0 measured decision 4's line: 5.5% of the route-labelled seed misses are items absent from every local
store, against 20%. The full claims extract waits.

**How any download is done** (Nick, 2026-10-31: 5 TB on Drive):
- on Colab, streamed from the source, keeping only the columns needed;
- the extract kept on Drive as the master copy, so the raw dump is never stored;
- what Cubby looks up copied to a local SSD (the spinning disk is too slow for lookups).

## What happens to the blocked orders
- **0043** (namesakes at the source): its guarantees stay (no value on the wrong item, answer each or
  ask). Its tiers are what R2 replaces. Items 2 and 3 of its check (fold the items that hold nothing;
  refuse an "each" with several values) wait for R2 and R4.
- **0044** (several values): its design is R4's first arm (preferred rank, single-value constraints).
  It is rebuilt from R4's result.
- **0039** (the question reader): replaced by R3's result.
