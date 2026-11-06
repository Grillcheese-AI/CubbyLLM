# 0055 — The full knowledge base at home: Wikidata hot and cold, so the benchmarks read the real ceiling
status: in progress
written: 2026-11-06 by Cowork · approved: 2026-11-06 by Nick ("perfect lets put that as a top priority so we can have the true numbers for the benchmarks. Right now its very low as we do not have the full kb") · hypothesis: the system paper; the design is `docs/research/2026-11-06-fact-store-synthesis.md`

**Priority: top** (Nick). It goes before 0054's full run, the rest of 0053, and every other open order. 0054's full run
waits for this order, then reads before (today's knowledge base) and after.

## Why
Today Cubby answers from:
- a 4.3M-fact Wikidata5M world;
- a small wiki world;
- live Wikidata, through its search API.

SimpleQA's "3 of 860" was the wiki world. 0054 run now would measure the store, not the system.

The panel and its check (the synthesis) give the design: **one hot store, Cubby's own, in Rust, and the full graph as
a cold tier.** The full graph's numbers come from Willerval, Diefenbach and Bonifati (ESWC 2024): 19.2 billion
triples, built on a 16 GB desktop in about two days, 340 GB.

**What it brings, in plain words:** Cubby keeps a fast pocket edition of what it needs and the complete Wikidata on
the shelf beside it, so the benchmarks show what the system can do, not what its small store happened to hold.
- **Over a conventional model:** a model's knowledge is whatever fit in its weights at training. Here every fact is
  stored exactly, with its source, and is added or corrected in one write, without retraining.

## The machine today (read 2026-11-06)
- 48 GB of RAM. Five drives with 47-57 GB free each. No Java.
- **A 2 TB NVMe is needed before the full builds:**
  - the dump, if stored (100-150 GB compressed);
  - the cold tier (about 340 GB) and its build space;
  - the hot store (25-90 GB, estimated).

  This is Nick's purchase. Until it is in, steps 0 and 1 run: they need no local room.
- **Update (2026-11-06, later): no purchase needed to start.** Nick is freeing the system NVMe (94 GB free, about
  210 GB once his Steam library has moved). The 1 TB hard drive is back and reads cleanly.
  - **Hot store:** on the system NVMe.
  - **Cold tier:** on the 1 TB hard drive. That drive is slower, which is acceptable for the rarely used tier.
  - The 2 TB NVMe becomes an upgrade, not a blocker.
- **Nick's 5 TB Google Drive holds the dump and every build product.** Builds can run on Colab against it. Serving
  still needs the local NVMe: a lookup from Drive is far too slow for the VM's hops.

## Build
0. **The coverage audit** (outside `cubbyllm/`): 1,000 questions.
   - **Amended 2026-11-06 (Nick: "stop it now", "amend step 0").** Read live, the audit costs about 13 hours (about
     50 s a question on Wikidata's API), and neither of its jobs needs the live API: the split does not gate the
     download (below), and the store check needs a reference, which the same audit gives against the dump. So:
     - **Step 0 is the audit's code** (`validation/fact_store/`, `e41d481`) **and the 51-question live sample** it read
       before it was stopped (SimpleQA only, kept off-repo). The sample is the live reference for the store check.
     - **The 1,000-question split is read in step 1, from the dump** (its statements in place of the API), in
       minutes.
   - **The questions:** SimpleQA 400, Mintaka 300 (with its French), MKQA 300 (French).
   - **Per question:** is the gold a value of a statement on the asked item, elsewhere in Wikidata, or not in Wikidata?
   - **Where a set has no gold ids** (SimpleQA): every candidate item for the name is checked (no fame ordering), and
     ambiguity is counted.
   - 100 disagreements are read.
   - **The split is the ceiling the full base can reach.** A low share means text is the next lever, not a stop:
     Nick's priority stands.
1. **The dump, downloaded once to Nick's 5 TB Google Drive** (Nick, 2026-11-06, answering where the download may go:
   his 5 TB Drive). This is done from a Colab session signed in to the account that owns that Drive. The compressed
   dump is kept there as the source copy, so a parser bug means a reparse, not a re-download. Then one pass over it,
   never unpacked to disk, writes two things:
   - **Counts:** items; statements by rank; qualifiers and references; EN, FR and `mul` terms; bytes by section. This
     settles the panel's conflicting sizes.
   - **A 1% hash sample** (not the first Q-ids). The hot store is built from it, and on it are measured:
     - bytes per statement, and the build's time and peak RAM;
     - lookup latency, in memory and from SSD;
     - name resolution: latency, and EN/FR recall on the audit's entities that fall in the sample;
     - **measured, not gating:**
       - inverse, symmetric and chain folding, from Wikidata's own property constraints, with exceptions counted;
       - place ids in S2 (Hilbert) order against Q-id order: pages read on geography questions;
       - neighbourhood preloading against a plain cache;
       - how often a coarse community map prunes a relation search.
   - **Where:** a Colab session Nick runs, reading the dump from his Drive. The counts and the sample are written
     back to Drive.
2. **The full builds, after the NVMe:**
   - **Hot, Cubby's store** (the synthesis, section 3):
     - dense ids;
     - statements at every rank, flagged;
     - time qualifiers hot, the others cold;
     - references deduplicated;
     - exact values (no floats; dates keep their precision and calendar);
     - EN, FR and `mul` names in an FST, exact and accent-folded, with a bounded typo path;
     - a reverse index;
     - a small overlay for live values.
   - **Cold:** the full Wikidata in HDT. Either built with qEndpoint's k-HDTCat (GPL-3.0, Java 17, used as a
     separate tool and never linked into Cubby's code), or taken from the prebuilt index its image downloads (its
     date and contents recorded). Which one is Nick's call at download time.
   - **Both are resumable in chunks.** This PC has shut down under long jobs before.
3. **Wiring, behind a flag (off by default):** the VM's hops read the hot store, then the cold tier, then live
   Wikidata.
   - The live value wins for time-sensitive facts (the standing rule).
   - The cold tier is read in-process from the HDT file. If only a server works, that is a question for Nick: the
     serving code never touches the network.
4. **0054 reruns with the flag on.** Its full run is the "after"; its 50-question preflight on today's base is the
   "before".

## Acceptance (pre-registered)
- **Tests:**
  - 100,000 entities round-trip with 0 differences against a second, independent parser;
  - 1,000 items re-fetched at their recorded revision agree;
  - a rebuild gives identical chunk hashes;
  - with the flag off, today's answers are unchanged, bit for bit;
  - no stored value is a float;
  - no candidate list is ordered by sitelinks, pageviews or Q-id.
- **Reported:**
  - the stream's counts;
  - the hot store's size and bytes per statement;
  - build time and peak RAM;
  - lookup latency p50, p95 and p99, in memory and from SSD;
  - name resolution latency and recall, EN and FR;
  - the audit's split, per set.
- **The lines:**
  - a fact lookup from SSD at most 1 ms p95; name resolution at most 50 ms p95;
  - every fact the audit found in live Wikidata (step 0's live sample) is found in the store, unless it changed after
    the dump's date (listed);
  - with the flag on, 0 wrong on the existing batteries (E8 reworded, the emitter families, the stack's 20, 0038's
    scoreboard), and more answers spoken.
  - **Size:** the hot store at most 90 GB, the panel's top estimate. Above that, the projection is reviewed with
    Cowork before the full build.
- **Kill:**
  - any stored value that differs from its source: the build is not faithful;
  - or, in 0054 with the full base, Cubby's wrong answers above 1% of its spoken ones.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Where the store lives:** in the VM's datastore (the cubelang repository), or in a crate beside it. Claude Code's
  choice, recorded. Main cubelang's `target/release` binary is never overwritten.
- `cubbyllm/` never touches the network. The audit, the stream and every download run outside it.
- **Downloads over 1 GB:**
  - The dump to Nick's Drive has his yes (2026-11-06).
  - Any other large download (a prebuilt HDT index, a second dump) waits for his yes in chat.
  - The Drive's account is named in no committed file.
- No external model builds stored knowledge.
- **Licences recorded:** Wikidata CC0; qEndpoint GPL-3.0 as a tool only; every crate's licence checked.
- No local path in a committed file.

## Out of scope
- **Text** (Wikipedia, books): passages, extraction, and their compression (hypertokens or zstd). The next order,
  from this one's numbers.
- **Following Wikidata's change stream:** after the first full build.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's addendum (2026-11-06): the hot store chosen by real use
Nick found the Wikidata Query Logs dataset (Walter and Bast, SIGIR 2026, arXiv 2602.14594; the
`ad-freiburg/wikidata-query-logs` repository). Its organic split holds about 860,000 distinct real queries that
people ran against Wikidata between June 2017 and March 2018, from Malyshev et al.'s 2018 logs. The queries are
human-written; only the dataset's questions are model-written, and those are not used here.
- **First:** the licence of the logs and of the released files is checked and recorded. The paper says "permissive"
  without naming one. If it does not allow this use, this addendum stops.
- **From the real queries, with no question text:**
  - each property and qualifier, counted by how many distinct queries use it;
  - the query shapes, counted.
- **Used for the projection:**
  - the properties and qualifiers real queries use go hot, ranked by use; the rest go cold;
  - reported: the share of real queries whose properties all sit in the hot store, at the hot store's size.
- Nothing else in 0055 changes.
## Cowork's addendum (2026-11-06): QLever as the reference engine
From the Freiburg group's repositories (`ad-freiburg`; Nick: "yes"). Its licence is checked and recorded first; it
is expected to be Apache-2.0.
- **QLever indexes the whole Wikidata dump** in about 4.5 h with about 20 GB of RAM on a desktop CPU, by its own
  Wikidata recipe (`qlever-control`, `Qleverfile.wikidata`).
- **Its two jobs here:**
  - the reference the hot store and the cold tier are checked against: the stream's counts, and a 100,000-entity
    sample's facts, read both ways;
  - the engine 0054's query-log benchmark uses to recompute its gold answers on the same dump.
- **Serving does not change:** the cold tier stays the HDT file read in-process. QLever runs as a server, and the
  serving code never touches the network unless Nick decides a local server is fine.
- **Space:** its index size is measured. If the 1 TB drive cannot hold both it and the HDT tier, Cowork and Nick
  choose one, and the report says what each would cost.
## Cowork's addendum (2026-11-06): the source dump is the WikiKGQA 2026 challenge's own
Nick chose it ("Challenge dump"; his yes for the download): `wikikgqa-wikidata-all-dump.ttl.bz2`, 121.3 GB, CC0,
from QAWiki v2026.3 on Zenodo (DOI 10.5281/zenodo.22661413).
- **Why:** it is the exact Wikidata the WikiKGQA 2026 challenge was scored on, and the record also holds every team's
  submitted answers. Built from it, Cubby is scored on the challenge's test exactly as the entrants were (0054's
  addendum). Every other benchmark runs on the same snapshot too.
- **The trade-off, accepted:** the snapshot is some months old. Live Wikidata still answers time-sensitive facts (the
  standing rule).
- **What changes in step 1:**
  - the stream parses RDF (Turtle), not JSON: statements, qualifiers and references all appear as the statement
    nodes of Wikidata's RDF format;
  - each item's revision is read from its `schema:version`, for the revision cross-check;
  - QLever's recipe and the HDT tools read this format directly.
- **The dump's date** (the challenge's "Wikidata base version") is recorded in the report.