# 0009 — Capacity: from ~37 pieces a scene to 10,000 facts, at a fixed state and a constant-time read
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("we also need to grow the fact base / capacity
considerably": KBLaM holds 10,000 triples, a scene holds ~35) · hypothesis: H-K10

## Why
A scene is one superposition of `d = 1024`. At 37 written pieces, the 128k-way decode sits at 0.758 per piece, the
random-code ceiling (`diag_fact_cleanup_run3_2026-10-21.json`, `sim_hrr_ceiling_2026-10-21.json`). Splitting the
scene by address already fixes that offline: K = 4 gives 0.996 whole answers over 128k
(`diag_fact_cleanup_grid_run3_2026-10-21.json`). Scaled up, with K buckets chosen by a hash of the address and a mean
load of ~10 pieces a bucket, the ceiling at **10,000 facts** (46,000 pieces) is **0.999 whole answers** over the whole
vocabulary (`validation/logs/sim_bucket_capacity_2026-10-22.json`):
- 4,600 buckets, **9 MB (bf16)** of state per stream, still fixed-size;
- a read touches one bucket, so its cost does not grow with the number of facts;
- no ring of ids is needed, since the decode is over the vocabulary.

And writing needs no forward pass. The scene is built at the input level, as the sum of bound embeddings, so the
host can add a fact straight into the state: 10,000 facts load in milliseconds.

- **What it brings:** the model holds 10,000 facts at once in 9 MB, reads any of them in the same time, and loads
  them without reading them as text.
- **Advantage over conventional models:** KBLaM attends over every knowledge token, so its cost grows with the base,
  and a context window pays for every token in its KV cache. Here the read is constant-time and the memory about
  1 KB a fact.
- **In plain words:** it can keep thousands of facts at hand at once, and finding one takes no longer than when it
  knew thirty.

## Build
- **A. The capacity curve, offline first** (`validation/diag_capacity.py`; no training, CPU). Use run 3's trunk with
  run 4's copy, as `diag_swap_copy.py` assembles it, or run 5's once it exists.
  - **Write** N wiki-world facts straight into a bucketed scene, for N ∈ {37, 100, 1,000, 10,000} and
    K ∈ {1, N/40, N/20, N/10}.
  - **Ask** 200 questions about written facts and 50 about facts that were never written.
  - **Report:**
    - per piece and whole answer, for the cleanup over 128k (the algebra) and for the trunk with its copy;
    - how often an unwritten fact gets an answer;
    - state bytes, and the read time per question.
  - **Equivalence check:** at N = 37, K = 1, compare the direct write with the same facts as text in the document.
- **B. Build, flag-gated.** `SpatialLink(parts=K)` (`CB_SPATIAL_PARTS`, meta `spatial.parts`, absent = 1 =
  bit-identical; this is Addendum 2's B8).
  - The scene state is (B, K, d). `bucket = hash(address) mod K`, deterministic and documented. A write adds to its
    bucket, and a read unbinds its bucket. Decode = forward.
  - A host API writes facts directly: `write_facts(state, ids, places)`. The scene state saves and loads with the
    store.
  - For the copy's candidates at low load, take the vocabulary (`table`) or a small per-bucket ring, whichever A
    measures better. Keep the whole state fixed-size either way.
  - With order 0006's text keys, the bucket comes from the pointed tokens' ids, so it stays computable at ask time.
    Note how, without building it here.
- If A shows the trained copy needs retuning at low load, a copy-only run (0005's `CB_TRAIN_ONLY=copy`) on bucketed
  documents is the budget. Cowork runs it on the GPU.

## Acceptance (pre-registered)
- **A, at 10,000 facts with K = N/10:**
  - the cleanup ≥ **0.95** whole answers (the ceiling is 0.999, and run 3's table behaved like random codes);
  - unwritten facts answered ≤ **5%**;
  - state ≤ **16 MB** bf16 per stream;
  - read time within **1.5×** of the N = 37 time.
- **The trunk with its copy at 10,000 facts:** ≥ **0.80** exact, from A if the trained copy carries over, else after the
  copy-only retune.
- **Kill:** the cleanup at N = 1,000 with K = N/10 below 0.80 per piece. Then the hash routing or the learned
  table's geometry breaks the scaling, and it is reported, not patched.
- **Tests:** parts = 1 is bit-identical to today; a direct write is equal to a write through a forward; decode =
  forward with K > 1; full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. CPU for A. No bulk downloads: the wiki world's facts are already in the cache.

## Out of scope
- Growing the host store itself, a Wikidata ingestion, which needs Nick's yes for the download. Free-text keys
  (0006). Consolidation into weights.

## Addendum (Cowork, 2026-10-22): several values for one fact
Order 0007 found that several values written at one address superimpose ("lawyer" + "politician" read
"lawyerician"); today the reader defers them to the walk. This order owns how facts are written: give each value
its own sub-address (the fact's key bound with a value index), read in order until a read comes back empty, and
measure it on 0007's multi-valued facts (9 of 47 in its battery).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
