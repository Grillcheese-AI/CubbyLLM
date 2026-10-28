# 0046 — Retrieval, measured before it is built: experiments R0 to R8
status: in progress (Claude Code, 2026-10-28)
written: 2026-10-28 by Cowork · approved: 2026-10-28 by Nick ("tested before implementing"; a plan doc and an experiments-only order) · hypothesis: none (each experiment below is pre-registered)

## Why
Retrieval decides which item a name means, and which value a relation gives, with tiers written one
scoreboard row at a time (0037, 0043), scored on those same rows. Nothing measures them on names they
were not written for. Nick, 2026-10-28: "the whole retrieval side needs to be rethought and
replanned, and tested before implementing." `docs/RETRIEVAL.md` is the plan, written from three
independent plans (Cowork's sealed one, CubbyConsul's board, Gemini 4 Argon). This order runs its
experiments. It builds nothing on the serving path.

**What it brings, in plain words:** we learn which way of finding things works on questions nobody
tuned Cubby on, in English and French, before any of it is built.
- **Over a conventional model:** a model's retrieval is judged by its average score. Here each stage
  is measured apart, every miss says why, and a design that speaks one wrong value is killed.

## Build
All of it lives in `validation/` (network fetchers in `cubby_serve/`), never in `cubbyllm/`. Data sits
off-repo under `$CUBBY_DATA_ROOT`. Only ids, counts and the fame set are committed.

1. **The sets**
   - **Builders** (`validation/retrieval_sets/`) for:
     - Mintaka (EN + FR);
     - QALD-9-plus (EN, FR; its Wikidata queries);
     - WebQSP_EL and GraphQ_EL;
     - SimpleQuestions-Wikidata;
     - ShadowLink (Top, Shadow, Neutral, Tail);
     - AmbER;
     - CronQuestions.
   - **One schema** per row: id, language, question, gold mention items, gold property when the set has
     one, gold answers, source, split.
   - **Wikipedia titles to Wikidata ids:** through Wikidata by title, cached.
   - **Licences:** confirm each in the report.
   - **Downloads:** ask Nick before any download over 1 GB.
   - **Splits:** dev and test, fixed by seed 0. Tune on dev only. Read test once, at the end of each
     experiment.
   - **Today's rows** (the 187 and the stack's 20) are regression only.
2. **The fame set** (`validation/retrieval_sets/fame_set.py`), built mechanically:
   - **Names:** every label that Wikidata5M gives to two or more items of one kind (P31, one `subclass
     of` step), where at least two of those items carry a relation from a fixed list (place of birth,
     place of death, date of birth, country, capital, occupation).
   - **Sample:** 500 names, drawn by seed 0.
   - **Bare-name questions,** EN and FR, from fixed templates ("Where was {name} born?" / "Où est né
     {name} ?"). Gold: no single item may be picked.
   - **Qualified variants:** the same question with one discriminator from one item's own facts (its
     occupation, a place it is in). Gold: that item. Spot-checked by hand: 50 rows, the check in the
     report.
   - **Freeze:** committed with its SHA-256 before R2 runs. Not changed after.
   - Built by name collision, not by sitelinks or degree.
3. **The relation lexicon:** property labels and aliases, EN and FR, from the property table that
   exists, frozen with its SHA-256 before any held-out row is read.
4. **The harness** (`validation/retrieval_bench/`): runs each arm of each stage on a set and writes
   `validation/logs/retrieval_R<n>_<date>.json`. Every row records its candidates (unordered within a
   kind), the stage that refused, and the reason.
5. **The experiments,** in this order: R0, then R1, then R2, R3 and R4 (in parallel), then R5 and R6,
   then R7, then R8. Each is described in `docs/RETRIEVAL.md`. The lines are below.

## Acceptance (pre-registered)
- **R0, the denominator**
  - Today's pipeline on every dev set. Each refusal is labelled by stage and route:
    - reading;
    - no candidate;
    - several candidates;
    - wrong candidate;
    - relation;
    - several values;
    - item absent from every local store (Wikidata5M, DBpedia's titles, EN/FR Wikipedia's titles).
  - English and French apart.
  - Also reported: whether the local Wikipedia text kept its links and redirects.
  - The dump line is decision 4 in `docs/RETRIEVAL.md`, fixed before R0 counts.
- **R1, finding**
  - Arms: the search API (today); exact; + normalized; + trigram codes; + BM25 as supplement only
    (fills the set when the others give fewer than K, never displaces).
  - Metric: recall@K, K in {5, 20, 100}, with names absent from the index's build split scored apart,
    plus a typo set (one or two letters changed in 20% of names).
  - **Gate:** the best local arm's recall@20 is at least 0.90 on Mintaka EN and 0.85 on Mintaka FR, and
    above the API's on every set. The trigram arm is kept only if it adds at least 5 points on the
    typo set.
  - **Kill:** no local arm beats the API at @20 on Mintaka.
- **R2, choosing** (on R1's candidates)
  - Arms: today's tiers; BM25 cue overlap with each item's facts and text; qFHRR context against one
    prototype per item; each plus the usage table between kinds (only if decision 1 allows it).
  - **Gate:**
    - fame set, bare names: 0 single picks;
    - qualified variants: 0 wrong, at least 70% resolved;
    - ShadowLink: Shadow's accuracy among spoken picks within 5 points of Top's;
    - on Mintaka and WebQSP_EL test: no more wrong picks than today's tiers (and at most 1 in 50), with
      more questions decided.
  - **Kill:** any bare-name single pick, or a Shadow-Top gap over 10 points.
- **R3, the relation**
  - Arms: today's grammar; the frozen exact lexicon over the spans left after the names; the qFHRR
    codebook behind a rank-normalized margin set on dev.
  - Metric: property recall@5 given the gold item, on SimpleQuestions-Wikidata, QALD-9-plus (EN, FR) and
    Mintaka (EN, FR).
  - **Gate:** at least 0.85 on SimpleQuestions-Wikidata and 0.80 on QALD-9-plus FR. The qFHRR arm is
    kept only if it adds at least 3 points at the same wrong rate.
  - Report the share of each set today's grammar parses.
- **R4, several values and time**
  - Arms: the source's own order (preferred rank, then single-value constraints, then the values with
    no stated end); + an interval filter for an explicit time; + "present tense means now", run only as
    Nick's decision 2 says.
  - Sets: CronQuestions (single-entity questions) and the scoreboard's 44 + 2 rows.
  - Wikidata5M holds no ranks or qualifiers. For these items, fetch the statements by id (rank, start,
    end, point in time) through the source's cache, and record the date of the fetch.
  - **Gate:** 0 wrong; at least 20 of the 46 rows spoken; at least 60% of CronQuestions single-entity
    dev questions spoken.
  - **Kill:** any wrong value, such as Valladolid or Toledo beside Madrid.
- **R5, the asker's memory** (synthetic dialogues, 200 per language)
  - **Gate:** referents resolved at least 95%; 0 values supplied by memory; 0 crossings between askers;
    ties untouched unless decision 3 allows it.
- **R6, passages**
  - A reader's facts from retrieved passages (500 sampled).
  - **Gate:** 0 erroneous admissions to the canonical store (latent until corroborated, as today);
    passage recall@10 reported.
- **R7, scale**
  - The name index and the shortlist at 5M, 20M and 75M names.
  - Reported: recall, p50 and p95 latency, memory and build time, with the machine and corpus stated.
  - **Gate:** p95 lookup at most 50 ms at 75M. Nick sets the memory line from the numbers.
- **R8, end to end**
  - The arms that passed, together, against today's pipeline on each held-out test set, EN and FR.
  - **Gate:** spoken at least today's.
  - **Kill:** any wrong spoken value.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim).
- **Nothing on the serving path changes.** No flag, no default, no tier.
- **No external model in any arm,** and none to build the stores. The fame set's questions come from
  templates, not from a model.
- **Nothing is ranked by fame within a kind:** sitelinks, degree, link counts and description length
  decide nothing, and candidates cross unordered.
- **No hand rule is written for a failing row.** A failure is reported as what it is.
- **Licences:** the sets stay off-repo when their licence asks it; logs hold ids and counts only.

## Out of scope
- Building any stage for serving: an order per stage, after its experiment passes and Cowork has
  read it.
- The 959 GB dump: only on decision 4's line.
- The trunk (0045).

## Report
<!-- Claude Code fills this in per experiment, then sets status: review after R0 and R1 (Cowork reads them before R2) -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
