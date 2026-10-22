# 0011 — Fact shards in readable words: names keep their spaces and case; no "of of"
status: in progress
written: 2026-10-22 by Cowork (Nick: "why is there no spacing between words?") · approved: 2026-10-22 by Nick · hypothesis: H-K10 (data)
**Priority (Nick): take this next after 0006, before 0007** (0007, 0010 and 0006's templates use these shards).

## Why
The far-fact answers read `fortpeckdam highrelease2011` and `rocket useinchina13thcentury`. The model copies
exactly what was written. What was written is wrong: `make_fact_shard.words()` treats any name with an underscore
as a relation (`WORKS_FOR` → `works for`), so it lowercases it and only swaps `_` for a space. The wiki world's
76,901 `TIMELINE_EVENT` objects are event ids, `FortPeckDam_ConstructionBegan1933` and `Rocket_UseInChina13thCentury`,
and 77,575 of its 365,923 objects carry an underscore. Those lose their CamelCase splits and their capitals. The
question template also doubles a preposition: `VARIANT_OF` → "What is the variant of **of** Spanish Dancer?".
Plain CamelCase names render fine (`HexabranchusSanguineus` → "Hexabranchus Sanguineus").

- **What it brings:** the facts Cubby recalls read like words, "Fort Peck Dam: construction began, 1933", so
  answers are speakable as they are. Free-text questions (0006) and the casual talk adapter (0010) train on real
  phrasing.
- **In plain words:** the answers come out as normal words instead of squashed-together labels.

## Build
- `words()` gets three cases:
  - **relations** (UPPER_SNAKE): as now;
  - **entities** (CamelCase): as now;
  - **event ids** (`Subject_EventCamel`): each part split on CamelCase and digit runs, case kept. For example
    `FortPeckDam_ConstructionBegan1933` → "Fort Peck Dam construction began 1933" (exact rendering Claude Code's
    call; say it in the docstring).
- The question template drops a trailing preposition the relation already carries ("What is the variant of
  Spanish Dancer?"), or rephrases. No doubled word, checked over all 2,902 relations.
- Rebuild the far-fact shard under a new name (`facts_far1k_v2`, same seed and documents otherwise), keep the old
  one, and copy it to the Drive.
- **No training:** probe run 5's final on the new shard's held-out documents (`exp_hypertoken_recall`, CPU, 64 docs),
  to show that the copy carries over to the new spelling.

## Acceptance (pre-registered)
- Tests: the three cases render as specified; no question in the rebuilt shard holds a doubled word; old shards are
  untouched. Full `pytest tests`.
- Run 5 on `facts_far1k_v2` (64 held-out docs): exact within 0.05 of its 0.946 on the old shard. A bigger drop says
  the copy learned the old spelling's statistics. Report it; don't patch it.

## Out of scope
- Retraining. The villages shards. The store's own labels (the host speaks those, not the shard).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
