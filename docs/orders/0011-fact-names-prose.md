# 0011 — Fact shards in readable words: names keep their spaces and case; no "of of"
status: review
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
- **Commits:** `fa1d8e0` (claim), `6dd5485` (`--names v2`, `facts_far1k_v2`, the shard check), this one (run 5's probe,
  the report).
- **Files:** `validation/fact_names.py` (new, 50 lines: `words`, `question`, `doubled`), `validation/make_fact_shard.py`
  (`--names v1|v2`, default v1; the build log counts doubled words and no longer carries local paths; 254 -> 260 lines),
  `validation/diag_fact_names.py` (new: two shards compared from their bytes), `tests/spatial/test_fact_shard.py`
  (+3 tests). Logs: `validation/logs/make_fact_shard_facts_far1k_v2_2026-10-21.json`,
  `diag_fact_names_facts_far1k_v2_2026-10-21.json`, `hypertoken_recall_spatial_run5_on_far1k_v2_2026-10-21.json`.
- **Rendering (`--names v2`):**
  - relations as before (`WORKS_FOR` -> works for);
  - entities as before (`KeikoMatsuzaka` -> Keiko Matsuzaka), and an all-caps name kept as it is (`ABBA`, `1 BCE`,
    `FIDO2`; v1 lowercased these 1,979 names too);
  - event ids part by part, split on CamelCase and digit runs, case kept, joined by spaces:
    `FortPeckDam_ConstructionBegan1933` -> "Fort Peck Dam Construction Began 1933", `Rocket_UseInChina13thCentury` ->
    "Rocket Use In China 13th Century". Nothing is lowercased: an id cannot tell a word from a proper noun ("china"
    would be wrong; "Began" is only stiff). The order's example lowercases "construction began"; that rule would
    also lowercase China.
  - 79,810 of the 374,975 names render differently. v1 (the default) rebuilds the first 300 documents of
    `facts_far1k_lead` byte for byte.
- **Questions -- one change from the order.** Dropping the doubled "of" ("What is the variant of Spanish Dancer?")
  asks the inverse in English: "the variant of X" is a variant OF X, while `Spanish Dancer VARIANT_OF Y` says
  Spanish Dancer is the variant; "What is the capital of Paris?" for `Paris CAPITAL_OF France` shows it plainly.
  So the 646 relations ending in "of" ask subject first: "What is Spanish Dancer variant of?", with a "the" where the
  name's last word is the relation's first ("What is Camp Album the album of?", 340 facts). The order allowed
  "or rephrases". Over all 365,923 facts: questions with a doubled word outside the names, 101,246 (v1) -> 0.
- **Tests:** `tests/spatial/test_fact_shard.py` 7 passed (the three kinds; no doubled question over all 2,902
  relations, skipped without the wiki world; the readable name's pieces written at their addresses). Full suite:
  see order 0012's report (run the same night; no failure in this order's files).
- **Measured:**
  - `facts_far1k_v2` (facts_far1k_lead's arguments, seed 2, `--names v2`): 20,000 documents, 159,882 facts, 100,152
    questions, as before. From the bytes: the same written and asked addresses and the same number of questions in
    all 20,000 documents; written pieces 741,267 -> 706,581 (spaced names tokenize shorter); questions with "of of"
    27,585 -> 6 (titles that begin "Of" after the template's "of": "the author of Of Mice And Men", kept); 0 with a
    doubled word outside a name. On the cache and the Drive (md5 equal; no sync duplicates); `facts_far1k_lead`,
    `facts_far1k` and `facts_far1k_text` md5 unchanged.
  - **Run 5 (nothing trained), the last 64 held-out documents, CPU:** `facts_far1k_v2` **0.931** exact (first token
    0.988), the same 64 documents of `facts_far1k_lead` **0.941** (first token 1.000), 321 questions each. The drop is
    0.010; the line was 0.05 against 0.946 (all 2,002 questions): pass. The copy carries over to the new spelling.
- **Not verified here:** the held-out set is 64 documents (321 questions), not the full 400; which 3 answers (and 4
  first tokens) the new spelling costs was not broken down.
- **Found on the way:**
  - 528 facts of the wiki world name a Python repr (`<class 'wikikg.relations.X'>`, `<__main__.DynamicRelation
    object at 0x...>`) as a subject or object, and one object is `_` (`BryantPark LOCATION_OF _`); a relation is
    spelled `SP0USE_OF`. The store's labels are out of scope; they are in the shards as written (v2 keeps `_`).
  - Entities with a glued digit run stay glued (`Of Human Bondage1964 Film`): entities were "as now".
  - Fact sentences can still repeat a word where a name meets the relation ("Horror Comics Genre genre ...",
    "Camp Album album of ..."): only the questions were in scope.
  - Committed build logs from 2026-10-20 (`make_fact_shard_*_2026-10-20.json`) carry the cache's and the tokenizer's
    local paths; new logs don't. Not edited.
- **Questions:** make `--names v2` the default for new shards (0007, 0010 and 0018 will build from these)? It is a
  flag today because CLAUDE.md wants new behaviour off by default.
