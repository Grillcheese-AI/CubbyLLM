# 2026-09-13 — the gen-3 builder: certified chains in, free-text questions out, the plan never the model's

Agreed on 2026-09-12 after the ceiling probe: the free-text holes the gate exposed — verb forms ("was
born"), ask-type forms ("on what day, month, and year"), possessives, canonical labels for plain words,
appositives — are patched by a dataset built the reverse way. Not questions with answers a model wrote,
but **chains the store already certifies**, each with provenance, worded as questions; and the plan the
emitter must learn to emit is the **host's** — the seed entity and the question's own relation words,
which the host resolves the way it resolves them at serve time. Nick's rule stands: an LLM's only job is
the dataset, and this first pass uses none at all.

## The builder (`standin/data/build_gen3.py`)

**The certified pool.** (1) The encyclopedia's frame-read facts (`standin/encyclopedia.py`, exp_r16:
10,790 over 29 volumes, sentence on record), declared and added to the wiki world under Wikidata's
labels; (2) a seeded sample of the wiki world's own facts for the relations that have wordings (its
labels are its own: `birth date`, `birthplace`, `death date`, `alma mater`); (3) two-hop chains off
those: a first hop as a noun phrase ("the spouse of X", "the child of X") and a second held fact off its
object.

**The wordings.** A table per relation, each with the relation words the plan binds: `When was {S}
born?` → `born`; `On what day, month, and year was {S} born?` → `born`; `What is {S}'s date of birth?`
→ `date of birth`; `When did {S} die?` → `die`; `Where was {S} born?` → `born`; `Who is {S} married
to?` → `married`; `What was {S}'s profession?` → `profession`; `Which university did {S} attend?` →
`attend`; 50 wordings over 17 relations. Not `When is X's birthday?` — that is P3150, day and month,
another property.

**The gate.** Every (question, host plan) goes through `learn_and_answer` with a source that has nothing
to fetch (the store must already hold the chain), the local property table as the resolver, the coverage
rule, the typed answer class, the resident VM — and the record enters only when the VM verifies and the
answer is the chain's object. Every refusal is counted by wording and by reason. **That count is the
coverage rule's own measurement on free text**, and the first probe read like one.

## What the probe found, and the five rules it forced (1,500 encyclopedia + 900 wiki + 420 two-hop chains)

Probe 1: 10,991 asked, **5,449 certified (49.6%)**. The refusals, by class, each a host defect:

1. **`unknown_relation` 2,375** — `When did X die?`: the table has `died`, not `die`; `Which university
   did X attend?`: `attended`. → `PropertyAliases.relations` tries a wording's regular inflections after
   the exact tier (first word: -s/-es/-d/-ed/-ing and their stems, two steps: `establishing` →
   `establish` → `established`), each an exact lookup. `die` → date of death / place of death (the ask
   type splits them); `attend` → educated at / participant in; `work for` → employer.
2. **the store's own wording** — the wiki world says `birth date`, `birthplace`, `alma mater`; Wikidata's
   labels are `date of birth`, `place of birth`, `educated at`; the plan says `born`. Lever 4 kept the
   labels the store holds and the store held neither — `unknown_relation`, then a fetch, with the fact in
   the store all along. → The table names every wording of a label's property (`PropertyAliases.wordings`),
   and a held wording that names that property **alone** is lever 4's target (`born` → `birth date`; a
   wording two properties share is no evidence of either). And the reverse: a **held** relation whose walk
   finds nothing, when the store holds the same property under another wording (`birthplace` beside the
   encyclopedia's `place of birth`), is translated to it and walked once more (`learn.resolve_siblings`) —
   the sibling must be the only other held one, the ask type narrowing two.
3. **`plan_does_not_cover_question` 1,476** — `In which city was X born?`: `city` is a relation word the
   store knows, read as a dropped hop; `Where was X born?` — `where` names no ask type, so `born` stayed
   an ambiguity. → The typed answer class grows a **place** ask (`where`, or a place-class noun — city,
   country, county, district … — as the ask), name-valued: its words are the ask's frame, `born` narrows
   to `place of birth`, and a date answered to a *where* is refused as a type mismatch.
4. **`plan_does_not_cover_question`, two-hop** — `Who is the child of X married to?`, `When was the
   father of X born?`: hop 1 a noun phrase before the entity, hop 2 a verb after it; neither the
   answer-first nor the entity-first reading. → `covers()` v5, the **mixed reading**: inner hops, the
   entity, then the rest in walk order. The residual rule is untouched; a dropped hop still fails.
5. **the tail split** — `married of anne of cleves` split at the last ` of ` into `married of anne` |
   `cleves`: an unknown relation nobody could resolve, a seed that is nobody. → `split_tail`: for an
   unheld relation the **question decides** — the split whose entity the question names, unless the
   question itself joins the relation to the next part with `of` (`the place of birth of jean` keeps
   `place of birth`). And in the held tier, a relation the store states once and no source declared may
   be a fragment (`spouse of anne`); a shorter reused or declared relation wins it.

Plus one carry-over: a relation translated twice (`born` → `place of birth` by the ask type → `birthplace`
by the sibling step) keeps the question's own word on record, so coverage still sees `born`.

Every rule pinned (`validation/test_search_learn.py` +6, `test_property_aliases.py` +2,
`test_answer_type.py`, `test_plan_verify.py` +1); the whole suite green at 450.

**Probe 5 (the rules in): 10,464 asked, 8,325 certified (79.6%)**, 16,650 records (a plan and a chain
record per certified question), 194 s, 17,425 VM calls — and **no coverage refusal left** (1,476 → 0).
What remains refused, by class: `ambiguous_hop` 1,009 — multi-valued relations (two spouses, three
children, four awards, a town in two counties): no single answer, correctly refused, a list-answer shape
for later; `unknown_relation` 736 — wordings the table has no form of (`pass away`, `county` as a
relation, `educated` alone), which stay refused rather than invented; `retrieval_exhausted` 295 —
`attend` / `study` against a store that holds `alma mater` for few of the sampled people;
`verified_other_answer` 27 — the wiki world's full date beside the encyclopedia's year for the same
person (both true; not this chain's record); `answer_type_mismatch` 13 — `Deadmau5 is the birth date of
Joel Thomas Zimmerman`, a corrupt wiki-world fact the typed answer class refused three times over. Zero
wrong answers certified, by construction: the answer must equal the chain's object. `When was X born?`
524/527, `Where was X born?` 414/433, `When did X die?` 428/434, `Who is X married to?` 66/127 (60 of
the rest have two spouses on record).

`standin/data/out/gen3_probe.{jsonl,manifest.json}` (off-repo); the full build follows.
