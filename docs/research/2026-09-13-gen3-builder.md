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

## The full build, and the merge

All 10,790 encyclopedia facts, 300 wiki-world facts per relation (15 relations), 200 two-hop chains per
noun (7): **16,626 chains, 63,138 questions asked, 52,538 certified (83.2%)**, 105,076 records, 108,154 VM
calls, 1,050 s. By wording: `When was X born?` 3,509/3,535; `Where was X born?` 2,532/2,606; `When did
X die?` 3,185/3,207; `In which district is X located?` 1,261/1,333; `Who is X married to?` 317/501 (181
with two spouses on record); `Who wrote X?` 0/307 (`wrote` names `notable work`, the inverse — refused,
rightly). Refused: `unknown_relation` 4,915 (`pass away`, `county`, `educated`: no form in the table),
`ambiguous_hop` 3,963 (multi-valued — and two `Benedetto Accolti` in one encyclopedia, 1415 and 1497,
refused as one ambiguous entity), `retrieval_exhausted` 1,219, `verified_other_answer` 169 (the wiki world
beside the encyclopedia: `Date17750120`, a corrupt wiki value, among them), `answer_type_mismatch` 43.

**The merge** (`emitter_sft_v13e.jsonl`): gen 2 unchanged (11,845 records) + a **capped, wording-stratified
sample of 12,000 certified train questions** (24,000 records: a plan and a chain each; round-robin over
source × wording, seeded, so every one of the 41 wordings and both sources are in) + **all 5,234 held
questions** (10,468 records; entities no training record used — exp_r17's set). 46,313 records, 38,760
train rows after repeat, against gen 2's 14,760: the free-text records are the majority of the training
signal without drowning the grammar's chains. The cap is `--merge-cap` (0 = all 47,304); the full set is
kept off-repo for a later, larger generation.

## The held split as a gate: the gen-2 emitter's baseline (exp_r17), and lever 7

`validation/exp_r17_gen3_heldout.py`: 600 of the 5,234 held questions (seed 7), any emitter GGUF, the
same gate and store the builder used, scored verified / correct / near / **wrong** against the record's
gold. The gen-2 emitter, which never saw a free-text wording: **599 plans, 481 verified, 481 correct,
0 wrong**, 114 coverage refusals, 315 s. Read against the wordings, the refusals are two things. The
emitter **re-types the entity and garbles it** — `karol burgmann` for Karl Brugmann, `jean louis bastu`
for Barthou, `vincente scamozzi` — with the right relation; and it **reads the ask words as relations** —
`day`, `month`, `year` as three hops of "On what day, month, and year was X born?", `year of death`,
`died` + `city`, `city of death` for "In which city did X die?" (0/14).

The first is the host's to fix, and it is **lever 7** (`learn.snap_seed`): the question is the only source
of the entity's spelling, so an emitted seed the question does not contain is replaced by the question's
closest n-gram (difflib ≥ 0.85, a unique best, at least two words or six characters), on record as
`snapped`, and the gate runs on the snapped plan as on any other — a wrong snap is a plan the gate
refuses. Pinned. With it: **599 plans, 511 verified, 511 correct, 0 wrong**, 82 coverage refusals
(`_gen2_snap`). The 82 that remain are the second thing — the relation words — which is what the 12,000
gen-3 training questions teach, and what the A/B measures: gen 3 masked and gen 3 full against this 511
at 0 wrong, on the same 600.

## The wordings a model writes (`standin/data/gen3_llm_wordings.py`) — dataset phase, $0.18

The deterministic table has 50 shapes; SimpleQA has appositives, inversions and descriptors the table
lacks. Under the rule that an LLM's only job is the dataset, a frontier model (Gemini 3.8 Flash, effort
low, cached) was given 600 certified one-hop facts — subject, relation, value, and the encyclopedia's
sentence where there was one — and asked for three natural questions each whose only answer is the
value, naming the subject, varying the shape. It writes **questions only**; the host then does what it
does at serve time — the seed is the fact's subject, the relation words are the wording's own n-gram the
property table resolves to the fact's relation — and the gate certifies. **1,797 wordings, 1,098
certified (61.1%)**: encyclopedia facts 927/1,392, wiki facts 171/405. The certified ones are the shapes
the table has none of: `In what town or city was Judson Harmon, the Ohio governor, born?`, `Madame de
Stael was born in which year?`, `What was the birthplace of the dramatist Victorien Sardou?`, `What is
the birth year of Madame de Stael?` (`birth year`, a P569 alias). The refusals: 364 for coverage — the
model adds descriptors the store reads as relations (`primary church employer during his career in
gospel music`, `legendary radio personality credited as an author`), which is the residual rule doing its
job on a question that asks more than the fact states; 297 with no relation wording the table knows
(`passed away`, `employed the musician`, `work as`); 32 `ambiguous_relation` (`In what Orkney parish was
X born?` — `parish` was not a place class, and `which Bavarian town` had an adjective between the ask and
its noun: both fixed in `_ASK_PLACE`, pinned). And one **wrong answer the gate caught**: `Where was John
Joseph Sirica when he died?` answered `1992` — `when` inside the question read as the ask; now the
leading interrogative decides (`where` first), pinned, and the answer-type check refuses the year.
The 1,912 records (a plan and a chain per certified wording, minus 284 whose text a table wording
already produced) join `emitter_sft_v13e` whole, by their chains' split: **48,225 records, 40,560 train
rows**. `gen3_llm_wordings.{jsonl,manifest.json}` (off-repo; manifest in `validation/logs/`).

`exp_r17_gen3_heldout_gen2.*`, `exp_r17_gen3_heldout_gen2_snap.*`;
`standin/data/out/gen3_free_text.{jsonl,manifest.json}`, `emitter_sft_v13e.{jsonl,manifest.json}` (off-repo;
the manifests are copied to `validation/logs/`).
