# MoWM as the world model behind the bridge (H-F2 M4)

2026-09-29. Status: built, tested in both repos, screened once (`validation/exp_f2m4_possibility.py`); not
on the serve path until `CubbyServe.mount_oracle` is called. Uncommitted, for review.

## What it brings

The worlds layer of the eight. Today `standin/worlds.py` routes a text to the fact store whose best hit scores
highest ("MoWM v0") and the world model behind `bridges/world_model.py` is a fake. This puts ONE question in
front of every write to a store and every route -- *which world is this, and is it possible there?* -- and
lets MoWM answer it, or the stores themselves when MoWM is not installed.

The advantage over a conventional model's "does this sound right": the answer names the world that gave it,
the members that back it, and the margin by which that world won. A fact can be refused with the member it
clashes with (auditable, ledgered), or held as latent until a source attests it, instead of being either
trusted or lost. Nothing crosses back but symbols and scores.

## Which worlds -- one interface, three kinds

| kind | what it is | covers a text by | supports | contradicts |
|---|---|---|---|---|
| fact world | a `FactStore` / `CapsuleStore` per domain, era or source: what someone STATED | its best hit | members at or above `tau_answer`; an exact member is `why='member'` | a stored (subject, relation) with a different object -- `serve.contradiction`'s rule, now `bridges.possibility.same_subject_clash`, applied once for every writer; looked for in the routed world, then (`cross_world`) in every other world holding the subject |
| law world | a MoWM `World` with axioms (or a `Knows`): how things BEHAVE | cosine to its axiom centroid (the open-set rule, exp_m3_open_set) | `best_member`; `predict` cleaned up against members | an injected rule; the world model never learns the fact grammar |
| time world | a fact world sliced by when the fact held (`FactStore.times`; the era/year worlds of exp_m3_temporal_causal) | as a fact world | as a fact world | as a fact world |

A law world answers in knowledge, never state (the oracle line in `worlds.py`). `exclude X` in a branch is
removing X from its year-world before asking again -- the Branch's `T^t` role; not yet wired to the branch
runner.

## What a possibility branch returns

```
Possibility(world_id, verdict, score, margin, support, predicted, latent, why)
```

- `verdict`: **POSSIBLE** -- the routed world covers the text and holds it (`member`) or a member clears
  `tau_answer` (`resembles`), and nothing it holds contradicts it. **IMPOSSIBLE** -- the routed world covers
  it and a member contradicts it. **UNKNOWN** -- no world covers it (`below_tau_match`, `ambiguous_margin`),
  or the routed world neither holds nor contradicts it (`covered_unsupported`); a latent world takes it.
- `score`: the best member similarity, or the routing score when nothing matched. `margin`: best world minus
  runner-up (the route-vs-spawn signal). `support` / `predicted`: `((member, similarity), ...)`, symbols
  only. `latent`: the world the fact waits in.

`possible` means "in a world's domain and not ruled out by it", never "true"; truth still comes from a
verified walk.

## How the capsule stores read it

`CapsuleStore(oracle=...)` (or `serve.mount_oracle`, which sets it on every mounted store):

- **IMPOSSIBLE** -> `add` returns False; `last_refusal` and `ledger` carry the world, the verdict and the
  member it clashed with. `serve.learn` answers "That can't be, as far as the *geo* world knows: Paris is
  the capital of France."
- **UNKNOWN** -> kept with `meta.latent = <world>` and provenance `'<source> (latent)'`: learn.py's existing
  tier, so a chain resting on it is not spoken until a source attests it (`CapsuleStore.attest`).
- **POSSIBLE** -> kept with `meta.support`.
- After a keep, `oracle.learn(fact, world)` (when the oracle has it) makes the fact a member of that world:
  a latent world fills with what waits in it and the next such fact routes there -- spawn a latent world,
  verify later.
- `route_world(worlds, text, oracle)` asks the oracle first and falls back to best-top-score.
- `PossibleRelations(oracle)` is a `plan_verify.KnownRelations`: a relation the oracle finds UNKNOWN lands in
  `unknown_relations`, the spawn-a-latent-world branch of that verdict.

Saved stores carry the verdicts (`capsules.jsonl`) and the refusals (`ledger.jsonl`); a loaded store does not
re-judge.

## Code

CubbyLLM: `cubbyllm/bridges/possibility.py` (Possibility, Verdict, PossibilityOracle, `StorePossibility`,
`PossibleRelations`, `same_subject_clash`, `relation_members`), `standin/capsules.py` (oracle, ledger,
attest, latent_facts, persistence), `standin/worlds.py` (`route_world(..., oracle)`; the store's matrix is
now stacked once per add, not per query), `standin/serve.py` (`mount_oracle`, the learn line),
`cubbyllm/reasoning/index.py` (`TripleIndex.about`, `relations`). Tests:
`tests/bridges/test_possibility.py` (20), `standin/tests` unchanged (run with `-p no:hypothesispytest`).

mowm (mowm -> cubbyllm only): `mowm/bridges/possibility.py` (`MoWMPossibility`: fact worlds from texts,
`mount` for seeded domain worlds, `learn`, `routing='member'|'centroid'`, latent spawn through
`MoWMRouter.maybe_spawn`, `Knows` surface `covers`/`answer`), `mowm/world.py` (member matrix and centroid
cached per axiom count: a production-width fact world was a 100 MB copy per query). Tests:
`tests/test_possibility.py` (17, k=4 l=32, model-free encoder).

## Screen and gates (pre-registered)

`validation/exp_f2m4_possibility.py`: the chain-QA fact pool split into one fact world per relation family
(8 worlds), 400 true facts held out, 400 contradictions planted (object swapped within the relation), the
same worlds for every oracle. Gates: `mowm_member` routing >= the store rule - 2 pts; planted caught >= 90%;
true facts refused <= 2%; members asked back possible >= 98%; store `possible()` <= 5 ms mean. Kill: MoWM
routing loses > 2 pts or refuses > 2% of true facts -> the bridge stays the stores' own rule (membership
only) and MoWM keeps the law worlds.

### Smoke (n=300, `exp_f2m4_possibility_smoke.json`)

Store rule: routing 0.897 (parity 1.00 with `route_world`), planted caught 0.96, true refused 0.00, members
1.00, 2.0 ms. MoWM under the **centroid** rule at the open-set taus: abstained on 65% of questions and on
54% of its own members -- a member sits ~1/sqrt(n) from a centroid of n quasi-orthogonal facts, so an
absolute floor calibrated on domain passages does not transfer to a 180-fact world; 0.74 accuracy when it
did route; planted caught 0.51. That is why the oracle's default is **member** routing (best member per
world, one matmul each: MoWM's `query_worlds` in its exhaustive form and the rule `route_world` applies)
and the centroid rule is kept for worlds of few coherent axioms. World construction at production width
took 15 s/world in HYLA init; fact worlds are now built with `n_hylas=0` when no prediction is wanted.

### Full run (n=1500, 8 worlds of 43-840 facts; `exp_f2m4_possibility.json`) -- all gates PASS

| oracle | routing | abstain | planted caught | true refused | members possible | possible() ms |
|---|---|---|---|---|---|---|
| store (`route_world` rule) | 0.937 (parity 1.00) | -- | 0.990 | 0.0125 | 1.00 | 4.0 |
| mowm, member routing | 0.937 (0.972 when routed) | 0.117 | 0.990 | 0.0125 | 1.00 | 4.6 |
| mowm, centroid routing | 0.830 (0.837 when routed) | 0.682 | 0.443 | 0.0075 | 0.39 | 2.7 |

Readings. (1) MoWM under member routing reproduces the store rule verdict for verdict -- parity by
construction, which is the point: the same worlds now live in one router that also holds law worlds,
spawns latent worlds and predicts. (2) The first run checked the contradiction in the routed world only and
caught 0.918: of its 33 escapes, 30 had the same subject and relation as the original, but 29 had routed
AWAY from the fact's home world -- the swapped-in object ("Taxxon", "Magyar Köztársaság") pulls the best
member into 'misc' (840 facts), where nothing holds the subject (`exp_f2m4_possibility_run1_routed_only.json`).
Hence `cross_world`: after the routed world, every other non-latent world that holds the subject is asked
(an index lookup each, `TripleIndex.about`; the MoWM adapter keeps the same subject index) -- 29 more caught
(`contradiction_elsewhere`), 4 left: the original was stored in the 'X is the country Y is in' template and
the index split its subject differently from the planted 'of' form. Time worlds are the case to turn
`cross_world` off. (3) The 5 true facts refused (1.25%) are second values of multi-valued relations
(component of, characteristic of, time zone of, described by source of): the cost of treating every relation
as functional -- bounded, and the existing gate's choice. (4) The centroid rule is a recorded negative for
fact worlds: 68% abstain, and 61% of a world's own members sit below the floor. MoWM builds the 8
production-width worlds in 0.24 s with `n_hylas=0`.

## Not done

- The serve path does not mount an oracle by default; `CubbyServe.mount_oracle(StorePossibility(brain.worlds))`
  is the one line, and MoWM's adapter needs the v4 table loaded (1.1 GB) plus the stores' texts.
- Temporal `exclude` through a year-world and the branch runner (O3).
- Law worlds mounted from `mowm.domains` answer in `'<formula> is the formula of <name>'`; nothing asks them yet.
- Planted contradictions are within-relation object swaps; a multi-valued relation (child of) counts a true
  second value as a clash -- the held-out refusal rate is what bounds that cost.

## Science worlds from MegaScience (2026-09-29): the importer, first sample

`standin/data/import_science.py` (pinned by `standin/tests/test_import_science.py`, 5 tests) turns textbook
Q/A rows (MegaScience / TextbookReasoning, CC-BY-NC-SA-4.0: research worlds only, and every world carries the
license) into fact worlds, one per subject area, in three re-runnable stages:

- **fetch**: rows through the Hub's datasets-server rows API at seeded, spread offsets (no file download).
- **extract**: the dataset LLM proposes `(subject, relation, object)`; the HOST keeps one only when both ends
  are copied from the row (after LaTeX and symbol transliteration), the relation is in a closed vocabulary of
  17 (none contains " of ", so "moment of momentum" stays one subject under `parse_fact(known=...)`), the
  template `"<object> is the <relation> of <subject>"` round-trips, and every name is ASCII.
- **build**: every kept fact is judged by the worlds' contradiction rule (`same_subject_clash`, the rule
  `StorePossibility` applies) across all science worlds, by index lookup, so the build stays linear. A first
  statement is **latent** whatever it resembles (one row is one unverified source); a second row stating it
  **attests** it; a clash on a functional relation is **refused** into the ledger -- by default none is (see
  the 30K read below: in textbook science a second unit or symbol is a notation, not a contradiction) -- and a
  second object is otherwise a new latent fact. `serve_api --science <folder>` mounts the ATTESTED facts only.

**Sample (144 non-math rows, extractor Nemotron-3-Ultra on OpenRouter's free tier, reasoning off):** 720
candidates, 424 kept (59%); rejects: 216 paraphrased rather than copied, 45 outside the vocabulary, 19
non-ASCII, 8 bare variables, 8 too long. Built: 6 worlds (medicine 154, biology 131, chemistry 62, physics
42, cs 33, economics 2), 424 latent, 0 attested, 0 refused, 55 second objects on many-valued relations.
Nemotron-3-Super on 40 of the same rows: 49% kept, similar quality, at a seventh of Ultra's paid price.
Math rows (64% of the split) are left out: they are exercises, not world facts.

**What the sample says.** Kept facts are mostly real and general ("salicylaldehyde is the product of
Reimer-Tiemann reaction", "kJ/kg is the unit of specific latent heat of vaporisation", "condenser is the
component of Rankine cycle"); the noise is exercise-specific statements that passed the copy check ("t = l *
sin(delta + alpha) is the formula of thickness"). That noise is what the latent tier is for: an exercise's
own statement is almost never restated by another row, so it stays latent and never mounts. Two rules came
out of the run: `value` left the vocabulary (in exercises it is the exercise's number), and `formula` is
many-valued (the one refusal was "D_c = 4r" against "D_c = sqrt(3)L" for the same cube diagonal, both true).
**Nothing is attested at 144 rows**, so nothing mounts yet: attestation needs either scale (restatements
across the 235K non-math rows) or a second source (the wiki world, a curated constants table).

**What it brings to CubbyLLM:** science enters as facts with a row, a license and a status, and the VM can
only speak one after a second statement agrees. A conventional model absorbs the same data into its weights,
exercise numbers included, and cannot say which of them it is sure of.

### The 30K slice (2026-09-29, running): first read at 4,888 rows

30,000 non-math TextbookReasoning rows (medicine 10,805, biology 6,931, physics 5,504, chemistry 4,311, cs 2,267,
economics 182), fetched through the rows API with math dropped locally, extracted by Nemotron-3-Super (reasoning
off) in 24 shards under a $0.17-per-shard cap (~$0.00013 a row; the model's provider rate-limits at ~140 rows a
minute, so the slice takes ~3.5 h). Read at 4,888 rows with `extract --cached-only` (what the running shards
have already paid for), then `build`:

- 17,762 candidates, **9,252 kept (52%)**; rejects: 4,131 paraphrased, 3,339 outside the vocabulary, 451 too
  long, 370 non-ASCII, 156 bare variables.
- 6 worlds, 9,168 facts: **61 attested (0.7%)**, 9,107 latent, 1,950 second objects on a (subject, relation)
  already held.
- The attested facts are the textbook canon: "H_2O is the formula of water", "Hz is the unit of frequency",
  "G is the symbol of gravitational constant", the Arrhenius and Darcy-Weisbach equations, "vitamin B6 is the
  other name of pyridoxine", the congenital rubella syndrome triad. Restatement is what makes a fact common
  knowledge, and it is what the gate keeps.
- **No relation is functional in textbook science.** With {abbreviation, symbol, unit, discoverer} functional,
  68 facts were refused; a sample of 15 held 13 truths (unit systems: "ft*lb" vs "J" for work; notations:
  "Nm^2/kg^2" vs "N*m^2/kg^2", "s" vs "seconds"; codes: "A" vs "Ala" for alanine) and 2 real errors ("T_L is
  the unit of temperature"). String inequality is not a contradiction here: `FUNCTIONAL` is now empty by
  default (`build --functional` restores a set), a second object is a new latent fact, and the contradiction
  check waits for a law world that can decide it (dimensional analysis for units).
- Two statements from one dataset are weaker than two sources; the attested tier is "restated in the corpus",
  and a second source (the wiki world, a constants table) is the next attestation.

Finish when the shards are done (`standin/data/out/science/progress30k.txt` reaches 24 lines):
`import_science.py build --facts standin/data/out/science/facts30k_s*.jsonl --out standin/data/out/science/worlds30k`,
then `serve_api ... --science standin/data/out/science/worlds30k` mounts the attested facts.
