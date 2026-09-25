# 2026-09-24 — the skill library: kinship algebra learned overnight, answered through the VM

**The ask** (Nick, 24 Sep): self-evolution level 3, the skill library
([sleep cycle](2026-09-24-sleep-cycle.md), level table). His choices:

- **Kind:** both, rules first. Composition rules now; CubeLang helpers (recurring sub-programs in the emitter's
  programs) later.
- **Gate:** zero counterexamples.
- **Checking:** every rule application goes through the VM.

**What it brings to CubbyLLM:** a question no fact answers, *"how is B related to A?"*, is now answered by
composing the chain of facts between the two. CLUTRR's own question is the example. It is also level 3 of
self-evolution running: the night's episodes become rules, and tomorrow's asks use them.

## A skill

```
Rule("father", "sister", "aunt")      the SISTER of the FATHER of x is the AUNT of x   (walk order)
```

A fact store cannot hold this. It is a rule about facts. Until today the engine could walk the chain but could
not name what the chain composes to (WO-2.6).

## Where rules come from, and the gate

**Episodes.** An episode is a chain of relations and the relation it composes to, stated by a bench's gold or by
an asker who said a spoken relation was right. The loop's own answer never counts: a rule confirmed by answers
it produced itself would be the loop judging itself (invariant 6).

**Candidates.** A candidate premise (X, Y) is read off every episode that splits into two spans the library
already composes. A single relation composes to itself. So a rule that is only ever seen inside longer chains
gets learned once its neighbours are known (closure). The night runs until a round adopts nothing.

**The gate.** A candidate is adopted only if all three hold:

- every instance of the premise names the same conclusion;
- it has at least 2 instances;
- with the rule added, **every past episode** still derives its own relation or nothing. None may derive a
  different relation, and none may derive two.

**Retirement.** When a new episode is derived wrongly, the weakest rule its derivation used is retired, and this
repeats until no episode on record is wrong. Retired, never deleted: the ledger (`skills.jsonl`) keeps the
adoption line, and the retire line names the episode. A retired rule is not adopted again.

## Using it

`derive` composes a chain over **every bracketing** (CYK). The chain composes only if every bracketing that
completes gives the same relation. Two relations for one chain is a split: it is refused, and it is evidence
against a rule.

`relate` runs `derive` on every path from A to B in the store (`TripleIndex.paths`, forward facts, up to 10
hops, at most 16 paths). It speaks only if every path that composes agrees, and each of those paths certifies in
the VM.

- **A path with no rule doesn't block the others.** Composition is associative, so a missing rule is missing
  knowledge, not contrary evidence.
- **A path that composes but fails the VM does block.**

**VM check.** Each spoken derivation is certified in the VM together with its facts. Facts and rule steps are
bound in one chain program, two bindings per frame, and each is recovered and checked by `pipeline._check_chain`
— the one place a chain is certified. The VM certifies binding fidelity, as it does for facts. A rule's *truth*
comes from the gate.

## Wired

| Piece | What it does |
|---|---|
| `cubbyllm/reasoning/skills.py` | `Library` (ledger replay, `adopt` / `retire`), `derive`, `mine`, `reverify`, `certify`, `relate` |
| `TripleIndex.paths` | every simple forward path between two entities; refuses on more paths than it follows |
| `standin/ask.py` | `relation_ask` (four wordings, all read as "B is the R of A") routes before the profile ask; `AskLoop(skills=)`, `AskLoop.relation`; the record carries every path, its status and its steps |
| `sleep.py` | a `skills` phase: tonight's episodes join an append-only record (`skill_episodes.jsonl`), and the library is mined against the whole record |
| sleep routes | `no_path` → source_gaps · `no_rule` → skill_gaps · `rule_split` → skill_review · `relation_paths_overflow` → capability_gaps |
| `serve_api --skills` | reads the ledger at boot |
| `POST /ask/feedback`, `AskLoop.feedback` | an asker confirms, corrects or teaches a relation; support counts distinct askers |
| `skills_contested.jsonl`, `python -m cubbyllm.reasoning.skills` | the premises a night kept out, most one-sided first; the host's `adopt` (minority discounted, refused against the record) |
| `dashboard/control_panel.html` | right / correct it / teach, after a relation ask |

**Tests:** `tests/reasoning/test_skills.py` (17) and `standin/tests/test_ask_relation.py` (5). The second
includes *day 1 refuses, the night learns, day 2 answers a four-hop question no episode was about*.

## Measured: CLUTRR's own question (`validation/exp_r34_clutrr_relations.py`)

The setup is the path a live loop would take:

- **Day 1:** the train split (9,074 records, 2–3 hops) as refused relation asks, each with gold.
- **Night:** one sleep night.
- **Day 2:** the test split (1,146 records, 2–10 hops). Each record's own story graph is its store, and the
  real cubelang VM checks every answer.

Log: `validation/logs/exp_r34_clutrr_relations.log`.

| Depth | n | Correct | Refused | Wrong |
|---:|---:|---:|---:|---:|
| 2 | 38 | 38 | 0 | 0 |
| 3 | 105 | 43 | 62 | 0 |
| 4 | 190 | 144 | 46 | 0 |
| 5 | 174 | 138 | 36 | 0 |
| 6 | 107 | 87 | 20 | 0 |
| 7 | 144 | 106 | 38 | 0 |
| 8 | 150 | 113 | 37 | 0 |
| 9 | 119 | 81 | 38 | 0 |
| 10 | 119 | 77 | 42 | 0 |
| **all** | **1,146** | **827** | **319** | **0** |
| **≥ 4** | **1,003** | **746** | **257** | **0** |

The night adopted 109 rules in 3 rounds, in 6 s.

**Controls:**

- **`null`:** the edges into B are relabelled with a token no episode used. 1,146/1,146 refused, 0 breaches.
- **`shuffled`:** the library's conclusions are permuted, as an instrument check. It spoke 29 wrong answers on
  150 questions, so the harness sees wrongs when they happen.
- **Cost:** 21,832 VM calls over all arms, about 10 s of VM time.

Verdict: **survives the kill clause.**

### Readings

1. **The first real CLUTRR score, not a CLUTRR-derived one:** 827 correct, 319 refused, **0 wrong**. At the 4–10
   hop depths no training record reaches: 746/1,003, 0 wrong. The engine was never told a rule; the night
   learned all 109 from 2–3-hop records.

2. **The gate rejected 8 rules that the 2-hop records alone would have adopted.** Every one is a real ambiguity.
   - (son, grandfather) is stated as *father* 156 times and *father-in-law* 16 times. A grandparent through
     a child is either A's parent or A's in-law, and only the 3-hop records show both.
   - (husband, brother) → *brother* is contradicted by longer chains.

   The `pairs_only` arm (2-hop records alone) adopts all 8. It answers 11 more questions (838), refuses 27 as
   splits, and on this test set says nothing wrong — by luck of the split, not by construction. The whole-record
   gate is also why depth 3 reads 43 rather than 86: the depth-3 test chains are exactly where those ambiguous
   compositions sit.

3. **206 of the 319 refusals are the VM threshold, not the rules.**
   - They are true bindings recovered at 0.447–0.473, against the served 0.4736 for a two-binding frame.
   - That threshold is exp_r11's lowest observed true binding, not a separation bound: at two bindings the
     control floor is about 0.03 (exp_r28).
   - A **diagnostic** run at 0.2 (`exp_r34_clutrr_relations_tau_diag.log`, never a result): 1,033/1,146
     correct, 950/1,003 at ≥ 4, **still 0 wrong**, 36 wrong in `shuffled`.
   - Recalibrating the gate is Nick's call.

### The second split: train on 2–4 hops (`exp_r34_clutrr_relations_gen_train234_test2to10.log`)

| Arm | Correct | Refused | Wrong | ≥ 4 hops correct / wrong |
|---|---:|---:|---:|---:|
| `rules` (whole-record gate) | 552 | 496 | **0** | 469 / 0 |
| `pairs_only` (2-hop records alone) | 727 | 318 | **3** | 629 / 3 |
| `null` | 0 | 1,048 | 0 (0 breaches) | — |

- **The ablation spoke wrong answers.** `pairs_only` says *mother* three times where gold says *mother-in-law*. It
  adopted (husband, brother) → *brother* from the 2-hop records. The whole-record gate keeps that rule out,
  because longer chains contradict it. The harness's verdict line counts both arms, so this log reads `KILLED`.
  What it killed is the 2-hop-only gate, which is not the one that serves. The criterion stays as it was
  written; this paragraph is the reading.
- **The price of zero counterexamples is coverage.** The 2–4-hop night keeps out 16 premises. Some are real
  ambiguities, like (son, grandfather), which is *father* 247 times and *father-in-law* 14. Others look like
  label noise: (husband, father) is *father-in-law* 114 times and *father* twice. Two records keep out a rule
  that 114 support, and 395 of the 496 refusals are `no_rule`.
- **The night now shows these to the host.** It writes `skills_contested.jsonl` and lists them in its report,
  most one-sided first. Whether a minority is noise is a host decision, and nothing adopts a contested rule
  on its own.
- **The host can adopt a contested rule** with `python -m cubbyllm.reasoning.skills adopt --sleep <dir>
  --premise "husband then father" --conclusion "father in law" --reason "..."`.
  - The minority episodes are discounted on the ledger, with the reason. They stay on record and stop being
    evidence.
  - The adoption is refused if any other episode would then derive wrongly. The host overrides a minority,
    never the record.
- **What that would cost here** (a diagnostic, `exp_r34_clutrr_relations_gen_train234_host_diag.log`, never a
  result). Adopting the four premises at least 95% one-sided — (husband | wife, father | mother) →
  father- / mother-in-law — gives 560 correct and **10 wrong**, against 552 and 0.
  - All ten are CLUTRR's gold saying *father* for a wife's father, or *mother* for a husband's mother. That
    happens in stories whose graph holds a two-hop shortcut next to the four-hop proof chain.
  - By the benchmark the rule is wrong there, and by kinship the gold is. Either way the kill line is scored
    against gold, so the strict gate is what kept the record clean on a noisy benchmark. On this evidence the
    host should not adopt these four.

**The robustness splits can't be measured from this pull.** In `rob_train_{sup,irr,disc}`, the story edges
include the noise facts, but `edge_types` labels only the path. 346 of 447 test records have four edges and two
types. The noise can't be put in the store without guessing its relations, so no robustness number is posted.

## The asker teaches (live)

- **The feedback call.** `POST /ask/feedback {id, verdict?, relation?, asker?}` goes to `AskLoop.feedback`.
  Every ask record now carries its `id`. The feedback is written to the history as its own line, and the
  sleep cycle folds it into the record.
- **A stated relation becomes an episode.** When an asker states the relation for a relation ask, that ask
  becomes an episode.
- **A correction can make a defect.** A relation the asker states against a spoken one makes that answer a
  defect.
- **One asker can't put a rule in alone.** An asker's episodes carry the asker as their source, and a rule's
  support counts distinct sources (`test_one_asker_cannot_put_a_rule_in_two_can`).
- **The panel.** After a relation ask, it offers *right* and *correct it* on an answer, and *teach* on a
  refusal.

## Real families: Wikidata (`validation/wikidata_kin.py`, `validation/exp_r35_wikidata_kinship.py`)

**Why not reuse CLUTRR's rules.** A rule's guarantee reaches only as far as the episodes that gated it. In
CLUTRR a wife's son is a son; in a real family he may be a stepson. So the real world gets its own episodes.

**The episodes.**

- **Gold.** Wikidata's own "relative" statements with a "kinship to subject" qualifier (P1038 + P1039): 2,536
  pairs over 70 kinship terms.
- **Chains.** The family graph Wikidata holds around each pair: father, mother, child, sibling and spouse for
  11,376 people, 55,055 edges.
  - Each edge is named by the person's sex (a male child is a son). That is the host's six-word lexicon,
    `skills.GENDERED`, not a learned rule.
  - Each edge also gets the edge back that Wikidata itself declares (P1696: father and mother are the inverse
    of child; sibling and spouse are symmetric). `skills.family_facts` does both, and the live loop uses the
    same function.
- **Grain.** The kinship terms' own hierarchy (P279). "father's brother" is an "uncle", so the library learned
  term entailment:
  - two terms contradict only when neither entails the other;
  - a rule concludes the finest term its instances stated and enough of them confirm;
  - comparable conclusions speak the finest.
- **Split.** By subject, 2,050 train and 486 test. One night.
- **Inverse rules.** (relation, sex) → inverse, learned from pairs Wikidata states both ways and from family
  edges it states on both people. `relate_back` uses them when the store leads only from B to A.

**Result: KILLED** (`validation/logs/exp_r35_wikidata_kinship.log`). The night adopted 255 rules and 109
inverse rules. On the 486 test pairs, 255 have no path within four hops and 78 have more paths than the loop
follows.

| Arm | Spoken | Correct | Coarser | Finer | **Wrong** |
|---|---:|---:|---:|---:|---:|
| `rules` | 70 | 48 | 2 | 12 | **8** |
| `reversed` (the 337 pairs stated both ways, asked the other way) | 34 | 17 | 2 | 9 | **6** |
| `inverse` (the same questions, from the B-to-A paths through inverse rules) | 22 | 12 | 3 | 3 | **4** |
| `clutrr_transfer` (CLUTRR's library on real families, a diagnostic) | 75 | 28 | 39 | 0 | **8** |
| `null` | 0 | — | — | — | 0 breaches |
| `shuffled` (instrument check) | 18 | 0 | 0 | 0 | 18 |

**What the wrong answers are.** Most are not a rule composing wrongly:

- **Two true relations, one gold.**
  - *Wife's father* where Wikidata says *adoptive father*, and *son-in-law* for *adopted son*: the same two
    people, related both ways. A son-in-law adopted as heir is common in some families.
  - *Brother* for *twin*: Wikidata's hierarchy does not make a twin a brother, so the scorer calls it
    incomparable.
  - "Incomparable" is not "contradictory". The hierarchy can say what entails what, not what excludes what.
- **Gold that contradicts itself or the graph:** *son-in-law* where Wikidata's qualifier says *father-in-law*;
  *great-grandmother* over a three-hop mother–father–mother chain the qualifier calls *grandmother*;
  *husband's father* against *wife's father*.
- **A rule too specific for its evidence:** the inverse of *niece* for a woman came out *father's sister*,
  because every training niece was a brother's daughter. It met a sister's daughter in test. Zero
  counterexamples in training is not truth.

**Fixed on the way, and recorded because the first run's log was overwritten:**

- **The null arm was fabricating facts.** It relabelled a *direct* fact into one the store does not hold and
  counted 10 breaches from it. A stated fact is now left out of the null arm.
- **`inverse_of` borrowed the inverse of a coarser term** through the hierarchy. That spoke *ancestor* for a
  grand-nephew and for a father, and the first run's inverse arm had 8 wrong. It now takes an exact key only.

**Reading.** On a clean closed world (CLUTRR) the gate holds at zero wrong. On real Wikidata it does not, and
the reason is the scoring's model of kinship as much as the rules. Real kinship terms overlap: people are
related in more than one way, and the hierarchy knows subclasses, not exclusions. Refusals dominate either way:
333 of 486 test pairs have no usable path (255 none within four hops, 78 too many).

**The live loop.**

- **Wired:** a relation ask on an empty store now fetches both families through the source, gated and with
  provenance (`AskLoop._fetch_family`). It turns a one-way store round with inverse rules when any are adopted.
- **Not done:** the live ledger is not seeded with the Wikidata night's rules, and on this result it should
  not be.

## The second kind: CubeLang helpers (`cubbyllm/reasoning/helpers.py`, `validation/exp_r36_emitter_helpers.py`)

**Nick's choice:** emitter sub-programs first.

**What a helper is.** The emitter's arithmetic programs are straight lines of binary steps. When the first of
two steps feeds only the second, the pair is one expression, for example `(a / b) * c`. That shape recurs with
different numbers across thousands of verified programs, and a helper is the shape as a CubeLang function of its
leaves:

```
function div_mul(p0: quantity, p1: quantity, p2: quantity): quantity {
    create t : quantity; assign t = arg0; div t, arg1; mul t, arg2; return t;    # CALL binds arg0.., not names
}
...
let s1 = div_mul(48, 2, 3);        # was: create s0; assign s0 = 48; div s0, 2; create s1; assign s1 = 0; add s1, s0; mul s1, 3;
```

**The gate** is the library's own: every past episode re-verifies. A helper is adopted only if every program on
record that contains its shape, rewritten to call it, runs in the real VM to exactly the result it ran to before.
Then the whole record is rewritten with all adopted helpers at once and run again. A helper that the joint
rewrite breaks is retired, never deleted. Helpers go on the same ledger as rules, with `kind: helper` and their
source.

**Wired.** `eval_emitter_vm.run_vm` supplies the definition of any adopted helper a program calls
(`with_helpers`). A program that calls none is left unchanged. Every emitter eval and the ES loop run through
it.

**Measured** on v14e_nochain's verified arithmetic programs, with the real VM
(`validation/logs/exp_r36_emitter_helpers.log`):

| | Train (5,819) | Val (329, held out) |
|---|---:|---:|
| Programs that call a helper | 5,551 | 315 |
| Programs whose result moved | **0** | **0** |
| `solve` statements | 91,016 → 41,339 (−54.6%) | 5,258 → 2,341 (−55.5%) |
| `solve` characters | 2.79 M → 1.30 M | 160.8 k → 74.0 k |

- **Adoption.** 32 helpers were adopted, one per two-op shape: `add_of_mul`, `add_add`, `mul_add`, `div_mul`
  and so on. None were rejected, because the rewrite is exact.
- **Served path.** I took 60 val programs, kept only the helper calls and removed the definitions. Through
  `run_vm`, 60/60 ran to their own result.
- **Instrument check.** I put each of the 16 helpers whose inner op is `sub` or `div` through the same gate
  with that op's operands swapped. **All 16 were rejected**, so the gate does look.

**Reading.** On the programs a helper touches, the emitter's output would be about half the statements and half
the characters, and every rewritten program is VM-equivalent to the verified original. **Whether the emitter
gets more right is not measured.** That needs an adapter round on the rewritten set
(`standin/data/out/exp_r36/emitter_sft_arithmetic_helpers.jsonl`), and the names are shapes (`sub_of_mul`), not
meanings. It is H-A14.

## Advantage over conventional models

- **Explicit.** Each rule is a line on a ledger with the episodes behind it. One counterexample retires it,
  without retraining anything.
- **Depth for free.** It trained on 2–3 hops and answered 4–10 with zero wrong: ten relations need rules that
  each passed the gate, not a ten-relation example.
- **It refuses the guess.** A composition the record shows to be ambiguous never becomes a rule, so the
  question is refused. A model that learned "the son's grandfather is the father" from short examples would
  say it.

## Not done

- **Training on helpers:** the rewritten set exists; no adapter round has used it (H-A14).
- **A kinship model of contradiction:** generation, line and sex per term, so *twin* and *brother* stop
  counting as a contradiction and *great-grandmother* against *grandmother* keeps counting as one. Until then
  the real-world kill line is scored against a model that cannot tell the two apart.
- **Tau for two-binding frames:** a decision, above.
