# THE NEUTRAL PRIOR — a research-design competition
GrillCheese Research Lab · 2026-09-14 · every number below is from a captured log in `validation/logs/`

## 0. RULES
Tags: `[M]` measured, log cited · `[P]` projected · `[ASSUM]` assumption with basis · `[NM]` not measured.
**A fabricated number, result or citation costs −5. A design that violates an invariant scores 0, however
good it is otherwise.** If you need a number that is not here, derive it and show the arithmetic, or write
"no source — assumption" with the basis. Length is not a virtue; a mechanism specified precisely enough to
implement beats an essay about tradeoffs.

## 1. THE QUESTION, IN THE FOUNDER'S WORDS
> "When ambiguous it should either select **A** based on context or **B** ask the user: do you want to know
> more about: *choices*."
>
> "We should elaborate an algorithm that allows **the VM to still be neutral while scoring most requested
> answers**, this way we will eventually be able to **guess the context if none**."

Design that algorithm. The hard part is the word *neutral*: this system's entire claim is that a spoken
answer is one a verifier certified, not one a ranker preferred. A popularity prior is exactly the kind of
thing that quietly turns a verifier into a recommender. Specify how a prior can inform the system without
ever touching what makes an answer true.

## 2. THE SYSTEM, AS BUILT
A question is answered by a **program that a separate machine executes and checks**, never by generated prose.

    question → emitter (a small local LM) writes a CubeLang program: bind frame, SEED/HOPk, "..."
             → host disposes: does the plan COVER the question? (refuse if not)
             → walk the fact store hop by hop → chain program → Rust VM: BIND_ROLE / UNBIND + cosine cleanup
             → per hop: recovered symbol == the walked object AND similarity ≥ τ_vm, plus an ABSENT_ROLE control
             → verified answer + auditable trace   OR   honest refusal + partial trace
             → every outcome harvested as supervision

Seven invariants. The ones that bind this question:
1. **The model proposes, the host disposes, the VM decides.** No answer is spoken that a VM did not certify.
2. **A refusal is a result.** Refusing with a reason is always available and always acceptable.
3. **Provenance per fact.** Every stored fact carries its source; every spoken answer carries its chain.
4. **The kill line: 0 wrong answers spoken.** A "near" answer is not correct. A wrong answer is a defect that
   becomes a pinned regression test the same day.
5. The serving path calls **no external LLM**. Sources may be fetched; language is never outsourced.

## 3. WHERE AMBIGUITY ACTUALLY ARISES (all `[M]`, today)
A question names an entity by a string. The source (Wikidata) must turn that string into ONE item.

* `2026-09-12`: the proposer's seed `james young` matched the first of several items labelled *James Young*;
  a wrong birth year was verified and **spoken**. `[M]` That is the only class of wrong answer this system
  has produced from ambiguity, and the rule since is: **a label shared by two or more items is refused,
  never picked by rank.**
* That rule then refused most famous people. `[M]` *Marie Curie* is a physicist, a 2022 book edition, a
  Montreal metro station and a ferry. *Jim Haslam* is a man's label and his son's alias.
* Four deterministic tiers now resolve it, none of them a rank `[M]`:
  1. **linked** — the entity is the object of a fact this source already served, so the item is the one that
     claim pointed at (keyed by the fact, never by the name);
  2. exactly one exact hit (label or alias equal to the query);
  3. **the question decides** — among several exact hits, those whose claims carry the property the walk
     needs next (a book edition has no date of birth);
  4. label over alias (*Jim* over *Jimmy*); Wikimedia meta-pages (disambiguation, list, category) dropped first.
* What survives all four is genuinely ambiguous, and the system now **asks the asker** (B): the record
  carries `clarify: {question, choices:[{label, detail, item}]}` and the answer re-asks with the chosen item.
* Today's first cut of the prior `[M]`: a host-side tally of the items askers picked, consulted only when no
  tier decided, only when it has a **unique maximum**, recorded on the answer as `how: asker history (n of m)`,
  never written or read by the benches. The VM is untouched by it.

Scale, for calibration `[M]`: 552,297 facts in the wiki world; 2,794 distinct relations; a fetch admits up to
~180 facts about one entity; SimpleQA 600 → 177 questions reach a fetch; 6 verified, 5 correct, 1 near,
**0 wrong**; a VM verification costs 0.7 ms and a per-user VM 1.69 MB `[M]`.

## 4. WHAT IS ALREADY DECIDED (do not re-propose)
* Asking the user when genuinely ambiguous — built.
* Using the question's own next relation to choose among items — built.
* Refusing rather than guessing when nothing decides — built, and it is the fallback the prior must beat
  *without* costing the kill line.
* Logging the asker's choice — built.

## 5. WHAT TO DESIGN
Answer all five. Be specific enough to implement.

**Q1. The neutrality boundary.** State precisely which decisions a prior may touch and which it may never
touch, as a rule someone can check in code review. Where exactly does "scoring" stop and "verifying" begin?
Is *choosing which item to fetch* already a violation, given the fetched facts are then gated and verified?
Give the test that distinguishes a legitimate use from an illegitimate one.

**Q2. The estimator.** A tally of human picks is sparse, biased by who asks, and drifts. Specify the
estimator you would actually use — its state, its update, its decision rule, and the threshold at which it
is allowed to decide rather than ask. Say what it does on one observation, on a 51/49 split, and on a sense
that was popular last year and is not now. Cheap beats clever: it runs per question on a CPU.

**Q3. Guessing the context when there is none.** The asker's tally is one signal. Name the others this
system already has and could use in the same neutral way — and for each, why it is evidence about *which
item was meant* rather than about *what is true*. Rank them. Say which you would NOT use, and why.

**Q4. Keeping it honest.** A prior that decides silently is a recommender wearing a verifier's coat. Specify
what the user and the log must see when a prior decided, and design the measurement that would catch the
prior making the system worse — including what you would hold fixed, what the control arm is, and what
number would make you switch it off.

**Q5. The failure you expect.** Name the way this goes wrong that its designers will not see coming, and the
cheapest instrument that would detect it early.

## 6. SCORING
+3 a mechanism specified precisely enough to implement, with its edge cases · +3 a neutrality rule that
survives a hostile reading · +2 an estimator that is correct on sparse and drifting data · +2 a measurement
that could actually falsify the design · +2 naming a signal or a failure the lab has not considered ·
−5 any fabricated number or citation · 0 for anything that lets an unverified answer be spoken.
