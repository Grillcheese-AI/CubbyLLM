# Synthesis: the qFHRR panel (2026-11-12)

Cowork's reading of `2026-11-12-qfhrr-panel.md` against its own answer, sealed and committed before the run
(`2026-11-12-qfhrr-cowork-sealed.md`). The brief: `2026-11-12-qfhrr-brief.md`. Eight members, each lab's top model on
the day (GPT-6.1 Sol Pro, Claude Fable 5.2, Gemini 4 Argon, DeepSeek V4 Pro, Qwen 3.8 Max, GLM-5.3, Grok 4.7, Kimi
K3.1 Pro), high effort, 64k tokens each: 8 answers of 8, $2.65 reported by OpenRouter.

**These are model opinions.** Nothing below is adopted until an order pre-registers it and its line is read.

## Where all eight agree
1. **Namesakes: no score chooses, ever.** qFHRR, BM25 or any margin stays out of the choice; B1q (one role per word,
   a cue checked against each candidate's statements, refusal otherwise) stays the chooser. Its two known faults are
   fixed outside the table: the French relation word "né" (and its inflections) before any accent folding, and the
   Rosebank case, where the gold's "located in" is one level below the cue.
2. **Names are found by the exact and typo indexes, not by qFHRR.** A better text teacher will not close the BEIR gap
   (v4 gained 18% and kept the 2.3-times gap).
3. **Relations: the missing-fact handover is the worst failure in the brief.** A ranker over the item's own
   properties hands over a near property the item does carry (36 of 38), and the VM certifies the wrong answer. The
   relation step must be able to refuse. Bounding the candidates to the item's own properties is right.
4. **The asked kind is not a hard filter** as it stands: incoming "who" and item-valued "how many" must stay in.
5. **Growing the model is not the lever** for names, namesakes, routing, known-corpus retrieval, the n-gram memory or
   reading placed facts. It is plausible only for reading relation wordings (French above all), and only after the
   cheap levers fail.
6. **Loci tokens read; they never find or choose.** Same name, same key: the host must choose (or refuse) before it
   writes a scene, and write only that item's facts. A relation's key should be the canonical property the host
   resolved, not the question's words.

## Where they split
- **Does qFHRR keep a place in relation choice?** Five keep it as a proposer among the item's own properties behind
  a missing-fact guard (Fable, Gemini, Qwen, GLM, Kimi; also Cowork). Three take it off the picker: the grammar,
  then the lexicon, else refuse (Grok); a property must have label or alias support in the question (DeepSeek); the
  property asked is established before availability is looked at (GPT).
- **How absence is detected.** The best property overall against the item's best (Gemini; Cowork's absence test);
  properties that items of its class usually carry and this one lacks, counted from the store (Fable); independent
  lexical support (DeepSeek, Kimi, GPT); a "none" answer for every proposer (GLM).
- **"Located in."** Seven would let a bounded, VM-checked chain make a candidate hold the cue. Grok would use the hop
  only as a veto: it can turn a pick into undecided, never create a pick. The veto is the safer first step.
- **French.** A multilingual teacher (Gemini, Kimi, GLM); French wordings from the store's labels into the lexicon
  only (Grok, GPT, DeepSeek); in steps, stopping at the first that passes: a stemmer, French words tied to English
  codes through paired labels, then a multilingual teacher (Fable; close to Cowork's alignment).
- **Size, if ever.** Fable: one doubling of the trunk, about 0.9B (estimate). Gemini: 1.2 to 1.5B by widening.
  Qwen: 1 to 3B. Kimi: a 1B pilot. GPT, DeepSeek, GLM, Grok: no size can be read from the evidence.
- **Open-web entity search.** Retire qFHRR from it and use BM25 (DeepSeek, Kimi), or diagnose first (Fable).

## What the panel adds to Cowork's sealed answer
1. **An ablated missing-fact set** (Fable, Gemini): on rows where the item holds the gold, hide the gold statement
   and require a refusal. Hundreds of missing-fact rows instead of 38, so an absence guard can be set on one half and
   read once on the other. Cowork's absence test needs exactly this; without it, 38 dev rows decide nothing.
2. **QALD EN is 0.755, below the 0.80 gate set for QALD FR** (Fable). Fixing French alone cannot pass that gate.
3. **The veto form of "located in"** (Grok), above.
4. **A selection seat for the 450M** (Fable, Gemini): the item's own property list in the prompt, trained to pick
   one or answer "none of these", from store records and the question sets only. It is the trained form of Cowork's
   decode-time bound, and its "none" option attacks absence. The two combine.
5. **Diagnostics before any entity fix** (Fable): the IDF an unseen word gets, the share of query words out of
   vocabulary, and dense recall searched exhaustively. The 2.3-times gap mixes code quality with shortlist loss.
6. **Loci keys** (Fable, Gemini, GLM): a role code on subject and relation so the key stops being symmetric (the
   direction of "who was born in Paris?"); an index code for each value of a multi-valued property; the loci read
   compared with the store's value as an exact string, falling back to the store; and an unwritten address reads as
   "does not know", the opposite of qFHRR's handover.
7. **B1q at scale** (Gemini, GLM): latency at 20 or more candidates is not measured, while a fifth to a third of
   mentions name more than 20 items; intersect sorted posting lists instead of walking each candidate.
8. **Audits first** (Gemini, Fable, Grok): the second bare pick, the 61 undecided and 154 refused (adjective cues such
   as "French" for France may be a share), and the wrong rows left on Mintaka and WebQSP_EL.

Not taken up by any member: Cowork's graph codes learnt as phase rotations (RotatE, ComplEx). Nobody argued against
them either. They stay the longest lead and the last in line.

## Order candidates, in Cowork's recommended order
Each is a candidate for an order; Nick decides which are written.
1. **B1q repairs (namesakes).** The audits above, then: the EN and FR relation lexicon with its inflections, checked
   before folding; a short-token guard against acronym aliases; "located in" as a veto. Read once on fame seed 1
   (never read). Lines: 0 bare picks, 0 wrong qualified, resolved at least 0.70, at most 1 wrong in 50 on Mintaka and
   WebQSP_EL, p95 at most 50 ms, measured at 20 or more candidates too. CPU only.
2. **The missing-fact guard (relations).** Build the ablated set first (half to set the guard, half read once). Read
   the overall-against-own contrast and the class-expected guard side by side. Lines: handovers at most 4 of 38 (the
   450M's count) and at most 10% on the ablated read; SimpleQuestions recall@5 at least 0.85. CPU only.
3. **A direction-aware asked kind.** "Who" from the value's side keeps incoming properties; "how many" keeps
   item-valued ones; an empty cut refuses. Line: handovers not above 36 with the kind on; place and time keep their
   gains.
4. **French, in steps.** The store's French labels into the lexicon; then French words aligned to the English table
   through paired labels; a multilingual teacher only if both fail. Line for this order: QALD FR recall@5 at least
   0.70 at a wrong rate at or below the lexicon's (15.1%), routing at least 95% of the teacher. The 0.80 gate also
   needs QALD's English ceiling raised; recorded as its own open question.
5. **The 450M's selection seat.** The item's property list in the prompt, the same list as the allowed tokens at
   decode, "none of these" trained in. Lines: R3o's gate, handovers at most 4, wrong rate at or below the lexicon's.
   About one A100 hour: Nick's call.
6. **Loci keys, on the CPU toy first.** Canonical property keys, role codes, value-index codes; then the 450M.
7. **Growth: not now.** Only if the selection seat fails on coverage of hard questions and not on safety; then a
   doubled trunk against the 450M on the same data, with Fable's lines.

Parked: open-web entity search. Cubby finds its own items by index; the BEIR diagnostics run only if open-corpus
entity search becomes a job.
