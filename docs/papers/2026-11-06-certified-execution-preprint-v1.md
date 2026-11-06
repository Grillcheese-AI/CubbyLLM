---
title: "From probabilistic Chain-of-Thoughts to Certified Execution"
subtitle: "A language model that proposes programs, and a virtual machine that decides what may be said"
author: "Nicolas Cloutier, GrillCheese Research Lab (Lévis, Québec)"
date: "Preprint v1, November 2026. Results are partial; this version will be updated with the full evaluation."
---

# Abstract

Hallucination is a recognized, widespread problem with language models. Hallucinations are hard to detect and
prevent in the current opaque reasoning architectures. We present a new architecture in which the model only
proposes: it generates CubeLang programs. CubeLang is a VSA (vector symbolic architecture) language we designed in
Rust so the model can carry out tasks in a safer way. A virtual machine (VM) marks each CubeLang program as
"certified" or "not certified", and the model only speaks if the VM certifies the program. Note that "certified" is
not the same as "correct". A certified answer follows from the facts and rules it used. A program can run cleanly
with the wrong relation word, so if a fact or a rule is wrong, or the gold disagrees, a certified answer can still
be wrong. On CLUTRR, where the programs come from a rule library learned overnight with no language model, the
system answers 827 of 1,146 questions correctly and refuses 319, with 0 wrong. That
includes 746 of 1,003 at depths of 4 to 10 hops, which no training example reaches. In a null control, where the
evidence for each answer is relabelled, it speaks on 0 of 1,146. On the other hand, real Wikidata families are
harder: 8 of the 70 answers it gave were wrong. We are actively working on bringing Wikidata families to 0 wrong.
This preprint describes the architecture, the results measured so far, and the evaluation protocol we fixed before
running the full evaluation.

# 1. Introduction

A language model writes its reasoning as text, one token at a time, and every token is a probabilistic guess.
When a chain of thought is wrong, it looks exactly like a right one. Nothing outside the model checks it before the
answer is given, so the error is silent.

Our approach separates proposing from deciding. The language model never states a fact. It writes a short program
in CubeLang, a language for a vector symbolic architecture (VSA). A virtual machine runs the program against a fact
store and returns a verdict. The answer is spoken only when the VM certifies the program. Otherwise the system
refuses, and it says why.

This changes what an error looks like. A wrong certified answer always has a cause we can point at: a wrong fact
in the store, a wrong rule, or a gold answer that disagrees with the graph. A refusal says what was missing. Both
can be fixed, and fixing them does not require retraining the model.

Our contributions:

- **An architecture where the model proposes and never judges.** The VM is the only truth gate, and no model judges
  another model anywhere in the loop (Section 3).
- **A definition of certification, and of its limits** (Section 4).
- **Results measured so far,** with who proposed each answer: a learned rule library on CLUTRR (0 wrong on two
  splits), a reasoning pipeline on a 552,297-fact Wikidata world, SimpleQA through the gate, and a failure on real
  Wikidata families that we report as it is (Section 5).
- **A pre-registered protocol** for the full evaluation, with its pass and kill lines written before the run
  (Section 6).

# 2. Related Work

**Reasoning in natural language.** Frontier models score highly on reasoning benchmarks. However, their reasoning
cannot be checked, so a wrong chain looks exactly like a right one. This leads to numerous problems and silent
errors that are nearly impossible to debug, because the process is opaque. For example, DeepSeek-R1-Zero uses a
large mixture-of-experts model and pure reinforcement learning (GRPO) to generate long, natural-language
chain-of-thought traces [1, 2]. However, the tokens are still probabilistic guesses: a single unverified arithmetic
or entity slip in the middle of the chain poisons the final answer silently.

**Memory by lookup.** Qwen3.8-Flash-Next, an experimental preview, gets closer to a solution by offloading a
51B-parameter table of bigram and trigram embeddings to host RAM, read by deterministic addressing [3]. The main
problem is that exact addressing is brittle: a single synonym or paraphrase misses the entry completely and returns
nothing. We measured an alternative in the hypervector space, where a reworded n-gram still reaches its memory row
(Section 5.6).

**Frontier assistants.** The most widely used models, Claude Opus 5.5 and GPT-6 [4, 5], answer from an opaque
context window and from factual knowledge stored in their weights. This is expensive and exposed to prompt
injection and silent hallucination. Nothing checks the answer before it is given, and behaviour changes with
updates the user does not control.

**Programs as reasoning.** Program-aided language models [6] and Program of Thoughts [7] let a model write code
that an interpreter runs. Logic-LM [8] and LINC [9] translate a problem into logic and let a symbolic solver decide.
These are the closest in spirit: the model proposes and an exact engine computes. Two things differ here. First,
those systems reason over premises given in the question, while CubbyLLM answers from a fact store and refuses when
the store cannot support an answer. Second, the interpreter in those systems returns a value; our VM returns a
verdict, and nothing is spoken without it.

**Question answering over Wikidata.** GRASP won the WikiKGQA 2026 challenge, and GRISP lets a small model write a
query skeleton whose entities are filled in by search [10]. The challenge's dataset, QAWiki, is a community
resource of hand-crafted question and query pairs [11]. Walter and Bast also released a benchmark built from real
Wikidata query logs [12]. These systems are scored by F1 on the query's results. We report right, wrong and refused
instead, and we plan a matched comparison on the challenge's own Wikidata snapshot (Section 6).

**Knowledge-graph platforms.** Several 2026 open-source systems converge on a similar store: facts with two
timelines, ontology rules run forward, and provenance for each fact. TrustGraph builds its graph with an LLM and
answers with an LLM over graph context [13]. Semantica adds deterministic reasoning, two timelines and W3C PROV-O
lineage, and settles conflicting sources by voting or source credibility [14]. Utopia compiles ontology axioms into
forward-chained rules over a graph with two timelines, but its answers are written by an LLM with citations [15].
PULSE-S, from the same group, is a deterministic language for executable knowledge graphs, with Lean 4 proofs of
some of its properties [16]; its repository shows no question answering from natural language. As far as their
public pages show, none of these systems refuses with a reason or publishes a measured wrong rate. A citation tells
the reader where to look. A certificate lets the reader check the answer.

**Abstention.** SimpleQA already grades answers as correct, incorrect or not attempted [17]. We use the same three
outcomes everywhere, and we never report accuracy alone.

**Vector symbolic architectures.** CubeLang builds on holographic reduced representations [18], hyperdimensional
computing [19] and the VSA tradition [20]. Our own prior work used VSA transitions for a mixture of world models
(MoWM) [21] and learnable reasoning rules on hypervectors with hidden Markov models [22].

# 3. Architecture

CubbyLLM is not a small language model. It is an infrastructure of eight layers in which the language model is
deliberately the smallest and most replaceable layer:

1. **Compute:** `grilly`, a Vulkan compute framework (any GPU, no CUDA).
2. **Representation:** one VSA algebra with sparse block codes. Text is encoded with no neural model in the path.
3. **The VM:** CubeLang, written in Rust, deny-by-default, verify-before-execute.
4. **The symbolic boundary:** only a symbol and its similarity cross it, never raw hypervectors.
5. **Worlds and knowledge:** MoWM [21], and a fact store with a triple index as the serving form.
6. **Reasoning:** the certified chain-of-thought pipeline.
7. **The host:** routing, and every spoken word exits through a VM-mediated ASK.
8. **The trunk:** the language model, which proposes programs and speaks, and never judges.

Every spoken answer is a program the VM ran against a fact store, or a refusal with a reason.

## 3.1 Seven invariants

1. The VM is the only truth gate. No LLM judges anything in the loop, and the model never scores its own thoughts.
2. The model proposes and the host disposes. Identity, provenance, execution, reward, retirement, and what the model
   is shown are host decisions.
3. The don't-know contract: an unverified claim is never spoken.
4. Retire, never delete. Every record carries a hash of the store's snapshot and a git revision.
5. A serving budget: a model on a 12 GB GPU, VM calls in milliseconds, and training rounds of tens of minutes on one
   80 GB GPU.
6. No self-play or self-judging shortcuts.
7. Ported, never linked: nothing depends on another repository at runtime.

## 3.2 The pipeline

A question goes through these steps:

1. A proposer writes a plan: a seed entity and up to *k* hops.
2. A disposer checks the plan before anything runs: are the relations known, and does the plan cover the question?
3. A lookup-first walk finds the facts in the triple index.
4. The plan becomes a CubeLang program, which the VM runs.
5. The VM returns an answer or a refusal.

When a refusal is caused by the store (a missing fact), the host may consult a source. A new fact enters the store
only through a gate, with its provenance, and the walk runs again.

There are several proposers, and each answer records which one proposed it:

- the host's grammar;
- an everyday table of common question shapes;
- a rule library learned overnight (Section 5.1);
- an episodic memory of certified chains;
- the language model's program emitter.

The proposer never matters for safety, because all of them go through the same VM.

## 3.3 The models

The trunk is base450m, GRL's own model: a 450M-parameter hybrid (MinGRU recurrence with windowed attention, hidden
size 1024, 32 layers) pretrained on about 4 billion tokens and served natively on Vulkan. The program emitter is a
LoRA adapter on it. Before base450m existed, a third-party model (LFM2.5-2.6B) sat behind the same interface, and
several of the results in Section 5 were measured with that stand-in in the trunk's seat. We say so on each result.

## 3.4 Facts proposed by a model are never spoken

A model may still be used as a source of candidate facts, but only in a latent tier. Its facts are stored with
their provenance and are never spoken until a second, independent source attests them. Section 5.5 shows why this
matters.

# 4. What "certified" means

A CubeLang program binds the facts it uses into a VSA frame. The VM unbinds each hop and recovers the fact. A hop
counts only when its recovered similarity is at or above a floor set by the frame's size, while a control role,
one that was never bound, stays below that floor. The verdict covers the whole chain. A resident VM process checks
a three-hop answer in about 2 ms.

A certificate states that the answer follows from the facts and rules it used. It does not state that those facts
are true. A program can run cleanly with the wrong relation word, and a certified answer can still be wrong in
three ways:

- a fact in the store is wrong;
- a rule is wrong, or applies more widely than it should;
- the gold answer disagrees with the graph, or the graph holds a second true answer that the gold does not list.

This is why we always report wrong answers, and why we read every wrong answer the system speaks and classify its
cause. A wrong certified answer is a bug with an address. A wrong answer from an opaque model has no address.

# 5. Results so far

Every number below comes from a logged experiment. Each result says who proposed the answers. Most of the results so
far were measured with no language model in the proposer's seat, or with the 2.6B stand-in; this is the main gap
the full evaluation closes (Section 6).

## 5.1 CLUTRR: rules learned overnight, applied through the VM

CLUTRR asks how two people in a story are related [23]. Each story's relations are given as facts. A rule library
learns composition rules (for example, a father's brother is an uncle) overnight, from one day of refused questions
whose gold relation is known. A rule is adopted only when it re-derives every past episode with zero
counterexamples. Each application of a rule is certified in the VM with its facts. No language model is involved
in this result.

| split `gen_train23_test2to10` (test, 1,146) | result |
|---|---:|
| rules adopted | 109, in 3 rounds |
| correct / refused / wrong | **827 / 319 / 0** |
| at 4 to 10 hops (no training record reaches them) | **746 of 1,003, 0 wrong** |
| null control (the edges into the answer relabelled) | 0 of 1,146 spoken |
| shuffled library (a test that the measure can see wrong answers) | 29 wrong of 150 |

The controls matter as much as the headline. The null control shows that the system does not speak when the
evidence is not there. The shuffled library shows that the measure can catch wrong answers when the rules are
wrong; 0 wrong is not the result of a test that cannot fail.

The zero-counterexample gate rejected 8 rules that the 2-hop records alone support. For example, (son,
grandfather) is *father* 156 times and *father-in-law* 16 times, which is a real ambiguity that only the 3-hop
records show.

206 of the 319 refusals are true answers recovered just under the served threshold (0.447 to 0.473, against
0.4736). At a diagnostic threshold of 0.2, the system reads 1,033 of 1,146 correct, still with 0 wrong. We did not
adopt that threshold: recalibrating is a decision we take on a separate set, not on the test.

| second split `gen_train234_test2to10` (test, 1,048) | result |
|---|---:|
| the gated library | **552 correct, 0 wrong** (469 of 903 at 4 hops or more) |
| ablation: 2-hop records only | 727 correct, **3 wrong** |

The ablation's 3 wrong answers (*mother* for *mother-in-law*) come through a rule the whole-record gate keeps out.
That is the cost and the value of zero counterexamples: fewer answers, and no wrong ones.

## 5.2 Real Wikidata families: killed as built

We ran the same method on real kinship from Wikidata. Wikidata's own kinship statements are the gold, the family
graph is gendered with Wikidata's declared inverses, and there are 2,050 training and 486 test pairs, split by
subject.

| real families | spoken | wrong |
|---|---:|---:|
| the gated library (255 rules, 109 inverse rules) | 70 | **8** |
| reversed questions | 34 | 6 |
| the inverse arm | 22 | 4 |
| null control | 0 breaches | |
| shuffled library | | 18 |

Our pre-registered kill line was any wrong relation spoken, so this result is killed as built. Most of the wrong
answers have the same cause: two true relations against one gold answer. Examples are a son-in-law who is also an
adopted son, or a brother who is also a twin. The class hierarchy says what entails what, not what excludes what.
Some golds contradict the graph. One is a real induction failure: the inverse of *niece*, for a woman, was learned
as *father's sister* from brothers' daughters only. The next step is a model of contradiction (generation, line and
sex per kinship term) rather than more rules. Exact checks of asymmetry and disjointness, from an OWL 2 RL reasoner,
are one part of it.

## 5.3 The reasoning pipeline on curated and Wikidata worlds

These results were measured with the 2.6B stand-in as the emitter.

| an 800-question evaluation (1,241 facts, 165 relations) | result |
|---|---:|
| certified coverage | **574 of 800 (0.718) at precision 1.000** |
| refusals with a named reason | 226: 152 missing edge or plan coverage, 28 no seed fact, 9 ambiguous (candidates named), 37 unparseable |
| misparsed plans stopped before any walk | 90 of 92 |
| time for a certified 3-hop answer, resident VM | 2.0 ms |

| a Wikidata world (552,297 facts, 2,967 relations; the emitter has seen 165) | result |
|---|---:|
| 300 canonical two-hop chains | 300 of 300 certified |
| 200 chains in 3 surface forms outside the grammar | grammar 0 correct; emitter **228 correct, 0 wrong** |
| 600 free-text questions about held-out entities (no training record names them) | **531 correct, 0 wrong** |
| an episodic memory of certified chains, 300 out-of-grammar questions on chains it has seen | 272 correct, 0 wrong |

Every wrong answer this pipeline ever spoke on the 800-question set was a pick between several true facts. An
ambiguous hop is now a refusal.

On a third world (Wiki5M), we asked 1-, 2- and 3-hop questions in their template wording and in an LLM's
rephrasing, with each chain's facts in the store. Across 17,415 scored questions, the pipeline spoke **0 wrong**
answers. The rephrasings are evaluation data only; no external model writes training records.

## 5.4 SimpleQA through the gate

On all 4,326 SimpleQA questions, against a store that holds the answers to only 3 of them, the system speaks 0
answers and makes 0 wrong. 3,858 questions become plans, 3,825 are refused with a reason, and 33 walks fail. This is
the intended behaviour: free text against a store that cannot answer it is refused, not guessed.

With Wikidata as a source and 600 SimpleQA questions, it certifies 6 answers: 5 correct, 1 counted as near the
gold, and 0 wrong, with no model API called. We also put a frontier model
(Gemini 3.8 Flash) in the emitter's seat, behind the same disposer, walk and VM, as a ceiling probe. It certified 4
(3 correct, 0 false facts). The ceiling on free text is not the proposer. It is the store, which the full knowledge
base addresses (Section 6).

The everyday table answers 65 of 187 WebQuestions questions with 0 wrong, with no model in the proposer's seat.

## 5.5 A model as a fact source

On the same 600 SimpleQA questions, we used the 2.6B stand-in as a source of facts in the latent tier (Section
3.4). It proposed 450 facts. The system held 7 answers from them, and **all 7 were wrong**. Against Wikidata's
facts, the model's facts agreed once and disagreed 28 times (dates of birth: 0 of 22). A model's parametric facts on
this long tail are nearly always wrong. The latent tier is what made that safe to measure: nothing it said was
spoken.

## 5.6 GRL's own model

**Arithmetic through the VM, one step at a time.** The emitter never writes the whole program. Each turn it proposes
one operation on numbers the host placed, the VM runs it, and the host writes the value back. Every intermediate
number is the VM's, never the model's.

| 236 held-out word problems (base450m with the emitter adapter) | result |
|---|---:|
| whole program in one pass, then one certified step at a time | 5.5%, then **22.9%** correct (the 2.6B stand-in in one pass: 85.2%) |
| templates seen in training / never seen | 24.6% / 20.4% |
| speak only when base450m and the 2.6B stand-in agree | **0.8% of questions answered wrong**, precision 0.964 |

The 450M model is weak as a proposer, and we say so. The point of this result is that a weak proposer can be made
safe: two independent proposers that must agree, and a VM that computes every number.

**An n-gram memory in the hypervector space.** An n-gram is bound from word codes (60,147 words, 10,240
dimensions), position by position, and recalled by a Hamming scan. On 597k corpus trigrams with one word swapped:

| word codes | a paraphrase reaches the row | a different fact (*france* for *germany*) | a random word |
|---|---:|---:|---:|
| the shipped table | 22.3% | 0% | 0% |
| a paraphrase model's codes | 74.7% | 10.7% | 0% |
| the same, refined with WordNet (test words held out) | **74.3%** | **4.0%** | 0% |

An exact n-gram address reaches nothing on a paraphrase. One distance cutoff sets the trade-off against reading a
different fact. The paraphrase list is small (35 hand pairs), and the language-model test has not run yet.

## 5.7 What isolation costs

One resident CubeLang process per user session means no user's frame, knowledge path or program can reach another's.
With 256 sessions on one machine, each user costs **1.69 MB** of marginal memory, a spawn and first call take 19 ms,
and the VM runs 5,211 programs per second from 8 threads (p99 2.77 ms). Isolation is not the expensive part of
serving; the shared language model is.

# 6. The full evaluation, pre-registered

The results above are scattered, and most have no language model writing programs. The full evaluation reports
everything in one run, with each answer's proposer. We wrote its protocol and its lines before running it.

**The knowledge base.** The full evaluation runs on a store built from the complete Wikidata dump released with the
WikiKGQA 2026 challenge (121.3 GB, CC0), so every benchmark runs on one fixed snapshot. The current results are
limited by coverage: the store holds the answer to very few of the free-text questions, so the system mostly refuses.

**The benchmarks** (full test splits):

- SimpleQA (4,326) [17];
- Natural Questions, open (3,610) [24];
- WebQuestions (2,032) [25];
- CLUTRR (1,146) [23];
- the public split of AA-Omniscience (600) [26].

**The arms:**

- **C:** the full stack with every proposer, the VM certifying. C speaks only what is certified.
- **C without the VM:** the same proposals, spoken without certification. This is read from C's own records, with no
  second pass.
- **M:** the 450M model alone, closed book, always answering.
- **F:** frontier models' own published numbers, labelled as other models, other graders and other settings.

**The grading** uses no judge model. A spoken answer that matches a gold answer or an alias after normalization is
right. Every unmatched answer that C speaks is read by hand, with its program and its certification trace.

**What is reported:**

- right, wrong and refused, per benchmark and arm;
- the wrong rate among spoken answers, with a 95% interval;
- refusals by reason (no plan, a relation the store lacks, a missing fact, a failed walk, not certified);
- the answers by proposer.

**The line:** C's wrong answers are at most 1% of its spoken answers on every benchmark, and fewer than C without the
VM. **The kill:** C above 1% on a benchmark. The claim then narrows to the benchmarks where it holds, with the cause.
The stack is frozen for the run, nothing is tuned on these test sets, and every number is written once.

**A matched comparison.** QAWiki's test (150 questions in English and Spanish) comes with every challenge entrant's
submitted answers [11]. Built on the challenge's own Wikidata snapshot, Cubby answers the same questions, and we
score every entrant's answers by the same rule as Cubby's: right, wrong and refused. The challenge's own F1 is
reported beside it.

# 7. Limitations

- **Coverage.** Without the full knowledge base, the system refuses most free-text questions. A refusal is honest,
  but it is not an answer.
- **Who proposed.** Most results so far were measured with no language model, or with a third-party stand-in. The
  results of GRL's own 450M model as a proposer are modest.
- **Certified is not correct.** A wrong fact in the store gives a wrong answer with a valid certificate. We report
  every wrong answer, but the store's quality bounds the system's.
- **Real kinship is not solved.** The method that gives 0 wrong on CLUTRR gives 8 wrong of 70 on real Wikidata
  families.
- **The threshold.** The similarity threshold was set on earlier experiments. A lower one would answer more on
  CLUTRR; we have not adopted it, and its choice must be made on a separate set.
- **Languages.** The system serves English and French. Spanish in the matched comparison will be reported as
  Spanish.

# 8. Conclusion and next steps

The model proposes, the VM certifies, and the system speaks only what is certified. On CLUTRR this gives 827
correct answers of 1,146 and 0 wrong, including depths no training example reaches, while the controls show that
the measure can see wrong answers. On real Wikidata families it does not hold yet, and we have shown why.

The next version of this paper adds the full evaluation of Section 6. The work after that:

- **Time in the store:** every fact keeps when it held and when the store learned it, so any past answer can be
  replayed exactly.
- **Questions about periods of time,** refused when the dates are too coarse to decide.
- **Certificates in W3C PROV-O,** so standard tools can read the proof of an answer.
- **Domain worlds:** finance (FIBO with GLEIF's records) and automotive, added as data, not as training.
- **Mutation tests:** small faults planted in the VM and the rules, each of which must be caught.

# References

1. DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). arXiv:2402.03300, 2024.
3. Qwen Team. Qwen3.8-Flash-Next, an experimental preview. August 2026.
4. Anthropic. Claude Opus 5.5 System Card. September 2026.
5. OpenAI. GPT-6. September 2026.
6. Gao, L. et al. PAL: Program-aided Language Models. ICML 2023.
7. Chen, W. et al. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. TMLR 2023.
8. Pan, L. et al. Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning. Findings of EMNLP 2023.
9. Olausson, T. et al. LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers. EMNLP 2023.
10. Walter, S. and Bast, H. GRISP. arXiv:2604.21133, 2026. GRASP: winner of the WikiKGQA 2026 challenge.
11. QAWiki v2026.3, the WikiKGQA 2026 dataset. Zenodo, doi:10.5281/zenodo.22661413. CC0.
12. Walter, S. and Bast, H. A benchmark from Wikidata query logs. SIGIR 2026. arXiv:2602.14594.
13. TrustGraph. github.com/trustgraph-ai/trustgraph.
14. Semantica. github.com/semantica-agi/semantica.
15. Utopia, DeepLethe. github.com/deeplethe/utopia.
16. PULSE-S, DeepLethe. github.com/deeplethe/pulse-spatial.
17. Wei, J. et al. Measuring short-form factuality in large language models (SimpleQA). arXiv:2411.04368, 2024.
18. Plate, T. A. Holographic Reduced Representations. IEEE Transactions on Neural Networks 6(3), 1995.
19. Kanerva, P. Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. Cognitive Computation 1(2), 2009.
20. Gayler, R. W. Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience. ICCS/ASCS 2003.
21. Cloutier, N. MoWM: Mixture of World Models via Hypernetwork-Generated VSA Transitions. Zenodo, March 2026.
22. Cloutier, N. Learnable Reasoning Rules for Neuro-Vector-Symbolic Architectures via Hidden Markov Models. Zenodo, March 2026.
23. Sinha, K. et al. CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text. EMNLP 2019.
24. Kwiatkowski, T. et al. Natural Questions: A Benchmark for Question Answering Research. TACL 2019.
25. Berant, J. et al. Semantic Parsing on Freebase from Question-Answer Pairs. EMNLP 2013.
26. Artificial Analysis. AA-Omniscience, public split. Hugging Face: ArtificialAnalysis/AA-Omniscience-Public.
