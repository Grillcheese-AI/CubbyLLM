---
# v1 published on Zenodo, 2026-11-06: doi:10.5281/zenodo.24020073 (CC BY 4.0). Upload v2 as "New version" on that record.
title: "From probabilistic Chain-of-Thoughts to Certified Execution"
subtitle: "A language model that proposes programs, and a virtual machine that decides what may be said"
author: "Nicolas Cloutier, GrillCheese Research Lab (Lévis, Québec)"
date: "Preprint v1, November 2026. Results are partial; this version will be updated with the full evaluation."
---

# Abstract

Hallucination is a recognized, widespread problem with language models, and hallucinations are hard to detect and
prevent in the current opaque reasoning architectures. We present a new architecture where the model only proposes,
in this case by generating CubeLang programs. CubeLang is a VSA (vector symbolic architecture) language we designed
in Rust to allow the model to carry out various tasks in a safer way. CubeLang programs are marked as "certified" or
"not certified" by a virtual machine (VM), and the model only speaks if the VM certifies the program. Note that
"certified" is not the same as "correct". A program can run cleanly with the wrong relation word: a certified answer
follows from the facts and rules it used, so if one of those is wrong or the gold disagrees, a certified answer can
still be wrong. On CLUTRR, where the programs come from a rule library learned overnight with no language model, the
system gets 827 of 1,146 correct, 319 refused and 0 wrong, including 746 of 1,003 at depths of 4 to 10 hops that no
training example reaches. In a null control where the evidence for each answer is relabelled, it speaks on none of
the 1,146 questions. On the other hand, real Wikidata families are harder: 8 of the 70 answers it gave were wrong, and
we are actively working on bringing them to 0 wrong. This preprint describes the architecture, the results measured
so far and the evaluation protocol we fixed before running the full evaluation.

# 1. Introduction

A language model writes its reasoning as text, one token at a time, and every token is a probabilistic guess, so a
wrong chain of thought looks exactly like a right one. Nothing outside the model checks the chain before the answer is
given, so the error stays silent and hard to debug.

Our approach separates proposing from deciding. The language model never states a fact on its own; it writes a short
program in CubeLang, a language for a vector symbolic architecture, and a virtual machine runs that program against a
fact store and returns a verdict. The answer is spoken only when the VM certifies the program, and otherwise the
system refuses and says why.

With this design, every wrong certified answer has a traceable cause, whether a wrong fact in the store, a wrong rule,
or a gold answer contradicting the graph. A refusal, on the other hand, names the missing piece. Both get fixed in the
store or the rules, without retraining the model.

This paper makes four contributions. We describe an architecture where the model proposes and never judges, with the
VM as the only truth gate (Section 3). We also define what certification means and where it stops (Section 4). We then
report the results measured so far, with who proposed each answer (Section 5). They include a learned rule library on
CLUTRR with 0 wrong on two splits, a reasoning pipeline on a Wikidata world of 552,297 facts, SimpleQA through the
gate, and a failure on real Wikidata families that we report as it is. Finally, we give the protocol of the full
evaluation, with its pass and kill lines written before the run (Section 6).

# 2. Related Work

Frontier models score highly on reasoning benchmarks; however, their reasoning can't be checked, so a wrong chain looks
exactly like a right one. This leads to numerous problems and silent errors which are nearly impossible to debug, as
the process is opaque. For example, DeepSeek-R1-Zero uses a large mixture-of-experts model and pure reinforcement
learning (GRPO) to generate long, natural-language chain-of-thought traces [1, 2]. However, the tokens are still
probabilistic guesses, and a single unverified arithmetic or entity slip in the middle of the chain poisons the final
answer silently.

Qwen3.8-Flash-Next, an experimental preview, is closer to a solution by offloading a 51B-parameter table of bigram and
trigram embeddings to host RAM, read by deterministic addressing [3]. The main problem with this is that exact
addressing is brittle: a single synonym or paraphrase misses the entry completely and returns nothing. In Section 5.6
we measure an alternative in the hypervector space, where a reworded n-gram still reaches its memory row.

The most popular models, Claude Opus 5.5 and GPT-6 [4, 5], answer from an opaque context window and from factual
knowledge synthesized in their weights. This is expensive and prone to prompt injection and silent hallucination,
since nothing checks the answer before it's given. Their behaviour also changes with updates the user doesn't control.

Program-aided language models [6] and Program of Thoughts [7] let a model write code an interpreter runs, and Logic-LM
[8] and LINC [9] translate problems into logic for a symbolic solver. These come closest to our work in spirit, since
the model proposes and an exact engine computes. However, they reason over premises given in the question, while
CubbyLLM answers from a fact store and refuses when the store cannot support an answer, and their interpreter returns a
value where our VM returns a verdict required before speaking.

On Wikidata, GRASP won the WikiKGQA 2026 challenge, and GRISP lets a small model write a query skeleton whose entities
are then filled in by search [10]. The challenge's dataset, QAWiki, is a community resource of hand-crafted question
and query pairs [11], and Walter and Bast also released a benchmark built from real Wikidata query logs [12]. These
systems are scored with F1 on the query's results. We report right, wrong and refused instead, and we plan a matched
comparison on the challenge's own Wikidata snapshot (Section 6).

Several open-source systems released in 2026 converge on a similar store, with facts on two timelines, ontology rules
run forward and provenance for each fact. TrustGraph builds its graph with an LLM and answers with an LLM over the
graph's context [13]. Semantica adds deterministic reasoning, two timelines and W3C PROV-O lineage, and it settles
conflicting sources by voting or by source credibility [14]. Utopia compiles ontology axioms into forward-chained
rules over a graph with two timelines, but its answers are still written by an LLM with citations [15]. PULSE-S, from
the same group, is a deterministic language for executable knowledge graphs with Lean 4 proofs of some of its
properties [16]. Its repository shows no question answering from natural language. As far as their public pages show,
none of these systems refuses with a reason or publishes a measured wrong rate. The main problem with a citation is
that it only tells the reader where to look, while a certificate lets the reader check the answer.

SimpleQA already grades answers as correct, incorrect or not attempted [17], and we use the same three outcomes
everywhere in this paper, never accuracy alone. Finally, CubeLang builds on holographic reduced representations [18],
hyperdimensional computing [19] and the wider VSA tradition [20]. Our own prior work used VSA transitions for a mixture
of world models, MoWM [21], and learnable reasoning rules on hypervectors with hidden Markov models [22].

# 3. Architecture

CubbyLLM is not a small language model. It is an infrastructure of eight layers where the language model is
deliberately the smallest and most replaceable one. At the bottom, `grilly` is a Vulkan compute framework that runs
on any GPU without CUDA, and above it sits a single VSA algebra with sparse block codes, in which text is encoded with
no neural model in the path. The third layer is the VM: CubeLang, written in Rust, deny-by-default and
verify-before-execute. A symbolic boundary comes next, and only a symbol and its similarity cross it, never raw
hypervectors. Then come the worlds and the knowledge (MoWM [21], and a fact store with a triple index as its serving
form), the reasoning layer with the certified chain-of-thought pipeline, and the host, which routes requests so that
every spoken word exits through a VM-mediated ASK. The last layer is the trunk, the language model, which proposes
programs and speaks but never judges. Every spoken answer is a program the VM ran against a fact store, or a refusal
with a reason.

The system holds to seven invariants. The VM is the only truth gate, so no LLM judges anything in the loop and the
model never scores its own thoughts. The model proposes and the host disposes, which means identity, provenance,
execution, reward, retirement and what the model is shown are all decisions of the host. An unverified claim is never
spoken, and nothing is deleted: records are retired, and each one carries a hash of the store's snapshot and a git
revision. Serving fits a model on a 12 GB GPU with VM calls in milliseconds, and a training round takes tens of
minutes on one 80 GB GPU. There are no self-play or self-judging shortcuts, and nothing depends on another repository
at runtime, since code is ported rather than linked.

## 3.1 The pipeline

A question first goes to a proposer, which writes a plan: a seed entity plus up to *k* hops. Before anything runs, a
disposer checks the plan for known relations and question coverage. A lookup-first walk then finds the facts in the
triple index, the plan becomes a CubeLang program, and the VM runs it, returning an answer or a refusal. When the store
causes a refusal, for example through a missing fact, the host may consult a source, but new facts enter the store
only through a gate, with provenance, before the walk runs again.

There are several proposers in the system: the host's grammar, an everyday table of common question shapes, a rule
library learned overnight (Section 5.1), an episodic memory of certified chains, and the program emitter of the
language model. Each answer records which one proposed it. Note that the choice of proposer never matters for safety,
as all of them go through the same VM.

## 3.2 The models

The trunk is base450m, GRL's own model, a 450M-parameter hybrid of MinGRU recurrence and windowed attention (hidden
size 1024, 32 layers) pretrained on about 4 billion tokens and served natively on Vulkan, with the program emitter as
a LoRA adapter on top. Before base450m existed, a third-party model, LFM2.5-2.6B, sat behind the same interface, so
several results in Section 5 were measured with that stand-in in the trunk's seat, and we say so on each of them.

A model can still supply candidate facts, but only in a latent tier, stored with provenance and never spoken until a
second, independent source attests them. Section 5.5 shows why this matters.

# 4. What "certified" means

A CubeLang program binds the facts it uses into a VSA frame, and the VM unbinds each hop to recover the fact. A hop
only counts when its recovered similarity is at or above a floor set by the frame's size, while a control role that
was never bound stays below that floor, and the verdict covers the whole chain. A resident VM process checks a
three-hop answer in about 2 ms.

A certificate proves the answer follows from the facts and rules used, not that those facts are true. A program can
run cleanly with the wrong relation word, so certified answers still fail through a wrong fact in the store, a wrong or
overly broad rule, or a gold answer contradicting the graph or missing a second true answer the graph holds. This is
why we always report wrong answers, and why we read and classify every wrong answer the system speaks. A wrong
certified answer is a bug with an address, unlike a wrong answer from an opaque model.

# 5. Results so far

Every number in this section comes from a logged experiment, and each result names its proposer. Note that most
results here involve no language model in the proposer's seat, or the 2.6B stand-in, and closing that gap is the main
job of the full evaluation (Section 6).

## 5.1 CLUTRR: rules learned overnight, applied through the VM

CLUTRR asks how two people in a story are related [23], and each story's relations are given as facts. A rule library
learns composition rules, such as a father's brother being an uncle, overnight from one day of refused questions whose
gold relation is known. A rule is adopted only when it re-derives every past episode with zero counterexamples, and
every application of a rule is certified in the VM with its facts. No language model is involved in this result.

| split `gen_train23_test2to10` (test, 1,146) | result |
|---|---:|
| rules adopted | 109, in 3 rounds |
| correct / refused / wrong | **827 / 319 / 0** |
| at 4 to 10 hops (no training record reaches them) | **746 of 1,003, 0 wrong** |
| null control (the edges into the answer relabelled) | 0 of 1,146 spoken |
| shuffled library (a test that the measure can see wrong answers) | 29 wrong of 150 |

The null control shows that the system does not speak when the evidence isn't there, and the shuffled library shows
that our measure does catch wrong answers when the rules are wrong, so the 0 wrong is not the result of a test that
cannot fail. The zero-counterexample gate also rejected 8 rules that the 2-hop records alone support. For example,
(son, grandfather) is *father* 156 times and *father-in-law* 16 times, a real ambiguity that only the 3-hop records
show.

Of the 319 refusals, 206 are true answers recovered just under the served threshold (0.447 to 0.473, threshold
0.4736). A diagnostic threshold of 0.2 reads 1,033 of 1,146 correct, still 0 wrong, but we did not adopt it, as
recalibration needs a separate set, never the test.

| second split `gen_train234_test2to10` (test, 1,048) | result |
|---|---:|
| the gated library | **552 correct, 0 wrong** (469 of 903 at 4 hops or more) |
| ablation: 2-hop records only | 727 correct, **3 wrong** |

The ablation's 3 wrong answers, *mother* for *mother-in-law*, come through a rule that the whole-record gate keeps
out. This is the cost and the value of zero counterexamples, as the system gives fewer answers but none of them are
wrong.

## 5.2 Real Wikidata families: killed as built

We ran the same method on real kinship from Wikidata, using Wikidata's own kinship statements as gold, the family graph
gendered with Wikidata's declared inverses, and 2,050 training and 486 test pairs split by subject.

| real families | spoken | wrong |
|---|---:|---:|
| the gated library (255 rules, 109 inverse rules) | 70 | **8** |
| reversed questions | 34 | 6 |
| the inverse arm | 22 | 4 |
| null control | 0 breaches | |
| shuffled library | | 18 |

Our pre-registered kill line was any wrong relation spoken, so this result is killed as built. Most wrong answers
share one cause, two true relations against a single gold answer (a son-in-law who is also an adopted son, a brother
who is also a twin), because the class hierarchy encodes entailment but not exclusion. Some golds contradict the graph,
and one is a real induction failure: the inverse of *niece*, for a woman, was learned as *father's sister* from
brothers' daughters only. The next step is a contradiction model, with generation, line and sex per kinship term,
rather than more rules, and exact asymmetry and disjointness checks from an OWL 2 RL reasoner form one part of it.

## 5.3 The reasoning pipeline on curated and Wikidata worlds

All results in this section use the 2.6B stand-in as emitter.

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

Every wrong answer this pipeline ever spoke on the 800-question set was a pick between several true facts, so an
ambiguous hop is now a refusal. On a third world, Wiki5M, we asked 1-, 2- and 3-hop questions both in their template
wording and in an LLM's rephrasing, with each chain's facts in the store, and across 17,415 scored questions the
pipeline spoke no wrong answer. The rephrasings are evaluation data only, since no external model writes training
records.

## 5.4 SimpleQA through the gate

On all 4,326 SimpleQA questions, against a store that holds the answers to only 3 of them, the system speaks no
answer and makes no mistake: 3,858 questions become plans, 3,825 of them are refused with a reason and 33 walks fail.
This is the intended behaviour, as free text against a store that cannot answer it is refused, not guessed.

With Wikidata as a source on 600 SimpleQA questions, it certifies 6 answers, with 5 correct, 1 counted as near the
gold and 0 wrong, and no model API called. As a ceiling probe, we also put a frontier model, Gemini 3.8 Flash, in the
emitter's seat behind the same disposer, walk and VM. It certified 4 answers, 3 of them correct and no false facts.
The ceiling on free text is not the proposer but the store, which is what the full knowledge base addresses (Section
6). The everyday table, with no model in the proposer's seat, answers 65 of 187 WebQuestions questions with 0 wrong.

## 5.5 A model as a fact source

On the same 600 SimpleQA questions, we used the 2.6B stand-in as a source of facts in the latent tier. It proposed 450
facts, the system held 7 answers from them, and all 7 were wrong. Against Wikidata's facts, the model's facts agreed
once and disagreed 28 times, and none of its 22 dates of birth agreed. A model's parametric facts on this long tail
are nearly always wrong. The latent tier is what made that safe to measure, since nothing it said was spoken.

## 5.6 GRL's own model

For arithmetic, the emitter never writes the whole program. Each turn, it proposes one operation on numbers the host
placed, the VM runs it and the host writes the value back. Every intermediate number is the VM's and never the
model's.

| 236 held-out word problems (base450m with the emitter adapter) | result |
|---|---:|
| whole program in one pass, then one certified step at a time | 5.5%, then **22.9%** correct (the 2.6B stand-in in one pass: 85.2%) |
| templates seen in training / never seen | 24.6% / 20.4% |
| speak only when base450m and the 2.6B stand-in agree | **0.8% of questions answered wrong**, precision 0.964 |

The 450M model is a weak proposer, and we say so. This result shows that even a weak proposer becomes safe when two
independent proposers must agree and the VM computes every number.

We also built an n-gram memory in the hypervector space, where an n-gram is bound from word codes (60,147 words,
10,240 dimensions) position by position and recalled with a Hamming scan. We tested it on 597k corpus trigrams with
one word swapped.

| word codes | a paraphrase reaches the row | a different fact (*france* for *germany*) | a random word |
|---|---:|---:|---:|
| the shipped table | 22.3% | 0% | 0% |
| a paraphrase model's codes | 74.7% | 10.7% | 0% |
| the same, refined with WordNet (test words held out) | **74.3%** | **4.0%** | 0% |

An exact n-gram address reaches nothing on a paraphrase, while here one distance cutoff sets the trade-off against
reading a different fact. The paraphrase list is still small, with 35 hand pairs, and the language-model test has not
run yet.

## 5.7 What isolation costs

We run one resident CubeLang process per user session, so no user's frame, knowledge path or program can reach another
user's. With 256 sessions on one machine, each user costs 1.69 MB of marginal memory, and a spawn and first call take
19 ms. The VM runs 5,211 programs per second from 8 threads, with a p99 of 2.77 ms. Isolation is not the expensive
part of serving; the shared language model is.

# 6. The full evaluation, pre-registered

The results above are scattered, and most have no language model writing programs, so the full evaluation reports
everything in one run, recording each answer's proposer. We fixed its protocol and pass lines before running it.

The full evaluation runs on a store built from the complete Wikidata dump released with the WikiKGQA 2026 challenge
(121.3 GB, CC0), so every benchmark runs on one fixed snapshot. Right now the results are limited by coverage, as the
store holds the answer to very few of the free-text questions and the system mostly refuses.

The benchmarks are the full test splits of SimpleQA (4,326 questions) [17], Natural Questions open (3,610) [24],
WebQuestions (2,032) [25], CLUTRR (1,146) [23] and the public split of AA-Omniscience (600) [26]. There are four arms.
Arm C is the full stack with every proposer and the VM certifying, and it speaks only what is certified. The second
arm is C without the VM, the same proposals spoken without certification, read from C's own records with no second
pass. Arm M is the 450M model alone, closed book and always answering. Arm F is the frontier models' own published
numbers, labelled as other models, other graders and other settings.

The grading uses no judge model. A spoken answer is right when it matches a gold answer or an alias after
normalization. Every unmatched answer that C speaks is read by hand, with its program and its certification trace. For
each benchmark and arm we report right, wrong and refused, the wrong rate among spoken answers with a 95% interval,
the refusals by reason (no plan, a relation the store lacks, a missing fact, a failed walk, not certified) and the
answers by proposer.

Our pass line is C's wrong answers at most 1% of its spoken answers on every benchmark, and below C without the VM.
C above 1% on any benchmark kills the claim as stated, narrowing it to the benchmarks where it holds, with the cause.
The stack stays frozen during the run, nothing gets tuned on these test sets, and every number is written once.

QAWiki's test, 150 questions in English and Spanish, also comes with every challenge entrant's submitted answers [11].
Built on the challenge's own Wikidata snapshot, Cubby answers the same questions, and we score every entrant's answers
by the same rule as Cubby's, right, wrong and refused, with the challenge's own F1 reported beside it.

# 7. Limitations

Without the full knowledge base, the system refuses most free-text questions; a refusal is honest, but it is not an
answer. Most results so far come from no language model or a third-party stand-in, and GRL's own 450M model gives
modest results as a proposer. Certified is not correct either: a wrong fact in the store yields a wrong answer with a
valid certificate, so store quality bounds system quality, even with every wrong answer reported. Real kinship remains
unsolved, since the method giving 0 wrong on CLUTRR gives 8 wrong out of 70 on real Wikidata families. The similarity
threshold came from earlier experiments, and a lower one would answer more CLUTRR questions, but we have not adopted
it, as that choice needs a separate set. Finally, the system serves English and French, so the Spanish half of the
matched comparison will be reported as Spanish.

# 8. Conclusion and next steps

In CubbyLLM the model proposes, the VM certifies, and the system speaks only certified answers. On CLUTRR this gives 827
correct answers out of 1,146 with 0 wrong, including depths no training example reaches, while the controls show our
measure does catch wrong answers. On real Wikidata families it does not hold yet, and we have shown why.

The next version adds the full evaluation of Section 6. After that, we are working on time in the store, where every
fact keeps both its validity period and its recording date so any past answer replays exactly, and on interval
questions, refused when dates are too coarse to decide. We also plan certificates in W3C PROV-O, readable by standard
provenance tools, and domain worlds for finance (FIBO with GLEIF records) and automotive, added as data rather than
training. Finally, mutation tests will plant small faults in the VM and the rules, each of which must be caught.

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
