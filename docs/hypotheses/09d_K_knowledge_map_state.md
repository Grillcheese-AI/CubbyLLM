<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 9d. Group K — Knowledge, the map, the state: the pause of 2026-10-20 (added 2026-10-20)

**Why this group exists.** Nick, 2026-10-20: "lets pause and do a thorough research … we need to find something that
will really challenge everyone … plan / test some new hypothesis that are or seem to be way ahead … lets not be afraid
of complex problems"; then: "the status is only for inspiration we wont take their research we will create our own";
"also search github for inspiration … some hidden gems are in small repos sometimes"; and two questions that turned into
hypotheses of their own: "décupler [tenfold] the information without trillion of tokens and still have the same power …
10 fold could be 1,000,000 fold too … 40B token to train the model to be par to a 1T token model" (H-K7, data
efficiency) and "kv cache is one of the problem when its massive its energy consuming a lot" (H-K8); a repo to read,
`google-research/swirl-dynamics` (H-K9); and "if tokens are the problem why not simply replace them by something more
efficient? like hypertokens" (H-K10). **The rule for this group:** the field's results are inspiration, named once per
entry so the difference is explicit; every hypothesis is ours, stated in the algebra we already have (H-S12's Hilbert
space: a unit vector per place, binding as motion, similarity as a kernel), with a pre-registered gate and a kill, and
the cheap half of each run the same day where it could be. **The thread through all ten:** one space for place (H-S12),
fact (H-K1), memory (H-K8) and time (H-K4), so what, where and when compose by binding; the trunk learns *when* to read
(a gate, a query), the algebra guarantees *what* comes back, and the VM verifies before anything is spoken. A
conventional model has none of these as objects — it has weights that may or may not have memorized them.

**What the field is doing, read for inspiration only (2026-10-20).** Test-time memory in the trunk: Titans (a neural
memory updated by its own surprise), Titan-Llama (memory modules on a frozen Llama-3.2-1B, ~1% of the parameters
trained), Nested Learning / HOPE (layers as learners at different update frequencies, a continuum memory), Gated
DeltaNet-2 (erase and write decoupled, channel-wise, 1.3B/100B tokens), Kimi Linear and the Qwen hybrids (3:1
linear:attention), "Language Models Need Sleep" (offline passes that consolidate evicted context into fast weights),
HeLa-Mem (Hebbian associative memory for agents); representation: "Attention as Binding" (attention is approximate VSA;
explicit binding would be exact), Haga's disentangled successor information (grid cells and word embeddings from one
factorization), grid-cell path integration as a group representation (`ruiqigao/grid-cell-path`), the capacity analysis
of vector-symbolic superpositions (the d/(2 ln M) law); knowledge: "Physics of Language Models 3.3" (2 bits per
parameter, ~1,000 exposures to store a fact), Memory Layers at Scale (product-key memory, 128B memory parameters),
Hrrformer (holographic attention everywhere — linear time, lost accuracy); the bench the field now scores on: ARC-AGI-3
(interactive, hidden rules, Oct 2026 leaderboard: all closed models, the best in the sixties); and the small repos:
`swirl-dynamics` (DySLIM — regularize a learned dynamics to keep the true system's invariant measure, so long rollouts
stay on the attractor; flow-map matching and MeanFlow — one-step jumps over a time interval with the semigroup
consistency Φ_{s→u} = Φ_{t→u} ∘ Φ_{s→t}; "Evolve Smoothly" — latent dynamics in the parameter space of a small network,
kept low-dimensional by a consistency term; debiasing by optimal transport / rectified flow between a cheap model's
distribution and the expensive one's; WENO-NN — a learned weight inside a numerical scheme that keeps the scheme's
guarantees). Links in the appendix row.

**H-K1 — The fact store as a map: entities are places, relations are displacements, composition is path integration,
relation discovery is unbinding — in the same Hilbert space the trunk binds places into.** Source: Nick, 2026-10-20;
H-S12's encoding (`cubbyllm/spatial/ssp.py`). Inspiration: rotation-based knowledge-graph embeddings (RotatE) score a
triple by a rotation per relation; ours uses the spatial link's phases (one space for the village and the graph),
inverses are free (φ → −φ, so a backward hop is the same operator), composition is exact algebra (S(e0) ⊛ D1 ⊛ D2, never
touching the store), and every proposal is verified by the VM before the loop speaks it — the map proposes, the walk
checks. **What it brings:** "how is o related to s" by unbinding (D = S(o) ⊛ S(s)^-1, ranked over the 2,902 relations) —
a question a conventional model answers from memorized text if at all; two hops in one read; the store's capacity in
bytes, not tokens (H-K7). **Mechanism, built:** `validation/exp_k1_map.py` — entity phases (E, F) learned, relation
phases (R, F) learned, score `mean_f cos(θ_s + φ_r − θ_o)`, softmax over the true answer and 8,191 shared negatives in
both directions (SparseAdam on the 375k-row table: a touch is a step of `lr` radians), against TransE at the same width
(2F real dims) and budget; the wiki world (`standin/data/out/wikikg`: 374,975 entities, 2,902 relations, 365,923
triples, 5% held out, 2,000 two-hop paths whose both hops are training triples). **Gates (pre-registered):** (1)
composition — two-hop Hits@10 by path integration ≥ 0.6 × one-hop Hits@10; (2) relation discovery — unbinding Hits@3 ≥
0.5 over 2,902 relations; (3) one hop — MRR within 10% of TransE's at the same width and budget. **Kill:** gate 3 worse
than TransE by more than 30%, or gate 1 below 0.3. **First read (2026-10-20; F = 256, 10 epochs, 8,191 shared negatives,
3,000 held-out triples, 2,000 two-hop paths; `validation/logs/exp_k1_map_{map,transe}_2026-10-20.json`):** the map — one
hop MRR 0.383 / Hits@1 0.290 / Hits@10 0.542, relation discovery by unbinding MRR 0.180 / Hits@3 0.188, two-hop by path
integration Hits@10 0.980; TransE at the same width and budget — one hop MRR 0.185 / Hits@1 0.091 / Hits@10 0.328,
relation MRR 0.037 / Hits@3 0.041, two-hop Hits@10 0.956. Gate 1 passes (two-hop / one-hop Hits@10 = 1.81, the paths
being over training hops); gate 3 passes at 2.1× TransE instead of within 10%; **gate 2 fails** at Hits@3 0.188 — 4.6×
the baseline but far from 0.5, on a graph of mean degree 1.9 where most entities carry one or two facts and the
displacement between a held-out pair is generalisation, not fit. Caveats, stated: the map's learning rate was probed at
2 epochs (0.05 > 0.2 > 0.5 > 1.0; SparseAdam's first touch steps ~3× lr on a row touched twice an epoch) and its
relations start uniform on the circle (a relation is a displacement, distinct from the start — it cost one hop at 2
epochs and bought relation discovery); TransE's rate was probed the same way (0.01 of 0.003 / 0.01 / 0.03) and it
memorised the training set (loss 0.13) — a stronger baseline would be regularised, so the 2× is a first read, not a
claim. Status: **OPEN — gates 1 and 3 passed, gate 2 failed as pre-registered; next: the map's phases as the fact link's
codes (H-K10 gate 4), and relation discovery re-read on the dense core of the graph (entities with ≥ 5 facts).**

**H-K2 — Surprise-gated episodic writing: the store keeps what the trunk did not predict, and erases what it later
confirmed.** Source: Nick's "find something that will really challenge everyone" and H-E20's gate 2 (beyond-window
recall at chance with a store that holds every aged token: 15.9% vs 16.8% shuffled). Inspiration: Titans (surprise
drives the memory update), Gated DeltaNet-2 (erase and write as separate gates); ours is a *write mask on the row store
we already have* — the token's own loss at training (its surprisal at serving) decides whether its (k, v) enters the
store, the gate's later read error decides erasure, and the VM's verification is the confirmation signal on the host
side. **What it brings:** a store of news — a needle is surprising by construction, filler is not; capacity scales with
what was unexpected, not with length. **Advantage over conventional models:** a KV cache keeps everything and eviction
heuristics score rows after the fact by attention mass; ours decides at write time by prediction error, which is the
definition of information. **Mechanism (to build, flag-gated):** `CubbyMemory` with `CB_MEM_WRITE=surprise` — rows
ranked by the token's loss within each window, the top `mem_write_k` kept, the rest never written; erase on a confirmed
read (the read's output matched the next token's logit). **Gates:** (1) the needle probe (`exp_needle_forward.py`,
4,096, three depths): beyond-window recall ≥ 50% with a store of 256 rows chosen by surprise where the same 256 rows
chosen by recency sit at chance (< 20%); (2) held-out text loss within 0.01 of the ungated store; (3) the store's row
count at 4,096 ≤ 1/8 of the ungated one. **Kill:** surprise-chosen rows no better than recency-chosen by 10 points.
Status: **OPEN — not built; runs on the grown model's second round.**

**H-K3 — Continual learning by construction: a frozen trunk, and a sleep pass that distills the day's VM-verified facts
into the context adapter's bases, per context.** Source: Nick, 2026-10-20; H0 (the context channel), H-A7 (the learning
gate). Inspiration: Nested Learning (levels at different update frequencies), "Language Models Need Sleep" (offline
consolidation of evicted context); ours consolidates *only what the VM verified*, into the adapter the architecture
already has, with the learning gate (H-A7) deciding whether the update is CE-neutral before it is kept, and the trunk's
goldens unchanged by construction (the trunk is frozen; features off = bit-identical). **What it brings:** a model that
is better tomorrow at what it checked today, for an adapter-sized update, with forgetting measured as a number rather
than hoped away. **Advantage:** conventional models retrain, fine-tune with drift, or retrieve; ours learns facts by
writing rows and consolidates them by sleeping, with the trunk's language untouched. **Gates:** (1) after one sleep pass
over 1,000 verified facts, recall of those facts through the adapter alone (store removed) ≥ 80%; (2) text held-out loss
within 0.01; (3) the previous day's 1,000 facts recalled ≥ 70% after the next day's pass (forgetting bounded). **Kill:**
gate 1 below 50% or gate 3 below 40%. Status: **OPEN — not built; the sleep pass is `exp_if_follow.py`'s sibling (H-E20
gate 4).**

**H-K4 — A clock in the state: a one-dimensional fractional-power time code bound into memory keys and the scene, so
before / after / when are displacement queries.** Source: Nick, 2026-10-20; H-S12 (the same encoding, dims = 1); Group P
(keyframes need a coordinate). Inspiration: RoPE is this code inside attention only; ours is a readable coordinate the
trunk, the host loop and the VM share, and a what–where–when triple is one bound vector `e ⊛ S(p) ⊛ T(t)`. **What it
brings:** temporal questions by algebra — "what happened between t1 and t2" is a box kernel read (H-S12's closed form),
"which came first" is a sign. **Gates:** (1) a synthetic diary (events with times): between-t1-and-t2 recall ≥ 90%
through the box read; (2) in the trunk, with `T(t)` bound into the episodic keys, the needle probe's beyond-window
recall at 4,096 at least the untimed store's, and "which came first" questions ≥ 80% where the untimed store sits at
chance. **Kill:** gate 2's needle recall drops by more than 5 points (the clock costs the read). Status: **OPEN — the
host side is a `Scene` with dims = 1 (cheap, next); the trunk side follows H-K2.**

**H-K5 — Interactive hidden-rule learning in BlockWorld: a private ARC-AGI-3-like bench where the rule is a chemistry
the agent must act to discover, propose as a program, and verify with the VM.** Source: Nick ("something that will
really challenge everyone"); ARC-AGI-3 (interactive, hidden rules; the field's current yardstick). Inspiration: the
bench's shape; ours is in our own world (BlockWorld's `VoxelGrid`, chemistry, `orders.py`), scored by the VM, never by a
model. **What it brings:** a measurable test of learning-by-doing — propose → act → verify — where verification is in
the loop, not in the weights, and the transcript of actions is the evidence. **Advantage:** a frontier model given the
same observations must guess the rule from text; the loop runs the experiment. **Gates:** (1) 20 hidden rules; the loop
recovers ≥ 60% with ≤ 50 actions each; (2) a frontier model (when OpenRouter credit exists), given the same observation
transcripts and the same action budget, scores ≤ the loop + 10 points; (3) every rule the loop states is VM-verified
before it is spoken (0 wrong spoken). **Kill:** the loop below 30%. Status: **OPEN — not built; the rule generator is
the first piece (`cubby-blockworld`, private).**

**H-K6 — Provable binding heads: an explicit bind / unbind head with unitary role vectors in place of some attention
heads, so role-filler structure is a theorem where attention has a tendency.** Source: "Attention as Binding" (attention
≈ VSA, approximately); H-B6 (the trunk is already shaped to be bindable). Ours is the exact version: roles are unitary
(`SpatialLink.unitary`), binding is the circular convolution the spatial link uses, unbinding is its conjugate, and a
role-swap is a different vector by construction. **What it brings:** who-did-what-to-whom that cannot be confused by
swapping the names; composition that survives nesting. **Gates:** (1) on a role-filler task with swapped roles, a bind
head ≥ 95% where an attention head of the same width ≤ 80% at the same training budget; (2) text loss within 0.02 with 1
of 4 heads replaced. **Kill:** gate 2 fails. Status: **OPEN — not built.**

**H-K7 — Décupler: the same power from twenty-five times fewer training tokens (40B for what a 1T-token model knows and
does), and on facts from a million times fewer, because a fact does not have to be learned from sightings and a textbook
does not have to be scraped.** Source: Nick, 2026-10-20 ("décupler [tenfold] the information without trillion of tokens
and still have the same power … 10 fold could be 1,000,000 fold too, the higher the better … per example we need 40B
token to train the model to be par to a 1T token model"). Inspiration: the exposure law ("Physics of Language Models
3.3": ~2 bits per parameter, ~1,000 sightings to store a fact, wording diversity needed to extract it); the data-quality
result (phi: a 1.3B model on ~7B tokens of textbook-quality data matching models trained on far more web); the
data-constrained scaling law (Muennighoff et al. 2023: repeating data up to ~4 epochs costs almost nothing against fresh
data); RETRO and kNN-LM, Memory Layers at Scale (retrieval as knowledge). Ours: the store is H-K1's map plus the exact
store, algebraic and VM-verified, not a dense index; the textbooks are written by our own loop (the VM, the walks, the
scene, paraphrased by rule — no external model in Cubby's loop, OpenRouter only to build datasets); and the trunk's part
is a *route* (a gate and a query, as in H-S12), not the facts. **Where the factor comes from, honestly split.** (a)
**Facts — unbounded.** Most of a trillion-token corpus is its long tail: names, numbers, dates, who-did-what, each seen
many times so the weights keep it. **Measured on our own trunk** (`validation/exp_k7_exposure.py`, 2026-10-20: a
1.06M-parameter `train_base` trunk, 1,000 synthetic facts in a Markov language, one pass, 10 min on the CPU): exact
recall 0.00 / 0.00 / 0.50 / 0.99 at 1 / 10 / 100 / 1,000 exposures with paraphrased wordings (first object token 0.00 /
0.01 / 0.74 / 1.00), 0.00 / 0.00 / 0.30 / 0.86 verbatim, language held-out loss 6.96 → 2.67 — **gate 1 passes**: the law
holds here, and wording diversity helps extraction as in its paper. A fact in the store costs one write (the exact store
recalls it at 1.00 by construction, the vector map at H-K1's Hits@1): a fact scraped a million times is written once.
(b) **Language — bounded by quality, not by the store.** On the same data, 25× fewer tokens costs ~2.5× the reducible
loss (the Chinchilla exponent, D^−0.28); what beats the law is what the tokens are. Ours write their own textbooks:
verified facts in many wordings, verified reasoning traces (BlockWorld chemistry, wiki-world walks that the VM checked),
the scene's questions with answers guaranteed by the algebra — repeated up to ~4 epochs under the learning gate (H-A7
decides whether an update is CE-neutral). (c) **Structure at the input — paid in algebra, not tokens.** A village bound
as a scene is learned from zero text about space (H-S12 gate 1: ≥ 90% against < 60% for the text control at the same
tokens); the same mechanism carries a fact link (H-K1's phases bound into entity tokens, read mid-trunk) and a clock
(H-K4). **What it brings:** the fact power of a trillion-token model from a store the size of a disk, the language of
one from textbooks the loop wrote, and facts that update without a gradient step. **Advantage over conventional
models:** they buy every fact with ~1,000 sightings of web text and cannot change one without training again; ours
writes it once, checks it with the VM, and spends its tokens on language. **Gates:** (1) ✓ the exposure curve (above);
(2) tokens-to-recall — the trunk + store pair (the trunk routes, the store answers) ≥ 90% of held-out facts at one
exposure where the trunk alone needs ≥ 100 (≥ 100× fewer fact tokens for the same recall); (3) the corpus accounting —
the share of fact tokens in our cache (decode a sample, count entity / number / date tokens a store could hold),
reported either way, ≥ 30% expected; (4) **the 40B-vs-1T test, scaled down** — two trunks of the same size (20M-class, a
Colab job): one on the full text budget, one on 1/25 of it plus the store, the loop's textbooks and the input channels;
the pair ≥ the full trunk on fact QA (through the loop), within 0.15 nats on the text held-out loss, ≥ 90% of it on
BlockWorld tasks; (5) at size — the 450M continued with the fact link answers wiki-world questions at ≥ 2× the text
control at the same tokens. **Kill:** gate 2 below 50% (the route does not learn; the host loop stays the only path), or
gate 4's text loss worse by more than 0.4 nats (the textbooks do not close the law's gap; then the claim is facts only).
Status: **OPEN — gate 1 passed 2026-10-20 (`validation/logs/exp_k7_exposure_{paraphrase,verbatim}_2026-10-20.json`);
gates 2–5 to build, gate 5 on H-S12's mechanism (places → entities).**

**H-K8 — A holographic state in place of the KV cache: beyond the window, one vector per chunk of C tokens, the
superposition of the chunk's bound (key, value) pairs with unitary keys, read by unbinding; bytes per token flat in
context length.** Source: Nick, 2026-10-20 ("kv cache is one of the problem when its massive its energy consuming a lot
… if we are able to do something like that it would be immense power in the end"). Inspiration: Hrrformer (holographic
attention everywhere; linear time, lost accuracy on language), linear-attention states (Gated DeltaNet-2, Kimi Linear: a
d×d matrix per layer), the VSA capacity analyses; ours keeps exact attention inside the window (512), goes holographic
only beyond it, in chunks with a summary key for selection, in the same algebra as place and fact
(`SpatialLink.unitary`, `bind`, `unbind` are the operators), and — from `swirl-dynamics`' DySLIM — asks that the state's
statistics stay stationary over long rollouts. **The law, derived and measured (`validation/exp_k8_holo.py`,
2026-10-20):** with random unitary keys and unit values every decoy's dot with a read is N(0, C/d), its maximum over N
candidates ~ √(2 C ln N / d), so a nearest-value decode over N holds while C ln N < d/2 (one vector alone: M ln M <
d/2). Measured: the law's crossing is where the decode sits at 0.68–0.70 at every d (512 / 1024 / 2048) — gate 1 as
written (90% within 25% of the law) **fails**, and the corrected rule is that 90% holds to about half the law (M = 31 /
55 / 97). The chunk store at N = 4,096: at d = 1024 (the grown model's width) chunks of 16 decode 1.00 / 1.00 / 0.97
over query noise σ = 0 / 0.5 / 1.0, chunks of 32 0.97 / 0.90 / 0.53 (**gate 2, C = 32, σ = 0.5: 0.90, at the line**),
chunks of 64 0.61 / 0.51 / 0.14; at d = 2048 chunks of 32 decode 1.00 / 1.00 / 0.98 and 64 0.97 / 0.95 / 0.64; summing
all chunks (no selection) decodes nothing (0.00) and top-4 summed is worse than top-1 — selection is the whole
mechanism; the read is attenuated (cos 0.16–0.25 to the true value) though decodable. **Bytes per generated token at the
grown geometry (D = 1024, L = 64, window 512, 16 memory layers), bf16:** full attention at N = 1M touches 262 GB per
token; the window + row store we have, 2.2 GB (resident 35 GB); window + chunks (C = 32), 1.15 GB (resident 2.2 GB); at
N = 4,096 all three sit on the 64-layer window's floor of 128 MB — the energy argument is a long-context argument, and
the resident saving (16×) is the larger one. **What it brings:** a context of a million tokens held in 2 GB and read at
a cost flat in its length. **Advantage:** a KV cache's per-token bytes grow with N and so does its energy; ours is
bounded by construction, and the beyond-window read is one Fourier-domain product, no softmax over rows. **Gates:** (1)
**failed as written** (the law marks the 0.68 point; recorded, not re-registered); (2) **passed at the line** (0.90);
(3) model level, to build — a tiny trunk with the holographic store (`mem_kind=holo`, flag-gated) on a synthetic recall
task matches the row store within 5 points and beats window-only by 30; (4) the grown model's variant at 4,096: needle
recall ≥ the row store's with ≤ 1/16 of its resident bytes; (5) stationarity — the state's norm and the read's SNR drift
< 10% over 100k tokens of rollout (DySLIM's question, asked of our state). **Kill:** gate 3 fails (the trunk cannot use
a 0.16-cos read; then rows stay, chunks go). Status: **OPEN — gates 1–2 run
(`validation/logs/exp_k8_holo_2026-10-20.json`); gate 3 is the next build (a `CubbyMemory` sibling, < 300 lines, off by
default).**

**H-K9 — Jumps in time with a composition law: a learned flow map Φ_k on the trunk's state that gives the state k steps
ahead in one application, trained with the semigroup consistency Φ_j ∘ Φ_k = Φ_{j+k}, so prospection's keyframes cost a
read, not a rollout.** Source: `google-research/swirl-dynamics` (flow-map matching, MeanFlow: one-step jumps over a time
interval with the consistency of the flow; DySLIM; "Evolve Smoothly"), read at Nick's request 2026-10-20; Group P
(keyframes, branching futures); H-S12 (composition is binding — the displacement algebra already composes exactly, the
learned part is the trunk's dynamics). **What it brings:** a 64-token lookahead for one application instead of 64
forwards; a planner that can afford to look; the energy argument applied to thinking ahead. **Advantage:** conventional
models plan by generating; ours would jump in state space and verify the landing with the VM. **Gates:** (1) on
BlockWorld traces, Φ_k's state after k = 8 / 32 actions decodes the right world facts ≥ 80% (the rollout's own decode =
100%); (2) consistency error ‖Φ_j(Φ_k(s)) − Φ_{j+k}(s)‖ / ‖s‖ < 0.1 on held-out traces; (3) planner success with
keyframes ≥ 90% of token-by-token rollout at ≤ 1/8 the forwards. **Kill:** gate 1 below 50%. Status: **OPEN — not built;
traces come from `cubby-blockworld` (private).**

**H-K10 — Loci tokens (named *hypertokens* until 2026-10-22): the unit the trunk reads is a bound vector, not a text
piece — identity ⊛ place ⊛ time ⊛ role ⊛ store address — so a verified fact is one input unit that the trunk reads back
by algebra, and a span of language is one unit of a length chosen by information rather than by a merge table.** Source:
Nick, 2026-10-20 ("if tokens are the problem why not simply replace them by something more efficient? like hypertokens
(tentative name)"; "and for language are we condemned to use tokens?"). **Name (2026-10-22, Nick).** Called
*hypertokens* until 2026-10-22, renamed because the word is taken twice. zip2zip (Geng et al. 2025, arXiv 2506.01084)
uses it for tokens merged by LZW compression at inference, the sense H-C1, H-C2 and H-C6 use. Augeri 2025 ("Hypertokens:
Holographic Associative Memory in Tokenized LLMs", arXiv 2507.00002) uses it for symbolic codes built from
error-correcting codes, written into the *prompt* of an unmodified model in rare or private-use tokens, and retrieved by
the model's own attention and an inner-product argmax; v1 names no model, dataset or baseline. A **loci token** is
different in kind: it is the trunk's own token embedding bound to a place at the input (`e ⊛ S(p)`, a fractional power
encoding). A scene per document sums the tokens that write. A token that asks reads its place back inside the trunk by
unbinding, exact by the algebra rather than found by attention, and a copy path cleans the read up against the
document's written ids into the output (order 0001). The name is from the method of loci: things put at places, recalled
by going back to the place. File and log names keep the old word (`exp_k10_hypertoken.py`, `exp_hypertoken_recall.py`,
`hypertoken_recall_*`), so every earlier link still resolves. H-S12 is the first loci token (a token bound with its
place), the scene `V = Σ e_i ⊛ S(p_i)` is the first loci-token store, and the read mid-trunk (`SpatialLink.read`) is how
the trunk uses one. Inspiration, named for the difference: byte patches sized by entropy, sentence-level concept units,
text diffusion over spans, multi-token prediction — none keeps an exact decode; ours does (the lattice theorem). **Three
problems one unit causes, and what the loci token does to each.** A subword carries ~1 byte of identity and no
structure, so the trunk rebuilds structure from statistics — that is the 100–1,000 sightings of H-K7; with the structure
bound in at the input, a fact given *once* is read back exactly (the read is algebra, the gate learns *when*). Every
piece is a forward pass, so the cost is paid per piece, not per meaning — a span of n pieces as one unit runs the trunk
once (H-K8's energy argument compounds). A fact of ~20 pieces is 20 places to be wrong — a fact as one unit is decoded
whole from the store. **Two halves, honestly separated.** *Input (cheap, ours to choose):* a span is `Σ_i e(piece_i) ⊛
S(i)`, a bag with order, exactly decodable by unbinding at `S(i)`, length bounded by the capacity law (M ln M < d/2: ~55
pieces at d = 1024, ~35 at d = 256); the boundary comes from the host (a word, a name, a fact, a sentence — the ask
loop's grammar and `route()` are the linker's start) or from an entropy rule; language words stay words — loci tokens
replace the fact-bearing spans and add structure to the rest. *Output (the hard half):* predicting the next span is
predicting a vector, not a distribution over 128k symbols — a vector cannot say "70% Paris, 30% Lyon"; ambiguity becomes
a superposition, and a superposition decoded blindly is a blend. Routes: keep piece-level output for free text and use
loci tokens at the input only (this entry's gate), or emit a point in the Hilbert space that the *store* decodes —
nearest entity, unbind for relation and object — exact for the structured part of what the model says and free of the
vocabulary softmax (Group C's bottleneck), silent on the rest. **What it brings:** a fact seen once and answered; the
trunk run once per meaning; the fact half of the corpus at one unit per fact (H-K7's factor before any sightings
argument). **Advantage over conventional models:** their token is a text piece and structure is a hope; ours is a bound
vector and structure is a theorem. **Mechanism, to build (flag-gated, < 300 lines):** `validation/exp_k10_hypertoken.py`
— a tiny `train_base` trunk with `SpatialLink` (the existing channel, unchanged): documents of 12 facts `s r o` with the
object's token placed at the fact's address `cell(s) + offset(r)` (an entity's cell a hash into a 1,024³ lattice, a
relation's offset a lattice step the kernel separates), ~150 tokens of filler language, then questions `Q s r HERE → o`
with `HERE` placed at the same address — the questions fall beyond the attention window (64), the facts are random per
document (never seen in training), so the answer can come only from the scene's algebraic read; the control is the same
documents with nothing placed (text; the plain path, bit-identical). **Gates (pre-registered):** (1) one sighting —
held-out documents' questions answered ≥ 90% through the channel where the text control sits at chance beyond the window
(and at ≥ 80% *inside* the window, to show attention can do it only within reach); (2) the text goldens bit-identical
with the channel off (SPEC) and the filler language's held-out loss within 0.02 of the control's; (3) spans — a word of
up to 8 pieces presented as one bound unit at the input, the trunk's next-piece loss on the following text within 0.05
of the piece-by-piece baseline at ≤ 1/4 the forward passes; (4) output — on the wiki world, the trunk's emitted vector
decoded by the store (nearest entity) answers ≥ the text head's exact-match on the same questions. **Kill:** gate 1
below 60% (the trunk cannot learn to query the scene for a fact it was shown once; then loci tokens stay a host-side
object), or gate 2's loss worse by 0.05. **First read (2026-10-20, `validation/exp_k10_hypertoken.py`, a 3.4M-parameter
`train_base` trunk, D = 256, window 64, 8 random facts per document shown once, questions 120+ tokens later, 1,000 steps
× 16 documents, 25 min on the CPU per arm; `validation/logs/exp_k10_hypertoken_*_2026-10-20.json`):** before any
training the scene decodes the object at the fact's address at 0.84 by nearest entity (cos 0.17–0.35 to the true
embedding: the attenuated read of H-K8, with `HERE`'s own bound embedding at the same address). *The loci-token arm* —
the question's `HERE` token carries its address and the read's query **is** that address (the gate and the decode `o`
learned, the gate started open at 0 for a trunk trained from scratch): held-out answers 0.00 → 0.21 (step 200) → 0.53 →
0.81 → 0.86 → 0.92 → **0.94** at 1,000 — **gate 1's first clause passes** (≥ 90%); *the text control*, the same
documents with nothing placed: **0.00** throughout, the answer beyond attention's reach; the filler language's held-out
loss 2.075 in both arms — **gate 2 passes**; the in-window control (4 facts, no filler): **0.22** at 1,000 steps —
attention learns the in-window copy slowly, the second clause not shown at this budget (the algebraic read reached 0.94
*beyond* the window in the same steps). *The learned-query arm* — H-S12's route, the trunk asked to recover the address
from its own state through `q = W rms(h)`: **0.00** for all 1,000 steps, the gate drifting shut (−4 → −4.08 from the
design's start; 0 → −0.36 when started open). Read plainly: the algebra delivers, the trunk learns to decode an
algebraic read in a few hundred steps when the address arrives with the token, and *recovering* an address from the
hidden state is the hard part at this scale — which is exactly H-S12's gate 2 risk, now measured small. **Built the same
day, flag-gated:** `SpatialLink(query="placed")` (`CB_SPATIAL_QUERY=placed`, meta `spatial.query`, absent at the default
so every earlier meta still matches): a placed token's query is its own place `S(p)` from the input, exact; the unplaced
keep the learned query; the default path is bit-identical (`tests/model/test_spatial.py`, decode = forward with places
asked). **Second read — the route (2026-10-20, same trunk and budget;
`validation/logs/exp_k10_hypertoken_*_2026-10-20.json`).** The learned route is teachable: a *route loss*, `1 −
cos(unitary(q), S(p))` at the placed tokens — the side file knows every question's address, so the trunk's own query is
taught where to look, for free at training time — takes `--query learned` from 0.00 to **0.82** (route cosine 0.54).
What did not work, recorded: the gate pinned open for 250 steps then released (an outside suggestion,
freeze-and-release) 0.00 alone and 0.77 with the route loss; a per-frequency complex query head (the shape of the
deconvolution, d parameters instead of d²) 0.69 with the route loss — the dense head can also project the mixing layers'
noise out, the diagonal one cannot; a *matched* read (the query's conjugate at its own magnitudes instead of unitary(),
to stop a noise-dominated frequency being stretched to modulus 1) 0.19 — unitary() is right for this read; the dense
head started at the identity 0.66. Controls asked for by the outside review and run: attention over the *whole* document
(window 256) **0.03** at 1,000 steps — the algebraic read is the sample-efficient path, not a stand-in for a longer
window; a *mismatched address* (every `HERE` carrying another fact's valid address) **0.00** on every arm — the channel
relies on the address alone, no other path carries the answer; and the trunk's own gated recurrence (MinGRU, a dense
state past the window) is the text control that sat at 0.00. Built: `CB_ROUTE_LOSS=λ` in `train_base` (the query tapped
in the forward; needs `CB_GRAD_CKPT=0`), `validation/make_fact_shard.py` (the wiki world's facts as loci tokens: a
fact's address is the subject's hashed cell plus the relation's step, a name a span up the z axis read piece by piece;
`facts` / `facts_text` built, 20,000 documents, 240,208 facts, 119,839 questions, on the cache and the Drive), and
`make_spatial_shard.py --ask placed` (`spatial_here` / `spatial_text_here`: every village question carries its address).
Status: **OPEN — gates 1–2 passed with the address given, the learned route at 0.82 when taught; next: the 450M
continued on `spatial_here` + `facts` with `CB_SPATIAL=1 CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1` against the text
controls at the same tokens (H-S12 gates 1–2 and this entry's gate 4 in one run), then spans (gate 3).** **Run 1 at the
450M (2026-10-20, the same runs as H-S12's first read):** facts at addresses in-window (`facts`, 512-token documents) --
`link` **0.016** exact (first token 0.27), `text` **0.987**: the read did not learn, and the comparison was the wrong
one. Causes, in the shard and the scene: every placed token wrote into ONE scene per 1,024-token window -- two
documents, every name piece, and the questions' own `here` and answer pieces at the very addresses they asked about --
~150-250 items in d = 1024, past the capacity law's ~55 for 90% (H-K8); and the facts sat inside the attention window,
so `text` copied them. **Built for run 2 (`cad8da2`):** a thing WRITES into the scene and a question only ASKS there
(`.role.u8`; `Tokens.flags`), the scene restarts at every document (`SpatialLink.scene(starts=)`), and the facts move
beyond the window -- `facts_far1k` (6-10 facts, ~800 tokens of real text, then the questions; ~24 written items, the
first question 809 tokens past the last fact) against `facts_far1k_text`. Run 1's link final read with the roles:
villages 0.836 (12 held-out documents), `facts_far1k` 0.000 (never trained on it). **Run 2 (2026-10-20,
`spatial_link_2`: 3,058 steps = 0.40B tokens; the roles, a scene per document, `facts_far1k`):** villages **0.776**
exact (route alone 0.729); far facts **0.005** exact (first token 0.22, route alone 0.010) -- the read learned nothing
on the facts. `validation/diag_fact_read.py` (24 held-out documents, 120 questions, 446 later answer tokens, CPU) splits
it: first piece 0.242 with the places and 0.242 without (the read at `here` reaches nothing); the later pieces **0.096
with the places, 0.213 without, 0.229** with only the writing tokens bound -- every placed token was bound at the input,
so each answer piece, asking for the next, entered the trunk as `e (*) S(p_next)` and the trunk lost which token it had
just read; the read itself added ~0.02. Two causes: the ASK tokens' input binding (a question scrambled its own words)
and the first piece's address on `here`, three tokens before the answer. The text arm was not run (the link's gates fail
on its own numbers; gate 2 stands from run 1). **Built for run 3:** `SpatialLink(bind="writes")` /
`CB_SPATIAL_BIND=writes` (with roles, only a thing at a place is bound; an asking token enters as itself and asks with
`S(p)`; the default keeps every earlier run exact) and `facts_far1k_lead` (`make_fact_shard.py --lead`: run 2's
documents, ids identical by md5, the token right before each answer asking for its first piece -- every token that
predicts a piece asks for it). **Run 3 (2026-10-21, `spatial_link_3`, G4: 3,418 steps = 0.45B tokens;
`CB_SPATIAL_BIND=writes`, `facts_far1k_lead`; trained on the fused MinGRU kernel + flex attention at 99 KB tiles from
step 353, ~75.0k tok/s, 40.6% MFU):** villages **0.953** exact (first token 0.960; route alone 0.748) -- up from 0.776 /
0.793, past gate 1's link bar of 0.90 (the gate itself still wants `text` below 0.60, and in-window `text` read 0.995 in
run 1); far facts **0.021** exact (first token 0.229; route alone 0.014). `diag_fact_read.py` on 24 held-out documents:
first piece 0.250 with the places, 0.233 without; later pieces **0.309 with, 0.300 without** -- the scrambling is gone
(run 2: 0.096 against 0.213), the trunk's own guess at the rest of a name rose, and the read adds ~0.01-0.02. So with
every token that predicts a piece asking for it, and the asking tokens themselves, the read still does not carry an
open-vocabulary token at 0.45B: the villages' answers come from a few dozen names, the facts' from the whole 128k
vocabulary, and the read's vector (the input embedding of the token at the address, plus crosstalk) has to be decoded
through the 16 layers after the link into the untied head -- a copy the trunk learns for a closed set and has not for
all tokens, on ~1% of its tokens. Next (not run): a copy path -- the read scored against the embedding table straight
into the logits, gated (the vector already IS a token's embedding) -- or the read nearer the head. Text arm not run
(gate 4 fails on the link's number alone). **Order 0001 (2026-10-21, CPU; `docs/orders/0001-far-facts-local.md`): the
read carries the facts; the trunk never learned to copy them out.** Run 3's checkpoint, nothing trained
(`validation/diag_fact_cleanup.py`, 400 held-out documents, 9,451 piece reads;
`validation/logs/diag_fact_cleanup_run3_2026-10-21.json`): the scene unbound at each asking token's address and cleaned
up by cosine is the stored piece 0.758 of the time over the 128k vocabulary (0.457 whole answers) and **0.992 among the
document's own writers (0.965 whole answers)**. Cos to the target is 0.165, against 1/sqrt(37.5) = 0.163: the learned
table behaves like random codes, and the 128k-way decode sits on the random-code ceiling
(`validation/sim_hrr_ceiling.py`: 0.765 per piece at 37 items in d = 1024,
`validation/logs/sim_hrr_ceiling_2026-10-21.json`). The scene split by a hash of the address into K = 2 / 4 gives 0.859
/ 0.996 whole answers even over 128k (`diag_fact_cleanup_grid_run3_2026-10-21.json`). **Correction to run 3's reading:**
the villages never tested copying. A stored `pond` is answered ` pond` (another token), and 76% of village questions
answer "nothing": their 0.953 is empty-read detection plus a closed-set map. *The toy*
(`validation/exp_k10_far_facts.py`, D 256, 1,000 steps x 16 documents, teacher-forced): the 450M-like combination
(vocabulary 8,192, 4 facts of 1-5 pieces, 1 question a document, L 8 read at 4) reads **0.000 on 3 seeds**
(`exp_k10_far_facts_combo_s*`); the vocabulary step alone 0.000, the pieces step 0.221, an 800-token gap 0.941 (distance
is not the cause). *The fix:* a copy path (`cubbyllm/model/spatial_copy.py`, `CB_SPATIAL_COPY=ring`, absent = nothing
built). At asking tokens, `p = (1 - lam) p_LM + lam p_copy`, where `p_copy` is the read's cosine softmax over a fixed
ring of the last 256 ids the document wrote (the decode state stays fixed-size) and lam is read-aware
(`CB_SPATIAL_COPY_BETA`: the gate also sees the best match against a threshold). It reads **0.930 / 0.908 / 0.895**
exact on 3 seeds, the language within 0.005 (`exp_k10_far_facts_combo_ringb_s*`). The gate from the stream alone reads
0.922 but ends the name only 0.469 of the time (it cannot see an empty address); the copy over the whole vocabulary
reads 0.357. Added to run 3 with nothing trained: far facts 0.000 -> **0.258** exact (first piece 0.250 -> 0.800),
villages unchanged at 0.982 (`diag_copy_step0_run3_ringb_2026-10-21.json`, 24 documents). **What it brings:** a fact
seen once, 800+ tokens back, answered by its address, with a decode state of the scene plus 256 ids a stream.
**Advantage over conventional models:** they recall that far only through a KV cache that grows with every token. Next:
run 4 (`spatial_link_4`: the copy added to run 3) -- far facts >= 0.50 exact, villages >= 0.90, text mix within 0.02 of
run 3's. **Run 4 (2026-10-21, `spatial_link_4`, G4: 846 steps = 0.11B tokens, 0.5 h; run 3 continued with the copy
added, `CB_SPATIAL_COPY=ring`, tau 3.04, beta 0.115; 68k tok/s, 36.9% MFU against run 3's 40.6%, compiled, no per-step
recompiles):** far facts **0.761** exact (first token 0.975; 2,002 held-out questions; run 3: 0.021) and villages
**0.953** (run 3: 0.953) (`validation/logs/hypertoken_recall_spatial_run4_link_2026-10-21.json`). Pass on both. The text
mix **fails**: 2.2596 against run 3's 2.2224 (+0.037; the line is 0.02). That is the schedule, not the copy. Every
pure-text source rose +0.07-0.10 when the learning rate re-warmed from 3e-6 to 3e-4 with a fresh optimizer, peaked at
+0.13-0.17 by step 500, and only partly came back in the decay, ending +0.05-0.06. Those sources have no asking token,
so the copy never touches them (`spatial_run4_link_metrics_2026-10-21.jsonl`). **The gain lives in the copy alone**
(`validation/diag_swap_copy.py`, 64 held-out documents a shard, CPU; `diag_swap_copy_run4_2026-10-21.json`): run 3's
trunk with run 4's trained copy (1,027 parameters: the gate's d + 1, tau, beta) reads far facts **0.844** exact (first
0.988) against run 4 as trained at 0.738, villages 0.958 against 0.960. The trunk's drift cost the facts 0.11, and its
text is run 3's bit for bit, since the copy leaves every non-asking position unchanged. **What it brings:** the far
facts go from 0.021 to 0.84 on a trained 450M by training ~1,000 parameters, with its language untouched. **Advantage
over conventional models:** they add knowledge by fine-tuning, which moves every weight and the language with it, or by
context, which grows the KV cache. Next: train the copy alone on run 3's frozen trunk (order 0005), and report where the
rest of the gap to the read's 0.965 goes. **Run 5 (2026-10-21, `spatial_link_5`, G4: 674 steps, 0.17 h, 183k tok/s; run
3 with the copy added and `CB_TRAIN_ONLY=copy`: only the copy's 1,027 parameters train, at 3e-3, on the two asking
shards):** far facts **0.946** exact (first token 0.994; 2,002 held-out questions; run 3: 0.021, run 4: 0.761) and
villages **0.953** (run 3: 0.953) (`validation/logs/hypertoken_recall_spatial_run5_link_2026-10-22.json`). Every one of
the 21 text sources stands within 0.005 of run 3's (worst 0.0000), and `validation/diag_frozen_check.py` finds 0 of the
326 trunk and 5 router tensors moved (`diag_frozen_check_run5_2026-10-22.json`). The copy ends at tau 3.04 → 4.40, beta
0.115 → 0.009, and gate weight norm 2.62. The probe's split (64 documents;
`spatial_run5_link_metrics_2026-10-22.jsonl`): first piece 0.997, later pieces 0.978, the name's end 0.938; the gate's λ
is 0.80 where the asked address holds a piece and 0.056 where it is empty. The read's own ceiling among the writers is
0.965 whole answers, and run 5 reads 0.946. **What it brings:** a trained 450M answers facts it was shown once, 800+
tokens back, at 0.95 exact, by training 1,027 parameters, with its language provably unchanged. **Advantage over
conventional models:** fine-tuning would move every weight to add this, and a long context would pay a growing KV cache;
here neither. **In plain words:** it remembers what it was told once, far back, and nothing else about how it talks
changes. **Order 0006 (2026-10-22; CPU toy: d 256, 8 layers, 1,000 steps;
`validation/logs/exp_k10_free_text_<arm>_s<seed>_2026-10-21.json`, copy-alone reruns `..._alone_2026-10-22.json`):**
text keys `u(W e(s)) ⊛ u(W e(r))` and a pointer that reads a free-text question's subject and relation words. On
subjects never seen in training, asked in seen phrasings, the pointer reads 0.854 / 0.899 / 0.870 exact (3 seeds);
hashed addresses with the learned route read 0.000 and a learned query on text keys 0.006, so the pointer is what makes
text keys reachable. With the address given, text keys read 0.848 and hashed places 0.857: no cost as addresses. The
pre-registered lines missed (U 0.848 < 0.90; P on held-out phrasings 0.816 / 0.694 / 0.785, 1 of 3 >= 0.80; absent facts
answered 5.3-9.9% > 5%) for one cause, the mix: the copy alone picks the right answer on 0.955-0.974 of present facts,
mixed 0.846-0.857, and a hard switch at λ >= 0.5 recovers only to 0.875; about 12% of present facts are answered wrong.
Next: the switch trained alone (order 0021). **Order 0011 (2026-10-21):** readable fact names; run 5, nothing retrained,
reads 0.931 on the new spelling against 0.941 on the same 64 documents of the old
(`hypertoken_recall_spatial_run5_on_far1k_v2_2026-10-21.json`): the copy carries over. **Order 0021 (2026-10-22, the
same toy; `validation/logs/exp_k10_switch_<arm>_s<seed>_2026-10-22.json`,
`k10_policies_<arm>_s<seed>_2026-10-22.json`):** training the switch alone on the frozen toy lifts U from 0.848 to 0.879
but leaves present facts answered wrong at 7.3-9.2% and absent ones answered at 6.9-10.2%. The read is not the limit:
one threshold on the best match separates present from absent at a balanced 0.947-0.963. The fallback is: when the gate
closes, the trunk speaks, and at an asking position its pick is never the fact. **Copy or refuse** (the copy's whole
answer when every piece clears the threshold, NONE otherwise) on the same models: present wrong 0.6-1.3%, absent
answered 2.9-4.9%, refused 12-16%, exact 0.830-0.872 (threshold 0.17-0.18, picked on the evaluation documents). Facts
come from the copy or not at all. **Order 0022 (2026-10-22; `exp_refuse_450m_end_2026-10-22.json`,
`k10_refuse_<arm>_s<seed>_2026-10-22.json`):** copy or refuse as the contract, with the contract saying the answer's end
(`end_id`). On the 450M (run 5, nothing retrained; theta 0.117 fit on 200 calibration documents, measured once on 200
test documents): present wrong **1.7%**, absent answered **0.0%**, refused **32.9%**, exact 0.654, against the mix's
12.4% wrong and 100% absent answered. Both wrong-answer lines hold; the kill line (refusals at most 25%) fires. Absent
questions are already all refused at this theta, so the threshold is spent on present facts answered wrong: one scalar
separates present from absent, not right from wrong among present facts. On the toy, thresholds fit at the line's edge
on calibration missed it on test (2.3-3.4% present wrong). Next: a per-answer confidence learned on calibration
documents, its threshold fit with a conformal risk bound (order 0029). Status: **OPEN -- far facts by address solved at
the 450M (run 5: 0.946 exact, villages 0.953, text identical; order 0005); free-text keys reach unseen subjects on the
toy (0006: 0.85-0.90 with seen phrasings); copy or refuse is the contract (0021, 0022): on the 450M it meets both
wrong-answer lines (1.7% present wrong, 0.0% absent answered) at 32.9% refused. Next: a learned confidence to cut the
refusals (0029), learn on a miss (0007), creativity (0008), capacity (0009).**
