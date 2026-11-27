<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 5. Group C — Vocabulary, tokenization, and the softmax bottleneck (a core design requirement, not an optimization)

**Reframed 2026-07-23.** CubbyLLM's vocabulary will be significantly larger than cubby-lm's 32k — explicitly *because*
the mechanisms below can remove the full-vocabulary softmax step that would otherwise make a large vocab expensive to
train and serve. Everything in this group graduates from "worth checking if it helps" to "the output-head mechanism
CubbyLLM's vocab strategy actually depends on." The open question is no longer whether to pursue this, it's which
specific mechanism, and how large "much larger" ends up being — both still genuinely open (see H-C6).

**H-C1 — Dynamic hyper-encoder embeddings are the enabling mechanism for a much larger vocabulary, not an optional
parameter-count optimization.** Source: ZeTT (hypernetwork predicts token embeddings for any target vocabulary on the
fly) and zip2zip, motivated by the observation that static embedding/unembedding tables scale linearly with vocab size
and can reach up to 93% of total model params in typical LLMs at 100k+ vocab — the exact regime CubbyLLM is now
explicitly targeting, unlike cubby-lm's comparatively modest 32k. Status: SPECULATIVE mechanism (ZeTT/zip2zip are
external research, not yet reproduced against CubbyLLM's actual target scale), but no longer optional to check — it's
close to a load-bearing assumption behind the vocab requirement itself. **Validate** by prototyping a hyper-encoder
generating embeddings on the fly for a synthetic vocabulary at the target order of magnitude, and confirming (a)
generated-embedding quality is competitive with a static lookup table at a small/medium vocab size where both are still
cheap enough to compare directly, and (b) the hyper-encoder's own compute cost per token doesn't reintroduce the cost
problem the static table was supposed to avoid. **Kill criterion:** generated embeddings measurably underperform static
ones at matched vocab size, or the hyper-encoder's forward cost scales badly enough to erode the "fast" half of the goal
— either result means the large-vocab requirement needs a different enabling mechanism, or a smaller target vocab after
all.

**Validation result (2026-07-23 campaign): kill criterion HALF-triggered at toy scale.** On the real spm32k token stream
(cubby-lm valid.txt, V=8k kept, d=64, identical budgets): the **quality clause FAILS** for a pure surface-form generator
— static table val loss 3.886 (ppl 48.7) vs char-CNN hyper-encoder 4.389 (ppl 80.6), +0.50 nats; surface form alone
cannot recover what a table memorizes. The **cost clause PASSES decisively** — 1.26µs/token to regenerate the full
table, cacheable at inference, embedding params V-independent (77k vs 578k). Read alongside zip2zip's actual design
(static base vocab + composed hypertokens), this steers toward a **hybrid** (static core + generated/composed tail), not
pure generation — see H-C6. Toy-scale caveat stated in the log; re-check at target scale before betting the vocab
strategy. `validation/exp_c1_hyperencoder.py`.

**H-C2 — zip2zip-style LZW hypertoken compression reduces autoregressive step count 20–60%, directly improving
latency.** Status: **VERIFIED (2026-07-23 campaign — measured on the real tokenizer, no model change).**
grillcheese_spm32k_v2 over 8MB of cubby-lm's valid.txt (2.07M base tokens): **per-document (streaming-honest) reduction
22.2%** — the claimed range reproduced at its low end; a *persistent cross-document dictionary* reaches **53.4%–66.6%**
(32k→992k hypertokens minted) — at/above the claimed top. The persistent-dictionary variant is the H-C6 dynamic-vocab
question in disguise: ~22% is free, the rest is bought with exactly the machinery H-C6 would have to decide to build.
`validation/exp_c2_lzw.py`.

**H-C3 — Removing the full-vocabulary softmax is the specific mechanism that makes a much larger vocabulary affordable
at the output head.** The general pattern (retrieve top-K via an ANN index, softmax only over those, instead of a
full-vocabulary softmax) is exactly what cubby-lm's `VSABindingHead` cosine-similarity-against-a-frozen-D=10240-codebook
already did structurally, so it's not a foreign idea to graft on — it's an extension of something the predecessor
already had the right shape for. GrillCheese's `gcheese-faiss/` module is a small-scale (sub-100k item) working
existence proof this pattern is production-viable: a hand-written Vulkan GPU reimplementation of brute-force kNN (built
because CUDA faiss-gpu doesn't run on their AMD hardware), running *side by side* with a separate softmax-attention
memory-read path that they are actively optimizing — i.e., a real system where both approaches coexist and get compared.
Status: PLAUSIBLE, and the clearest "this is already almost what we do" finding in the whole survey. Self-reported
performance figures (sub-2ms @ 10k memories, 2-5x GPU speedup) are explicitly flagged by their own `CRITICAL_ISSUES.md`
as unvalidated planning targets, not measurements. **This is no longer a formalize-what-already-exists nice-to-have —
it's the specific mechanism the new, larger vocabulary is being designed around, so it needs validating at the target
vocab scale, not at cubby-lm's old 32k.** **Validate** by building an explicit ANN-retrieval output head (or a
Sylph-style sigmoid-decomposed alternative, per H0) at a representative large-vocab scale and benchmarking
latency/accuracy against a full-softmax baseline at the same scale, not against cubby-lm's existing numbers.

**Validation result (2026-07-23 campaign): mechanism confirmed REQUIRED; bounds measured (CPU, batch=1).** Full-head
cost is linear in V (measured 7.8× for 8× vocab at d=2560; 47.9ms/token at the real 32k×10240 shape). The retrieval
floor (exact scoring over a K=1024 shortlist) is ~flat at 0.2–1.5ms — a **32× win at the real head shape**. Naive IVF
over *random* codewords is the pessimistic bound and it's poor (37–80% top-1 agreement, falling with V) — index quality
on *trained* embeddings is the real engineering question before committing to a specific ANN. The decisive number is
memory: **V=1M fp32 codebook = 10–41GB** (d=2560/10240; the 1M row is explicit linear extrapolation, not measurement) —
at large V the softmax bypass isn't an optimization, it's the only way the head exists. One correction to this
hypothesis's framing: the current `VSABindingHead` does NOT avoid the O(V·D) cost — it's still a full-vocab matmul; it
saves *parameters*, not compute. `validation/exp_c3_latency.py`.

**Order 0083 (2026-11-25, Code B): route 5 in the served stack (`CB_HEAD_SHORTLIST`, home card); killed, the switch
stays off.** On the programs seat it passes every line: 347 of 348 greedy programs identical (99.7%), every answer the
same on 1,200 rows, the live eight the same. On the talk seat (H-E8, talk_v3a) 546 of 594 drafts are identical (91.9%,
under the 99% line); reworded correct goes from 302 to 291, wrong stays 1. The decode step: route 5 20.45 ms, the 32k
head 20.32, the full 128k head 21.60 (p50).

**H-C4 — A trained flat-taxonomy domain classifier is a usable baseline for the "automatic specialization" routing half
of the goal, not the end state.** Source: Novel_GNN_Arch's `capsules/domain_classifier.py` + `domain_head.pt` — a small
(~570K-parameter), actually trained and evaluated 2-layer MLP (frozen sentence embedding → 512 → 88 sigmoid domain
outputs, `scripts/test_domain_head.py` reports accuracy/Hamming-loss/F1, though the achieved numbers weren't reproduced
this session, only the code and eval harness were read). Status: this is the single most mature, lowest-integration-risk
artifact found anywhere across all four surveys — real weights, real eval script, minimal dependencies (`torch`,
`transformers`, `scikit-learn`). But it routes into a fixed flat 88-domain taxonomy via a plain MLP, not a
learned/generative specialist space — it's a stepping-stone baseline for H0's routing half, not the hypernetwork-based
specialization mechanism itself. **Validate** by using it directly as a baseline router to benchmark any new
context-conditioned routing mechanism against — this is close enough to drop-in that skipping it would be wasted effort.

**Validation result (2026-07-23 campaign): VERIFIED and executed end-to-end offline.** `domain_head.pt` loads with
`weights_only=True` (clean tensor dict): exactly **569,944 params**, shapes 1024→512→88 matching the code. The eval
harness ran on CPU against the shipped 3-record fixture (metrics print: exact-match, Hamming, micro/macro F1). Two
precision notes: the module emits **logits** (sigmoid applied downstream; trained with `BCEWithLogitsLoss`), and the
shipped `file_capsules.jsonl` is **0 bytes** — no headline accuracy is reproducible from the repo as shipped; embedding
fresh text needs `BAAI/bge-m3` (1024-dim, not in the local HF cache — would require a download). Baseline: usable as
claimed. Its historical metrics: unverifiable without regenerating a test set.

**Reinforced by H0 part 2 (2026-07-23):** the learned-context experiment (`exp_h0b_learned_context.py`) makes this
artifact's role precise and load-bearing rather than optional. When H0's context is inferred by a router trained
*online*, the router itself catastrophically forgets its own routing (slots collapse 6→3 distinct, task-0 confidence
→37%) — so the context source **must be an offline-pretrained classifier over all domains**, which is exactly what
`domain_head.pt` is (trained once on 88 domains). H-C4 moves from "stepping-stone baseline" to "the specific shape the
routing half needs": upstream, pretrained, frozen at inference.

**H-C5 — Economics claims in cubby-concepts' `budget-hardware/` need scrutiny before use.** `revenues-simulation.txt`
projects a $956M five-year valuation; the ~100–1000x cost-per-query claim lives in
**`simulatorandprofitability.txt:138`** (citation corrected 2026-07-23 — `vs-Kimmy-K3.txt` contains no numeric
multiplier, only "a fraction of the cost"). Status: SPECULATIVE — both are explicitly flagged by the source repo's own
survey as unmeasured narrative, not benchmarked data; no methodology backs either number. **Treat as directional
motivation only.** Do not use these specific figures in any real financial planning until CubbyLLM has an actual
measured cost-per-query from a running prototype.

**Validation result (2026-07-23 campaign): scrutiny performed, verdict unchanged and sharpened.** The 100–1000× figure
rests on unsourced round numbers ($15,000 vs $15.00 per 1M queries; 700W GPU node vs 20W "neuromorphic-assisted" node; a
retail API price treated as competitor TCO) — assertion-only, zero benchmark data or citations. The $956M is
arithmetically self-consistent (12× a self-asserted $79.72M Y5 revenue) but every driver is assumed (TAM, 219% YoY
growth, 8% terminal share, 12× multiple, $0.22/1M-token COGS), and the file carries internal inconsistencies (narrative
"$58.9M EBITDA" vs table 74.93; malformed 9-label/7-column table). Directional motivation only — confirmed.

**H-C6 — CubbyLLM's vocabulary may not need to be a fixed-size discrete set at all.** Added 2026-07-23. zip2zip's
inference-time LZW hypertoken compression and ZeTT's target-vocabulary-agnostic embedding generation both point toward
an open-ended vocabulary — new hypertokens formed and embedded on demand, no fixed upper bound chosen in advance —
rather than simply picking a bigger fixed number than 32k. Status: SPECULATIVE, and genuinely undecided: "much larger"
could mean a large-but-fixed vocabulary (say 250k–1M, sized once, a more conservative extension of what cubby-lm already
did) or an effectively open/dynamic one (no fixed size, tokens/hypertokens minted at inference time, closer to zip2zip's
actual proposal). These have different engineering consequences and different new failure modes (a dynamic vocab risks
unbounded cleanup-memory growth and variable novel-hypertoken quality; a fixed-but-large vocab reintroduces a version of
the static-table cost problem at a bigger constant). **Validate** by deciding this explicitly as a design choice before
implementation begins — it's not a detail to leave implicit, since H-C1's hyper-encoder and H-C3's softmax-bypass
mechanism need to be built differently depending on the answer.

**Validation result (2026-07-23 campaign): decision package assembled.** Every measurement this campaign produced points
the same way: H-G1 (fixed-vocab pipeline proven to 65,536, one order of magnitude of headroom demonstrated), H-C1 (pure
surface-form generation loses +0.50 nats at toy scale; generation *cost* is negligible/cacheable), H-C2 (22% step
reduction free; persistent hypertokens buy 53–67% but require dynamic-vocab machinery), H-C3/H-C7 (direct readout fine
to V=1M; memory is the wall; retrieval head required either way).

**DECIDED (user, 2026-07-23): HYBRID.** Fixed core vocabulary — next step a 128k–256k BPE on the proven H-G1 pipeline —
plus a retrieval output head from day one, with the hypertoken/dynamic tail deferred as a later, separately-gated
addition. H-C1's hyper-encoder and H-C3's softmax-bypass mechanism should be scoped to this shape.

**H-C7 — A much larger vocabulary strains the VSA binding head's codebook capacity and needs its own sizing pass.**
Added 2026-07-23, a direct consequence of combining H-B4's capacity law with this group's new vocab requirement. If the
binding head's codebook doubles as the word-vocabulary representation (as in cubby-lm's D=10240 MAP-bipolar codebook,
per the architecture map), the resonator-network capacity law verified in Group B (P_cap ≈ 0.64·D^(1/(F-1)), empirically
~80–100 distinguishable codewords per factorization slot at D=10000, F=3) means a significantly larger vocabulary may
need a larger D, a re-architected codebook (e.g., compositional/factored token identities instead of one codeword per
token), or a deliberate decision to keep the VSA binding space and the token vocabulary as two different-sized spaces
connected by a projection, rather than one shared space. Status: SPECULATIVE — this hasn't been worked through as its
own design question yet, it's a mathematical consequence of two already-gathered pieces of research meeting for the
first time in this revision. **Validate** by computing the actual required D for the target vocabulary size once one is
chosen (H-C6), against the same capacity formula already verified in cubby-concepts' test suite, before assuming
cubby-lm's D=10240 sizing transfers to a much larger vocab.

**Validation result (2026-07-23 campaign): RESOLVED with numbers — the fear was aimed at the wrong place.** **Direct
one-codeword-per-token readout does NOT strain at D=10240**: max-crosstalk grows only as √(2lnV/D), leaving a +0.50
cosine margin for a cos=0.55-quality query even at V=1M (empirically 100% top-1 at every V tested up to 1M). What DOES
break is **flat factored identity**: per-slot resonator capacity at D=10240 is F=3: 65, F=4: 14 (verified law), so V=1M
at F=3 needs 100/slot (over capacity) and F≥4 is hopeless; single-token factored readout at 1M would need D≥24k (F=3) or
D≥121k (F=4). Chained F=2 hops (H-B4) remain the escape hatch if factored identities are ever wanted. The real
large-vocab pressure is the flat codebook's memory + O(V·D) compute — i.e., H-C3's problem, not binding-algebra
capacity. `validation/exp_c7_capacity.py`.
