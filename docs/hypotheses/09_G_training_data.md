<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 9. Group G — Training data (`$CUBBY_TRAINING_DATA`)

Surveyed 2026-07-23, after the vocab/from-scratch correction above — this folder wasn't part of the original research
corpus; the user pointed to it directly as material for testing and for building CubbyLLM's own training set. It's
substantial: roughly 210GB across 20 top-level files and 16 subdirectories, almost entirely real (not synthetic) data —
logged conversations, instruction data, Gutenberg/Wikipedia/news text, and an already-trained tokenizer history. One
finding that came out of this survey is covered above instead of here, since it's substantively about the binding head:
the `grilly.experimental.vsa.ops` discovery is H-B5. Everything else training-data-specific is below.

**H-G1 — Real tokenizer-scaling history should ground the H-C6 vocab-size decision, not replace it.** `tokenizer/`
(15GB) holds three actually-trained, actually-validated fixed vocabularies, each with a real `report.json`:
`grillcheese_spm20k_v3` (19,947, SentencePiece unigram), `grillcheese_spm32k_v2` (32,000, SentencePiece unigram — this
is the exact tokenizer cubby-lm's frozen trunk uses, trained on a rebalanced corpus with French boosted to 27%, 15.16GB
corpus), and `grillcheese_bbpe65k_v3` (65,536, byte-level BPE, the largest ever built here, trained on a 6.32GB
subsample of a 115GB pool). No open-ended/dynamic tokenizer has ever been built in this pipeline — the only precedent
anywhere in this stack for a very large *discrete* symbol space is the SVC subject space in `svc_targets/` (208,538
distinct subjects), a different subsystem entirely (VSA role-filler targets, not an LM tokenizer). Status: VERIFIED
(real files, real stats, independently sampled this session). This doesn't answer H-C6, but it grounds it: "much larger
than 32k" has a proven, low-risk next step one order of magnitude away (65,536, using tooling that already works),
against the higher-risk, unprecedented open/dynamic path (H-C1, still SPECULATIVE in this project). **Validate** by
costing out a straightforward next fixed-vocab step (e.g. 128k–256k BPE, same pipeline as the 65.5k build) before
committing to the harder dynamic-embedding path — the fixed path has a working reference here; the dynamic path doesn't,
in this folder or anywhere else surveyed.

**Validation result (2026-07-23 campaign): the next fixed-vocab step is COSTED and it's trivially cheap.** Byte-level
BPE (the same HF `tokenizers` machinery the 65k build used) trained on a 200MB slice of real corpus takes ~9.9s at
**both** 65,536 and 131,072 vocab — the **65k→131k cost ratio is 1.00×** (vocab size barely touches BPE training time).
Extrapolated to the production-size ~6.3GB subsample: **~5 minutes single-machine CPU**. So the vocabulary-size lever
was never the expensive part — the real cost/risk is choosing and pinning the corpus mix (H-G3; note the 65k build's own
report shows NSFW-labeled sources contributed ~136k lines to *tokenizer training* itself, not just pretraining).
`validation/exp_g1_bpe_cost.py`.

**H-G2 — `temporal/nyt_data/` removes the biggest practical barrier to building the Zero-Forgetting Stability
Benchmark.** 294 files literally named `<year>_<month>.json`, spanning 1851–2024, each a real NYT Archive-API record set
with explicit ISO `pub_date` timestamps, headlines, sections, and keywords — already bucketed by year and month. A
smaller, coarser `historical/` subdirectory (382MB) has date-*range*-tagged historical events and conflicts with
causal-link annotations, a secondary option. Status: VERIFIED the data exists and is dated as described (independently
sampled this session); PLAUSIBLE, not yet proven, that it's sufficient for the benchmark's actual needs — coverage is
sparse (294 files, not every month present), so density across whatever time range gets chosen needs checking. Note: the
three `temporal_corpus*.txt.meta.json` files at the parent folder's top level describe a further-processed corpus
(assembled from this data plus Gutenberg books via `book_subj_map.json`, up to 3.59GB in the v3 build) that **no longer
exists on disk anywhere in this folder** — only the raw source material survives; whatever consumed those files will
need the corpus regenerated, not just located. **Validate** by building the actual benchmark harness (still the missing
piece — the data existing isn't the same as the test existing) against `nyt_data/`, sorted and sliced by `pub_date`.

**Validation result (2026-07-23 campaign): DONE — the dated harness now exists and ran**
(`validation/exp_g2_nyt_forgetting.py`: 8 year-spread eras 1852→2024, 128 headlines each, keys = sign-bundled word
hypervectors of real headlines, written chronologically through the same four rules as the synthetic benchmark).
Findings: real headline keys are **3.1× more correlated** than random bipolar (mean |cos| 0.0547 vs 0.0176), and that
correlation costs every candidate ~2–3× recall at N=1024 — the honest-regime penalty, now measured. **The synthetic
ranking SURVIVES**: NLMS best overall (18%, clean chronological forgetting curve 0%→61% oldest→newest era), SDM still
the only order-agnostic method (~9–12% flat across 150 years), flat-decay keeps only the newest era (24%), Hopfield
totally collapsed (0%). Whichever ordering survived both regimes is the one to trust — and it's the same ordering.
`validation/logs/exp_g2_nyt_forgetting.log`.

**H-G3 — `unified/` (120GB, the corpus this pipeline currently treats as its best pretraining mix) needs a deliberate
cleanup pass before it becomes CubbyLLM's training set.** Specific, independently-confirmed issues: it mixes explicitly
NSFW-labeled story sources (`bluuwhale_nsfwstory2`, `mickume_alt_nsfw`, ~3.1GB combined) into the general pool with no
visible separation or filter; its composition has visibly drifted since its own tokenizer-training report was generated
(files that report references are no longer present in the current listing) — a reproducibility gap; there's meaningful
cross-directory duplication elsewhere in the corpus (`factual/` and `gbooks/` are both Gutenberg pulls with likely
overlapping IDs; `jsonl/` and `knowledgetxt/` share a byte-identical 4.8GB `wikibooks_corpus.jsonl`; `kbtxt/` has
confirmed duplicate files under different names); and at least one alternative build (`v3_3/`) has a broken 0-byte file
where 500k QA records were expected (`stackexchange_qa.500k.jsonl`). Status: VERIFIED — every issue listed was
independently sampled and confirmed this session, not inferred. This is exactly what the user meant by "create a better
dataset": the raw material for a strong corpus is here, but assembling it as-is would inherit real, fixable problems.
**Validate/apply directly:** before any real training run, decide explicitly whether NSFW-labeled sources are in scope
at all and if so how they're tagged and gated rather than silently blended in; run an ID-overlap check between
`factual/` and `gbooks/`; and pin/snapshot whatever mixture actually gets used, since `unified/`'s current drift makes
past results hard to reproduce.

**Validation result (2026-07-23 campaign): the named ID-overlap check is DONE and the fear was mostly unfounded on this
axis** — `factual/` (1,976 book txts, 0.90GB) vs `gbooks/` (3,612, 1.87GB) share only **9 Gutenberg IDs (0.5% of
factual/, ~0.00GB)**; the two pulls are effectively disjoint (`validation/logs/exp_g3_overlap.log`, name+size comparison
— no content hashing of the 120GB). The OTHER confirmed issues (byte-identical 4.8GB `wikibooks_corpus.jsonl` in two
places, `kbtxt/` dups, NSFW blending, report drift, the 0-byte `stackexchange_qa.500k.jsonl`) stand as found by the
survey and still need the cleanup/pinning pass before training.

**Validation result (2026-08-14): H-G3 CLOSED — the 2026-07-30 cache build already was the cleanup pass; verified
against disk, with one stale pin fixed and one accepted overlap documented.** Every gate checked against the real
artifacts (`$CUBBY_DATASETS\_pipeline\corpus_sources.json` + `$CUBBY_TRAINING_DATA\token_cache`): **(1) NSFW scope:
excluded structurally** — the `unified` source is an explicit 11-file allowlist (C4-realnewslike, twteval, infini-news,
orca-agentinstruct, nemotron-CC math/HQ/personas, factnet, ultradata-math ×3); `bluuwhale_nsfwstory2`/`mickume_alt_nsfw`
are absent from the config, not merely filtered. **(2) Dedup gates: resolved** — wikibooks consumed as ONE cleaned copy
(`wikibooks_clean.jsonl`, neither raw duplicate path); `v3_3/` used only for `fineweb_edu.2m.jsonl` (the 0-byte
stackexchange file unreferenced); `gbooks/`+`factual/` deferred out of pretrain entirely; open_thoughts reserved for
SFT. **(3) Reproducibility: all 13 stored `.sig` signatures byte-identical to a live recompute** — nothing has moved or
changed since build. **(4) The pin was stale and is fixed**: the doc's `5e7f8b73…` predated the same-day quarantine
re-tokenization; the live `WeightedCorpusPipeline.manifest_hash()` over the real config is
**`e03c991e2728759d25c67d285943baf44042dbebb4cdc12d72a7b39e76d2835e`**, now re-pinned in `corpus_manifest.md`, whose §0
table also carried four pre-quarantine token counts (books/verified_facts/kbtxt/wikibooks, +239M total) — corrected to
disk-measured values. **(5) The true cache size is 17,744,604,983 tokens (17.74B), not the "14.37B" this project's
session memory carried** — every budget derived from 14.37e9 scales ×1.23 at one epoch (H100 ~101 hrs $203–304) and the
Chinchilla-2B target is ~40B tokens ≈ 2.25 epochs (H100 ~228 hrs $457–685; B200 ~100 hrs $451–651; A100 ~724 hrs) —
still well under the panel's estimate, which itself assumed only ~14B tokens. **(6) One accepted overlap documented, not
hidden**: `wiki_full` (3.04B) remains in the cache despite the manifest's own "DROP — superseded" dedup note; the routed
`wikipedia` (125M) overlaps it by ≤0.7% of the cache — accepted for this build, revisit only on a rebuild. **The
pretraining gate is OPEN**: config-pinned, hash-verified, scope-explicit.

**H-G4 — `socratic_test_text.jsonl` is a ready-made general-capability regression check, not a forgetting test itself,
and can be used immediately.** Confirmed to be the GSM8K test split (1,319 grade-school math word problems,
chain-of-thought format, `#### <answer>`-terminated) sitting directly in the training-data folder despite being an eval
set. Status: VERIFIED. Useful run alongside the Zero-Forgetting benchmark (H-A3/H-A4/H-G2) specifically to distinguish
"forgot a learned fact" from "general reasoning capability regressed" — a narrow forgetting probe alone wouldn't catch
the second failure mode. Nothing to validate here — this is the one finding in the whole survey that's already exactly
the shape it needs to be.

**Also confirmed:** `labeling/svc_domain_labels.json` uses the identical 88-category flat domain taxonomy already found
independently in AURA_GENESIS and Novel_GNN_Arch (H-C4's context) — a third independent sighting, reinforcing that this
is a real, deliberately shared convention across this user's projects rather than coincidence. One hygiene issue worth
fixing if this taxonomy gets reused: 16 categories are listed twice, once capitalized (e.g. `History`, index 38) and
once lowercase (`history`, index 82), fragmenting what should be one category into two valid-but-separate codes.

**H-G5 — Synthetic domain-tagged Q&A distilled from the domain-sorted corpus is the SFT-phase instruction data, and its
domain tags feed the router — a later phase, not a pretraining input. Added 2026-07-25.** Once the corpus is cleaned and
domain-partitioned (the book pipeline → `$CUBBY_DATASETS\domains\<domain>\`, and the facts pipeline →
`domains/verified_facts/<era>/<region>/<topic>/`, both built this session), a frontier model reached via OpenRouter can
generate question/answer (and instruction/conversation) pairs *from* that content — distillation from a stronger model,
not the model training on its own output. The CubbyLLM-specific reason to do it deliberately rather than grab a generic
instruction set: **tag every generated pair with its source domain.** That both populates the functional domains the
taxonomy already reserves (`QA`, `instruction`, `conversation`) *and* gives the θ=f(c) router (H-C4 / H0) exactly the
per-domain instruction data that trains specialization — generic untagged Q&A would not. The tooling exists
(Novel_GNN_Arch `capsules/openrouter_encoder.py`, `file_topics_openrouter2.csv`). Status: **SPECULATIVE** — nothing
built or measured, and explicitly a *later-phase* item, for two sequencing reasons: (a) instruction-tuning data is
consumed *after* a base model is pretrained on raw text, so this neither blocks nor is blocked by pretraining; (b) at TB
scale (the facts corpus alone is headed there per H-G3 / the facts pass) exhaustively Q&A-ing every source is a large,
real OpenRouter spend — worth doing once, on a *pinned* corpus (H-G3), not on a still-converting one. It augments SFT;
it does not replace the raw domain text pretraining actually learns from. **Validate** by prototyping the generator on a
small per-domain sample (~5 sources/domain): lock the output schema and a quality bar (answer grounded in the source
passage; valid format; domain-appropriate), and measure per-domain generation cost — cheap, and it de-risks the bulk run
without committing the spend. Defer bulk generation to the SFT phase. **Kill criterion:** if the generated pairs don't
measurably beat the instruction data already in hand (`instruct_55k_clean.jsonl` etc.) at SFT or at improving domain
routing, or if per-domain cost at corpus scale is prohibitive relative to that gain, don't do the bulk spend — fall back
to existing instruction sets plus targeted generation only for under-covered domains.

**H-G6 — `AutomatedScientist/wikikg-trajectories` (MIT) is the real fact store AND the chain/tool-calling corpus:
Wikipedia already in the template-fact shape the planner, the triple index and the VM consume. Added 2026-09-04 (owner:
"gold, Wikipedia already formatted for the VM").** 365,923 (subject, relation, object) triples from 45,416 Wikipedia
articles, 1.5M two-hop random-walk paths with directions, and 2M tool-calling trajectories whose two tools —
`query_relations(subject, rel_type)` and `get_neighbors(entity, direction)` — are exactly `TripleIndex.hop` /
`by_subject` / `by_object`, i.e. the store's own operations. Claim: verbalized as "O is the R of S" template facts it
becomes the serve store (the live store today is the harvest sample's fact pool — no "capital of France"), the chain
family regenerated on real entities (the harvest corpus is typo'd Wikidata: "Huamn"), and the tool-calling family
grounded in the store. Kill criterion: fewer than 80% of triples verbalize into facts `parse_fact` reads back, or the
index at full size costs more than the serve host can hold, or VM-verified chains over it fall below the harvest's
precision.

**Validation result (2026-09-04 screen, `validation/exp_g4_wikikg_screen.py`, log
`validation/logs/exp_g4_wikikg_screen.log`): the store half is VERIFIED, the question half needs two maps.** 359,385
template facts from 365,923 triples (dropped 3,967 self-loops/empties), `parse_fact` round-trip 100% on 20k;
**`TripleIndex` over all of them builds in 16.5 s at 284 MB peak** — a cosine `FactStore` would need 14.7 GB of float32
rows, so the wiki world is lookup + a lexical fallback (the pure-python IDF overlap re-tokenizes the store per query: an
inverted index is a prerequisite). Relations: **2,902 types**, not the card's 180+ — X_OF forms 27.7% of triples
(direct), a 41-entry noun map 45.0%, a generic lowercase phrase 31.2% (HAS_PART 32k, IS_A 13.7k, CREATED_BY 7.4k…) → a
noun map for the top ~100 relations first. Paths → grammar questions: **13.6%** of two-hop paths are direct without an
inverse-relation map (17,272 of 25,194 sampled hops need one; every direct path parses). VM sample 300 (lookup walk, no
search, real cubelang): 130 verified = the walk's end, **162 verified with a different valid end** — the graph is
many-valued (TIMELINE_EVENT alone is 21% of triples), so a random-walk end is not a gold answer: chain questions come
from functional hops (unique lookups) or are set-valued; 8 exhausted (the one-candidate walk's frontier case). No
precision problem: every verified chain is two real triples. Build order in `TODO.md`.

**Validation result (2026-09-04, same day, `validation/exp_g4b_wiki_world.py`, log
`validation/logs/exp_g4b_wiki_world.log`): the two maps built and the world SERVES.** `standin/data/wikikg.py`: a
163-entry relation map (noun, inverse noun, reading direction) covering 96.6% of triples, inverse readings for 56.3% →
**552,297 template facts** (356,902 forward + 195,395 inverse), `TripleIndex` build 46 s / 613 MB peak; `FactStore`'s
lexical fallback is now an inverted index (6 ms/query at this size, near-universal tokens skipped past 10k facts);
`serve.py --wiki` / `serve_api.py --wiki` mount it, and an exact hop-0 hit in any world's index routes the question
there. Chain questions from FUNCTIONAL hops: **12.5% of random-walk paths** (24,965 of 200k), **VM sample 300/300
verified to the one gold answer** (87 ms/question) — the chain family's real-entity corpus exists. The honest bound:
**SimpleQA reachability 19.9% entity-known, 0.1% lookup-reachable** (3 of 4,326) — the graph is 45k articles; SimpleQA
is the search-and-learn loop's number, not the store's. Status: store VERIFIED at serve scale; questions VERIFIED (VM);
coverage BOUNDED.
