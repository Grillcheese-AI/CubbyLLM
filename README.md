# CubbyLLM

CubbyLLM is **not a small language model**. It is an eight-layer infrastructure in which the language model is
deliberately the smallest and most replaceable layer: **compute** (`grilly`, Vulkan, any GPU, no CUDA) →
**representation** (one VSA algebra, sparse block codes, text encoded with no neural model in the path) →
**the VM** (CubeLang: Rust, deny-by-default, verify-before-execute) → **the symbolic boundary** (only
`(symbol, similarity)` crosses it, never raw hypervectors) → **worlds / knowledge** (MoWM; `FactStore` +
`TripleIndex` is the serving form) → **reasoning** (the verified CoT pipeline) → **the brain / host**
(neurochemistry → route → cortices; every spoken word exits through a VM-mediated ASK) → **the trunk**, the LM,
which proposes programs and speaks and never judges. Every spoken answer is a program the VM ran against a fact
store, or a refusal with a reason. Honest by construction rather than by judgement; one stack for cloud API and
enterprise on-prem.

> **Status caveat, first.** The serving stack today runs a third-party GGUF model (LFM2.5-2.6B, fine-tuned on
> Colab) as a **stand-in** behind the trunk interface. GRL's own 2B trunk is designed and validated but not yet
> trained. Every number below measures the *host architecture* — the gates, the VM, the harvest loop — never the
> CubbyLLM model. The trunk drops in behind the same `Emitter` interface unchanged.

## The seven invariants

1. **The VM is the only truth gate.** No LLM-as-judge anywhere in the loop; the model never scores its own thoughts.
2. **The model proposes, the host disposes.** Identity, provenance, execution, reward, retirement, and what the model is shown are host decisions.
3. **The don't-know contract.** An unverified claim is never spoken.
4. **Retire, never delete.** Every record carries a store-snapshot hash and a git revision.
5. **Serve-time budget.** A small model on a 12 GB GPU; VM calls in milliseconds; training on one 80 GB GPU for tens of minutes per round.
6. **No self-play or self-judging shortcuts.**
7. **Ported, never linked.** Nothing depends on another repository at runtime.

## Where it stands — 11 September 2026

The reasoning pipeline (`cubbyllm/reasoning/`) is wired end to end and measured on two worlds:

```
question ──► plan ──► plan_verify (disposer) ──► lookup-first walk ──► CotChain program ──► CubeLang VM ──► answer / refusal
          └─ a refusal the store caused ──► source ──► gate ──► store (+provenance) ──► walk again   (learn.py, lever 3)
              │              │                        │                                        │
     grammar or emitter   relations known?     TripleIndex, exact hop 0,          per-hop similarity ≥ frame-size floor,
     (CotPlan: SEED,      plan covers the      paraphrase tier on hops ≥ 1        control role below the floor; verdict
      HOP1..HOPk)         question?            (Jaccard ≥ 0.6)                    over a resident process, ~2 ms/question
```

| measured on the 800-question eval (1,241 facts, 165 relations) | |
|---|---:|
| verified coverage, lookup-first + disposer | **0.718** (574/800) at **precision 1.000** — every wrong answer the pipeline ever spoke here was a pick between several true facts; an ambiguous hop is now a refusal (lever 3) |
| honest refusals with a named reason | 226 (152 no-such-edge or plan-coverage · 28 no seed fact · 9 ambiguous, candidates named · 37 unparseable) |
| misparsed plans the disposer stops before any walk | 90 / 92 |
| VM-verified 3-hop answer, wall, resident VM | **2.0 ms** (65.7 ms per-process; CoT is now faster than chase-only) |
| gen-2 emitter (`emitter_v12e`) vs gen 1 on the 79 held-out B questions | 10 vs 5 accepted + gold hop; 4/4 verified correct, 0 wrong — bar met |
| gen 2 outside the grammar, arms B + C (exp_r7 --exclude) | **18/116** by emitter plan alone (B 7/79, C 11/37) after coverage levers 1–3, 0 wrong; 14 more C composed through the grammar (GoT-1 slice, exp_r8), 0 wrong |

| measured on the wiki world (552,297 facts, 2,967 relations — the emitter has seen 165) | |
|---|---:|
| 300 canonical two-hop chains (exp_g4b) | 300/300 VM-verified |
| matched pairs, 200 chains × 4 surface forms (exp_r9) — out of basin | grammar **0** correct · emitter **228** correct · **0 wrong** on every form |
| canonical / have / relative / possessive, emitter | 110 / 80 / 127 / 21 of 200, 0 wrong; the possessive residue is the emitter folding a hop into the seed, refused correctly |
| canonical form | grammar 189/200 · emitter 110/200 · 0 wrong |
| SimpleQA, 4,326 free-text questions through the whole gate (exp_r10) | 3,858 plans → 3,825 refused with a reason + 33 walk failures; 0 verified, **0 wrong** (the store holds 3 of the answers) |
| search-and-learn (exp_r11): held-out facts, then Wikidata on 600 SimpleQA | held-out: 200 refused → 195 learned and verified, 0 wrong; poisoned source, either order: ambiguous refusals, 0 wrong. Wikidata: 1,715 facts learned with provenance; with the source resolving relation words to its labels (`born` → `date of birth`, lever 4) **4 VM-verified, 3 correct, 0 false facts** (the 4th: Wikidata's transliteration and granularity vs the gold's). A WordNet+WOLF synonym oracle (`reasoning/lexicon.py`, lever 5) adds the bilingual axis — a French question verifies against an English store — and changes nothing on English factoids. With the typed answer class, the near rule and the local property table (12,684 Wikidata properties, EN+FR labels, aliases and datatypes, built once): **6 verified, 5 correct, 1 near, 0 wrong, 0 API calls** on the same 600 — the wording step (`born` → `date of birth`, a two-sense wording narrowed by the question's ask type and the property's datatype) never touches the network; 845 s against 1,395 |
| the offline Source (`standin/wikitext.py`): regex frames over 4.7M local article openings, provenance per sentence, EN + FR | same 600 SimpleQA: 9 facts admitted, all re-read true against their sentences; **2 verified, 2 correct, 0 wrong**, 0 API calls. Run 1 stored one false fact (`British is the author of The Roar`) — two source hazards, fixed and pinned. Half the API source's verified count, offline, with the sentence on record |
| the ceiling probe: a frontier model (Gemini 3.8 Flash) in the emitter's seat, same disposer / walk / VM / kill line (`standin/openrouter.py`; the serving model never calls it) | 600 questions, $0.27: the model declines 403 outright, plans 195; **4 verified, 3 correct, 0 false facts** vs the stand-in emitter's 6 / 5 / 0 the same day. Run 2 spoke one wrong answer — the source picked among several items labelled `James Young`; an entity label shared by two items is now a refusal, never a pick (pinned). What the probe exposed is the host's: lever 6 (`learn.resolve_wordings`, the question's own word for a canonical label, `founded` → `inception`) and a coverage rule that is wording-bound and vocabulary-sensitive. The ceiling on free text is not the proposer |
| the hippocampal side cortex (`reasoning/hippocampus.py`, exp_r13): certified chains as episodes, 256-bit codes, recalled as plan candidates through the same gate | matched pairs, 100 seen chains × 3 out-of-basin forms: **272/300 correct, 0 wrong** (possessive 90/100 — the emitter's 21/200 hole was folding a hop the memory has no reason to fold); 100 unseen chains: 7 verified by rebinding a remembered shape to the new entity, 0 wrong (ceiling 11); 1,200 VM walks, no model in the loop, 37 s. Memory proposes, the host disposes, the VM decides; consolidation retires by utility and deletes nothing |
| the encyclopedia Source (`standin/encyclopedia.py`, exp_r16): 29 OCR'd volumes of a 2005 encyclopedia, entries by headword, frames over each entry's opening, provenance = volume + headword + sentence; no model, no network | volume 13: 718 entries, 204 persons, **577 facts** in 16 s; cross-checked against Wikidata where the name resolves to exactly one item: years **121/125 agree** (the 4 disagreements: a calendar-style year, one entity the search resolved to the father, one genuine), places 58 agree + 6 spellings + 13 the same place at another granularity or name + 2 genuine disagreements; 50 hand-read: 49 true, 1 partial place (fixed, pinned). The OCR's hazards — line-break hyphens, `bom`, captions and bylines run into names and places, a missed headword handing the next entry's birth sentence to the previous person — each a rule, each pinned. All 29 volumes: 22,213 entries, 3,601 persons, **10,790 facts in 16 s**; 60 hand-read: 57 true, 3 the rules above (fixed, pinned) |
| the gen-3 builder (`standin/data/build_gen3.py`): certified chains in, free-text questions out, the plan the host's — each (question, plan) certified by the serving gate itself before it becomes a record; no LLM | 16,626 chains × 50 wordings: **63,138 questions, 52,538 certified (83.2%)**, 105k records in 1,050 s. The first probe certified 49.6%, and its refusals were the coverage rule's own measurement on free text — five host rules later (inflections in the property table; lever 4 through the property's every wording, both ways; a *place* ask type; `covers()` v5, the mixed reading; the question decides an unheld tail's split) coverage refusals went 1,476 → 0. On the held split (entities no record trains on), the gen-2 emitter — which never saw a free-text wording — plans 599/600 and verifies **511 correct, 0 wrong** once lever 7 snaps its garbled seeds to the question's spelling; the 82 it still misses read the ask words as relations, which is what gen 3 trains |
| hdc through the gate (`validation/exp_r18_hdc_rephrased.py`): Wiki5M 1/2/3-hop questions, each in its template wording and an LLM's rephrasing, with the chain's facts — a matched-pair set at 10,000 with nothing missing from the store | 8,840 chains × 2 wordings, **0 wrong** on every row scored. The probe found three gate defects, each pinned: a relation ending in `of` (`instance of`) split at the wrong ` of `; the walk's overlap tier read an *inverse* relation where the table's exact alias named the right one (lever 4 now runs ahead of the walk); and the **split wording** (`covers()` v6: `Which country does X hold citizenship in?` covers *country of citizenship*, word by word, the residual rule untouched). Rephrased one-hop certified went **31.5% → 87.1%**, two-hop 27.8% → 55.1%, three-hop 11.0% → 35.1% (265 rephrasings that dropped the chain's last hop counted, not scored; 0 wrong of 17,415). The battery caught the place ask over-reaching twice: 2 wrong on exp_r9 (a class noun followed by `of` is a relation) and 1 on the emitter arm (`In which country is the location of formation of X?` — the asked class noun is a hop when the store holds it as a relation, exactly); the emitter arm's 11 wrong on its first run all became rules or exclusions — **0 wrong of 1,180** on the rerun. 26,216 gen-3 records with two- and three-hop free-text wordings; exp_r17's bar moved 511 → **531 at 0 wrong**, exp_r9 228/0 and exp_r11 7/5/2/0 unchanged |
| the loop, live (`standin/ask.py`, `serve_api --ask`, the panel's ask box): one natural question at a time, every step streamed to the control panel as it happens, links drawn between questions that share a fact or an entity | The first evening found **one wrong answer spoken** — `Who is the mother of Justin Trudeau?` → Pierre Trudeau, verified: the synonym oracle's *verb* sense of "mother" (beget, sire, father) reached lever 4 behind a property the table knew and the store lacked. Two rules, pinned: the first resolver that names a wording decides (a known, unheld relation is fetched, never paraphrased), and the lexicon words relations with noun senses only. Entity resolution went from "a shared label is refused" (which refused Marie Curie — a physicist, a book, a metro station, a ferry) to three deterministic tiers: the item a served claim pointed at, one exact hit, the question's next hop then label over alias; Bill Haslam's father's given name and Marie Curie's birthplace verify. The gate's snapshot hashed 552k texts per fact (0.35 s) — a running set hash now. exp_r11 on the same 600 SimpleQA with all of it in: **6 / 5 / 1 / 0** (one near traded for an mbiguous_relation refusal; 2,406 facts admitted against 1,328 as shared names now resolve) |
| LFM2.5-2.6B as a fact Source (`standin/lfm_source.py`, the **latent** tier: a model's fact is stored with provenance and never spoken until a second source attests) | Same 600 SimpleQA: 450 facts admitted latent, 0 verified, **7 answers held — all 7 wrong**; against Wikidata's cached facts, **agree 1, disagree 28** (dates of birth 0 / 22). The base model's parametric facts are nearly always wrong on this tail; the latent tier is what made that safe to measure — nothing it said was spoken |
| one VM per user — what isolation costs (`validation/exp_r20_vm_per_user.py`): a resident CubeLang process per session, so no user's frame, knowledge path or program can reach another's | 256 sessions on one machine: **1.69 MB per user marginal** (592 users per GB; summed RSS says 6.8 MB each — the executable's pages counted once per process and mapped once by the OS, a 4× over-count that would become a wrong capacity plan), **19 ms** median spawn + first call, **5,211 programs/s** from 8 threads (3.62× the 1,438/s serial baseline), p99 2.77 ms, 506 MB released on close. Flat from 16 to 256 sessions. Isolation is not the expensive part of serving — the shared emitter is |
| "Who is X?" — the retrieval program (`standin/ask.py`: SEED + ASK, no hop; the host runs it, the VM recovers each fact, the facts are the reply; a mounted talk adapter paraphrases them and the paraphrase is kept only when every name and number in it is in the facts) | *who is marie curie?* → nine facts bound, nine recovered by the VM, the grounded paraphrase spoken; *where is Lévis?* → country, administrative entity, waters, inception; *what is Ottawa?* → `ambiguous_entity` with the four Ottawas named. `standin/data/build_profile_sft.py`: **30,964 (question, program) records** so gen 3 writes the program itself (who 12,000 · what 12,000 · where 6,964) |

The reading: the grammar gets the templates; the emitter gets the shapes the grammar cannot parse; the disposer
and the VM keep the wrong count at zero on both; free text against a store that cannot answer it is refused, not
guessed. The loop closes: emitted plans the VM verified (`cot_harvest_r7*.jsonl`) were the first training records
that did not come from the grammar, and gen 2 was trained on them.

Full record: `docs/research/2026-09-11-plan-verify.md` (the disposer, the resident VM, the emitter's plans walked,
arm C, gen 2, matched pairs), with `2026-09-11-entry-diagnosis.md` and `2026-09-11-hop0-search-is-dead-code.md`.

## Run it

Needs a built cubelang binary (`cargo build --release` in the sibling repo; found via `--exe`, `$CUBELANG_EXE`,
or a `cubelang/` checkout beside this one) and a Python environment with the package deps. The GGUF stand-ins
live in `standin/models/` (not tracked).

```
python -m pytest validation/test_plan_verify.py validation/test_plan_verify_vm.py validation/test_pipeline_plan_refusal.py -q

python validation/exp_m3_cot_pipeline.py --lookup --verify-plan vm --resident     # the harvest, plan-verified, resident VM
python validation/exp_r6_plan_verify.py --real --vm                              # disposer: host verdict == VM verdict
python validation/exp_r7_emitter_planned_walk.py --exclude                       # the emitter's plans, walked and verified
python validation/exp_r8_decomposition.py                                        # GoT-1 slice on arm C
python validation/exp_r9_matched_pairs.py --n 200 --resident                     # generality: 4 forms on the wiki world
python validation/exp_r10_simpleqa.py --resident                                 # free text through the whole gate

python standin/serve.py                                                          # the stand-in brain (see standin/README.md)
```

Every experiment writes `validation/logs/<name>.{json,log}`; every number in the docs points at one of them.
Training notebooks (Colab, one 80 GB GPU, tens of minutes): `notebooks/standin_gen2_emitter_sft.ipynb` is the
current emitter recipe; `standin/data/build_gen2_partition.py` builds its data from the harvest.

## Layout

```
cubbyllm/            the package (Apache-2.0). `import cubbyllm` is torch-free.
  reasoning/         planner (the grammar), plan_verify (the disposer), retriever, index (TripleIndex),
                     programs (CotChain emission), pipeline (answer(): plan → verify → walk → VM)
  bridges/           cubelang_client (subprocess + resident CubelangSession, protobuf framing),
                     programs/*.cube (plan_verify, reasoning_bridge), world_model (the MoWM bridge contract)
  reasoning/learn.py search-and-learn: refusal → source → gate → store with provenance → walk again (sources live in standin/)
  reasoning/lexicon.py the synonym oracle: WordNet 3.0 + WOLF (FR) by synset, a second relation resolver (data built by standin/data/build_lexicon.py)
  reasoning/hippocampus.py the episodic side cortex: certified chains as 256-bit-coded episodes, recalled and rebound as plan candidates; retire, never delete
  reasoning/events.py every step of the loop as an event with its parent's id (question → plan → walk → hop → fact; fetch → gate; answer) — sinks: jsonl, memory, a live listener
  core/ model/ ops/ training/   the trunk design from the validation campaign
standin/             the serve stack (BSL-1.1): brain, thalamus, VM-mediated chat, cubby-man, emitter, SFT data builders
  sources.py         the Wikidata Source (API, cached; a name shared by items resolves by the claim that named it, by the hop the question needs, by label over alias -- else refused, never picked)
  ask.py             the verified-program loop on one natural question, as a service (serve_api --ask: POST /ask, live in /panel); the profile ask (who / what / where is X): the retrieval program, the VM recovers each fact, a grounded paraphrase
  data/build_profile_sft.py  the retrieval shape as training records: (question, SEED + ASK program) over the wiki world's entities, typed who / what / where
  wikitext.py        the offline Source: regex frames over local article openings, EN + FR, provenance per sentence
  encyclopedia.py    the encyclopedia Source: OCR'd volumes, entries by headword, frames over each entry's opening, provenance = volume + headword + sentence
  data/build_gen3.py the gen-3 builder: certified chains -> free-text wordings -> the host's plan, each certified by the serving gate; merges into emitter_sft_v13e
  lfm_source.py      the local base model (LFM2.5-2.6B) as a fact Source: few-shot, parsed, gated, provenance 'lfm (latent)' -- never spoken until a second source agrees
  openrouter.py      a frontier model as a proposer for probes and dataset building only — never the serving model
dashboard/           the three.js control panel (control_panel.html): every step of the loop (cubbyllm/reasoning/events.py) as a linked
                     graph, replay, a live stream from the stand-in server (/panel, /loop/stream), an ask box, and links between questions that share a fact or an entity; sample_events.jsonl to load
validation/          55 standalone experiment scripts + tests + logs/. Never imported by cubbyllm/.
notebooks/           Colab training runs, one per SFT round
docs/research/       dated findings and the outside-model competitions, scored against the measured record
```

Related repos, each ported from rather than linked: `cubelang` (the VM, Rust, BSL-1.1), `mowm` (Mixture of
World Models — implements the `WorldModelBridge`; the disposer's relation oracle is injectable so MoWM can be
the "possible edge" source), `grilly` (Vulkan compute and the VSA substrate), `cubemind` (the environment).

## Licensing

`cubbyllm/` — the trunk, the reasoning pipeline, the bridges — is **Apache-2.0** (`LICENSE`). `standin/` — the
serve stack — is **Business Source License 1.1** (`standin/LICENSE`, change date 2030-09-11 → Apache-2.0).
`cubelang` carries its own BSL-1.1. Licensing questions: licensing@grillcheese.ai.

## How it got here

**The validation campaign (2026-07-23).** `VALIDATION_REPORT.md` resolved the three structural decisions the
trunk is built around, all tracing back to one bet (`CUBBYLLM_HYPOTHESES.md`, Section 2): a model's active
weights as a function of context rather than a fixed stored state. The **VSA binding head** standardizes on
`grilly`'s `BlockCodeOps` (NVSA sparse block codes) — cubby-lm's head had no unbind at all. The **Hebbian memory
layer** forgot catastrophically under sequential learning (94% measured); the fix is context-conditioned
parameter generation, hardened, with a router pretrained offline and frozen — a Zero-Forgetting Stability
Benchmark now exists and ran on four candidates and on real correlated keys. The **vocabulary** is hybrid: a
128k–256k BPE core plus a retrieval output head from day one, hypertokens deferred. The campaign's caveat stands:
the bet only works hardened against its own drift, and context *inference* is the harder half.

**The stand-in serve stack (2026-08-30 → 2026-09-04).** While the trunk waits for its corpus, a fine-tuned open
model stands in behind the trunk interface so the host could be built and measured: a brain (neurochemistry →
route → thalamus → cortices), every spoken line through a CubeLang VM program, live learning of facts, a 3D
Pac-Man world (cubby-man) in which the model learns the maze from VM refusals, eleven SFT rounds whose
VM-verified reads drove the design to **two adapters on one base** — the program emitter (`e`) and the talk
cortex (`t`), never mixed — plus the signed certificate ledger and the encrypted vault. `standin/README.md` is
the record.

**The reasoning stack (2026-09-03 → 2026-09-11).** The GoT challenge (`docs/research/2026-09-03-got-challenge-scored.md`)
and the rung-1 generality-gate competition (`2026-09-10-rung1-generality-gate-competition.md`) put five outside
models against the measured record; their pre-checks reshaped the roadmap. What followed, each with a before/after
on the same store and seeds: the lookup-first walk over a TripleIndex; the entry diagnosis (absent facts 42%,
relation mismatch 58% of the residue); the finding that the plan never crossed the gate and the disposer that
closed it; the resident VM; the emitter's plans walked and harvested; gen 2 trained on the harvest and gated;
the matched-pair generality test on a world 445× the training store.

## Documents in this folder

`CLAUDE.md` — orientation for coding sessions: the from-scratch/no-frozen-trunk correction, the implemented
state, the sibling-repo anti-patterns to avoid.

`CUBBYLLM_HYPOTHESES.md` — every research claim as a falsifiable hypothesis with a kill criterion; most carry a
dated result. Read this first for the trunk. `VALIDATION_REPORT.md` — the campaign behind it, every number linked
to a script in `validation/` and a log in `validation/logs/`.

`docs/research/` — the dated findings (`2026-09-11-plan-verify.md` is the current one) and the outside-model
competitions with their scoring. `docs/ARCHITECTURE_VISION.md` and `VISION.md` — the north star: Cubby (the trunk)
+ CubeLang (the verified VM, an OS for AI) + cubemind (the environment); deny-by-default; the Brain-SDK contracts.

`PACKAGE_LAYOUT.md` — the target package layout, written before code could accumulate one by accident.
`docs/superpowers/` — the implemented package's design spec and plan. `TODO.md` — the working checklist.
`CUBEMIND_CLEANUP_PLAN.md` and `cubby-model-environment-map.md` — the verified state of the sibling repos this
project ports from.

## Training data and source repos

The training-data archive (~210 GB, surveyed 2026-07-23, kept off-repo): a real tokenizer history (19,947 →
32,000 → 65,536, costed to 131,072), a dated NYT archive (1851–2024) used for the forgetting benchmark on real
keys, and a 120 GB candidate pretraining corpus that still needs its dedup/content-filter pass — the one
substantial piece of Group G not done. See `CUBBYLLM_HYPOTHESES.md`, Group G.

`cubby-lm` and `cubemind` are the direct predecessors; CubbyLLM is a fresh design, not a fork — the package layout
mirrors cubemind's documented intent, the model architecture and vocabulary are new. `cubby-concepts`, `grilly`,
and three earlier sibling projects (AURA_GENESIS, GrillCheese, Novel_GNN_Arch) were surveyed for reusable
concepts; what each contributed is cited in `CUBBYLLM_HYPOTHESES.md`.
