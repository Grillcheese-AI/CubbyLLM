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
              │              │                        │                                        │
     grammar or emitter   relations known?     TripleIndex, exact hop 0,          per-hop similarity ≥ frame-size floor,
     (CotPlan: SEED,      plan covers the      paraphrase tier on hops ≥ 1        control role below the floor; verdict
      HOP1..HOPk)         question?            (Jaccard ≥ 0.6)                    over a resident process, ~2 ms/question
```

| measured on the 800-question eval (1,241 facts, 165 relations) | |
|---|---:|
| verified coverage, lookup-first + disposer | **0.710** (568/800), precision 0.993, 0 lost to the disposer |
| honest refusals with a named reason | 232 (155 no-such-edge · 40 no seed fact · 37 unparseable) |
| misparsed plans the disposer stops before any walk | 90 / 92 |
| VM-verified 3-hop answer, wall, resident VM | **2.0 ms** (65.7 ms per-process; CoT is now faster than chase-only) |
| gen-2 emitter (`emitter_v12e`) vs gen 1 on the 79 held-out B questions | 10 vs 5 accepted + gold hop; 4/4 verified correct, 0 wrong — bar met |
| arm C, the 37 the grammar declared unparseable | 17 verified (3 flat + 14 composed, GoT-1 slice), 0 wrong |

| measured on the wiki world (552,297 facts, 2,967 relations — the emitter has seen 165) | |
|---|---:|
| 300 canonical two-hop chains (exp_g4b) | 300/300 VM-verified |
| matched pairs, 200 chains × 4 surface forms (exp_r9) — out of basin | grammar **0** correct · emitter **201** correct · **0 wrong** on every form |
| canonical form | grammar 189/200 · emitter 107/200 · 0 wrong |
| SimpleQA, 4,326 free-text questions through the whole gate (exp_r10) | smoke n=60: 53 plans → 53 refused, 0 verified, 0 wrong; full run in progress |

The reading: the grammar gets the templates; the emitter gets the shapes the grammar cannot parse; the disposer
and the VM keep the wrong count at zero on both; free text against a store that cannot answer it is refused, not
guessed. The loop closes: emitted plans the VM verified (`cot_harvest_r7*.jsonl`) were the first training records
that did not come from the grammar, and gen 2 was trained on them.

Full record: `docs/research/2026-09-11-plan-verify.md` (the disposer, the resident VM, the emitter's plans walked,
arm C, gen 2, matched pairs), with `2026-09-11-entry-diagnosis.md` and `2026-09-11-hop0-search-is-dead-code.md`.

## Run it

Needs a built [cubelang](../cubelang) binary (`cargo build --release` in the sibling repo; found via `--exe`,
`$CUBELANG_EXE`, or `../cubelang/target/release/`) and a Python environment with the package deps (`.venv-dml`
on the dev box). The GGUF stand-ins live in `standin/models/` (not tracked).

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
  core/ model/ ops/ training/   the trunk design from the validation campaign
standin/             the serve stack (BSL-1.1): brain, thalamus, VM-mediated chat, cubby-man, emitter, SFT data builders
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

`D:\grillcheese_training_data` (~210 GB, surveyed 2026-07-23): a real tokenizer history (19,947 → 32,000 → 65,536,
costed to 131,072), a dated NYT archive (1851–2024) used for the forgetting benchmark on real keys, and a 120 GB
candidate pretraining corpus that still needs its dedup/content-filter pass — the one substantial piece of Group G
not done. See `CUBBYLLM_HYPOTHESES.md`, Group G.

`cubby-lm` and `cubemind` are the direct predecessors; CubbyLLM is a fresh design, not a fork — the package layout
mirrors cubemind's documented intent, the model architecture and vocabulary are new. `cubby-concepts`, `grilly`,
`H:\AURA_GENESIS`, `C:\Users\grill\Desktop\GrillCheese`, and `H:\Novel_GNN_Arch` were surveyed for reusable
concepts; what each contributed is cited in `CUBBYLLM_HYPOTHESES.md`.
