# The gap between the 2026 positioning and the repo

Written against the competitive-positioning draft (2026-09-19). Every claim in that
draft was traced to where the repo measures it. This is what is left to do — and,
separately, what the draft says that the repo does not yet support.

Two kinds of item below, tagged: **[BUILD]** is work, **[DOC]** is a sentence to
change. A [DOC] item is not a smaller thing than a [BUILD] item. A technical buyer
finds an overstated number in five minutes and then discounts the true ones.

---

## Already true, verified today — do not spend effort re-proving

| Draft claim | Where it lives |
|---|---|
| `import cubbyllm` is torch-free | verified by import probe: cubbyllm + reasoning.{pipeline,plan_verify,planner,index,learn,lexicon,hippocampus} + ops.vsa + bridges.cubelang_client all import with `torch` absent from `sys.modules` |
| 1.69 MB marginal RSS/user, 592 users/GB, flat 16→256 sessions | `validation/exp_r20_vm_per_user.py` |
| 2.0 ms median wall, VM-verified 3-hop, resident VM | README benchmark row, `exp_m3_cot_pipeline --resident` |
| precision 1.000 on the 800-question eval | README; 574/800 verified, 226 named refusals |
| 10,790 OCR facts / 22,213 entries / 16 s, no model, no network | `standin/encyclopedia.py`, exp_r16 |
| Rust VM as truth gate, deny-by-default, no self-scoring on the fact path | `cubbyllm/bridges/cubelang_client.py` + plan_verify |
| the learning gate separates at 3e-5 | `validation/logs/exp_a7_learning_gate*` — **ran once, on Colab** (see G3) |

---

## A. Blockers — the draft cannot be sent as written until these move

### A1 [DOC] "Production State" is a stand-in trunk
The column header says *CubbyLLM (Production State)*; the diagram inside it says
*Local GGUF 2.6B Stand-In*, which is the honest label. `cubbyllm/core`, `model/`,
`training/` are the trunk **design** from the validation campaign — `README` calls
them exactly that. The serving model is a fine-tuned LFM2.5-2.6B GGUF, and
`TODO.md`'s first open item is *"Cycle-one 2B rental — GATE IS OPEN"*: the trunk
has not been trained.

Either retitle the column (*Production State* → *Serving today*) or hold the draft
until the 2B run lands. The architecture claims are unaffected either way — the VM,
the disposer, the isolation economics are all real and all independent of the trunk.

### A2 [DOC] The Vulkan claim is about dispatch, not about the shaders
**Corrected after the owner pushed back, and the correction matters.** The first
version of this item said the compute layer was roadmap. It is not: grilly ships
**275 compiled SPIR-V shaders**, including a dedicated vector-symbolic set
(`vsa-bind`, `vsa-bind-batch`, `vsa-bundle`, `vsa-bundle-batch`,
`vsa-similarity-batch`, `vsa-fft-convolve`, `vsa-resonator-step`) alongside a full
training and inference library.

The real gap is one function. `cubbyllm/ops/vsa.py::_detect_backend()` returns
`grilly_bridge` when `_bridge.blockcode_bind` exists — but `_load_grilly_ops()`
unconditionally loads `grilly.experimental.vsa.block_ops.BlockCodeOps`, the
**Python** ops. So `bind` / `unbind` / `bundle` route through Python even when the
bridge is present and the SPIR-V is sitting right there. `TODO.md:37` states this
precisely.

A measurement caveat worth recording: `_detect_backend()` returns `numpy` on this
machine in **both** the default interpreter and the serving venv, because grilly
is not installed in either — it is a sibling checkout only. So a `numpy` reading
here says nothing about whether the Vulkan path works; it says grilly is not on
the path. The first version of this item drew the wrong conclusion from exactly
that reading.

So the draft's sentence is not false about the engine, only about the serving
path. Say *"our own Vulkan compute engine"* in the capability table and keep the
dispatch gap in the honest section until B1 lands.

### A3 [DOC] Precision 1.000 without coverage beside it
The draft leads with *"a measured precision profile of exactly 1.000"* and never
says 0.718. The README states both in one row: **0.718 (574/800) at precision
1.000**, with 226 named refusals. On free text the coverage story is much harder —
600 SimpleQA questions through the full gate produce **6 verified, 5 correct, 1
near, 0 wrong**.

Say both numbers, in that order, everywhere. *"1.000 precision at 0.718 coverage,
and every one of the 226 non-answers names its reason"* is a stronger claim than
1.000 alone, because it is the one a buyer cannot catch you on. The refusal
taxonomy is the product.

---

## B. Build work the positioning implies but the repo has not done

### B1 [BUILD] Wire `ops/vsa.py` to the bridge — cubemind already shows how

**This item was wrong twice before it was right. The numbers below are the third
measurement and the first correct one.**

grilly is now installed into the serving venv (`pip install -e`, additive: numpy
stayed at 2.5.1, nothing downgraded). The Vulkan device initialises — RX 6750 XT,
VMA allocator, C++ backend, cooperative-matrix and fp16 extensions.

**Measured against `BlockCodeOps`, at k=80 l=128, one-hot block codes, `(n, k*l)`
batches — which is how `cubemind/ops/block_codes.py` calls it:**

| | result |
|---|---|
| `blockcode_bind` vs `BlockCodeOps.bind` | **agree** — `allclose`, max abs diff 1.19e-07 |
| bind→unbind exact block recovery, both paths | **100%** |
| speed, batch 1 | 1.01× |
| speed, batch 64 | **1.74×** |
| speed, batch 512 | **1.83×** |

So the bridge is correct and it wins as soon as the call is batched. There is no
algebra question and no semantics risk. `TODO.md:37`'s caution was right to demand
the check; the check passes.

**What the two earlier wrong readings were.** Both were probe errors, recorded
because the failure mode repeats:

1. *"the tier does not exist"* — `_detect_backend()` returned `numpy`, read as
   evidence the Vulkan path was absent. grilly simply was not installed in either
   interpreter. A `numpy` reading means grilly is off `sys.path`, nothing more.
2. *"the ops disagree and are slower"* — fed flat `(k*l,)` arrays of dense
   Gaussians. The bridge wants `(n, k*l)` batches of ONE-HOT block codes. Wrong
   shape and wrong representation, so the 41.6 diff and the 0.223 round-trip
   measured nothing.

**The actual task, and it is small.** `cubemind/ops/block_codes.py` is a working
three-tier implementation of exactly this: bridge → `BlockCodeOps` → numpy, with
the reshape convention and per-path `try/except`. Port that structure into
`cubbyllm/ops/vsa.py`, whose `_load_grilly_ops()` currently returns the Python
`BlockCodeOps` even when `_detect_backend()` found the bridge.

Two details worth carrying over: the bridge returns `None` to mean *"declined,
fall through"* rather than raising, and `np.atleast_2d` + reshape restores the
caller's shape.

**Batch is where the win is.** At batch 1 the tiers tie. The VSA calls in the
reasoning path are single binds today, so wiring alone buys nothing measurable —
the gain arrives if and when binds are batched. Say that rather than quoting
1.83× out of context.

**On the wider grilly question.** The C++ core and its pybind11 bindings measure
well here: `grilly_core` imports standalone with 230 annotated symbols and only
`Device` / `TapeContext` to manage, and the kernels agree with the reference to
float32 precision. The weight is in the ~72K-line Python layer above them and the
~8M lines of vendored `external/`. Whatever grilly2 becomes, this evidence says
the compute core and the shaders are assets to keep, not things to rewrite.

### B2 [BUILD] The fastword table is on a drive that is not attached
`validation/exp_m3_cot_pipeline.py` hard-codes `V4_TABLE` to an absolute path on an
**external drive that is no longer attached** — found today when the serve stack
would not start without it. So the *"25 µs static distillation, 0.84× MiniLM
teacher"* layer is currently not reproducible on the dev machine. Rebuild the
table, resolve it the way the cubelang binary is resolved (env var, then a path
beside the repo) instead of a hard-coded absolute, and re-measure both numbers.

Also: the draft says 0.84×; `CUBBYLLM_HYPOTHESES.md:137` says **0.86–0.95×** on the
M3 screen. Pin which measurement the draft is quoting.

### B3 [BUILD] One external, reproducible benchmark
Every number in the draft is self-run on self-built evals. This is the single
highest-leverage credibility item and it is already scoped in `TODO.md:24`
(*"External evals to take: openai/simple-evals"*). SimpleQA is the honest fit —
and the 0-wrong result on 4,326 questions is a genuinely unusual thing to be able
to publish, even at low coverage.

### B4 [BUILD] Serving concurrency, not just isolation
The isolation number is excellent and the README already says what it does *not*
cover: *"Isolation is not the expensive part of serving — the shared emitter is."*
The draft's cost-profile section leans entirely on the 1.69 MB figure and never
addresses the emitter. A buyer sizing a deployment will ask about tokens/s per GPU
under N concurrent users. Measure it before someone asks.

### B5 [BUILD] Install-from-clean and run
*"Deployed out-of-the-box … with zero modification"* is a claim about packaging,
and nothing in the repo tests it. Needs: a clean-machine install from
`pyproject.toml`, the cubelang binary acquisition path, and a smoke test that
answers one question end to end. Today the serve stack also needs a GGUF that is
not tracked and a fastword table that is missing (B2).

### B6 [BUILD] The licence split will be asked about
`cubbyllm/` is Apache-2.0; `standin/` — the serve stack, the brain, the chat, the
emitter — is BSL-1.1. The draft sells the whole stack as one product. Decide what
an enterprise actually receives and say so in the deployment section.

---

## C. Second-order — worth doing before a technical audience, not before a first meeting

### C1 [DOC] "Zero self-scoring loops" is true of the fact path — be precise
It holds where it matters: the VM decides, the LM proposes. But the speech path
(`standin/grounding.py`, `saying.py`) is guarded by **host regexes and a
world-agnostic filter**, not by the VM. That is still not an LLM judging itself,
which is the claim that matters — so the claim survives, but say *"no model scores
its own output"* rather than implying every path ends at the VM.

### C2 [BUILD] "Nightly" security gate has run once
`exp_a7_learning_gate` ran on Colab, not on a schedule. The 48 ms is the evaluation
cycle, not a nightly cadence. Either automate it or call it *"a 48 ms pre-registered
promotion gate"* and drop the word nightly.

### C3 [BUILD] The refusal taxonomy deserves to be the headline artifact
226 refusals across four named classes on the 800-question eval; 3,825 named
refusals on SimpleQA. Nothing in the draft shows a buyer what a refusal *looks
like*. One page of real refusal records — `no-such-edge`, `plan-coverage`,
`ambiguous_entity` with the four Ottawas named — does more for the security
argument than the whole comparison table.

---

## Suggested order

1. **A3** then **A1** then **A2** — three edits, no build work, and they take the
   draft from checkable-and-wrong to checkable-and-right.
2. **B3** (external eval) — the one number that is not self-graded.
3. **B2** (fastword table) — it is blocking reproducibility of a layer the draft
   leads with, and it is a rebuild, not research.
4. **B5**, **B4** — the two questions a deployment conversation reaches within
   half an hour.
5. **C3** — cheap, and it is the best asset in the box.
6. **B1**, **B6**, **C2** — before a technical audience.

Everything in A is a sentence. Everything in B is measurable. Nothing here
questions the architecture, which is the part the draft gets right and the part
that is hardest to copy.
