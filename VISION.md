# Cubby — Vision

**An AI you own — that learns you, remembers forever, and can never quietly go wrong.**

Cubby is a safe, self-improving AI system: a continually-learning model (**Cubby**, the trunk) that runs on a verified, capability-based reasoning machine (**CubeLang**, the VM). Think of it as an *operating system for AI* — the intelligence learns; the VM executes it safely; nothing runs until it's been verified. It borrows the shape of a "living, verified blockchain" — everything checked before it executes, sandboxed, deterministic, evolvable — and applies that idea to AI. **No financial component; only the safety and verification concept.**

> This is the simple, front-door version. The depth lives in the docs linked at the end — this page is the map, not the territory.

---

## The vision

Three parts, one system:

- **Cubby — the intelligence.** An efficient, from-scratch model that learns continually and in real time *without forgetting what it already knew*. It remembers you. It specializes to your world automatically.
- **CubeLang — the safe execution substrate.** A reasoning VM where programs are **verified before they run** and a tamper-proof core can't be broken by anything a program does. This is where Cubby thinks in a way you can trust.
- **The world it lives in.** A body and environment that host the VM, the memory, and the senses — including an affective (emotional) layer, because a system that can represent feelings can genuinely relate to yours.

The result is meant to reach the *capabilities* people expect from today's frontier models — while being built on the opposite philosophy.

---

## Why we do it

Today's big-cloud AI is stateless (it forgets you the moment you leave), it hoards your data on someone else's servers, it's unverified (you hope it behaves), and it's enormous and expensive. We want the capabilities without any of that:

- **You own your data.** Even using Cubby's cloud services, your logs stay **on your machine, encrypted**; data only ever travels strongly encrypted. Helping train Cubby is **opt-in, off by default** — and rewarded.
- **It remembers, and keeps learning.** Continual learning without catastrophic forgetting is the central bet (making the model's active weights a function of context, rather than a fixed state overwritten in place).
- **It's safe by construction, not by hope.** Deny-by-default: verify *containment* — that code can't escape the sandbox or touch the protected core — because that's the only kind of "safe" that holds against inputs you didn't foresee. Guardians sit at every entry and exit.
- **It's efficient.** Bounded memory, constant-time decoding — designed to run on your hardware, not a datacenter.
- **It can be empathetic — for real.** A hormonal/affective layer gives it internal states to represent, which is the honest machinery for relating to others' states rather than imitating empathy from training text.
- **It improves itself, safely.** Because programs are verified and the core is tamper-proof, the system can extend and repair itself *within provable bounds*.

Frontier capability, inverted values: **local, owned, safe, efficient, personal.**

---

## A little history

- **Bio-inspired roots.** The lineage runs through emotional-intelligence work (AURA Genesis — a spiking-neural "brain" with a hormonal system) into an AI meant to feel, not just predict.
- **A cube full of possibilities.** The core representation — Vector-Symbolic Architecture (binding concepts into high-dimensional vectors) — came from looking at how many states a Rubik's cube can hold.
- **From `cubby-lm` to CubbyLLM.** An earlier language model taught the hard lessons; CubbyLLM is the from-scratch "gen-2 hybrid" redesign — general capability *plus* automatic specialization.
- **Research-backed.** Grounded in a research program (Mixture of World Models; VSA/hyperdimensional computing) rather than vibes — with a validation campaign of falsifiable hypotheses, each with a kill criterion.
- **The foundation is real.** In 2026 the episodic-memory layer was verified at scale (memory carries recall past the attention window), and the CubeLang verification foundation was built and merged.

---

## What works today

Honest calibration — this is what's actually built and verified, not aspiration:

- **An efficient hybrid trunk + episodic memory** — verified at scale to recall information from far outside its attention window.
- **Continual learning that doesn't forget** — the context-conditioned mechanism, validated at small scale.
- **The CubeLang verified VM** — deny-by-default verification (`--strict`), a tamper-proof capability core, real VSA bind/recover reasoning, modular imports, and a raw-opcode escape hatch — the foundation cycle is merged.
- **A model in training** — a multi-billion-token trunk run underway, bits-per-character steadily falling.
- **A reasoning bridge** — a real, verified path for the model to hand a problem to the VM and get genuine symbolic reasoning back (in progress).

---

## Where it's going

The near and far horizon — specced or on the axis:

- **"Remember forever," as a program.** A memory service — itself a verified CubeLang program — where you submit content and it's remembered permanently: **domain-organized** (routed by topic), **searchable** by keyword or query, and, when you opt in, feeding **continual learning**. Nothing enters the shared world model until it's checked against real facts (including live web verification through a guarded gateway) so it can't be poisoned.
- **A capabilities showcase.** The "remember forever" demo as the homepage; a game as a second capability test; and eventually a **harness that lets any model plug into the VM** — the VM markets itself by being useful to everyone.
- **An affective cortex.** A hormonal system that modulates learning, attention, and caution — and a personality that *develops* through experience rather than being hardcoded, so the empathy is earned and individual.
- **A full immune system.** Intelligent guardians at every boundary that can only ever *restrict, never grant*, plus safe ingestion of the outside world.
- **Self-writing, self-repairing.** A code-generation specialist that produces verified CubeLang on demand, and self-repair as a permitted, bounded function call.

---

## Services

Cubby is offered at whatever level of trust and compute you want — and the privacy properties hold at *every* level:

- **Cubby Local (on-device).** The model, memory, and VM run entirely on your hardware. Fully private, works offline, nothing leaves the machine.
- **Cubby Cloud.** The cloud handles heavy compute when you want it — but it does *compute, not data-hoarding*: your logs stay on your machine, encrypted, and everything in transit is strongly encrypted. The cloud never becomes the home of your data.
- **The Memory Service ("remember forever").** Submit content; it's remembered permanently, domain-organized, and searchable. Opt in — off by default — to let it help train Cubby, and earn **bonuses and rebates** for contributing.
- **Reasoning / VM access.** Verified symbolic reasoning on demand — plus a harness so *any* model can plug into the CubeLang VM and use it as a safe reasoning substrate.

The through-line across all of them: **you own your data, and nothing runs unverified.**

## Cost savings

Cubby's efficiency isn't a slogan — it comes from specific design decisions: automatic in-context specialization instead of per-domain retraining, retrieval instead of giant softmaxes, bounded-state constant-time decoding, and running on local hardware. The table frames *where* the savings come from and *how much of it is proven*.

**On the numbers — read this first.** This project has a strict rule: **no fabricated measurements** (every quoted number must link the benchmark that produced it). So the **"real"** column is filled in only as savings are actually measured; **"planned"** is an explicit design *target*, not a promise; and empty cells honestly say "not yet measured." This table is a framework to fill in with linked results — never estimates dressed up as results.

| Cost dimension | Industry default | Cubby (planned) | Cubby (real) |
|---|---|---|---|
| **Specializing to a domain/user** | Fine-tune / retrain a model per domain (GPU-hours each) | Automatic in-context specialization (θ=f(c)) — no retrain | — *(not yet measured)* |
| **Output layer / vocabulary** | Full softmax over a huge vocab every step (O(V·D)) | Retrieval output head + hybrid vocab — bounded cost | — *(not yet measured)* |
| **Long context / memory** | Quadratic attention; KV cache grows with length | Bounded-state recurrence + O(1) decode + episodic memory | Bounded decode state + recall past the window — *verified (scale-tested)* |
| **Hardware to run it** | Datacenter GPU clusters | Consumer / local hardware | — *(not yet measured)* |
| **Staying current** | Periodic full retrain | Continual online learning (incremental, opt-in) | — *(not yet measured)* |
| **Data hosting** | Your data lives (and costs) on their servers | Local, encrypted — no server-side data store | Local-first by design |

*As benchmarks land in `validation/`, the "real" column fills in with linked measurements. If you have target figures or a costing basis for the "planned" column, hand them over and they go straight in — clearly marked as projections.*

## The through-line

One principle recurs at every layer: **deny-by-default; verify containment, not correctness; tamper-proof core, extend by permission.** It keeps reappearing because it's the only notion of "safe" that survives inputs you didn't foresee — which is the whole premise of a system that learns continuously and writes its own code. Everything else is an implementation detail of that one idea.

---

## Read next

| For | See |
|---|---|
| The detailed, calibrated architecture | `docs/ARCHITECTURE_VISION.md` |
| What's *proven* (hypotheses, kill criteria, results) | `CUBBYLLM_HYPOTHESES.md`, `VALIDATION_REPORT.md` |
| How to work in this repo | `CLAUDE.md` |
| Active design cycles | `docs/superpowers/specs/` |
| The reasoning VM | the `cubelang` repo |
| The integration home / world models | the `cubbyverse` repo |

*This is a living document. As the horizon becomes the present, move features up — and keep the honest line between what's built and what's the goal.*
