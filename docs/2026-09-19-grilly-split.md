# Splitting grilly into modules — what the code says about it

Owner, 2026-09-19: *"grilly is a monster lol — however the c++ side and bindings
are alright"*, then *"I would rework Grilly as modules like grilly-vsa grilly-snn
grilly-torch grilly-huggingface."*

This is the measurement behind that, taken from CubbyLLM's side of the fence. It
is a consumer's view, not a plan for grilly.

## What measures well, and should not be rewritten

`grilly_core` (the pybind11 `.pyd`) imports standalone with only the repo
directory on `sys.path`: **230 exported symbols**, full type annotations, and just
`Device` and `TapeContext` to manage. It initialises Vulkan on consumer AMD
hardware — RX 6750 XT, VMA allocator, cooperative-matrix and fp16 extensions — with
no CUDA anywhere.

And the kernels are correct. Against `BlockCodeOps` at k=80, l=128, one-hot block
codes in `(n, k*l)` batches:

| | |
|---|---|
| `blockcode_bind` vs `BlockCodeOps.bind` | agree, max abs diff **1.19e-07** |
| bind→unbind exact block recovery, both paths | **100%** |
| speed at batch 1 / 64 / 512 | 1.01× / **1.74×** / **1.83×** |

The 273 compiled SPIR-V modules are toolchain-independent — any Vulkan host
consumes them, including a Rust one. So the compute core, its bindings and its
shaders are assets. The weight is the ~72K-line Python layer above them and the
~8M lines of vendored `external/`.

## The proposed split, sized

Core Python is 72,031 LOC. Bucketed by keyword — approximate, and some
assignments are certainly wrong (`backend/attention.py` landed under huggingface,
`backend/capsule_transformer.py` under snn, both by vocabulary rather than by
intent):

| module | files | LOC | share |
|---|---:|---:|---:|
| `grilly-vsa` | 32 | 6,999 | 9.7% |
| `grilly-snn` | 39 | 10,934 | 15.2% |
| `grilly-torch` | 71 | 19,429 | 27.0% |
| `grilly-huggingface` | 11 | 4,404 | 6.1% |
| **shared core** | 77 | **30,265** | **42.0%** |

## The one thing that decides whether this works

**42% of the Python is shared, and 30K is not a thin base.** Four packages that
each depend on a 30K-line core have moved the problem rather than solved it — the
coupling that makes the current repo hard to reason about would survive the split
intact, now spread across four changelogs.

So the first question is not how to divide the four leaves. It is: **what is the
minimum `grilly-core` that `grilly-vsa` alone needs?** Device, buffer pool,
dispatch, shader registry — if that lands near 3–5K, the split is real and the
other three can take their time. If it stays near 30K, the split is cosmetic.

`backend/_bridge.py` is 2,715 lines and `backend/fnn.py` is 3,653; those two plus
`buffer_pool.py` are where the answer lives.

## What CubbyLLM needs from any of this

Three operations: `bind`, `unbind`, `bundle` — `grilly-vsa`, 9.7% of the Python.
Nothing in the reasoning path needs `grilly-torch`, and nothing in the serving
path needs a GPU at all (the precision, latency and isolation numbers were all
measured on CPU).

That is the strongest argument for the split from here: it would take CubbyLLM's
optional GPU dependency from *a 72K-line framework plus 8M lines of vendored
third-party source* down to *one small module and a C++ extension*. That is a
packaging claim the deployment story can actually make — see `B5` in
`2026-09-19-positioning-gap.md`.

One find worth flagging separately: `utils/vulkan_sentence_transformer.py` (1,586
lines) would land in `grilly-huggingface`. The fastword table's teacher is a
MiniLM sentence-transformer, so that module may bear on `B2` — rebuilding the
table — rather than being unrelated to CubbyLLM as the rest of `grilly-torch` and
`grilly-snn` are.

## Not recommended from this evidence: a Rust rewrite of the compute

The C++ core and bindings measure well and the shaders are portable. A Rust
grilly2 would be rewriting the half that works. The coherence argument for Rust —
CubeLang is already Rust, so one toolchain and shared buffer ownership — applies
to a future *host* layer, not to the kernels, and it competes for time with the
2B trunk run that `TODO.md:12` says the gate is already open on.
