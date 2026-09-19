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

## A core rewrite: retracted objection, and the case for it

**An earlier version of this file argued against rewriting the core.** That
argument rested on one measurement — `blockcode_bind` returning correct numbers
quickly at one shape — and kernel correctness says nothing about whether the core
is well structured. The owner's read (*"a patched over patched numpy framework
converted to c++"*) is supported by the structure, which the benchmark could not
see. Retracted.

**What the C++ actually looks like** (167 files, 34,590 LOC of own code, vendored
excluded):

| | |
|---|---|
| `cpp/src/device.cpp` | 585 |
| `cpp/src/buffer_pool.cpp` | 549 |
| `cpp/src/pipeline_cache.cpp` | 372 |
| **the real Vulkan core** | **~1.5K** |
| `cpp/src/autograd.cpp` | 1,684 (see the correction below) |
| files referencing `.spv` directly | **25**, headers included |
| pybind11 `m.def` registrations | 107, across ~12 `bindings_*.cpp` |

**Correction, before the argument.** An earlier version of this section said
`autograd.cpp` "reaches Vulkan directly with 56 raw `vk*` calls". Those symbols are
27 `VkDescriptorBufferInfo`, 27 `VkDescriptorSet`, one `VkPipeline`, one
`VkPipelineLayout` — descriptor *types* in signatures and locals, not `vkCreate*`
/ `vkCmd*` device calls. autograd passes descriptor handles around; it does not
stand up its own device or pipelines. The regex counted types as calls, and the
layering violation is milder than stated.

**What does hold: shader names are known to individual op families and headers
rather than to one registry** — even though `pipeline_cache.cpp`, `compute_backend.h` and `op_graph.h`
all exist to be that route. Two ways to reach the GPU, the same pattern as the two
unrelated VSA algebras (`blockcode_*` sparse float32, `vsa_*` bipolar int8) sitting
on one bridge with `_detect_backend()` gating on one of them.

**And this is why a rewrite is tractable rather than reckless: the spine is 1.5K
lines.** Device, buffer pool, pipeline cache. In Rust with `ash` that is a small
piece of work, and it is the piece that decides whether every op afterwards has
one path or two. The remaining ~33K of C++ is per-op host code — mechanical,
portable incrementally, and some of it deletable once a real registry exists.

**What carries over untouched:**

- the **273 compiled SPIR-V modules** — toolchain-independent, consumed by a Rust
  host exactly as by a C++ one;
- the **107-op binding surface**, which is a written specification of what the
  core must expose;
- `cpp/src/cubemind/block_ops.cpp` (467 lines) as the reference for the block-code
  algebra that `cubbyllm` and `cubemind` both depend on, now verified correct to
  1.19e-07 against the Python path.

**The sequencing question stands.** A core rewrite competes with the 2B trunk run
(`TODO.md:12`, gate open) and with gen-3 SFT. Nothing in the positioning document
depends on it — every number there was measured on CPU. The argument for doing it
first is not urgency but that every further op added to the current core makes the
second dispatch path more expensive to remove.

## "grilly core should be a drop-in replacement for torch"

Owner, 2026-09-19, and then the correction that matters: *"grilly was made as an
open source framework so people can use it as a torch drop in, not only us."*

So the surface is **not** defined by CubbyLLM's usage, and this document is a
consumer's note, not the design driver. What follows is offered as a first
milestone only.

### What CubbyLLM alone would need — a floor, not the spec

| | |
|---|---:|
| distinct torch symbols, whole repo | **110** |
| distinct torch symbols, `cubbyllm/` alone | **59** |
| distinct `nn.Module` classes | **13** |

The 13: `Conv1d`, `Embedding`, `GELU`, `GRU`, `Identity`, `LayerNorm`, `Linear`,
`Module`, `ModuleList`, `Parameter`, `Sequential`, `Tanh`. The heavy tail is
ordinary — `no_grad`, `randn`, `Tensor`, `from_numpy`, `cat`, `arange`, `zeros`,
`stack`, `gather`, `bmm`, `cross_entropy`, `normalize`, `Adam`.

Useful as **milestone 1** for two reasons that have nothing to do with CubbyLLM
being the customer: it is derived from a real training workload rather than from a
reading of torch's docs, and its acceptance test is already written and green —
`cubbyllm/`'s package suite, run under `import grilly as torch`. A public
framework needs a conformance matrix eventually; it needs one passing real
workload first, and this is one that exists today.

### The four entries that will be hard for anyone, not just here

- `torch.utils.checkpoint.checkpoint` — gradient checkpointing, not optional for
  a 2B on 12 GB. Re-entrant autograd is the awkward part of any autograd engine.
- `F.scaled_dot_product_attention`, `torch.nn.attention.flex_attention` —
  flash-attention-class kernels. grilly ships `attention-*.spv`, so possibly most
  of the way there.
- `torch.einsum` — a general implementation is real work; specialising the
  patterns actually in use is usually the better trade early.
- `torch.Generator` / `manual_seed` — bit-exact reproducibility across a backend
  swap is where "drop-in" quietly stops being true. Decide early whether to
  promise it, because users will test it.

### The differentiator, since it is a public framework

torch already runs everywhere torch runs. The reason to adopt a torch-compatible
Vulkan backend is the hardware torch serves badly: **consumer AMD and Intel GPUs,
and Windows.** torch's AMD path is ROCm, which is Linux-only; this repo's own
`TODO.md:49` records the local RX 6750 XT as *"no CUDA, ROCm Linux-only"*. That is
the gap — and it is the same gap for every hobbyist and small lab with a gaming
card, which is a large population with no good option.

Competitors to be honest about: `tinygrad` (Metal/AMD, small surface, not
torch-API), `candle` and `burn` (Rust, own APIs, not drop-in), `ggml`/`llama.cpp`
(inference only). **Nothing in that list is a torch-API drop-in that trains on a
consumer AMD card under Windows.** If grilly-core holds that position, the
narrowness of the surface at v1 matters much less than the fact that it runs where
the alternatives do not.

Which suggests the v1 claim is not *"a torch replacement"* — a claim that invites
an unwinnable comparison — but *"the torch API, on hardware torch will not train
on."* The conformance list then reads as scope, not as shortfall.

## Linking the C++, and the seams that are already cut

Owner: *"grilly2 core needs a rewrite, the vulkan code (all c++ now) is alright
and can be linked"*, then pointing at `cpp/python/`.

**The binding layer is already split by domain.** `cpp/python/` holds 26
`bindings_*.cpp` files and 5 `.inc` files, one per area, over a shared
`bindings_core.cpp` / `.h`:

| proposed module | binding files already present |
|---|---|
| `grilly-vsa` | `bindings_vsa_lm.cpp`, `vsa_explore_bindings.inc`, `block_ops_bindings.inc` |
| `grilly-snn` | `bindings_snn.cpp`, `bindings_eggroll.cpp`, `bindings_bandit.cpp` |
| `grilly-torch` | `bindings_autograd.cpp`, `bindings_optim.cpp`, `bindings_linear.cpp`, `bindings_conv.cpp`, `bindings_normalization.cpp`, `bindings_activations.cpp`, `bindings_loss.cpp`, `bindings_attention.cpp`, `bindings_pooling.cpp`, `bindings_mingru.cpp`, `tensor_ops_bindings.inc`, `elementwise_bindings.inc` |
| `grilly-huggingface` | `bindings_siglip.cpp`, `bindings_distillation.cpp`, `bindings_perceiver.cpp` |
| shared base | `bindings_core.cpp`, `bindings_core.h` |

This substantially changes the risk of the split. **The boundaries are not a
refactor to be argued about — they are already drawn, in C++, one file per
domain.** The 42% "shared core" figure earlier in this document is a measurement
of the *Python* layer, which never received the same treatment. The C++ side had
it all along.

So the split becomes: give each domain's bindings its own extension module over
the shared `bindings_core`, and let the Python packages follow the seam that
already exists rather than inventing one.

**On linking.** `CMakeLists.txt` builds `grilly_core_lib` as a STATIC library
separate from the Python module, so a second consumer links it today with no
restructuring. One detail decides how much glue: that is a C++ ABI — `extern "C"`
and the usual export macros appear nowhere in the family's own sources, only in
vendored `build2/_deps`. A grilly2 core written in C++ links it directly; one
written in Rust wants the `cxx` crate or a thin `extern "C"` shim over the entry
points it actually uses. The 107 registrations across those binding files are the
inventory of what that surface is today, and the list to prune against — not all
107 belong in a public v1.

(`grilly-rust` is not a rewrite in progress; it holds `grilly-model`, a separate
retrieval/benchmark line with MSMARCO and TruthfulQA artifacts.)
