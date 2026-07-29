# CubeMind cleanup — status, and what CubbyLLM actually needs from it

Written 2026-07-24, one day after everything else in this repo. `cubemind` (`C:\Users\grill\Documents\GitHub\cubemind`) is a real mess, confirmed by a fresh full-repo listing and a handful of direct checks against source today — not by re-reading old notes. This document lives in the **CubbyLLM** repo, not cubemind's, on purpose: it's not the cleanup plan itself (cubemind already has a better one than I'd write from scratch — see below), it's a record of what's actually true right now, what that plan doesn't cover, and specifically what CubbyLLM needs from cubemind versus what's cubemind's own general hygiene.

## cubemind already has a real refactor plan — this doc doesn't replace it

`cubemind/CUBEMIND_REFACTOR_PLAN.md` already exists, and it's good: grounded in an actual audit (import-usage grep, on-disk sizes, git-tracking checks), explicitly behavior-preserving (tests must stay green at every step, public API can't break), and phased by risk:

- **Phase 0** — baseline (branch, full test run, a real import graph saved to disk before anything changes).
- **Phase 1** — repo hygiene and size (highest relief, lowest risk): get `sandbox/` out of git tracking, move checkpoints/artifacts out of the repo, tighten `.gitignore`, resolve `cloned/`, delete empty/scratch dirs.
- **Phase 2** — dead code and ambiguous duplicates (medium risk): remove the importable `_archive/`, force an explicit MoQE decision (the plan flags that the decision record says MoQE is archived while the code still has ~130 live references — record and reality disagree), reconcile two committed `.patch.py` files.
- **Phase 3** — de-duplicate `core/` vs `functional/` (registry, kernels, routing all exist in both), and resolve the `model.py` / `models/` / `model/` naming collision.
- **Phase 4** — guardrails so it doesn't regress (document the canonical layout, ban `_archive` imports via lint, add a no-large-files pre-commit check).

Its own honest sequencing note is worth repeating here: Phase 1 gives ~90% of the relief for ~10% of the risk, and — its words — "a clean repo makes future work faster but does not move the valuation or land a customer." That framing should govern this doc too: the goal isn't to relitigate cubemind's plan, it's to track it accurately and scope CubbyLLM's actual stake in it.

## Freshly verified status (2026-07-24) — none of Phase 1–3 has actually been executed

Every checkbox in `CUBEMIND_REFACTOR_PLAN.md` is still unchecked, and today's direct checks confirm that's accurate — with one real exception. The plan's own prose numbers have drifted from current reality in the meantime, which is itself worth noting: this is the same "docs vs. code disagree" pattern this whole project has flagged repeatedly, now caught happening to the plan document itself in under what's evidently been a short window.

| Finding | Plan's original number | Verified today (2026-07-24) | Status |
|---|---|---|---|
| `sandbox/` tracked in git | 137 files, ~34.8GB | **137 files still tracked** (`git ls-files sandbox`); on-disk size now 13.87GB, `.git/` itself is 2.51GB | **Not done.** Disk usage dropped (likely manual deletion of untracked local files), but the actual git-history bloat this item exists to fix is untouched — same 137 tracked files. |
| `cloned/` (vendored) | ~12.8GB | **0 files, ~0GB** | **Done, and done right.** `INSTALL.md` now documents grilly as `pip install grilly` or a sibling `git clone`, exactly the "document as external deps" option the plan proposed — not just an emptied directory. |
| `model/` (repo-root Cubby training dir) | ~4.4GB | 0.3GB | Shrunk, but this doesn't touch the naming collision itself (see below) — likely just checkpoint cleanup. |
| `_archive/` live import refs | 3 | **3, unchanged** | **Not done.** Still importable from inside the package. |
| MoQE references | ~130 | **103** (simple case-insensitive match across `cubemind/**/*.py`; different counting method, same conclusion) | **Not done.** Still deeply embedded; the record-vs-reality contradiction the plan flags is still live. |
| `.patch.py` files | 2 (`mindforge.patch.py`, `moqe_distillation.patch.py`) | **Both still present**, unreconciled | **Not done.** |
| `core/` vs `functional/` duplication | `registry.py`, `kernels.py`, `routing.py` in both | **All 6 files confirmed present today** | **Not done.** |
| `model.py`/`models/`/`model/` collision | 3-way | **Confirmed, and it's worse than "3-way" suggests**: top-level `model/` (with `model/cubby/`, `model/cubby_grilly/` — this is where cubby-lm's actual trunk/trunk_torch backends are ported from), plus `cubemind/model.py`, plus `cubemind/models/` | **Not done.** |

Bottom line: nothing about the actual code structure has changed since the plan was written. The only real progress is `cloned/`, and it was done via the documented-external-dependency route, not a submodule.

## What the existing plan doesn't cover

A fresh top-to-bottom listing of the repo today turned up several things Phase 1–4 doesn't mention at all — worth folding in before treating that plan as complete, not worth a separate competing plan:

- **Five competing planning docs sit loose at repo root** with no index: `CUBEMIND_REFACTOR_PLAN.md`, `QC_APP_PLAN.md`, `QC_PERCEPTION_PLAN.md`, `TASKS.md`, `CHANGELOG.md`. None of these were read in depth for this doc beyond the refactor plan — worth at minimum a one-line root README section saying which doc governs what, so a new contributor (or a future Claude Code session) doesn't have to guess.
- **`.env` sits at the repo root.** It's correctly listed in `.gitignore` (verified), so this isn't an active leak — but it's worth explicitly confirming it was never committed before that line was added (`git log --all -- .env`), not something this pass checked.
- **`subdomain_taxonomy_patterns.jsonl` exists twice** — once loose at repo root, once at `data/subdomain_taxonomy_patterns.jsonl`. Not checked whether they're identical.
- **`gen_step_001000.md`** — a stray generation-output artifact sitting at repo root (gitignored, but still there on disk, and an odd thing to find next to `README.md`).
- **`webapp/` is an entire separate Next.js application** living inside the cubemind repo — its own `node_modules/`, `.next/` build output, its own `CLAUDE.md` and `AGENTS.md`. The refactor plan doesn't address it at all beyond noting its `node_modules/` size. Worth a deliberate call: does this stay a monorepo, or does the webapp get its own repo? That's a bigger decision than anything in Phase 1–4.
- **`gguf/bge-m3-q8_0.gguf`** — a large binary model file, correctly gitignored (`*.gguf`) but undocumented: why it's there, whether it's the same `BAAI/bge-m3` model the validation campaign flagged as needed (and missing from the local HF cache) for Novel_GNN_Arch's domain classifier eval. Possibly worth a pointer instead of a second copy somewhere else.
- **The `docs/` folder is under a hand-maintained embargo allowlist in `.gitignore`**, protecting content related to a NeurIPS 2026 submission (the I-RAVEN/I-RAVEN-X benchmarks and related architecture docs are explicitly embargoed — `benchmarks/iraven*.py`, `cubemind/perception/raven_renderer.py`, and `tests/test_raven_world_manager.py` are gitignored for the same reason). The `.gitignore`'s own comments warn against ever adding a broad un-ignore pattern here, which reads as a "this has already almost gone wrong once" note. **This is a real constraint on any cleanup work, not just cubemind hygiene** — anything touching `docs/`, the RAVEN benchmarks, or `perception/raven_renderer.py` needs to preserve the embargo, not just tidy freely.
- **The `.gitignore` itself has accumulated duplicate lines** from append-only edits — `CLAUDE.md` and `sandbox/mingru_baseline/bloopers.md` (three times) both appear more than once. Small, but it's the same sprawl pattern showing up in the meta-tooling, not just the code.

## Why this matters for CubbyLLM specifically

CubbyLLM doesn't need all of cubemind clean — it needs a small, specific slice of it stable and well-understood:

**H-B5 / the binding-head decision.** CubbyLLM's chosen algebra is `BlockCodeOps`, and cubemind's own `cubemind/ops/block_codes.py` is confirmed (again, independently, via `INSTALL.md`'s own verification snippet: `from cubemind.ops.block_codes import BlockCodes`) to be a thin, real wrapper over `grilly`. That one file — not the surrounding sprawl — is CubbyLLM's actual load-bearing dependency on this repo today.

**H-F1 (package layout).** CubbyLLM's `PACKAGE_LAYOUT.md` is explicit that CubbyLLM should mirror cubemind's *intended* structure, not its sprawl. Everything in the tables above is the concrete referent for "sprawl" — this audit is what that principle is being written against, not an abstract warning.

**H-F2 (the Cubby↔CubeMind bridges) and the VM-side unbind work (both on `TODO.md`).** Both land inside `cubemind/reasoning/` and `cubemind/execution/` — specifically `reasoning/vm.py` (where `UNBIND_ROLE` actually lives, confirmed by the validation campaign) and whatever carries `NoveltyToWorldBridge`. Neither of those paths is touched by the Phase 1–3 items above (they're about `sandbox/`, `_archive/`, MoQE, `core/`/`functional/`, and naming) — so CubbyLLM's bridge work isn't blocked by cubemind's cleanup, but it *is* worth re-checking `reasoning/vm.py` and the bridge code specifically stay stable if anyone starts executing Phase 2–3, since neither the refactor plan nor this doc has audited those two files' own internal health yet.

## Recommendation

Don't take on executing `CUBEMIND_REFACTOR_PLAN.md` as part of CubbyLLM's own work — it's real, it's already well-scoped, and it's cubemind's repo to fix on its own timeline. What belongs on CubbyLLM's side:

1. Treat this document as the thing to re-verify periodically, not `CUBEMIND_REFACTOR_PLAN.md`'s prose numbers directly — they're already stale one cycle in, and there's no reason to expect that to stop.
2. Before starting the VM-side unbind work or the bridge redesign (both already on `TODO.md`), do a narrow, fresh check of `reasoning/vm.py` and the bridge modules specifically — not the whole repo — since that's CubbyLLM's actual surface of contact.
3. If anyone does start executing cubemind's Phase 2 (the MoQE decision, `_archive/` removal) or Phase 3 (`core`/`functional` dedup), flag it here and re-check whether `ops/block_codes.py`, `reasoning/vm.py`, or the bridge code moved or changed shape — those are the only parts of this repo CubbyLLM actually depends on.

## Reference

- `cubemind/CUBEMIND_REFACTOR_PLAN.md` — the actual execution plan, phased and behavior-preserving. Authoritative for cubemind's own cleanup; this document doesn't duplicate its detail.
- `cubemind/INSTALL.md` — documents grilly as an external dependency (pip-installable or sibling clone), the correct resolution of the `cloned/` item above.
- `CUBBYLLM_HYPOTHESES.md`, Group F (H-F1, H-F2) and `VALIDATION_REPORT.md`, Group B/F — where cubemind's structure first became load-bearing for CubbyLLM's own decisions.
- `PACKAGE_LAYOUT.md` — the "mirror the intent, not the sprawl" principle this document's audit grounds.
- `TODO.md` — the VM-side unbind and bridge-redesign items this document's "why it matters" section connects to.
