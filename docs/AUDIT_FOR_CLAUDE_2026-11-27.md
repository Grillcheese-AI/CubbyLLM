# CubbyLLM Project Audit & Suggestions for Claude

**Date:** 2026-11-27  
**Scope:** Full repository audit — Claude workflow (Cowork + Code A/B), repository hygiene, code architecture, runtime flag management, test performance, `cubby_serve/` hardening, and strategic research bottlenecks.  
**Audience:** Nick, Cowork, and Claude Code (Code A & Code B).

> **Summary verdict:** CubbyLLM has unusually strong scientific discipline — pre-registered hypotheses, kill lines, held-out rules, executable architectural guards, and honest negative results. The main drag on velocity and quality right now is **operational and structural accretion**: unbounded append-only context files that waste Claude's context window, stale orientation docs, 362+ MB of accidentally tracked scratch binaries, 51 scattered runtime `CB_*` flags with no retirement policy, 116 lingering "one-cycle" shims, and growing heuristic complexity in `plan_verify.py` / `AskLoop` that is patching around a deeper proposer–gate interface mismatch.

---

## Table of Contents

1. [What Is Working Well (Keep & Protect)](#1-what-is-working-well-keep--protect)
2. [P0 — Making Claude Dramatically More Efficient](#2-p0--making-claude-dramatically-more-efficient)
3. [P0 — Repository Bloat & Broken Hygiene Guards](#3-p0--repository-bloat--broken-hygiene-guards)
4. [P1 — Code Architecture & Maintainability](#4-p1--code-architecture--maintainability)
5. [P1 — Test Suite Speed & Guard Accuracy](#5-p1--test-suite-speed--guard-accuracy)
6. [P1 — `cubby_serve/` Hardening & Concurrency](#6-p1--cubby_serve-hardening--concurrency)
7. [P2 — Strategic / Scientific Critique](#7-p2--strategic--scientific-critique)
8. [Suggested Action Plan (Ordered by ROI)](#8-suggested-action-plan-ordered-by-roi)
9. [Questions for Nick](#9-questions-for-nick)

---

## 1. What Is Working Well (Keep & Protect)

Before criticizing, these practices are genuinely rare and worth defending as the repo scales:

- **Falsifiable research discipline:** Pre-registered gates, kill lines, and held-out discipline (`CUBBYLLM_HYPOTHESES.md` §0). Keeping negative results on record (`0081`, `0083`, `H-E20`) prevents circular rediscovery.
- **Executable architectural boundaries:** `tests/test_guards.py`, `tests/test_package_rules.py`, and `tests/test_local_paths.py` turn repo conventions into enforceable code rather than wishful prose. All 26 tests in these three files currently pass.
- **Zero-wrong-spoken invariant & verifiable certificates:** Separating proposal (model) from disposal (host + CubeLang VM) and verifying CheezML certificates via independent replay (`0076`, `0085`) is a clean, defensible systems design.
- **Auth & path-safety fundamentals:** `cubby_serve/auth.py` implements PKCE S256, browser-bound login cookies, constant-time `hmac.compare_digest`, and `Origin` checks; static asset routes (`cubby_serve/board/api.py::_file`, `standin/pacui_threejs.py::load_asset`/`load_sfx`) validate filenames and paths properly.

---

## 2. P0 — Making Claude Dramatically More Efficient

This is the highest-ROI area for immediate impact on every Claude Code and Cowork session.

### 2.1 Context-Window Hemorrhage in Append-Only Markdown Files

Several files that `CLAUDE.md` and `CLAUDE.local.md` instruct Claude to read have grown into massive append-only logs:

| File | Current Size | Est. Tokens | Why It Hurts Claude |
|---|---:|---:|---|
| `docs/inbox.local.md` | **526 KB** (5,676 lines) | ~130k | `CLAUDE.local.md` says *"Read `docs/inbox.local.md` before starting or resuming any order."* Reading it blows up context; tailing/grepping it risks missing active notes. |
| `CUBBYLLM_HYPOTHESES.md` | **526 KB** (810 lines) | ~130k | Single paragraphs exceed 20,000 characters without line breaks. Line-based file reads truncate mid-paragraph; `CLAUDE.md` tells Claude to consult entries before editing code. |
| `docs/field-watch.local.md` | **425 KB** | ~105k | Hard to search/skim without loading huge chunks. |
| `docs/WORK_ORDERS.md` | **125 KB** | ~31k | Historical serve-stack era doc still pointed to by `CLAUDE.md`. |
| `docs/PATH.md` | **95 KB** | ~24k | Large monolithic prose document. |
| `TODO.md` | **67 KB** (59 lines!) | ~17k | Single bullet points >15,000 characters; stale since September 2026 and superseded by `docs/orders/`. |
| `docs/model-competition-for-cubby-finalconsensus.json` | **2.4 MB** | — | Raw model-panel dump tracked directly in `docs/`. |

**Recommendations for Claude:**
1. **Rotate `docs/inbox.local.md` immediately.** Move resolved notes older than 7–14 days into `docs/archive/inbox_YYYY-MM.local.md`. Keep `docs/inbox.local.md` under **5 KB** (only `new` / active threads + the last ~5 resolved entries). Add a rotation step to Cowork's weekly hygiene.
2. **Shard `CUBBYLLM_HYPOTHESES.md` into an index + per-group files.**
   - Create `docs/hypotheses/INDEX.md` (<15 KB): a crisp markdown table (`ID | Claim (1 line) | Status | Latest Order/Log | File`).
   - Split the body into `docs/hypotheses/{0_rules_context,group_A,group_B,group_C,group_D,group_E,group_F,group_G,group_K,group_P}.md` with normal paragraph breaks (<=120-char lines or one sentence/finding per bullet).
   - Keep `CUBBYLLM_HYPOTHESES.md` as a short pointer/index or auto-concatenated build if you want a single-file export for readers.
3. **Archive `TODO.md`.** Move it to `docs/archive/TODO_2026-09.md` (with a 5-line stub or removal from root) so Claude never confuses September's stand-in backlog with the active `docs/orders/` queue.

### 2.2 `CLAUDE.md` and `CLAUDE.local.md` Contain Stale & Contradictory Orientation

Fresh Claude sessions trust `CLAUDE.md` as ground truth, but right now it actively misleads them:
- **"Where things stand"** is dated `2026-10-21` (over a month old).
- **"Queue"** is dated `2026-10-31` — completely missing orders `0048` through `0087` (nearly half the project's work orders!).
- **"Known red"** (`CLAUDE.md:94`) still says `tests/test_guards.py::test_every_module_declares_wiring` fails on `cubbyllm.spatial` and `cubbyllm.reasoning.step_loop` — **false**; all 26 guard tests currently pass.
- `CLAUDE.local.md` still cites pre-0031 paths (`standin/books.py`, `standin/data/build_books_index.py`).

**Recommendations for Claude:**
1. **Keep `CLAUDE.md` timeless (<120 lines):** roles, workflow rules, invariants, coding/evidence rules, and commands — **remove hand-maintained status snapshots and queue lists from `CLAUDE.md`**.
2. **Auto-generate `docs/QUEUE.md` (or print live queue via `tools/queue.py`)** from `docs/orders/0*.md` headers. If a status section requires manual editing on every commit, it will always drift.

### 2.3 Work-Order Queue Rot: 13 Simultaneous `in progress` Orders

Audit of all **85 orders** in `docs/orders/`:
- `done`: **37**
- `open`: **25** (many untouched since October: `0008`, `0009`, `0010`, `0018`, `0019`, `0020`, `0023`, `0025`, `0026`, `0032`)
- `in progress`: **13** (`0024`, `0027`, `0029`, `0049`, `0052`, `0053`, `0054`, `0055`, `0067`, `0068`, `0080`, `0082`, `0085`)
- `proposed`: **5** (`0002`, `0003`, `0017`, `0042`, `0086`)
- `blocked`: **4** (`0039`, `0043`, `0044`, `0056`)
- `review`: **1** (`0048`)

`docs/orders/README.md` says *"works one order at a time"*, yet **13 orders are marked `in progress`** across only 2 coders + Cowork. Dependencies, assignees, and phase notes are stuffed into free-text parentheticals on the `status:` line (e.g., `status: open (queue: after 0055, 0010, 0059, 0060, 0061 and 0020)`).

**Recommendations for Claude:**
1. **Add structured frontmatter to orders** (update `docs/orders/_TEMPLATE.md` and migrate active orders):
   ```yaml
   ---
   status: in_progress   # proposed | open | in_progress | waiting_gpu | waiting_nick | review | blocked | parked | done
   assignee: code-a      # code-a | code-b | cowork | nick
   depends_on: [0082, 0083]
   hypothesis: [H-E15]
   updated: 2026-11-27
   ---
   ```
2. **Distinguish `waiting_gpu` / `waiting_nick` / `parked` from `in_progress`.** Enforce a strict WIP limit of **1 `in_progress` order per worker** (`code-a`, `code-b`, `cowork`). Multi-month umbrella orders (`0024`, `0027`, `0054`, `0055`) should either be closed as phase-1 done + phase-2 new order, or marked `parked`.
3. **Add `tools/queue.py`** (`python tools/queue.py --for code-a`) that parses order frontmatter, resolves `depends_on`, and prints: (a) currently active, (b) ready-to-claim (all deps `done`), (c) waiting on Nick/GPU, (d) stuck in `review` > 3 days (like `0048`).
4. **Close or triage `0002`:** Order `0002-wiring-guard.md` is still `proposed` even though `test_guards.py` already passes.

### 2.4 `.claude/` Configuration, Reusable Commands, and Multi-Session Locks

- `.claude/settings.local.json` currently contains **88 lines of stale one-off July/August bash commands** (specific `tee validation/logs/exp_c2_lzw.log` commands, hardcoded temp scratchpad paths, and drive-path `echo` commands), rather than clean permission globs for routine tools (`pytest`, `git status`, `git diff`, `ruff`, `tools/*.py`).
- There are **no `.claude/commands/`** (custom slash commands/prompts) to standardize recurring Claude Code rituals.
- **Code A / Code B resource contention:** Coordination for the shared home GPU and 12 CPU threads currently relies on reading a 526 KB inbox file before starting a run.

**Recommendations for Claude:**
1. **Clean up `.claude/settings.local.json`** into minimal, safe allowlists for read-only/test commands (`pytest`, `git status/diff/log`, `ruff check`, `python tools/*`).
2. **Create `.claude/commands/` workflow templates:**
   - `/start-order`: runs `tools/queue.py`, reads active inbox notes + order file, checks git worktree/branch hygiene, sets `in_progress`.
   - `/finish-order`: runs targeted tests + fast guards (`test_local_paths`, `test_package_rules`, `test_guards`), verifies `< 300 lines` on touched files, checks no `_`-prefixed logs staged, drafts the order **Report** with exact log links and commit hashes.
3. **Add a lightweight machine-local GPU/benchmark lock (`tools/hw_lock.py`):** A lockfile beside `paths.local.json` recording `{pid, session, order, started_at, resource: "gpu"|"cpu_bench"}` with stale-PID detection so Code A and Code B never collide on timing benchmarks or Vulkan GPU runs.

---

## 3. P0 — Repository Bloat & Broken Hygiene Guards

### 3.1 629.75 MB Tracked in Working Tree / 356.26 MiB `.git` Pack

`CLAUDE.md` states *"Files starting `_` [in `validation/logs/`] are scratch (untracked)"* and `PACKAGE_LAYOUT.md` states *"Corpora/checkpoints never in-repo"*. Both rules were silently broken before `.gitignore` was tightened:

- **`validation/logs/` has 2,068 tracked files**, of which **464 are `_`-prefixed scratch files**.
- **~295 MB of PyTorch checkpoints** tracked in git: 8 files in `validation/logs/_k10_ckpt/*.pt` (~37 MB each).
- **~67 MB of SQLite databases** tracked in git: 6 files under `validation/logs/_worlds_smoke/` (`fiction.sqlite` 42 MB, `reality_scholarly.sqlite` 20 MB, `quotes.sqlite` 4.5 MB, etc.).
- **Noise artifacts tracked in `validation/logs/`:** 75 `.err`, 70 `.out`, 38 `.py`, 25 `.done`, 15 `.cmd`, 2 `.ps1`, 1 `.u32`, 1 `.i16`.
- **Root-level stray artifacts tracked:** `cubby_mfu_pilot.zip` (71 KB), `cot_harvest_amb.jsonl` (2.0 MB), plus `docs/model-competition-for-cubby-finalconsensus.json` (2.4 MB) and `standin/data/music/neon_pixel_dash.mp3` (4.3 MB).
- **Untracked clutter at repo root:** `_brain_serve.err`, `_brain_serve.log`, `_panel.log`, `_panel2.log`, `_serve.log` (1 MB) — consider writing runtime server logs to a gitignored `logs/` or `$CUBBY_DATA/logs/` folder instead of the repo root.

**Recommendations for Claude:**
1. **Untrack the scratch artifacts** (`git rm --cached` on `validation/logs/_*`, `*.pt`, `*.sqlite`, `*.err`, `*.out`, `*.done`, `*.cmd`, `cubby_mfu_pilot.zip`, `cot_harvest_amb.jsonl`), moving any genuinely referenced artifacts to `$CUBBY_DATA` / Google Drive.
2. **Add a repo-hygiene guard to `tests/test_guards.py`:**
   - Fail if any git-tracked file matches `validation/logs/_*`.
   - Fail if any git-tracked file has extension in `{.pt, .pth, .safetensors, .gguf, .sqlite, .db, .zip, .u32, .i16, .err, .out, .done, .cmd}`.
   - Fail if any newly tracked file exceeds a size budget (e.g., `5 MB`, with an explicit small allowlist for `data/grillcheese_bbpe128k.json` and the two `.npz` files if they must stay in-repo).
3. **Move large static JSON logs out of `validation/logs/`** (or keep compact summary `.json` <= 200 KB in git and store per-row multi-MB dumps like `exp_r18_hdc_rephrased_host_lev8.json` [6 MB] on Drive with their SHA-256 recorded in the summary).

### 3.2 Packaging, Linting & CI Are Missing

- **`pyproject.toml` is still a stub:** `version = "0.0.0"`, `description = "CubbyLLM — context-conditioned LM (theta=f(c)). Package skeleton."`, `dependencies = []`. Anyone cloning the public repo gets zero declared dependencies (`numpy`, `torch`, `safetensors`, `tokenizers`, `grilly2`, `rdflib`, `pyshacl`).
- **No linter/formatter config:** `.ruff_cache/` exists, yet `[tool.ruff]` is not configured in `pyproject.toml`.
- **No CI (`.github/workflows/`):** Every guard test depends on Claude remembering to run `pytest`.

**Recommendations for Claude:**
1. Populate `pyproject.toml` with real metadata, `[project.dependencies]` (`numpy`), and optional extras (`train`, `serve`, `eval`, `dev` including `pytest`, `ruff`).
2. Add `[tool.ruff]` (e.g., `E, F, I, UP, B` rules, sensible line length like 120) so Claude doesn't introduce unused imports, inline `__import__("re")` hacks, or inconsistent formatting.
3. Add a lightweight `.github/workflows/ci.yml` running `pytest tests/test_guards.py tests/test_package_rules.py tests/test_local_paths.py` plus fast CPU unit tests on every push/PR.

---

## 4. P1 — Code Architecture & Maintainability

### 4.1 The `covers()` / `plan_verify.py` Heuristic Accretion Problem

`cubbyllm/reasoning/plan_verify.py` is **950 lines** (more than 3× the 300-line rule) — the largest production module in `cubbyllm/`. More importantly, it illustrates an architectural smell:

- `covers()` has accumulated six generations of surface-string heuristics (`v1`–`v6`): `_WHICH_CLASS`, `_PAREN`, `_ASK_PLACE`, `_strip_place_ask`, `_stem` with hardcoded English suffixes, `_find_split`, and now order `0087` proposes adding `cover_names.py` with capitalization heuristics to catch the "Ghana" bug.
- **Root cause:** `SlotTable` (`cubbyllm/reasoning/slots.py`) already parses entities and numbers with character spans *before* the emitter runs, yet `covers()` throws away structured span identity and tries to reconstruct coverage by scanning normalized flat strings!
- **Constructive alternative:** Pass the structured `SlotTable` (with entity/number character spans) directly into `verify_plan()`. Checking "did the plan use the entities that appear in the question?" and "is the seed actually a slot entity rather than a relation word?" becomes a **structural invariant over spans**, not a fragile capitalization/regex heuristic (which also breaks on lowercase user input, non-ASCII names, or French titles).
- Split `plan_verify.py` into focused `<300`-line modules: `relations_vocab.py` (`StoreRelations` / `KnownRelations`), `plan_coverage.py` (`covers`), `plan_segment.py` (`segment`), and `plan_verify.py` (`verify_plan`, `PlanVerdict`).

### 4.2 Line-Count Cap Violations & Mixin God-Objects

**31 files** across `cubbyllm/` and `standin/` exceed the 300-line rule (`cubby_serve/` complies well!). Top `cubbyllm/` offenders:

| Lines | File | Note |
|---:|---|---|
| 950 | `cubbyllm/reasoning/plan_verify.py` | Hot path; split by concern |
| 758 | `cubbyllm/reasoning/learn.py` | Search-and-learn loop |
| 735 | `cubbyllm/reasoning/skills.py` | Rule library + ledger |
| 602 | `cubbyllm/reasoning/pipeline.py` | Core walk + verification |
| 591 | `cubbyllm/reasoning/sleep.py` | Consolidation cycle |
| 517 | `cubbyllm/model/backbone/hybrid.py` | Mixers + RoPE + attention dispatch |
| 490 | `cubbyllm/bridges/cubelang_client.py` | CLI + protobuf + session transports |
| 416 | `cubbyllm/reasoning/school.py` | Arithmetic school |
| 387 | `cubbyllm/reasoning/step_loop.py` | Arithmetic step loop |
| 359 | `cubbyllm/reasoning/slots.py` | Slot annotation & filling |
| 341 | `cubbyllm/reasoning/helpers.py` | Miscellaneous grab-bag |

**Critique of mechanical mixin splitting:** `AskLoop(AskDated, AskProfile, AskRelation, AskMemory, AskRecall)` (`cubbyllm/reasoning/ask.py:49`) and `CubbyBrain(BrainRoute)` (`cubbyllm/brain.py:64`) were split via inheritance mixins in order `0031` to stay under 300 lines per file. That satisfies the line count *on paper* while hiding implicit coupling across 6 files (all mutating shared `self._forms`, `self.world`, `self.known`, `self._lock`, `self.contested`, `self._graph`), plus `AskLoop.__init__` takes **20 positional/keyword parameters**.

**Recommendation:** When refactoring oversized modules going forward, prefer **explicit composition** (passing narrow collaborators + a typed `AskConfig` dataclass) over inheritance mixins. It makes unit tests far simpler and stops Claude from having to read 6 mixin files to understand one method in `AskLoop`.

### 4.3 Runtime Feature-Flag Sprawl (51 Flags in Package, 215 Repo-Wide)

There are **51 unique `CB_*` flags** in `cubbyllm/` + `cubby_serve/` (215 across all `.py` files), read via scattered `os.environ.get(...)` calls:
- **No retirement policy:** Killed experiments remain wired into production hot paths forever (`CB_EMIT_TRIE` from killed `0081` is still imported and checked inside `AskLoop._ask` at `ask.py:213`; `CB_ASK_MEMORY` from killed `0083`; `CB_SOURCE_NAMESAKES` from `0043`). Every extra dead branch increases cognitive load and makes flag interaction bugs harder to diagnose.
- **Inconsistent read timing:** `cubbyllm/model/backbone/hybrid.py:81-83` reads `CB_ATTN_TRITON` and `CB_FLEX_OPTS` **once at module import time** (making test toggling fragile), while `cubbyllm/reasoning/ask_memory.py:21` reads `os.environ.get("CB_CERT_FACTS")` inside `ran()` on every single check.
- **Flag explosion in hot code:** `CB_RULES` appears in 13 package files, `CB_HELDOUT_FIXES` in 6, `CB_NAMESAKE_RELATION` in 5.

**Recommendations for Claude:**
1. **Create a single flag registry (`cubbyllm/core/flags.py`):** Every runtime `CB_*` flag is declared once with its default, owning order, lifecycle state (`experimental | graduated | deprecated`), and one-line docstring.
2. **Adopt a flag lifecycle rule in `CLAUDE.md`:**
   - **Killed & stays off:** Remove the runtime hook from `cubbyllm/` once the order reaches `done` (the experiment's exact commit hash remains in git and in the order's Report).
   - **Graduated (on by default for >= 2 cycles):** Promote from `os.environ` into a typed config field (`AskConfig` / `CubbyConfig`), removing the env var fallback or keeping a deprecation alias in one place.
3. **Pin with a guard test:** Fail if any module under `cubbyllm/` or `cubby_serve/` reads an unregistered `CB_*` env var directly.

### 4.4 Retire the 116 "One-Cycle" Shims in `standin/`

`tools/shim_count.py` currently reports:
> **`336 in-repo imports of shimmed names in code, 1 in notebooks (116 shims)`**

Order `0031` created these shims on 2026-10-22 for *"one cycle"*. Five weeks and 56 orders later, 116 shim files still sit in `standin/`, forcing defensive duck-typing workarounds (e.g., `cubbyllm/brain.py:80`: `# duck-typed: emitter vs standin.emitter import paths`) and polluting every symbol search Claude runs.

**Recommendation:** Schedule one clean, mechanical work order using `tools/moves_0031.json` to rewrite the 337 remaining imports, verify `tools/shim_count.py` prints `0`, and retire the 116 shims.

### 4.5 Smaller Code Smells & Stale Docs Worth Cleaning Up

- **Import-time global side effect:** `cubbyllm/brain.py:47-48` executes `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at module import time. Libraries should never mutate process-wide `sys.stdout`; move this to CLI entrypoints (`cubby_serve/api.py::main`, `cubby_serve/__main__.py`).
- **Inline `__import__("re")` / `__import__("pathlib")` anti-pattern:** `plan_verify.py` calls `__import__("re").compile(...)` 11 times (including inside `_WORD_RX()` on cache miss at line 693); `pipeline.py:97`, `wikidata_aliases.py:21`, and `cubby_serve/get.py:17` do the same. Both `re` and `pathlib` are stdlib modules — use normal top-level imports.
- **Unclosed file handles:** `cubbyllm/emit/seats_450m.py:109, 146, 155, 164, 177` and `cubby_serve/api.py:83` / `get.py:116, 122, 128` call `open(...)` without `with` context managers (leaks descriptors on Windows, especially in long-running servers).
- **Stale architecture docs:** `Architecture.md` still describes toy-scale `MinGRUBackbone` (d=128, L=2, July 2026) and makes no mention of `HybridBackbone`, `base450m`, or `spatial.py`. `cubbyllm/core/config.py:9` still claims `"Backbone shape is deliberately NOT here — that decision is still open (H-D1)."`.

---

## 5. P1 — Test Suite Speed & Guard Accuracy

Running just the three guard test files (`test_guards.py`, `test_package_rules.py`, `test_local_paths.py` — 26 tests total) currently takes **25.09 seconds**. Across **302 test files**, full-suite feedback loops are unnecessarily slow for Claude Code.

### Two Specific Bugs/Bottlenecks Found in the Guards

1. **`tests/test_guards.py::_production_importers()` has a false-negative bug AND is ~3× slower than necessary:**
   - Line 107 scans `repo.rglob("*.py")`, skipping only `tests/`, `.venv*`, and filenames starting with `test_` or `exp_`.
   - That means it AST-parses all **1,433 `.py` files** in the repo — including **341+ scripts in `validation/`**, `tools/`, and `standin/data/`.
   - **Bug:** If a non-`exp_` helper in `validation/` (e.g., `validation/sft/*.py` or `validation/builders/*.py`) imports a `cubbyllm/` module marked `Wiring.WIRED`, the guard **falsely treats it as wired in production**, even if zero production code calls it!
   - **Fix:** Restrict `_production_importers()` to `cubbyllm/`, `cubby_serve/`, and `standin/` (excluding `standin/tests/` and `standin/data/`). This makes the guard both **accurate** and **~3× faster**.
2. **`tests/test_local_paths.py` rescans every tracked text file on every run:**
   - Scanning >2,500 tracked files (including multi-MB JSON logs in `validation/logs/`) dominates the 25s runtime.
   - **Fix:** Support a fast mode (`CB_FAST_GUARDS=1` or `--staged` checking `git diff --cached --name-only`) for inner dev loops, while running the full scan on commit/CI. (Untracking the 464 scratch files in §3.1 will also cut scan time immediately.)
3. **Standardize pytest markers in `pyproject.toml`:** Tag slow integration/repo-scan tests with `@pytest.mark.slow` so Claude Code can run `pytest -m "not slow"` in ~2–3s during iteration and full `pytest` once before committing.

---

## 6. P1 — `cubby_serve/` Hardening & Concurrency

`cubby_serve/` is the BSL-1.1 product layer and already integrates OIDC (`cubby-auth`), roles, and API keys. Several real concurrency and security issues exist today:

1. **Data race in `CubbyBrain.turn()` under `ThreadingHTTPServer` (`cubbyllm/brain.py`, `cubby_serve/api.py:125`):**
   - `AskLoop` has `self._lock`, `Cubby450mTrunk` has `self._lock`, `CubbyAuth` has `self._lock`, and `_LOOP_RING` has `_LOOP_LOCK`.
   - **`CubbyBrain` has NO lock at all.** Concurrent `POST /turn` requests race on `self.chat.chem` (neurochemistry ODE), `self.chat.signals`, `self.chat.emotion`, `self.worlds`, and `self.events` (`collections.deque` iterated in `get.py:172` while mutated in `brain.py:106`).
   - **Fix:** Add `self._lock` around `CubbyBrain.sense()`/`turn()`/`trace()` (or move mutable conversation/mood state into a per-session object).
2. **Thread-unsafe shared cache in `cubby_serve/board/api.py:46-56, 94-103`:**
   - `_cache`, `_refreshing`, and `_rx` are read by HTTP handler threads while mutated by `start_refresh()`'s background thread with no lock. Notably, `_pattern()` calls `_rx.clear()` (line 101) on a miss, which can cause a concurrent thread to hit a `RuntimeError` or race.
3. **Unbounded request body reads (`cubby_serve/post.py:19, 32, 49`):**
   - `n = int(h.headers.get("Content-Length", 0)); h.rfile.read(n)` has **no upper bound** (a malicious or buggy client sending `Content-Length: 2000000000` can OOM the server) and does not guard against negative or non-integer `Content-Length` headers cleanly.
   - **Fix:** Extract one helper `_read_json(h, max_bytes=1_048_576)` returning `(body, err_code)` and use it across all POST routes.
4. **Hardcoded `Access-Control-Allow-Origin: *` (`cubby_serve/api.py:50, 57, 149`):**
   - Sent on every JSON response and on `/loop/stream` (including audit event streams). Even when `cubby-auth` is enabled, wildcard CORS is bad hygiene if exposed beyond localhost. Make allowed origins configurable (defaulting to `self.origin` when `--auth-issuer` is set, or explicit `--cors-origin`).
5. **Global mutable state vs. per-user isolation:**
   - `README.md` highlights per-user VM isolation (`exp_r20`), and `auth.py` resolves per-user `Identity` and `tenant`, yet `CubbyBrain` and `AskLoop` still keep one global `self.history`, `self.world`, and `self.chat.chem` shared across all callers (only partially scoped when `CB_API_ROLES=1`). Document this boundary explicitly or plan a `SessionState` container before any multi-user deployment.
6. **Structured logging instead of `print(..., flush=True)`:**
   - Silent `except Exception` catches in `post.py:84`, `ask_memory.py:67`, and `board/api.py:85` swallow tracebacks into `print` or HTTP 500 strings. Adopting Python's `logging` module with request IDs makes debugging live server issues dramatically easier for both Nick and Claude.

---

## 7. P2 — Strategic / Scientific Critique

Constructive observations on the research trajectory itself, based on `README.md` and orders `0078`–`0087`:

### 7.1 The Proposer–Gate Contradiction (Why Open-Domain Coverage Is <1%–3%)

Look at `README.md`'s 24 November 2026 numbers:
- **SimpleQA (4,326):** 6 spoken, 0 wrong, 4,320 refused (**0.14%** coverage).
- **Natural Questions (3,610):** 13 spoken, 2 wrong (+1 debatable) (**0.36%** coverage).
- **WebQuestions (2,032):** 62 spoken, 1 wrong (+2 debatable) (**3.05%** coverage).
- **AA-Omniscience (600):** 0 spoken (**0.00%** coverage).

Notice why **order `0081`** (constraining the 450M to emit only relations held by the store) was killed (198 → 188 answered on gen-3):
> *"because the gate reads the question's own word ('born') and the constraint took that word away."*

**This exposes a fundamental architectural tension:**
- The **store/VM layer** wants canonical ontology relations (e.g., `place of birth`, `country of citizenship`).
- The **`covers()` gate** currently requires **lexical surface overlap** between the emitted relation string and the question text!

That forces the proposer into an impossible bind: emit the store's canonical relation and fail `covers()`, OR emit the question's surface phrasing and fail `unknown_relation` (hoping lever 4/6 rescues it). Meanwhile, patching `covers()` with v1→v6 regexes and capitalization rules (`0087`) treats symptoms rather than the interface flaw.

**Constructive proposal — Span-Cited Plans:**
Extend the slot paradigm (`H-E15`) from entities/numbers to **relation justifications**:
- Instead of `covers()` reverse-engineering which words of the question correspond to `HOP1 "country of citizenship"`, have the proposer emit `(relation_id, span_id)` or annotate which spans of the question each hop accounts for.
- Then disposal splits cleanly into two independent, testable checks:
  1. **Structural span coverage:** Does the union of entity slots + relation spans cover the content words and constraints (years, numbers, entities) of the question? (Solves the `0087` "Ghana" bug structurally, with zero capitalization heuristics.)
  2. **Lexical/ontology validity:** Can that question span denote that canonical relation (checked via the deterministic relation-alias table / lexicon)? (Unblocks constrained decoding like `0081` because the model can now safely emit canonical store relations!)

### 7.2 Validation Script Duplication & Recipe Drift (`validation/` Has 341 Root Scripts)

In order `0082`, arm P had to be retrained because a batch-size recipe mismatch (`P8` vs `P32`) slipped into the run. With **341 standalone `.py` scripts** in `validation/` and **89 files** in `notebooks/`, every new experiment re-implements CLI parsing, run logging, recipe printing, VM setup, and freeze hashing.

**Recommendations for Claude:**
1. **Create `validation/_harness.py`** providing one reusable `RunContext`:
   - Auto-records git SHA, dirty status, VM binary SHA, adapter SHA, active `CB_*` flags, GPU/CPU hardware, and hyperparameters at run start **and** end (generalizing `0085`'s excellent freeze check to *every* experiment).
   - Provides a standard `--compare-recipe <reference_log.json>` check that warns/aborts if batch size, LR, or step count silently diverge from the reference run before burning Colab compute units!
2. **Archive completed experiment scripts** (`validation/archive/2026-07_campaign/`, `2026-08_standin/`, etc.) so `validation/` root contains only active trainers, active benches, and shared helpers (~20–30 files instead of 341).
3. **Add a tiny open-domain canary set (e.g., 200 stratified WebQuestions + NQ + SimpleQA dev rows)** alongside `gen-3` on proposer experiments (`0082`, `0084`, `0087`), so every proposer change shows whether it improves *real human phrasing*, not just synthetic store-chain templates.

### 7.3 Keep `0085`'s Freeze & Independent-Replay Pattern Everywhere

Order `0085`'s design — hashing weights + rule code + settings + VM binary at both start and end of a run, and testing the SHACL shape validator on 8 positive + 15 negative pin examples *before* trusting its verdict — is gold-standard ML engineering. Promote that pattern into `validation/_harness.py` so every future gated eval gets it for free.

---

## 8. Suggested Action Plan (Ordered by ROI)

| Priority | Action | Effort | Primary Benefit |
|---:|---|---|---|
| **1** | Rotate `docs/inbox.local.md` (<5 KB active); archive `TODO.md`; strip stale queue/status from `CLAUDE.md` | 30 min | Immediate massive token savings & fewer wrong assumptions by Claude |
| **2** | Add `tools/queue.py` + YAML frontmatter to active orders; triage the 13 `in progress` orders (1 active per worker) | 1 hr | Eliminates queue confusion between Cowork, Code A, and Code B |
| **3** | Untrack the 464 `_`-prefixed scratch files (362 MB `.pt`/`.sqlite`) + add repo-hygiene guard to `test_guards.py` | 1 hr | Clean public repo, faster git/grep/pytest |
| **4** | Fix `test_guards.py::_production_importers()` scope + add fast mode for `test_local_paths.py` + GitHub Actions CI | 1 hr | Accurate wiring guard, ~3× faster local checks, automated CI |
| **5** | Clean `.claude/settings.local.json` + add `/start-order`, `/finish-order` commands + `tools/hw_lock.py` | 1 hr | Faster Claude Code turns, zero A/B GPU collisions |
| **6** | Shard `CUBBYLLM_HYPOTHESES.md` into `INDEX.md` + `docs/hypotheses/group_*.md` | 1–2 hrs | Claude can read relevant hypotheses in <2k tokens instead of 130k |
| **7** | Add `cubbyllm/core/flags.py` registry + retire killed flags (`0081`, `0083`, `0043`) + fix `CubbyBrain`/board locks & POST body limits | 2–3 hrs | Removes dead code from `_ask()`, fixes real server race conditions |
| **8** | Run mechanical shim-retirement order (`tools/shim_count.py` 337 → 0; remove 116 `standin/` shims) | 2 hrs | Eliminates dual-import bugs & symbol-search noise |
| **9** | Populate `pyproject.toml` deps + `[tool.ruff]`; fix `brain.py` `stdout.reconfigure` and unclosed `open()` calls | 1 hr | Clean installability and automated static checks |
| **10** | Design order: Span-cited plans + split `plan_verify.py` (950 lines) + `validation/_harness.py` recipe-check | 1–2 days | Resolves the `0081`/`0087` proposer–gate tension at the root |

---

## 9. Questions & Nick's Rulings (2026-11-27)

Decisions confirmed by Nick on 2026-11-27:

1. **Git bloat cleanup:** **Safe `git rm --cached`** (no history rewrite / no force-push). Cleans tip immediately and pairs with a repo-hygiene guard in `tests/test_guards.py` → drafted as **order `0088`**.
2. **Killed-flag retention policy:** **Yes — remove runtime hooks** from `cubbyllm/` once a flag-gated experiment is killed and its order is `done` (reproducing ablations from git history + the order's commit hash) → drafted as **order `0090`**.
3. **Retiring the 116 `standin/` shims:** **Safe to retire** once in-repo imports hit 0 (move aside to `_discard/` or back up to another drive per the no-permanent-deletion rule) → drafted as **order `0091`**.
4. **`cubby_serve/` deployment timeline:** **No multi-user/public deployment soon** — keep `cubby_serve/` concurrency, CORS, and body-limit hardening at P1/medium (not blocking P0 work).
5. **Next steps:** **Approved to proceed** with low-risk local P0 cleanups and drafting `proposed` work orders:
   - **Done immediately (2026-11-27):** rotated `docs/inbox.local.md` (526 KB → ~8 KB active, full history preserved in `docs/inbox_archive_2026-10-22_to_2026-11-24.local.md`), added `tools/queue.py`, and added `.claude/commands/{start-order,finish-order}.md`.
   - **Proposed work orders ready for review/approval:**
     - `docs/orders/0088-repo-hygiene-and-fast-guards.md`
     - `docs/orders/0089-claude-docs-and-hypotheses-index.md`
     - `docs/orders/0090-flag-registry-and-killed-flag-cleanup.md`
     - `docs/orders/0091-retire-0031-shims.md`
