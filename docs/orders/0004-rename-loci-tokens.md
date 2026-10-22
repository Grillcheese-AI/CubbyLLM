# 0004 — Rename "hypertokens" to "loci tokens" in the code's prose
status: in progress
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("we need to rename ours") · hypothesis: H-K10

## Why
"Hypertoken" is taken twice. zip2zip (Geng et al. 2025, arXiv 2506.01084) uses it for LZW-merged tokens, and
H-C1/H-C2/H-C6 use it in that sense. Augeri 2025 (arXiv 2507.00002) uses it for error-correcting symbolic codes
written into an unmodified model's prompt. Ours, H-K10's unit, is now a **loci token**: the trunk's own embedding
bound to a place at the input and read back by unbinding (the method of loci). The difference and both citations
are in H-K10's **Name** paragraph (`CUBBYLLM_HYPOTHESES.md`). Cowork renamed the term in the docs it owns:
`CUBBYLLM_HYPOTHESES.md`, `CLAUDE.md` and `docs/ARCHITECTURE_DIAGRAM.md`.

## Build
- In docstrings, comments, printed text and notebook-builder prose, write "loci token(s)" where the H-K10 sense is
  meant. Files: `cubbyllm/model/spatial.py`; `validation/` (`exp_k10_hypertoken.py`, `exp_hypertoken_recall.py`,
  `exp_k10_far_facts.py`, `k10_far_docs.py`, `diag_*`, `make_fact_shard.py`, `make_spatial_shard.py`,
  `train_base.py`, `README.md`); `notebooks/_build_spatial.py` and `_build_grow_mem.py` (regenerate both notebooks).
  At first use in `spatial.py`'s docstring, add one line: "called hypertokens until 2026-10-22; see H-K10's Name".
- **Keep** every file name, log name, flag, meta key and JSON key: `exp_k10_hypertoken.py`,
  `exp_hypertoken_recall.py`, `hypertoken_recall_*` logs. Dated logs and the hypothesis record link to them.
  Where a kept name appears in prose, it stays as is.
- **Leave the zip2zip sense alone:** `exp_c2_lzw.py`, H-C1/H-C2/H-C6, `VALIDATION_REPORT.md`, and `README.md`'s
  vocabulary line. Also leave Nick's quotes, `docs/archive/` and the orders' history.

## Acceptance (pre-registered)
- In `cubbyllm/`, `notebooks/_build_*.py` and `validation/*.py`, `grep -i hypertoken` finds only kept file, log
  and module names, plus the one "called hypertokens until" line. The zip2zip-sense files are unchanged.
- The full `pytest tests` result is unchanged (406 passed, 5 skipped, the known guard). Both regenerated notebooks
  differ from the committed ones in text cells only.

## Constraints
- Rules in `CLAUDE.md` hold. Prose only: no behaviour, name or key changes. One commit.

## Out of scope
- Renaming files, flags or keys. The papers' prose (Nick's).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
