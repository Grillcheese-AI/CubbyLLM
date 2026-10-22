# 0004 — Rename "hypertokens" to "loci tokens" in the code's prose
status: done
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
- **Commits:** `1609209` claimed · the commit carrying this report (the rename; one commit, as the order asks).
- **Files:** `cubbyllm/model/spatial.py` (the one "were called hypertokens until 2026-10-22; see H-K10's Name"
  line, at the first use, in the `query="placed"` paragraph); `validation/exp_hypertoken_recall.py` (docstring, the
  printed header: "loci-token recall --"), `exp_k10_hypertoken.py` (docstring, the arm's printed name, a `--query`
  help string), `make_fact_shard.py`, `make_spatial_shard.py`, `train_base.py` (a comment);
  `notebooks/_build_spatial.py` and `_build_grow_mem.py`, both notebooks regenerated. 20 lines in all.
- **Tests:** full `pytest tests`: 406 passed, 5 skipped, 1 failed (the known wiring guard, order 0002), as before.
  Every edited `.py` parses to the same syntax tree once string text is ignored. No code, name, flag or key changed.
- **Measured** (against the acceptance):
  - `grep -i hypertoken` over `cubbyllm/`, `notebooks/_build_*.py` and `validation/*.py`, leaving out the
    zip2zip-sense `exp_c2_lzw.py`, finds 33 lines naming the kept files, modules and logs (`exp_hypertoken_recall`,
    `exp_k10_hypertoken`, `hypertoken_recall_*`), plus the one "called hypertokens until" line, plus two more.
    Those two are `exp_k10_hypertoken.py`'s arm name, `"hypertoken"`: the `--only` CLI value and the key its
    report is written under in `exp_k10_hypertoken.json`. They are a flag value and a JSON key, so they are kept.
  - The zip2zip-sense files are untouched: `exp_c2_lzw.py`, `VALIDATION_REPORT.md`, `README.md`,
    `validation/README.md`. `validation/README.md` was on the Build list, but its only mention is `exp_c2_lzw.py`'s
    row, in the LZW sense.
  - Notebooks: the cell counts, metadata and every non-source field are unchanged. `grow_mem_continue.ipynb` changes
    one markdown sentence. The other three changed lines are in **code** cells, all of them prose the Build asks
    for: two comments (`# ... loci-token shard(s) in the cache`) and the setup check's `assert` message ("push the
    loci-token commits first"). No executable token differs. The acceptance line "text cells only" holds in that
    sense; strictly, three code cells' text changed.
- **Not verified here:** nothing GPU-side is touched. Two printed strings changed (`exp_hypertoken_recall`'s header and
  `exp_k10_hypertoken`'s arm name, which its JSON also carries as the `"name"` value), and nothing in the repo
  parses either one. Dated logs keep the old text.
- **Found on the way:** none.
- **Questions:** should the `exp_k10_hypertoken.py` arm key `"hypertoken"` be renamed in a later order? It is
  user-visible as `--only hypertoken` and as the arm's key in its JSON.

## Review (Cowork, 2026-10-22)
- Checked: the grep over `cubbyllm/`, `notebooks/_build_*.py` and `validation/*.py` finds only kept file, module,
  log and key names, plus the one "called hypertokens until" line. Tests are as reported.
- The question: **keep** `--only hypertoken` and its JSON key. Both are a flag value and a key that the dated
  `exp_k10_hypertoken_*` logs carry, so renaming them would break the reading of those logs, for no reader's
  gain. The three code-cell comments and the assert message are the prose the order asked for. Done.

