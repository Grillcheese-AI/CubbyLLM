"""Run the cubemind `_archive` tests against the ARCHIVED modules, moving nothing.

Wired: STANDALONE (validation helper; never imported by cubbyllm/).

WHY THIS EXISTS (WO-2.7)
------------------------
The archive ships 22 test files, which is what makes "test it before porting it"
cheap. They cannot run as collected: they import `cubemind.execution.<name>`, and
those modules now live at `cubemind._archive.execution.<name>`. Rather than edit
that repo, this aliases each archived module into `cubemind.execution` in
`sys.modules` before pytest collects. Live modules of the same name
(`world_encoder`, `hyla`, `cvl`, ...) are left alone and resolve normally.

The archived modules import each OTHER through `cubemind.execution.<sibling>`, so
a single alphabetical pass fails whenever one loads before its sibling is
registered (`causal_graph` -> `data_normalizer`). Hence the retry loop.

Result as of 2026-09-15: **205 passed**; only `document_ingestor` fails to import
(it wants `cubemind.model1`), and it is 32 lines of plumbing nobody wants.

WHERE THE CHECKOUT IS
---------------------
Set `CUBEMIND_ROOT`, or pass `--root`, pointing at a cubemind checkout beside this
one. No path is hard-coded: `docs/WORK_ORDERS.md` rule, "No local drive paths in
committed files".

    CUBEMIND_ROOT=<a cubemind checkout> python validation/run_cubemind_archive_tests.py
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

argv = sys.argv[1:]
root = os.environ.get("CUBEMIND_ROOT")
if "--root" in argv:
    i = argv.index("--root")
    root = argv[i + 1]
    del argv[i:i + 2]
if not root:
    # a sibling checkout is the common layout; otherwise the env var is required
    guess = pathlib.Path(__file__).resolve().parents[2] / "cubemind"
    if (guess / "cubemind" / "_archive").is_dir():
        root = str(guess)
if not root:
    raise SystemExit(
        "set CUBEMIND_ROOT (or pass --root) to a cubemind checkout -- no path is "
        "hard-coded here; see this file's docstring")

REPO = pathlib.Path(root)
ARCH = REPO / "cubemind" / "_archive"
if not ARCH.is_dir():
    raise SystemExit(f"no cubemind/_archive under {REPO}")
sys.path.insert(0, str(REPO))

import cubemind  # noqa: E402
import cubemind.execution as live_exec  # noqa: E402

live_dir = REPO / "cubemind" / "execution"
live_names = {p.stem for p in live_dir.glob("*.py")}
arch_names = {p.stem for p in (ARCH / "execution").glob("*.py")} - {"__init__"}

aliased, shadowed, failed = [], [], []
pending = []
for name in sorted(arch_names):
    if name in live_names:
        shadowed.append(name)      # live wins; do not mask a module that still exists
        continue
    pending.append(name)

# Archived modules import each other through `cubemind.execution.<sibling>`, so a
# single alphabetical pass fails whenever a module is imported before its sibling
# has been aliased (causal_graph -> data_normalizer). Retry until no progress.
while pending:
    progressed = False
    still: list[str] = []
    for name in pending:
        try:
            mod = importlib.import_module(f"cubemind._archive.execution.{name}")
        except Exception as e:                                  # noqa: BLE001
            still.append(name)
            failed = [f for f in failed if f[0] != name]
            failed.append((name, f"{type(e).__name__}: {e}"))
            continue
        sys.modules[f"cubemind.execution.{name}"] = mod
        setattr(live_exec, name, mod)
        aliased.append(name)
        progressed = True
    pending = still
    if not progressed:
        break
failed = [f for f in failed if f[0] not in aliased]

print(f"aliased into cubemind.execution: {', '.join(aliased) or '(none)'}")
if shadowed:
    print(f"left alone (a live module of the same name exists): {', '.join(shadowed)}")
for name, err in failed:
    print(f"!! could not import archived {name}: {err}")
print()

import pytest  # noqa: E402

DEFAULT = [
    "test_vsa_translator.py",
    "test_decision_tree.py",
    "test_causal_graph.py",
    "test_event_encoder.py",
    "test_causal_codebook.py",
    "test_decision_oracle.py",
    "test_oracle_trainer.py",
    "test_future_decoder.py",
    "test_data_normalizer.py",
    "test_end_to_end_causal.py",
]
picked = argv or DEFAULT
paths = [str(ARCH / "tests" / p) for p in picked]
raise SystemExit(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", *paths]))
