# CubbyLLM target package layout — draft spec (H-F1)

Status: **PROPOSAL, 2026-07-23** — written as H-F1's validation step ("write the
target package layout as its own short spec before porting or writing any
code"). Nothing below exists yet except `validation/`. Revise freely; the point
is that structure gets decided *on purpose*, before code accretes it by
accident for a third time.

## Layout

```
CubbyLLM/
├── cubbyllm/                  # THE package — the only import root
│   ├── core/                  # config, constants, seeds, logging. No ML code.
│   ├── ops/                   # VSA algebra FACADE — the single place that
│   │                          #   touches grilly.experimental.vsa (H-B5).
│   │                          #   numpy fallback lives here too, mirroring
│   │                          #   cubemind's proven 3-tier pattern.
│   ├── model/                 # (dir named `model`, and NOTHING else named
│   │   │                      #   model.py / models/ anywhere in the repo)
│   │   ├── backbone/          # the trunk — Group D decision, built last
│   │   ├── memory/            # H0-hardened memory layer (Group A outcome)
│   │   ├── binding/           # binding head + real unbind (Group B outcome)
│   │   └── vocab/             # tokenizer adapter, embedding mechanism
│   │                          #   (static or hyper-encoder per H-C6), output
│   │                          #   head (retrieval per H-C3)
│   ├── training/              # data pipelines, loops, eval harnesses
│   └── bridges/               # ALL cross-repo interfaces (cubemind world
│                              #   model, CubeLang), first-class per H-F2 —
│                              #   never scattered duck-typed callbacks
├── validation/                # hypothesis experiments + logs/ (exists).
│                              #   Standalone by definition; never imported
│                              #   by cubbyllm/.
└── (docs at repo root)        # CLAUDE.md, CUBBYLLM_HYPOTHESES.md, this file
```

Mirrors cubemind's *intended* separation (core/ops/perception→model/execution→
bridges) without importing its sprawl. Perception/VM stay in cubemind — this
repo is the model, `bridges/` is the seam.

## Anti-sprawl rules, each mapped to an observed failure

| Rule | Failure it prevents (observed in siblings) |
|---|---|
| One implementation per component; alternate backends live BEHIND `ops/`, never as parallel trees | `trunk/` vs `trunk_torch/` duplication (cubby-lm) |
| No tracked `sandbox/`, `cloned/`, or vendor dirs; dependencies are dependencies (`grilly` is imported, not copied) | cubemind's ~35GB tracked `sandbox/`, `cloned/` |
| Nothing importable that is retired: deleted code lives in git history, not `_archive/` | cubemind's gitignored-but-importable `_archive/` (its two grounding encoders still point INTO it) |
| Exactly one thing named `model` (the directory). CI greps for `model.py`/`models/` and fails | `model.py` / `models/` / `model` collision |
| Every module's docstring declares `Wired: <path>` or `Standalone` — enforced by a repo lint, not convention | `SegmentMemory`/`MTP`/`CubeLangHead` built-but-never-wired trap |
| Corpora/checkpoints never in-repo; configs reference `D:\grillcheese_training_data` paths + a pinned manifest hash (H-G3) | cubby-lm's 2GB corpora inside the package tree; `unified/` silent drift |
| Any number quoted in docs links the `validation/logs/` file that produced it | self-reported numbers presented as measurements |

## Naming decisions made now (cheap now, expensive later)

- Import name `cubbyllm`, lowercase, no hyphen.
- Test tree `tests/` at root, mirroring `cubbyllm/` one-to-one.
- `validation/` keeps its own README and never graduates into the package —
  an experiment that becomes real code gets *rewritten* into `cubbyllm/` with
  tests, not moved.
