"""Scratch (untracked): write order 0045's report and set it to review."""
import io

REPORT = '''- **Commits:** `29d4409` in progress; this one: parts A-D, the readout module, the tests, the smoke.
- **Files:**
  - **A.** `validation/pod/prepare_inputs.py`: the slice (`make_base_cache.py --frac 0.3 --exclude-frac 0.2 --exclude-dir`), the
    decay shards, `facts_text_far` (`--far --mode text --doc 2048`), the epoch table for the most tokens an arm can see
    (`--card b200 --hours 14 --mfu 0.35`: 13.7B; every row at most 2 epochs, else "raise --frac"), the manifest
    (`pod_inputs.json`: size and sha256 of every file, the commit, the named roles), the staging folder (hard links), the
    upload (`upload_large_folder`, resumable; a 16 MB probe times the connection first; `--mbps` to skip it). `--dry` plans
    and prints, writes nothing.
  - **B.** `validation/pod/setup.sh` (the card, the stack -- a B200 on torch < 2.7 / CUDA < 12.8 is refused with what is
    missing, nothing downgraded -- the packages, the clone at the manifest's commit, tmux, how to watch); `run_pod.py`
    (the phases, the state file on the remote, the knobs written once: `HOURS_PER_ARM` and the smoke's steps are kept
    across pods); `pod_phases.py` (preflight, fetch, kernels, speed, the arms' environment: round 2's config and the tiny
    smoke's); `pod_train.py` (the arms with the `Watcher` uploading each slot as its sidecar turns valid, step 0 exact or
    no training, probes, readout, adapters, done); `pod_remote.py` (`FolderRemote` for the smoke, `HfRemote` for the
    private dataset repo; sha256-checked fetch; the token redacted from every line logged).
  - `validation/grow_mem_readout.py` (new): gates 0-3 from the metrics and the probes, one place; the notebook's readout
    cell now calls it (`notebooks/_build_grow_mem.py`, the `.ipynb` regenerated).
  - **C.** `validation/exp_if_follow.py`: ten rule kinds read off the held chats' prompts (words, sentences, forbid,
    require, language, list, json, no_markdown, case, first), 796 of the 3,186 held prompts carry one; ~300 drawn
    round-robin over the kinds; scored over the talk seat as the server runs it (`cubbyllm.emit.seats_450m` on grilly2)
    or over a replies file from any other path (`--dump` writes the prompts for it).
  - **D.** `docs/POD.md`.
  - `validation/train_base.py`: the in-context copy probe draws within a tiny vocabulary (the smoke's 40 tokens; a real
    vocabulary exactly as before). The file was over 300 lines already.
  - Tests: `tests/validation/test_pod_runner.py` (2), `test_grow_mem_readout.py` (3), `test_prepare_inputs.py` (3),
    `test_exp_if_follow.py` (3).
- **Tests:** 1134 passed, 1 failed (`test_every_module_declares_wiring`, the known red of order 0002), the suite without
  the runner's; the runner's two pass apart (about 3 minutes on this CPU).
- **Measured** (against each acceptance line, on this CPU):
  - **The runner end to end on the tiny shape:** `run_pod.py --smoke` runs every phase and writes its outputs
    (`bench.json`, `kernels.json`, both arms' finals and metrics, `grow_step0.json` exact, `probes.json`, `readout.json` /
    `.md`, `adapters.json`, the logs) in about a minute. **Killed mid-arm** (the process killed once a slot is uploaded)
    and started in an empty folder with the same remote, it runs the preflight and the fetch again, pulls the newest slot,
    and `train_base` resumes from it ("resumed ... @ step N" in the arm's log, N the uploaded step); the phases done before
    (kernels, speed) are not run again (no log of theirs in the new folder); the state counts the preemption. A third start
    finds everything done and runs nothing.
  - **A failing kernel test** (a fake test file that fails) records `CB_MINGRU_TRITON: off (tests failed, exit 1)` and the
    arms' environment carries the flag at 0. **Another card model** (the state says B200, the pod says H200) refuses before
    fetching: "compares nothing". A B200 on torch 2.5.1 is refused with "torch >= 2.7".
  - **The readout** gives gates 0, 1, 3a, 3b from fixture metrics as the notebook's cell did (its formulas restated in the
    test), the growth kill at +0.05, gate 0 failing on unequal hours or cards; gate 2 from the probes' reports (exact on
    facts_text_far against both floors, `_copy`, the needle's beyond-window cells against the control at chance).
  - **`exp_if_follow.py`:** each of the ten rules passes a hand-written good reply and fails a bad one; the draw is over
    the kinds; the score counts per kind.
  - **`prepare_inputs.py --dry`** on a toy cache: the planned blocks overlap none the first run (`--exclude-frac`) or the
    continuation (`--exclude-dir`'s `_subset.json`) took; a shard out of unseen blocks takes what is left and says so;
    the table flags a shard drawn more than twice; nothing is written.
  - **On the real cache:** at `--frac 0.3` the slice is ~5B tokens; a B200 control at 14 h and 35% MFU draws ~13.7B, so
    the table will say "raise --frac" on the big shards. Run `--dry` on the full cache and set `--frac` (or `--tokens`
    from the bench's measured tok/s) before building; the build itself is hours of disk, not done here.
- **Not verified here:** anything on a card (the kernel tests pass or fail there; `speed_check.py` runs only on CUDA;
  the probes and the adapters are skipped in the smoke and recorded as such); the upload (needs `HF_TOKEN` and
  `CB_POD_REPO`; the code path is `upload_large_folder` as documented); the prices.
- **Found on the way:**
  - `train_base.py`'s copy probe assumed a real vocabulary (ids 1000-30000): fixed for the smoke, bit-identical otherwise.
  - A new pod must run the preflight and the fetch again even when the state says they are done (its own card, its own
    empty disk); the first version skipped them and resumed into an empty corpus.
  - The adapters phase needs the emitter's and the talk's data in the inputs (`--emitter-data`, `--talk-data`); the order
    listed only the slice, the tokenizer and the start checkpoint. Without them the phase records "skipped".
  - `exp_if_follow.py` runs the talk seat on grilly2 (the serving path) or scores a replies file; a torch path would mean a
    torch LoRA loader the repo has only inside the trainers. Say if you want it.
- **Questions:**
  1. The epoch table at 14 h on a B200 asks for a bigger slice than `--frac 0.3`: should `--frac` follow the table (0.6-0.8
     of the cache, 25-35 GB to upload) or should `HOURS_PER_ARM` fall to what 0.3 feeds at 2 epochs (about 7 h)?
  2. The read count rule (16, or 8 under half the control's MFU) and the 92% memory line are in `pod_phases.py` as the
     order wrote them; the bench runs `speed_check.py` first and then the trainer's bench with its chosen kernels.
  3. The probes run on the card after the arms (1-2 h on the meter). Fine, or at home from the finals?
'''

p = "docs/orders/0045-rented-card-grow-and-memory.md"
s = io.open(p, encoding="utf-8").read()
s = s.replace("status: in progress (Claude Code, 2026-10-28)", "status: review (Claude Code, 2026-10-28)")
if "- **Commits:** `29d4409`" not in s:
    s = s.replace("## Report\n<!-- Claude Code fills this in, then sets status: review -->\n- **Commits:**\n- **Files:**\n- **Tests:**\n- **Measured:** <against each acceptance line>\n- **Not verified here:**\n- **Found on the way:**\n- **Questions:**",
                  "## Report\n" + REPORT.rstrip("\n"))
io.open(p, "w", encoding="utf-8").write(s)
print("ok", "status: review" in s, "- **Commits:** `29d4409`" in s)
