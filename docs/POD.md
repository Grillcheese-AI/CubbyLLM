# A rented card for H-E20 (order 0045)

The grown trunk with its episodic memory against the 450M continuation, both arms run to the end on one rented
card, resumable after a spot preemption. Nick rents and starts the card from this page; Cowork watches; the readout
(`runs/readout.md` on the remote) applies H-E20's pre-registered gates. Nothing here moves a number.

## Cards and image

| Card | Dense BF16 | Memory | Prices seen 2026-10-27 (recheck before renting) |
|---|---|---|---|
| H200 | 989 TFLOPS | 141 GB | ~$3.1-5 /h on demand, ~$2.3 /h spot |
| B200 | 2,250 TFLOPS | 180 GB | ~$3.75-8 /h on demand, spot from ~$3.2 /h |
| H100 80 GB | 989 TFLOPS | 80 GB | works, slower; the memory limits the micro-batch |

- **Image:** PyTorch on CUDA 12.8. A B200 (sm_100) needs torch >= 2.7 with CUDA >= 12.8; `setup.sh` refuses an older
  stack and says what is missing (it never downgrades anything). An H200 runs on the same image.
- **Disk:** at least 250 GB of local NVMe (3 x the inputs + 100 GB; the preflight checks).
- **Spot:** welcome. Checkpoints go to the remote every 15 minutes (`CB_CKPT_MIN=15`); a preemption loses at most that
  plus a fetch. Both arms must run on the **same card model**: a replacement of another kind refuses to resume
  ("compares nothing").

## Before renting: the inputs, from home

```
python validation/pod/prepare_inputs.py --src <full token cache> --dst <token_cache_r3> --exclude-dir <token_cache_cont> \
    --tokenizer <bbpe128k.json> --start <base450m_cont_stable_end.pt> --cont-metrics <base450m_cont_metrics.jsonl> \
    --emitter-data <emitter_sft_v12e_w_step.jsonl> --talk-data <talk_v2_2.jsonl> --hours 14 --card b200 --dry
```

`--dry` prints the slice's plan (blocks neither the first run nor the continuation took) and the epoch table for the
most tokens any arm can see; every row must stay at or under 2 epochs, else raise `--frac`. Then the same line without
`--dry` builds the slice, the decay shards and `facts_text_far`, stages the inputs beside the cache and, with
`--upload` and `CB_POD_REPO` + `HF_TOKEN` in the environment, uploads them to the private dataset repo (resumable:
rerun the same line after an interruption). The manifest `pod_inputs.json` records every file's sha256 and the commit
the pod checks out: push that commit first.

## The variables

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | the private repo's token; read from the environment, redacted from every log, never written |
| `CB_POD_REPO` | the private dataset repo holding `inputs/` (and receiving `runs/`, `logs/`, `pod_state.json`) |
| `HOURS_PER_ARM` | training hours per arm: 7 for a 20 h on-demand rental, 12-14 with spot |
| `CB_POD_STOP` | the provider's own stop command, run when the last upload is done (a card never sits billing idle) |
| `CB_REPO_URL` | this repo's git URL, cloned on the pod at the manifest's commit |
| `POD_WORK` | the local NVMe folder (default `/workspace/pod`) |

## The one command

```
export HF_TOKEN=... CB_POD_REPO=<user>/<repo> CB_REPO_URL=<this repo's git URL> HOURS_PER_ARM=14 CB_POD_STOP='<stop command>'
git clone "$CB_REPO_URL" /workspace/pod/CubbyLLM && bash /workspace/pod/CubbyLLM/validation/pod/setup.sh
```

It checks the card and the stack, installs what the image lacks,
clones the repo at the manifest's commit and starts `run_pod.py` in a tmux session named `pod`.

**Watch:** `tmux -L pod attach -t pod` (detach with Ctrl-b d); `tail -f $POD_WORK/logs/run_pod.log`; an arm's own log is
`logs/train_<arm>.log`, its metrics `runs/<arm>/grow_mem_<arm>_metrics.jsonl`. The remote mirrors them: `pod_state.json`
says the phase, the arm and the last step uploaded.

## The phases and their hours

| # | Phase | What | Hours |
|---|---|---|---|
| 0 | preflight | card, versions, disk, token, a write to the remote; any failure exits at once | minutes |
| 1 | fetch | the manifest's files to NVMe, sha256-checked, resumable | 0.5-1 (the slice) |
| 2 | kernels | the Triton kernel tests on this card; a failing kernel is turned off for the run and recorded | minutes |
| 3 | speed | `speed_check.py` and the trainer's bench per arm: the fastest micro under 92% of memory; the read count (16, or 8 when the grown arm runs under half the control's MFU), written to `runs/bench.json` before any training | 0.5 |
| 4 | control | the 450M from `stable_end`, round 2's config, `HOURS_PER_ARM` | `HOURS_PER_ARM` |
| 5 | grown | `grow_base.py` step 0 (exact, or no training), then the same config and hours | `HOURS_PER_ARM` + 0.5 |
| 6 | probes | `exp_hypertoken_recall.py` on facts_text_far (both finals and the 450M), `exp_needle_forward.py` at 1024 / 2048 / 4096 | 1-2 |
| 7 | readout | gates 0-3 (`validation/grow_mem_readout.py`) to `runs/readout.json` and `.md` | minutes |
| 8 | adapters | the emitter (v12e slots) and the talk adapter (talk_v2.2) on the winner, the notebooks' settings | 1-2 |
| 9 | done | the last upload; `CB_POD_STOP` | minutes |

Budget: 2 x `HOURS_PER_ARM` + about 4 h. With spot at 12-14 h per arm, plan a 32 h window and expect to lose 15 minutes
per preemption.

## A spot preemption

The pod disappears; nothing is lost but the minutes since the last checkpoint. Rent the **same card model**, set the
same variables, run the one command again. The new pod reads `pod_state.json` from the remote, fetches the inputs and
the newest slot of the arm under way, and `train_base.py` resumes from it; phases already done are not run again. The
state counts preemptions; the readout reports them with the hours and tokens per arm.

## After the run

`runs/readout.md` carries gates 0-3 with the tokens, tok/s and MFU per arm; `runs/adapters/` the adapters on the winner.
The serving gates (E6-E9, exp_r17, the step loop, `exp_if_follow.py`) run at home on grilly2, not on the meter. Record
the result under H-E20 in `CUBBYLLM_HYPOTHESES.md`, with `readout.json` and the metrics copied into `validation/logs/`.

## The smoke, at home

```
python validation/pod/run_pod.py --smoke --work <tmp>/work --remote folder:<tmp>/remote
```

A tiny shape on the CPU, minutes: every phase, a folder standing in for the repo. `tests/validation/test_pod_runner.py`
kills it mid-arm and resumes it in an empty folder.
