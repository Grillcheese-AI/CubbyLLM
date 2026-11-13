# 0071 - A logits hook in grilly2's generate: the host can steer each token as it is written
status: review (built 2026-11-12; the grilly2 branch waits for Nick's review)
written: 2026-11-12 by Cowork - approved: 2026-11-12 by Nick ("yes") - hypothesis: none

## Why
0067's step 0 read the decode loop (`python/grilly/infer/generation.py` in grilly2): `generate` takes only
`max_length`, `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `eos_token_id`,
`pad_token_id`, `attention_mask` and a streamer, and raises on anything else. The logits go straight to the sampler.
So the host can enforce only a required opening (a text prefix) and a length cap; every other rule can only be
checked after the reply is written. Three pieces of work wait on the same missing hook:
- **0067:** rules enforced while Cubby writes (a forbidden word masked, a list shape steered), not only checked after;
- **0068, route 1:** the programs seat choosing from an allowed list of tokens, exactly;
- **the relation step** (the qFHRR synthesis's selection seat): the 450M naming only one of the item's own properties.

transformers' `generate` has this hook already: `logits_processor`, a list of callables `(input_ids, scores) ->
scores` run on each step's logits before sampling, and `prefix_allowed_tokens_fn`. grilly2 follows transformers'
signature, so the same names keep it a drop-in.

- **What it brings:** the host can keep Cubby inside a rule while it writes, instead of finding the break afterwards.
- **Over a conventional model:** a hosted model is asked to follow a rule and may not; here the host can make some
  rules impossible to break, and checks the rest.
- **In plain words:** if you tell Cubby never to use a word, it can't, rather than being caught using it.

## Build
**A. grilly2** (a new branch from grilly2's `main`, pushed for Nick's review; `main` is not touched, per grilly2's
working agreement; grilly2's own `CLAUDE.md` holds there):
1. `generate(..., logits_processor=None, prefix_allowed_tokens_fn=None)`, transformers' names and meaning:
   - `logits_processor`: a callable or a list of callables, each `(input_ids, scores) -> scores`, run in order on
     each step's next-token logits (grilly tensors on the device) after the repetition penalty and before
     temperature, top-k, top-p and sampling, as transformers orders them;
   - `prefix_allowed_tokens_fn(batch_id, input_ids) -> list[int]`: a thin wrapper that builds an additive mask (0 for
     allowed ids, -inf otherwise) on the device for each row.
2. `None` for both leaves the loop exactly as it is today.
3. Works with padded batches and with rows dropped as they finish (each processor sees only the rows still running,
   with their own `input_ids`).
4. A helper for the common case: a mask of allowed ids as a device tensor, built once and reused across steps.

**B. CubbyLLM:** `cubbyllm/ops/grilly_seats.generate` passes `logits_processor` and `prefix_allowed_tokens_fn`
through when given (`cubbyllm/ops/` stays the only grilly importer). Default `None`; every seat unchanged.

## Acceptance (pre-registered)
- **Unchanged when off:** with both `None`, `generate` returns the same tokens bit for bit as the branch's base on
  grilly2's own test models, greedy and seeded sampling, equal-length and padded batches (pinned).
- **An allowed list holds:** a processor allowing a fixed set gives only tokens from it at every step, greedy and
  sampled, padded and not (pinned).
- **transformers parity:** the same processor on a small reference model (grilly2's pinned torch/transformers
  references) gives the same greedy tokens in grilly2 and in transformers (pinned where CI has the reference).
- **The seats:** CubbyLLM's served seats give identical output with the hook absent (the existing seat tests).
- **Cost, on the home card with the 450M, measured and reported:** per-step time with no processor, with a processor
  that returns its input, and with an allowed-list mask of 1,000 and of 18,000 ids. Line: the identity processor adds
  at most 2% per step. Kill: more than 10% for the identity processor (the hook would then need another shape before
  any consumer uses it).

## Constraints
- grilly2, never grilly v1. grilly2's `main` untouched; the branch waits for Nick's review.
- The hook is generic: no Cubby-specific code in grilly2.
- Processors run on device tensors; reading logits back to the host each step is not the hook's path (a test may).
- Rules in `CLAUDE.md` hold on the CubbyLLM side (< 300 lines a file, `__wiring__`, tests pin the claim).

## Out of scope
- The consumers: 0067's enforced rules (its step 4), 0068's route 1, the selection seat. Each uses the hook in its own
  order.
- Beam search, `stopping_criteria` and transformers' other generation arguments.

## Report
- **Commits:** grilly2 `03279a0` on the new branch `generate-logits-hook` (from `main` `8724f7a`; pushed, not merged;
  `main` untouched). CubbyLLM: this commit (the seam, its test, the bench and its log).
- **Files:** grilly2: `python/grilly/infer/logits_process.py` (new: `LogitsProcessor`, `LogitsProcessorList`,
  `PrefixConstrainedLogitsProcessor`, `SuppressTokensLogitsProcessor` with transformers' names, and
  `AllowedTokensLogitsProcessor`, the allowed-list helper: its 0/-inf mask built on the device once per vocabulary
  size); `python/grilly/infer/generation.py` (`generate(..., logits_processor=None, prefix_allowed_tokens_fn=None)`);
  `tests/python/test_logits_process.py` (19 tests). CubbyLLM: `cubbyllm/ops/grilly_seats.py` (`generate` hands both
  to the model only when given), `tests/ops/test_grilly_seats_hook.py`, `validation/bench_logits_hook.py`,
  `validation/logs/logits_hook_cost_2026-11-12.json`.
- **How it runs:** with either argument given, a step runs `sampling.processed_logits` (the repetition penalty alone),
  then the prefix constraint, then the caller's processors in order, then `next_token` with the penalty off
  (temperature, top-k, top-p, the draw), which is transformers' order (`_get_logits_processor` in 5.5.1). The token history is
  kept on the device whenever processors run (they read `input_ids`). With both `None` the step is the fused kernel
  as before. When a padded batch drops finished rows, processors see the running rows only; the prefix function still
  hears each row's batch index as given (the loop sets it).
- **Tests / measured, against each line:**
  - *Unchanged when off:* 60 `generate` outputs on grilly2's Qwen2 test weights (3 seeds x tied/untied x penalty
    1.0/1.3 x greedy, seeded sampling, left-padded greedy and sampled, padded with EOS and rows dropped) dumped on
    `main` and on the branch: **identical**. Pinned: `None` equals omitted; an identity processor changes no token,
    greedy, seeded sampling and with rows dropped (this pins the split path, penalty then sampling kernel, against
    the fused kernel). **Met.**
  - *An allowed list holds:* every new token in the set at every step, greedy and sampled, even and padded, as a
    processor and as a prefix function (8 cases); suppressed tokens never come; with rows dropped, each row keeps
    its own set and the function hears batch ids 1 and 2 after row 0 ends. **Met.**
  - *transformers parity* (5.5.1, torch 2.10 CPU, the references grilly2's CI pins): a bias processor,
    `SuppressTokens` and a prefix window that moves with length and row give the same greedy tokens as transformers'
    `generate`, penalty 1.0 and 1.3, even and padded; the test also checks they change what it says unsteered.
    **Met** (local; CI runs it where it has the reference).
  - *The seats:* grilly2's decode suites (`test_generation`, `test_sampling`, `test_cubby`, `test_lfm2`,
    `test_decode_stress`, `test_lora`): 116 passed. CubbyLLM: the seam passes nothing new to the model when the hook
    is absent (the call is the one before, word for word); `tests/emit`, `tests/ops`, `tests/speech`,
    `test_spatial_grilly` pass; under grilly2's venv an allowed list holds through `grilly_seats.generate` on a tiny
    Cubby model. **Met.**
  - *Cost on the home card, the 450M* (`validation/logs/logits_hook_cost_2026-11-12.json`; greedy, batch 1, a 32-token
    prompt, 64 new tokens, (t(65) - t(1)) / 64, median of 5 interleaved rounds): none **22.35 ms** a step; identity
    **22.42 ms (+0.35%)**; allowed 1,000 ids 22.50 ms (+0.68%); allowed 18,000 ids 22.73 ms (+1.73%); a prefix
    function returning 1,000 ids (mask built every step) 22.60 ms (+1.14%). **Line met** (identity <= 2%; kill
    > 10%). The rounds spread by about 0.4-0.7 ms within an arm, as large as the differences; every arm is
    inside 2% of none.
- **Not verified here:** CI's run of the parity test (lavapipe); a batch above 1 and sampled decoding in the cost
  bench (the tests cover their correctness, not their cost).
- **Found on the way:** grilly2's checkout on this machine now sits on `generate-logits-hook` (it was on
  `ci-and-release`, whose Python is the same as `main`'s); the stack imports grilly2 from that checkout, and with
  the hook absent it decodes exactly as before. Switch back with `git switch ci-and-release` in grilly2 if wanted.
- **Questions:** none blocking. 0067's step 4 can use `AllowedTokensLogitsProcessor` / `SuppressTokensLogitsProcessor`
  through `grilly_seats.generate` once Nick merges the branch (until then it runs from the branch's checkout).
