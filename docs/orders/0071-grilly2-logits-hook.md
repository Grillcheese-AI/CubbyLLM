# 0071 - A logits hook in grilly2's generate: the host can steer each token as it is written
status: open (queue: within 0067, before its step 4 builds the "enforced" class; then 0070 and 0069 as Nick set)
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
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
