# 0010 — Casual prose from Cubby's own 450M: a talk adapter taught by human dialogue, no external model anywhere
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("fix prose so we don't need an external model to word
everything in casual sentences"; data route chosen: open human dialogue corpora + rules) · hypothesis: H-E10 (+ H-E15)

## Why
The 450M already holds both seats with nothing third-party in the loop (H-E15, 2026-10-17: `Cubby450mTalk`, adapter
`talk_v2_2`), and spoke 0 wrong. But it answers like a form. Per H-E10, the adapter sees the host's canonical ask, 63%
of its mix is fact questions in one sentence shape, and the value check allows one value. So the LFM stand-in still
does the casual wording. H-E10's voice step planned "local dialogue sources, and a one-time model rewrite". Nick's
call, 2026-10-22: **no model rewrite**. The casual voice comes from what people wrote, and the facts come from the
host, checked as today.

- **What it brings:** Cubby talks casually in its own words, from its own 450M, with the stand-in retired from the
  talk seat.
- **Advantage over conventional models:** a chat model is fluent because it learned to say anything. Here the
  fluency is learned from people, and every fact in a reply is still tied to the host's block by the claim check,
  so sounding natural never buys a made-up fact.
- **In plain words:** it chats like a person, and it still only tells you facts it actually has.

## Build
- **Sources, licenses first.** Before any row is used, read each dataset's license file and record it in a
  manifest (`standin/data/talk_v3.manifest.json`). Prefer licenses that allow commercial use, and ask Nick about
  any non-commercial one (DailyDialog and EmpatheticDialogues may be NC: check, don't assume). Candidates:
  - **Topical-Chat** (Amazon; 10,784 human conversations grounded in Wikipedia, fun facts and news: a knowledge
    snippet beside the casual sentence that uses it, which is the supervision this needs; check its DATALICENSE);
  - **OASST2** (Apache-2.0; human-written replies; keep the human-written turns only);
  - **Claire** (French, already wired in `gap_families.py`; per-subset licenses);
  - the existing chat sources.

  Exclude every machine-generated set (SODA, synthetic chats).
- **Records (`standin/data/build_talk_mix.py --mix v3`).** Three kinds:
  1. **Knowledge → casual.** From Topical-Chat, the snippet, rewritten as the host's facts block, and the human
     turn that states it as the target. Kept only if the target passes the claim check against that block.
  2. **Casual chat** (no facts) from the human corpora, EN and FR, under the existing voice rules.
  3. **Today's fact records**, reweighted down from 63%. The share is chosen to hold `talk_v2_2`'s
     spoken-and-correct, not to change the mix for its own sake.

  Every fact target passes the name-and-number guard and the claim check. No target is model-written.
- **Training:** a LoRA talk adapter over `base450m_cont`, the same recipe as `talk_v2_2` (`notebooks/talk_sft.ipynb`;
  the builder `_build_talk.py`), as `talk_v3`. The notebook is for Cowork to run on Colab. Serving changes nothing
  but the adapter name.

## Acceptance (pre-registered)
- **The battery:** 300 held-out turns. 200 are fact turns (150 EN, 50 FR) from the H-E8/H-E10 dev sets and the
  wiki world; 100 are chat turns.
- **Hard lines:** **0 wrong spoken** with the guards on. Spoken-and-correct on fact turns ≥ `talk_v2_2`'s on the same
  battery, minus 0.02.
- **Casual** (the point):
  - **Nick's blind A/B** on 60 pairs (`talk_v3` against the LFM stand-in, order hidden): `talk_v3` judged as casual
    or more casual on ≥ 50%. The harness shows the pairs and records his choices.
  - **Proxies**, reported, not gates: the share of fact replies in the canonical one-sentence shape (target ≤ 30%,
    against `talk_v2_2`'s, measured first); distinct openings per 100 replies; contractions per reply; mean length.
- **Kill:** spoken-and-correct drops more than 0.05, or any wrong spoken. Fluency bought with grounding is not
  this order.
- Tests: the builder's records all pass the guards; the manifest lists every source with its license; full
  `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold: no external LLM on the serving path, **and here none in the data either** (Nick). No
  bulk downloads without Nick's yes: Topical-Chat and OASST2 are small (tens of MB); confirm before fetching.

## Out of scope
- The emitter seat. Multi-turn dialogue shape (H-E10 step 2, its own order). The loci copy inside prose: short fact
  blocks sit in-window, so attention copies them (run 1's in-window text read 0.987).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
