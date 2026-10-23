# 0035 — Villages, next: the host's three missing routes, why the copy did not fire, and the copy (then the link) taught on exported villages
status: proposed
written: 2026-10-23 by Cowork, from 0034's review · approved: — · hypothesis: H-S12 (gates 1 and 3), H-K10 · **BlockWorld stays private**

## Why
Order 0034 put Cubby's sense of place on 60 real BlockWorld villages. The host read them with 0 wrong in 3,600
questions and placed 20 of 20 plans on the cell it computed, but routes only 4 of 7 question shapes. The trunk (run 5)
read things at 0.006 and villager names at 0.000, below the kill line; renamed to shard words, the same villages read
0.58. So the trunk misses mostly on the words (an unseen thing is opened as "nothing", and the copy, which should
supply a written name's first token, does not fire) and partly on the layouts. The kill's next step, as written, is a
private shard of exported villages, not more synthetic ones.

What it brings: the trunk reading a real world's places and names, not only its own toy ones. In plain words: Cubby
learns to read real villages the way it already reads its practice ones.

## Build
- **A, the host's three routes** (public, `cubbyllm/reasoning/` and `cubbyllm/spatial/`): "n steps <direction> of X",
  "what is at x y z" and "where is X" reach the scene's exact readings, as the other four shapes do. Rerun 0034's
  questions on the same villages for the host only (they are fixed): 0 wrong, and the three shapes now routed.
- **B, why the copy did not fire** (no training): on 0034's villages, read the copy's gate (lambda) and its best match
  at every asking position, for names, things and "nothing", beside the far-fact documents it was trained on. Report
  whether the gate stays shut, or opens on the wrong writer.
- **C, the copy alone on a private village shard** (run 5's trunk frozen, `CB_TRAIN_ONLY=copy`, 1,027 parameters, as
  run 5 itself was trained): BlockWorld's exporter writes the shard in its own repo (other seeds than 0034's, so
  0034's villages stay held out), every thing and name written before the questions, the answer draws balanced (half
  pointing at a thing, half at nothing). The shard and its logs stay private; the run happens where the shard is (the
  local card or Colab with the private Drive).
- **D, only if C stays under its line:** the link continued on the same shard (the layouts), with the text sources as
  in run 3 so the language stays where it was.
- Each arm is measured once on 0034's 60 villages and their fixed questions (the same splits: things, names,
  "nothing" apart).

## Acceptance (pre-registered)
- **A:** 0 wrong on the 0034 questions; the three shapes routed.
- **C:** things at least 0.75 and names at least 0.60 on 0034's villages; the far facts (run 5's bench) within 0.02 of
  0.946; the text sources unchanged (the trunk is frozen).
- **D:** things at least 0.90, names at least 0.80, the text mix within 0.02 of run 3's.
- **Kill:** D below 0.75 on things: the places channel does not carry a real world at this budget; record it in H-S12.

## Constraints
- Rules in `CLAUDE.md` hold. BlockWorld's privacy as in 0034: the shard, the exporter, the layouts and the detailed
  logs stay in its repo or off-repo; this repo gets the generic code and counts.

## Out of scope
- New question shapes for the trunk beyond 0034's; the Unity view.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
