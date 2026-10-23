# 0035 — Villages, next: the host's three missing routes, why the copy did not fire, and the copy (then the link) taught on exported villages
status: review (A and B built and measured; C waits on its shard's shape: questions 1 and 2)
written: 2026-10-23 by Cowork, from 0034's review · approved: 2026-10-23 by Nick ("yes open 0035") · hypothesis: H-S12 (gates 1 and 3), H-K10 · **BlockWorld stays private**

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

## Report (A and B; C waits on its shard's shape, D on C)
- **Commits:** `7f30d4b` (A), `91a676d` (B), and this report.
- **Files:**
  - A: `cubbyllm/spatial/scene_ask.py` (new), `cubbyllm/spatial/scene.py` (the offset and occupant facts, 291
    lines), `cubbyllm/reasoning/ask.py` (283), `tests/spatial/test_scene_ask.py` (new), and the runner's host test.
  - B: `validation/diag_s12_copy.py` (new).
  - Nothing in BlockWorld changed for A or B. The detail logs are in its run folder; this repo has counts.
- **Tests:** suite 1,235 passed, 1 failed (the known red, order 0002), 6 skipped. `test_scene_ask.py` pins the
  three shapes, the scene's new facts, and the loop answering all three verified.
- **Measured.**
  - **A.** The host now reads "where is X", "*n* *d* of X" and "what is at *x y z*" off the question when a scene is
    mounted, and restates each in the scene's grammar (the position of X; the relation "*n* *d*"; the occupant of a
    cell). The scene states the new facts from its exact reading.

    0034's fixed questions on the same 60 villages, host only
    ([log](../../validation/logs/exp_s12_scene_files_2026-10-23.json)):

    | | routed | spoken (all right) | wrong | refused ("nothing") |
    |---|---|---|---|---|
    | cut | **1,800 of 1,800** (was 1,020) | 1,329 | **0** | 471 |
    | whole | **1,800 of 1,800** | 1,350 | **0** | 450 |

    Every question with an answer was spoken right; every empty one was refused. **A's line is met.**
  - **B** ([log](../../validation/logs/diag_s12_copy_2026-10-23.json)), no training. Run 5's copy at every asking
    token: gate λ, best match, and what it points at.

    | arm | λ things / names / nothing | first token right: things | names |
    |---|---|---|---|
    | 0034's villages, as written (`here` asks) | 0.06 / 0.02 / 0.00 | — | — |
    | the same, the lead token asking too | 0.96 / 0.99 / 0.38 | 0 of 329 | 3 of 120 |
    | the same, the answers in the listing's token form | 0.99 / 1.00 / 0.73 | **239 of 329** | **43 of 120** |
    | far facts (`facts_far1k_lead`) | 0.71 (points at the next token 1,005 of 1,410) | | |
    | run 5's own shard villages (`spatial_here`) | **0.00** at all 359 asking tokens | | |

    - **The gate stays shut on village documents, run 5's own included.** Its 0.953 on the shard's villages is the
      trunk's read alone. The copy opens only in the far facts' shape, the one it learned.
    - **The read mostly finds the right thing.** At `here` it points at the answer's first piece, in its written
      form, for 199 of 329 things.
    - **0034's documents give the copy nothing it can copy:**
      - it asks at `here`, which "?" or a newline follows, not the answer;
      - the answer's first token is never written: the listing writes `coal` at a line's start, and the answer after
        "Answer:" is ` coal`, another id.
    - **In the far facts' shape** (the lead token asks; "Answer:", a space, then the name in its written form), with
      no training: things 0.726 and names 0.358 on the first token.
      - Most misses are another piece of the same name (80 things, 69 names): a name's pieces all share one cell, so
        the read can return any of them. Few are a neighbour's piece (10, 3).
      - "Nothing" breaks there (3 of 391): the gate opens on empty cells too (λ 0.73).
- **Not verified here:** C and D (no shard, no training); the read with a name's pieces at their own addresses
  (question 1).
- **Found on the way.**
  1. **C as written cannot reach its names line.** The copy's 1,027 parameters (its gate, temperature and threshold)
     decide when to open and how sharply. They do not decide which written token the read points at: that is the
     read's, and the link is frozen in C. On village documents with a name's pieces at one cell, the read points at
     the answer's first piece for about a third of names, at best. What C's shard needs is the far facts' own shape,
     which is data, not training:
     - the token before each answer asks, at the question's cell;
     - each answer in the token form its name was written in;
     - each piece of a name at its own address (piece *i* at the cell + 6*i* z, as the far facts bind them; villages
       stand within 3 levels, so the pieces' addresses meet nothing).

     Training then has the gate to learn: open on a thing, shut on an empty cell. Today it opens on both.
  2. **Run 5's village reading never used the copy.** So run 5's 0.953 on its shard is no evidence that the copy works
     on places. Only the far facts show it working.
  3. With A, the host answers every 0034 question it should; "where is X" names a position, not a profile.
- **Questions.**
  1. **C's shard in the far facts' shape:** the lead token asking, the answers in the written form, the pieces at
     their own addresses, the answers balanced (half things, half nothing) as you wrote. Is that C, as you would
     have it? Without the shape, I expect it to stay where B measured it, whatever the training.
  2. **C's run:** the trunk frozen and 1,027 parameters, so a forward of about 1.8 s per 512-token document on this
     CPU. A run of a few thousand documents is hours here, not GPU time. May I run it on this machine (the shard
     stays in BlockWorld's run folder), or should it go to Colab?
