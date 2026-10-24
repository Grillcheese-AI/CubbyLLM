# 0035 — Villages, next: the host's three missing routes, why the copy did not fire, and the copy (then the link) taught on exported villages
status: review (2026-10-23: C passes its four lines, D not needed; Report C at the end)
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

## Review (Cowork, 2026-10-23): A and B accepted; C may start in the far facts' shape
- **A is met.** All 1,800 questions were routed in each view, with 0 wrong. Every answer was spoken right and every
  empty one refused. The log agrees with the table. The host now has all seven shapes on a real world.
- **B is the finding this order needed:** the copy never fired on places.
  - Run 5's 0.953 on its own villages was the read alone; its gate stays at 0.00 there.
  - In the far facts' shape, untrained, the first token is right for things 0.726 and names 0.358.

  H-S12 is updated with both.
- **Question 1 (C's shard in the far facts' shape): yes.** The shape is data and adds no parameters, so C is still
  "the copy alone":
  - the lead token asks, at the question's cell;
  - each answer is in the token form its name was written in;
  - each piece of a name sits at its own address (piece *i* at the cell + 6*i* z, as the far facts bind them);
  - half the answers point at a thing and half at nothing, so the gate learns to shut on an empty cell.

  **One addition, so the test stays fair:** 0034's 60 villages and their fixed questions are written in the same
  shape for C's measure. They are the same villages, questions and splits; only how a document is written changes.
  Report the old shape beside it too, so the change of shape and the training are seen apart.
- **The lines stay as written:** things at least 0.75 and names at least 0.60, the far facts within 0.02 of 0.946,
  and the text sources unchanged. Names start from 0.358 untrained, so they are the line to watch. If C stops between
  0.36 and 0.60 on names, report where the misses go (another piece of the same name, a neighbour, or "nothing")
  before D.
- **Question 2 (where C runs): this machine, yes.** The shard stays in BlockWorld's run folder, which is the privacy
  the order asks for. Run it in the background at low priority and keep 0036 going beside it. Stop it if it slows Nick's
  machine for other work. A few thousand documents at about 1.8 s each is fine. Say in the report how many, and the
  wall time.

## Report C (2026-10-23): C passes its four lines; D is not needed
- **Commits:** `c03cca4` (the far facts' shape, C's measure and shard builder), `93ba849` (the space form, C's shard),
  and this report.
- **The shape C trained in: the space form, not the bare one.** In the bare form (`render(lead=True)`) the trunk alone
  answers an empty cell with a village word, never "nothing" (0.018 of 391); the copy copies only written tokens, so a
  gate that shut on empty cells could not make them right there. In the space form (`render(lead="space")`: every name
  written and answered as " coal", the ":" of "Answer:" asking, the rest of your four points kept) the trunk alone says
  " nothing" for 0.987 of empty cells. So the work splits: the copy gives the things, the trunk gives "nothing", and the
  gate learns to shut on an empty cell (`exp_s12_copy_villages_run5_space_copy_off_2026-10-23.json`).
- **The shard** (`make_scene_shard --shape space`, kept in BlockWorld's run folder): 67 villages from 24 seeds outside
  0034's (1000-1129); 3,350 documents of 512 tokens; 92,103 questions (9,222 names, 36,850 other things, 46,031
  "nothing") (`make_scene_shard_villages_0035c_2026-10-23.json`).
- **The run** (`train_spatial_link_5c_2026-10-23.log`): from run 5's final checkpoint, `CB_TRAIN_ONLY=copy` (1,027
  trainable parameters, peak lr 0.003), the far facts' shard (`facts_far1k_lead`) and C's at equal tokens. 410 steps of
  8,192 tokens (3.36M tokens; one pass over C's shard, about 3,280 of its documents), 4.99 h on this machine's CPU at
  low priority.
- **The trunk is frozen** (`diag_frozen_check_run5c_2026-10-23.json`): against run 5's final checkpoint, 0 of 326
  tensors moved and 0 of 5 router tensors. Only the copy's four parameters moved (tau 4.40 -> 4.19, beta 0.009 ->
  0.028, the gate's weight norm 2.62 -> 3.94, its bias -2.10 -> -2.12). The text sources are therefore unchanged.
- **On 0034's 60 villages** (`exp_s12_copy_villages_run5c_2026-10-23.json`; run 5 beside it:
  `exp_s12_copy_villages_run5_space_2026-10-23.json`, `exp_s12_copy_villages_run5_2026-10-23.json`), whole answers right:

  | shape | answer | n | run 5 | run 5c | line |
  |---|---|---|---|---|---|
  | space (C's) | a thing | 329 | 0.818 | **0.845** | 0.75 |
  | space | a name | 120 | 0.775 | **0.817** | 0.60 |
  | space | "nothing" | 391 | 0.609 | **0.910** | |
  | lead (bare) | a thing | 329 | 0.815 | 0.942 | |
  | lead | a name | 120 | 0.667 | 0.850 | |
  | lead | "nothing" | 391 | 0.008 | 0.010 | (no "nothing" token in that form) |
  | 0034's own | a thing | 329 | 0.006 | 0.006 | (the copy does not fire there, as B found) |
  | 0034's own | "nothing" | 391 | 0.972 | 0.972 | |

  The gate learned what C was for: its mean on empty cells fell from 0.402 to 0.094 in the space form, while on things
  it stays at 0.891 and on names at 0.862.
- **Where the space form's misses go:** things, 51 missed (33 answered "nothing", 5 a neighbour's piece, the rest a
  later piece); names, 22 missed (15 "nothing", 3 a neighbour's piece, 4 a later piece); empty cells, 35 missed (31 a neighbour's piece,
  3 another token, 1 a farther thing's piece). **The 35 on empty cells, and the misses that name a neighbour, are wrong
  answers spoken, not refusals.** The 48 answered "nothing" for a thing or a name are wrong too: they deny what is
  there. On the kill line (0 wrong answers spoken) the trunk's place read is not yet an answer Cubby may give on its own;
  the host's A routes (0 wrong on all 1,800 questions) stay the speaking path.
- **The far facts** (run 5's bench, 400 held-out documents a shard; `exp_hypertoken_recall_run5c_cpu_2026-10-23.json`
  against `exp_hypertoken_recall_run5_cpu_2026-10-23.json`): `facts_far1k_lead` 0.9456 -> **0.9271** exact (2,002
  questions; first token 0.9935 -> 0.9950), a drop of 0.0185, inside the 0.02 line but close to it: 146 wrong against
  run 5's 109. `spatial_here` (run 5's own villages) 0.9526 -> 0.9526. The training probe showed the same: the far facts'
  probe sat at 0.871-0.906 from step 50 on, so the gate's new shut costs some far facts from the start rather than late.
- **Acceptance:** things 0.845 >= 0.75; names 0.817 >= 0.60; far facts within 0.02 (0.0185); text unchanged (frozen).
  **C passes, so D (the link continued) is not needed** by the order's own terms.
- **Not verified here:** a GPU run (none needed: C ran on this CPU, as you allowed); any shape but these three.
- **Questions.**
  1. The far facts' 0.0185 is inside the line by 0.0015. If you would rather keep the 0.946, one more C run with the
     far facts at twice the village share would show whether the gate can learn the empty cells without that cost.
     I have not run it: the line was met.
  2. The space form is where C works; 0034's own form does not use the copy at all. Should the shapes the host writes
     for the trunk (A's routes stay the speaking path) follow the space form from now on, or is that for 0032?
