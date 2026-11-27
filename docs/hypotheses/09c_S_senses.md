<!-- sharded: part of the hypothesis document; the rules (section 0) and the map are in CUBBYLLM_HYPOTHESES.md -->

## 9c. Group S — Senses: a sensory cortex learned by play (added 2026-10-10)

Origin: Nick, 2026-10-10 — Cubby (the BlockWorld agent) should navigate and manipulate from what he actually sees and
feels, so the same neurons work in other applications and the outside world; blocks stand for external objects; the game
is the safe sandbox where the trial and error happens; it must be a benchmark *and* a component of the full model /
agentic harness. Design: `docs/superpowers/specs/2026-10-10-sensory-cortex-design.md` (forks approved: headless eyes, a
sighted→blind ladder, a CubbyLLM plugin built on cubemind perception, CubeLang modding only after vision and in a
sandbox). The benchmark rungs S0–S6 live in the private sandbox repo. All entries below are **SPECULATIVE / UNRUN**.

**H-S1 — Salience is physical: a retina built from cubemind's bio-vision channels plus a centre–surround contrast
channel, read out as GIF spike rates, ranks emissive, bright, high-contrast and moving things above the background
without any rule saying they matter.** Source: Nick ("a glowing block will activate the neurons more as it brings
attention to it; same for bright vs dark colours or contrast"); `cubemind/perception/bio_vision.py` (Weber–Fechner
luminance, opponent colour, motion); Itti & Koch 1998 (bottom-up salience). Validate: S0 — over rendered frames with
known block ids, the salience rank of emissive / high-contrast / moving pixels vs a matched background sample. Kill:
those classes not ranked above background (AUROC near 0.5), or the ranking explained by luminance alone with contrast
and motion adding nothing. **First read (2026-10-10, `validation/logs/s0_salience_2026-10-10.json`; `standin/cortex/`,
12x16 tiles, 64x48 frames, 40 seeded scenes per class, each with four plain distractor blocks of the ground's kind;
top-1 = the target's tile is the most salient):** raw responses (the default) — glow 0.75, bright colour 0.50, contrast
0.38, moving body 0.90, control (a fifth plain block) 0.31. Itti–Koch map normalisation is worse on every class (glow
0.59, contrast 0.03, motion 0.41) and is off. **Luminance alone** (the ablation the kill clause asks for) — glow 0.81,
bright colour 0.53, contrast 0.56, moving 0.69, control 0.28: motion adds (0.90 vs 0.69), but for static targets the
colour and centre–surround channels add nothing over luminance and the saturation channel costs the white target (0.38
vs 0.56) against colourful distractors. Reading: glow passes; the kill clause holds for the colour/contrast channels as
weighted today. Next: weight them by learning (which tiles predicted something worth looking at), not by hand, and
re-read. Status: **OPEN — partly killed (colour/contrast weighting).**

**H-S2 — Depth plus proprio-driven path integration rebuilds the map the sighted agent gets for free, well enough to
plan on.** Source: cubemind's place / grid cells; the sandbox's ground-truth window. Validate: S1 per-frame occupancy
precision/recall, S2 map IoU after N ticks blind, S3 goal-reach success and path length vs the sighted agent at matched
ticks. Kill: S3 far below sighted while S1 occupancy is good — path integration, not vision, is the bottleneck, and the
fix belongs there before anything else is built on it. **First read (2026-10-10,
`validation/logs/s1_s2_where_2026-10-10.json`; `standin/cortex/where.py`; a sighted agent plays 2400 ticks in a new
procedurally generated world, the cortex maps from 64x48 depth frames every 4 ticks):** pose given — S1 solid precision
0.96, free precision 1.00, solid recall 0.99; S2 map IoU 0.77. Pose integrated from an odometer with a 3% gain error per
axis and per-tick noise — drift 0.91 blocks, S1 solid precision 0.37, recall 0.62, S2 IoU 0.36. A first re-localiser
(shift the pose within ±0.3 to agree with the map at 1-block resolution) diverges (drift 50.6, IoU 0.17) and is off.
Reading: vision is not the bottleneck (S1 high when the pose is right); path integration is, as the kill clause foresaw
— under a block of drift halves the map. Next: sub-cell surface matching of depth points (ICP-style) and place
recognition from what is seen, then re-read before anything blind is built on it. Second re-localiser (same day,
`validation/logs/s1_s2_where_2026-10-10b.json`): point-to-plane matching (translation only, only along directions the
surfaces constrain, capped per frame) against a 0.25-block surface map corrects a displaced frame in isolation (0.2 off
-> within 0.01) but in motion it is still worse than the odometer alone (drift 1.27 vs 0.90, IoU 0.06 vs 0.39): matching
a map built from the same drifting poses only re-confirms the drift, and the residual errors walk. It stays off. What
would work is the textbook remedy: landmarks seen again (loop closure) and online calibration of the odometer's gain
against frame-to-frame visual odometry. Status: **OPEN — the bottleneck located.** **S3 first read (2026-10-10,
`validation/logs/s3_blind_2026-10-10.json`; 6 seeded fields with walls, a step, a hazard and four food blocks, 3000
ticks, starting hungry, moves known but no kind of block): blind Cubby — no window, no item list, his eyes and touch
through this cortex into the map his mind plans on, position from his body — gets around and stays whole (cells stood on
119 vs sighted 212; harms 0.0, deaths 0.0) but finds food in 2 of 6 fields against the sighted 6 of 6 (first meal 2162
vs 769 ticks, capped). The gap is kinds, not ways: his eyes split one material into several kinds (each a new thing to
try) and merge look-alikes, so foraging by trying kinds is slow. Next: stabler kinds (cell votes are in; prototype
merging next), then re-read.** **S3 second read (same day): the gap was where he looked, not what he made of it.**
Merging prototypes that drift together (`KindCodebook.consolidate`) changed nothing
(`validation/logs/s3_blind_2026-10-10b.json`: still 2 of 6, first meal 2162). Tracing one field showed why: eating is
done looking straight up (so the use goes nowhere), and no walk ever brought the eyes back down — after his first meal
blind Cubby walked on staring at the sky, saw no new solid cell, and circled frontier cells that could never resolve.
Sighted Cubby never noticed, because the block window does not care where he looks. With the eyes settling back to ~20°
down on every move and sweeping while he waits (a gaze, not a new sense; `validation/logs/s3_blind_2026-10-10c.json`):
blind Cubby eats in **5 of 6** fields (was 2), first meal 1311 ticks (sighted 769), meals 1.17 (sighted 2.67), cells
stood on **215 vs the sighted 212**, kinds known 9.5, deaths 0, harms 0.33 (he found the hazard by touching it twice,
where DNA told the sighted one). Reading: with eyes that look where he goes, the blind map is as good as the window for
getting around; the remaining gap is slower food (trying kinds one by one), which is what a learned guess from looks to
use is for. The lesson for the cortex: an embodied eye needs a gaze policy, and the next one should be the cortex's own
fixation (salience) rather than a fixed rest angle.** **Re-run with events delivered (2026-10-15; the local driver
dropped every event when the agent was driven one tick at a time (found under H-S10); postmortem off, so only the fix
differs; `validation/logs/s3_blind_2026-10-15_events.json`, `validation/logs/s1_s2_where_2026-10-15_events.json`):** S3
unchanged in substance — blind Cubby eats in 5 of 6 fields, first meal 1316 ticks, meals 1.33, cells stood on 214.5
against the sighted 212, harms 0.17; the sighted run is identical. S1/S2 took a different walk (600 frames instead of
500), so its numbers move — pose given: map IoU 0.93 (was 0.77); pose integrated: 0.36 (0.39), drift 0.17 blocks (0.90);
re-localised against its own map: 0.07 (0.06) — and the reading holds: integration is the weak tier and re-localisation
as built does not help.

**H-S3 — A label-free kind codebook over fixated patches (spike→VSA block codes, prototype recall in the cortex's own
memory) separates the world's materials better than colour statistics, and binding the seen code with the felt code
makes "looks like this" predict "feels like this".** Source: the design's "what" stage; the sandbox knows each pixel's
true kind. Validate: S1 cluster purity vs true kinds against a colour-histogram k-means baseline; after touch, held-out
prediction of the tactile code from the visual code. Kill: purity not above the colour baseline; cross-modal prediction
at chance. **First read (2026-10-10, `validation/logs/s1p_s6_what_2026-10-10.json`; `standin/cortex/what.py`:
chromaticity + log brightness per surface pixel, online prototypes, no labels; a sighted agent plays 2000 ticks, pose
given):** purity 0.97 and completeness 0.91 (16 kinds learned for 13 true kinds) against the colour-histogram baseline's
0.89 / 0.76 — shading no longer splits a material (chromaticity is shading-invariant), so the codebook beats colour
statistics. Seen ⊗ felt: once a kind has been felt, sight predicts its feel almost exactly (error 0.002 per channel vs
0.037 for predicting the average feel), but for a kind touched for the first time, borrowing the feel of look-alikes is
WORSE than the average (0.175): in this world looks are only weakly tied to feel across materials. **Re-run with events
delivered (2026-10-15; the local driver dropped every event when the agent was driven one tick at a time (found under
H-S10); postmortem off, so only the fix differs; `validation/logs/s1p_s6_what_2026-10-15_events.json`):** a different
walk (206k pixels instead of 309k, 865 touches instead of 478); separation still passes — purity / completeness 0.94 /
0.83 against the colour baseline's 0.74 / 0.49 (a wider margin); a kind felt before is predicted almost exactly (0.0002
per channel against 0.032 for the average feel); a first touch by look-alikes is still worse than the average (0.088
against 0.032). Status: **OPEN — separation passes; look-alike transfer of feel fails as built** (next: a learned map
from appearance features to feel instead of nearest-prototype borrowing, and more first touches per run).

**H-S4 — Touch is learnable as its own channel: slip, roughness, softness and steadiness are told apart from contact
signals alone, and an unsteady feel predicts a fall before vision does.** Source: Nick (blocks that feel rough, slimy,
slippery; blocks that fall or disappear soon after being touched should feel like moving sand); the world's grip and
support physics. Validate: S6 — discrimination from touch only; lead time between an unsteady contact and the observed
fall. Kill: discrimination at chance, or no lead time (the channel carries nothing proprio does not). **First read (same
log):** over 478 floor contacts on 3 materials, a nearest-centroid read of the touch vector alone tells them apart 1.00
of the time — a check that the channel carries the difference, not yet a test (three materials, and the fall-prediction
half is not measured). Status: **OPEN.**

**H-S5 — Learning progress beats raw prediction error as the curiosity signal: with a noisy distractor in view (a
flickering light), an agent rewarded by the drop in its forward model's error keeps exploring the learnable world,
while one rewarded by the error itself gets stuck on the noise.** Source: Nick ("let it play with blocks for a while,
then it tries new things until it discovers what can be useful"); Schmidhuber 1991, Oudeyer et al. 2007 (learning
progress); Burda et al. 2018 (the noisy-TV failure). Validate: matched blind runs with the two intrinsic rewards and a
flickering emissive block placed in view; map coverage and affordances discovered at matched ticks. Kill: no
difference, or prediction error winning. **First read (2026-10-10, `validation/logs/p1_noisy_tv_2026-10-10.json`; the
forward model is `standin/cortex/forward.py`: per face of each cell seen (and per distance band, so haze is not
mistaken for change) a running prediction of its look, a never-seen face predicted from its region's usual look; its
error per cell is surprise, the older half of the cell's last 32 errors minus the newer half is learning progress, a
region's value the mean over its cells. Blind Cubby, fed, in the six S3 fields with a 3 x 2 "TV" of fixture blocks in
the middle whose look changes at random every tick; three arms differing only in what sends him back to a place he has
seen — "stale" (how long ago he looked, the reference agent), "error" (surprise) and "progress" (learning progress),
each ranking places on its own scale, the most interesting worth 20 steps; 3000 ticks):** the noise half **passes** —
the error-driven Cubby spends 0.43 of his time at the TV and sends 0.74 of his look-again choices to it (0.50–0.84 in
every field), the progress-driven one 0.085 of his time (less than the reference's 0.16: the TV is surprising and
teaches nothing, so he leaves it alone) and 0.05 of his choices. The coverage half **fails as built**: cells stood on
155 (progress) vs 149 (error) vs 239 (stale), map cells known 3445 vs 3290 vs 4134, kinds known 9.0 vs 9.8 vs 9.7 —
progress does not out-explore error, and both curiosity arms explore less than the reference, because looking again
(about 42 choices a run, each a walk and a look around) takes time from the frontier in a small field where, once it is
mapped, there is little left to learn but the noise. Reading: learning progress is the right signal to ignore a noisy
TV (the pre-registered contrast in attention is large and holds in every field), but as a pull on behaviour it is not
yet worth what it costs; by the kill clause on coverage this read is a **no-difference** between the two intrinsic
rewards. Next (a new read, not a re-tune of this one): progress only when it is more than its own noise (a per-cell
significance test instead of the raw drop), the pull scaled by how much is left unexplored, and a field where something
learnable changes (crops growing, water spreading) so there is real progress to find. **Re-run with events delivered
(2026-10-15; the local driver dropped every event when the agent was driven one tick at a time (found under H-S10);
postmortem off, so only the fix differs; `validation/logs/p1_noisy_tv_2026-10-15_events.json`):** the same reading —
the error-driven Cubby spends 0.37 of his time at the TV and sends 0.65 of his look-again choices to it, the
progress-driven one 0.05 and 0.01, the reference 0.11; cells stood on 159 (progress) against 158 (error) and 257
(stale), map cells 3514 against 3421 and 4269. Status: **OPEN — attention contrast passes, exploration benefit not
shown.**

**H-S6 — What the cortex learns in the sandbox transfers as machinery, not as appearance: on a new seed and recoloured
looks (and on Unity frames) the retina, salience and where stages hold their S0–S3 numbers without retraining, while the
kind codebook has to relearn — and relearns faster than from scratch.** Source: the design's transfer table (§6); the
honest limit that rendered blocks are flat-coloured cubes. Validate: S5. Kill: S0–S3 collapsing under the shift (the
stages overfit the palette), or the codebook relearning no faster than from scratch.

**H-S7 — Skills learned in the sandbox stay useful outside it when they are stored as programs over affordance roles and
abstract intents: given only a sense adapter and a motor adapter for a new environment, an agent carrying the sandbox
library reaches competence in fewer trials than a fresh agent with the same adapters, and a transferred skill that does
not hold there is retired for that environment alone.** Source: Nick ("the complicated part is to make the learned
skills something permanently useful outside the world"); the CubbyLLM rule that a new skill is a program, not a new
model; the fortune-cookie testimony mechanism (Beta trust over told facts) reused as trust in transferred skills.
Validate: sandbox-trained library vs fresh, same adapters, in at least two non-sandbox environments (Cubby-Man's maze; a
GUI task), trials-to-criterion and verdict precision of transferred skills. Kill: no sample-efficiency gain in either,
or transferred skills that keep firing after their verdicts fail (trust not doing its job).


**H-S8 — The brain learns as a whole: sensory cortices (eyes, skin, body, needs; later ears) that write percept tokens
into the language trunk's residual stream, routed to per-sense experts and trained together with it on one objective
(every part predicts what comes next, its own input and the others'), ground language in what is perceived better than
the same cortices trained apart and glued to a frozen trunk, without costing the language the trunk has.** Source: Nick,
2026-10-15 ("our brain does not train one cortex at a time; it is the sum of all the senses / cortex information that
drives learning"); design `docs/superpowers/specs/2026-10-15-whole-brain-design.md`; multimodal competition (arXiv
1905.12681, 2203.12221), Mixture-of-Transformers (arXiv 2411.04996), uncertainty-weighted losses (arXiv 1705.07115);
zero-exit admission from `cubbyllm/model/grow.py`. Validate: arms at equal compute — J (joint from step 0), S (cortices
trained alone, then attached to a frozen trunk), T (text only) — on held-out episodes and seeds of the private sandbox:
G1 grounded questions answered from percepts (J ≥ S + 5 points; T can only guess), G2 kinds recognised from fewer views
and first-touch feel predicted better than the cortex alone (J better than S), G3 held-out text loss within +0.02 nats
of T. Kill: J no better than S on both G1 and G2, or G3 failed — the cortices then stay separate and are joined the
cheaper way (S). Status: **OPEN — designed, not built.**

**H-S9 — Event → action → consequence: with Cubby's own acts as tokens between what he senses and what follows, and
counterfactual forks from the sandbox (the same state, the other acts tried), the whole brain of H-S8 learns what an act
causes, not only what follows what.** Source: Nick, 2026-10-15 (the ARC loop, after a sketch from Gemini, modified: the
action is Cubby's own act, not the stimulus); design `docs/superpowers/specs/2026-10-15-whole-brain-design.md` §4.7;
interventional vs observational data. Validate: arm J against J-shuffled (action tokens shuffled across ticks in
training), held-out forks and seeds: G4 the forward head's error on the consequence of each tried act ≥ 20% below
J-shuffled's; G5 the inverse head picks the act from (event, consequence) ≥ 2× chance and above J-shuffled. Kill: no
gain over J-shuffled on G4. Status: **OPEN — designed, not built.**

**H-S10 — Critical thinking in two modes: a postmortem after a surprise (locate the wrong channel, blame the feature
the prediction leaned on by counterfactual re-entry, correct a fast exception memory and the replay priority, widen the
spread instead when similar events disagree) and conditional branching before uncertain or irreversible acts (roll the
forward model per act and per likely outcome from a fork of the state, score by value + learning progress − risk) make
Cubby correct a mistake on its second occurrence and avoid harm, where habit and plain replay do not.** Source: Nick,
2026-10-15 ("if it got its prediction wrong it would adjust it for future similar events (in postmortem mode) and / or
evaluate all the possibilities in branches (when in conditional mode)"); design
`docs/superpowers/specs/2026-10-15-whole-brain-design.md` §4.8; H-P1/H-P2/H-P5 (forkable state, branching where
uncertain, counterfactual re-entry), H-E19 (blame at the first wrong step), prioritized replay (arXiv 1511.05952),
uncertainty arbitration between habit and deliberation (Daw, Niv & Dayan 2005). Validate, first in Cubby's current
mind, then in the brain: G6 second-encounter error on held-out surprise pairs ≥ 50% lower than replay alone at equal
compute, and on pure-noise events the spread widens instead of the prediction flipping (calibration no worse); G7 at
risky decisions, harms plus deaths ≥ 30% lower than acting on the best prediction, within the same decision-time
budget, with the chosen branch's predicted value correlated with what it delivered. Kill: no G6 gain over replay alone,
or predictions flipped on noise; or no G7 gain over habit. **First read, G6 at the level of the agent's affordance
memory (2026-10-15, `validation/logs/g6_postmortem_2026-10-15.json`; the prototype lives in the private sandbox agent's
memory: a prediction off by 0.6 or more that came from blocks like this one is blamed on the property bits that set the
block apart from the ones that misled it — any bit a block that went the usual way also had is cleared (re-entry from
memory), bits it has are preferred to bits it lacks, and failing any single bit two bits together are tried — and each
blamed bit becomes a lesson that every later block it bears on adds to (each block counted once, however often met) and
that is consulted before likeness for a block never tried; a wrong prediction from experience of the very block is not
blamed on a bit: the block now goes both ways, and after that neither way is a surprise. Protocol: 100 trials for each
of six kinds of prediction, five decided by one bit and one by two bits together — a dozen blocks behaving the usual
way, met twice each; a surprise block; then a never-met block sharing only the cause, and a never-met control block on
the usual side):** second-encounter error **0.36 against 0.79 for plain replay (−55%; gate ≥ 50%: passes)** and 0.72
with the surprise replayed three times (−50%). Per one-bit rule 0.30–0.33 against 0.65–0.94, the cause blamed in every
trial with about one bit blamed per surprise; the two-bit rule 0.54 against 0.68 (the right pair is among the blamed in
90% of trials, but credit is split over about six pairs). No spill: control-block error 0.16 against 0.15. Noise (one
bit makes a block go either way at random; after a surprise on one block, six more): the next block is predicted at
0.55 (unsure, not flipped), the prediction ends at 0.51 (replay 0.19, which never learns the bit), and calibration is
better (expected Brier 0.31 against 0.41) — but 14% of the predictions made during a run of the same outcome reach 0.8
or more, against 0% for replay, so the no-flip clause **fails** as set in the bench (flips at most replay's + 0.05). In
the live agent (2500 ticks, one world) the postmortem fires six times, each with a blame its voice can say ("a sharp
block never hurts to touch — it did; it is sharp"). Reading: one surprise now teaches the cause, not only the block,
and the lesson stays on the blocks it bears on; the price is that a run of chance outcomes on a noisy bit looks like a
rule for a while — the trade-off of learning from one example, not yet solved. Next: G7 (conditional branching before
risky acts) in the agent, then both modes in the brain's heads (H-S8). **G7 in the agent (2026-10-15,
`validation/logs/g7_conditional_2026-10-15.json`): habit judges every step by its best prediction (a floor holds and
does not hurt when that is more likely than not); conditional weighs a step over its outcomes — its cost plus the
chance of a hurt (from the floor, or from brushing a block beside it believed to hurt) times what a hurt is worth (15
steps) — and gives a floor it has never stood on no less risk than its own record with the unknown (how often a new
floor hurt it the first time, Laplace-smoothed: the forward head's own predicted error). Field: walls, deep water, four
food blocks and four burning floor patches that look like the ground around them, each a block never met; a hungry
start, moves known, blocks not; 3000 ticks.** Harms plus deaths 5.29 → 1.19 per run over 48 seeds (**−78%, 95% CI
59–91%; gate ≥ 30%: passes**), fewer in 38 runs and more in 3 (sign test p ≈ 1e-8), the first 16 seeds and a 32-seed
replication agreeing (−76%, −78%); meals 2.08 → 2.33 (difference CI −0.13 to +0.63: safety not bought with hunger),
cells stood on 198 → 196, decision time unchanged (2.32 → 2.26 ms per tick: the same search with different step costs).
Ablations on the first 16 seeds: weighing by likeness alone, without the record of the unknown, loses the gain (5.38
against 5.81, −8%); dropping the weight on what is beside a step keeps it (1.44 against 1.38) — the whole effect is
treating a never-tried floor as risky as the unknown has turned out to be. The calibration clause **fails**: the risk
given a never-tried floor does not tell the floors that hurt from those that do not (AUC 0.46; one base rate for every
unknown) and it runs high (0.44 predicted against 0.17 delivered). Reading: G7's harm half passes by a wide margin from
one ingredient — arbitrating by the agent's own track record with novelty (Daw, Niv & Dayan's uncertainty arbitration)
— not from branching over likeness predictions; a chosen branch's predicted value is not yet informative. Harness fix
found on the way: the local driver dropped every event when the agent was driven one tick at a time, as all per-tick
benches did (fixed; the reads it touched are re-run under H-S2, H-S3 and H-S5). **Baselines and controls (2026-10-15,
`validation/logs/g6_baselines_2026-10-15.json`, `validation/logs/g7_controls_2026-10-15.json`).** G6 against learners
that see the same blocks as bit vectors: second-encounter error 0.36 for the postmortem against 0.55 for naive Bayes
(the strongest; the postmortem 35% lower, paired-bootstrap 95% CI 32–39%), 0.73 for class-balanced logistic regression
(51% lower, CI 49–53%) and 0.88–0.93 for plain logistic regression at any of three ridge strengths; on noise the
learners that can generalise from one example flip at least as often as the postmortem (naive Bayes 0.20, balanced
logistic regression 0.21, against 0.14) — the no-flip clause fails for every fast learner here, and the postmortem
fails it least. G7 controls (16 seeds): **a correction to the reading above** — fearing every never-tried floor at a
fixed 0.5 gives the same harms as fearing it at the agent's learned record (1.38 against 1.38), so the active
ingredient is treating never-tried floors as uncertain, not learning how uncertain; within 3000 ticks the record stays
near its prior. The price of a hurt matters, with a plateau: 5 steps 2.44, 15 steps 1.38, 45 steps 1.31 harms plus
deaths (habit 5.81). In a control field whose new-looking floors do not hurt, caution costs little that can be
measured: meals 2.63 against 2.75 (difference CI −0.50 to +0.25), cells stood on 197 against 198, but the cautious mind
stands on a never-tried floor 35 times against 95 (it learns new floors more slowly). **A frontier LLM, and held-out
worlds (2026-10-15, `validation/logs/g6_llm_2026-10-15.json`, `validation/logs/g7_heldout_2026-10-15.json`).** G6 with
a frontier language model as the learner, told the same trials as text, features anonymised and renamed per trial:
Gemini 3.8 Flash's second-encounter error is 0.45 — the postmortem's is 21% lower on the same 600 trials (paired CI
19–24%) and lower on every rule — its control error 0.05 (the postmortem's 0.16 is plain replay's likeness, 0.15), and
on noise it flips as often as the other fast learners (0.21); GPT-5.5 at low reasoning effort, on 120 trials, 0.74. G7
on 48 held-out worlds (fresh seeds, the paper's own chemistry key, moves learned from scratch in a 2500-tick nursery
instead of a saved mind): harms plus deaths 3.85 → 0.58 (−85%, CI 66–96%), fewer in 41 worlds and more in 2 (sign test
p ≈ 2e-10), meals 1.67 → 1.92 (CI +0.08 to +0.44), decision time 2.71 → 2.59 ms per tick; in the control with nothing
that hurts (16 held-out worlds) a small cost: meals 1.94 → 1.88 (CI −0.19 to 0.00), cells stood on 195 → 191,
never-tried floors stood on 92 → 40. Status: **OPEN — G6 second-encounter passes and G7 harm passes (−78%); G6's noise
clause and G7's calibration clause fail; both modes still to move into the brain's heads.**

**H-S11 — The advisor loop: given a goal G, past trials P and a history of choices H, the whole brain (H-S8 to H-S10)
lists the options that really reach G with true pros and cons, picks the best under the user's criteria or asks the one
question that settles the pick, because each option is a branch its forward model played out, P is its postmortem
memory, and H reveals the user's criteria.** Source: Nick, 2026-10-15 ("I want to do G, I tried P, given that in the
past I did H, what are my options? And which choice you think would be best for me?"); design
`docs/superpowers/specs/2026-10-15-whole-brain-design.md` §1 (the advisor loop). Validate: held-out sandbox scenarios
(G, P, H), every option and claim run in a fork, against the trunk answering without branching and a general language
model answering from the same description, at equal compute: A1 ≥ 90% of listed options reach G; A2 ≥ 80% of the
distinct ways the forks find are listed; A3 ≥ 90% of stated pros and cons hold; A4 the best option under given criteria
picked ≥ 80%, and without them a question asked when the top options swap on an unweighted criterion (≥ 80%) and not
otherwise (≤ 10%); A5 a failed trial from P re-proposed without saying what changes in ≤ 5%, and the pick agrees with
H's revealed preference ≥ 70%. Kill: no better than the no-branching trunk on A1 and A3, or no gain from P and H over
answers given without them. **Benchmark built, baselines read (2026-10-15,
`validation/logs/advisor_bench_2026-10-15.json`; the benchmark itself lives in the private sandbox): 72 held-out
scenarios — three kinds of goal × three modes (criteria given; H only; neither), 24 per mode — every listed option
played out in a fork, plans in a six-step language, the user's criteria weighing time, harms, blocks used up, staying
dry and leaving the world as it was; three reference arms, each also run with P and H hidden:** *forks* (every way
played out on the truth — the ceiling Cubby's branching aims at) passes every gate (A1 = A2 = A3 = 1.00; right pick 1.00
with criteria given; asks 1.00 when the options trade off, 0.00 when one dominates; no failed trial repeated; pick from
H 0.96), so the gates are reachable — with H hidden its pick falls to 0.58 (it has to ask), the measured worth of H.
*prior* (the same decision rule without forks: costs averaged over other worlds) is close on validity and truth (A1
0.99, A3 0.97) and loses mostly on the pick (0.96 with criteria given; asks 0.91 when it should, 0.08 when it should
not), where it weighs a cost known from P against an estimated one; the facts the words leave out (whether the blocks in
hand stack here, whether a block can be broken) are where it lists options that do not work. *LFM2.5 1.2B* (local, all
layers on the GPU, JSON-constrained, the same words) fails every gate: A1 0.17 (47 of 274 listed options reach G), A3
0.79 over its 135 claims about options that work, right pick with criteria given 0.00, asks 0.09 when it should, pick
from H 0.00; with P and H hidden A1 falls to 0.02 (5 of 254) — P teaches it the form of a plan more than its content (15
of its 47 working options copy a trial that worked). Reading: the benchmark separates the arms and every number the kill
clause needs is measurable, but in these worlds branching on the truth beats a good no-fork prior by only 1–3 points on
A1 and A3 — too little room for Cubby's arm to show it beats the trunk without branching. Before that arm runs, the
worlds need more that only playing out shows (costs hidden from the words, plans that fail halfway, longer chains of
steps). **Level 2 (2026-10-15, `validation/logs/advisor_bench_l2_2026-10-15.json`): the same worlds plus a fence between
the player and the goal whose one gap mostly forces a detour (the median way gets 18% longer), and by one goal a pile of
the building material to quarry (a third way: break, carry, build — none of your own blocks used up, more of the world
changed).** *forks* still passes every gate (pick from H 0.88; 0.71 with H hidden). *prior* falls to A3 0.92 (0.90
blind) and asks 0.86 when it should (0.71 blind): the detours are what an estimate made without playing out gets wrong;
A1 barely moves (0.985). *LFM2.5 1.2B*: A1 0.18 (46 of 259), A3 0.79, right pick with criteria given 0.04; blind A1 0.00
(0 of 245). Reading: level 2 widens the forks–prior gap on true claims from 3 to 8 points and on asking from 9 to 14,
not on validity — these worlds still rarely make an option fail halfway. Status: **OPEN — benchmark built and baselines
measured; Cubby's arm waits on the whole brain (H-S8 to H-S10).**
**H-S12 — Space at the input, provably: a place is a unit vector in a Hilbert space (fractional power encoding, `S(p) =
IFFT(exp(iΘp/scale))`), bound into the token's embedding at the input and read back from the trunk's state through a
gated route the trunk itself drives, so a village is a scene the model can be shown without a word of text and asked
about with answers guaranteed by the algebra rather than hoped for from training.** Source: Nick, 2026-10-17 ("implement
spatial understanding provably at the input level; to build a village geo spatial understanding is important; Hilbert
spaces could be useful for the spatial features; it also needs to have a link route from the trunk so it knows when to
use it"); Plate 1995 (fractional powers of unitary vectors), Komer & Eliasmith 2019 (spatial semantic pointers), Frady,
Kleyko & Sommer 2021 (the kernel reading: `⟨S(p), S(q)⟩ = k(p − q)`), Skilling 2004 (the Hilbert curve); RoPE is the
same encoding in one dimension, inside attention — `S(p)` is its 3-D sibling at the input, for every mixer. **What is a
theorem and what is a capacity.** Theorems, each checked to float precision (`tests/spatial/test_ssp.py`,
`test_hilbert.py`, `tests/model/test_spatial.py`): moving is binding, `S(p + d) = S(p) ⊛ S(d)`; the origin is the
identity, so a token with no place is unchanged and every text golden is bit-identical with the link on (checked with
`torch.equal`); similarity is a kernel of the displacement alone, 1 at 0, ~0.6 one `scale` apart, ~0 two apart; a box's
vector is a closed form (a geometric series per axis, a 1,000-cell box costing one cell); a place reads back exactly on
the lattice; on the Hilbert curve consecutive cells share a face and every `8^j` consecutive cells fill an aligned `2^j`
cube — 512 tokens are an 8×8×8 block of the world where row-major order gives a 2×16×16 slab. Capacities, measured
(`tests/spatial/test_scene.py`): a scene vector `V = Σ code(name) ⊛ S(cell)` at the trunk's width d = 1024, 16 things in
a 32×32×4 box, reads back what-is-at-p and where-is-X at 1.00/1.00 and a box's contents at 1.00 (d = 512, 32 things:
0.95 / 0.66 — width buys capacity). **Mechanism, built (2026-10-17).** `cubbyllm/spatial/`: `ssp.py` (the encoding,
numpy), `hilbert.py` (the curve, vectorised, any dimension), `scene.py` (`Scene`: exact geometry — where / at / adjacent
/ nearest / within / offset / path — and the vector reading of the same scene, plus `facts(entity)` in the loop's `OBJ
is the REL of SUBJ` template: `position`, `<direction> neighbour`, `nearest <kind>`, `region`, derived on every call and
never stored, so the walk and the VM verify them like any other fact), `route.py` (the host's route: the scene is the
source when a relation word is the scene's or the seed is a thing in it). `cubbyllm/model/spatial.py` (`SpatialLink`):
**bind** at the input, `e ⊛ S(p)`; **scene**, `V_t = Σ_{i ≤ t, placed} e_i ⊛ S(p_i)` — the running sum of the placed
tokens' bound embeddings, causal like the context's mean, one accumulator per stream at decode, no codes but the model's
own embeddings (so `V ⊛ S(p)^-1` is the embedding of what is at `p`, already in the trunk's vocabulary); **read**
mid-trunk (layer L/2, through `HybridBackbone(read=)`, so the layers after it can decode what comes back), `g · o(V ⊛
unitary(q)^-1)` with `g = sigmoid(w·rms(h) + b)` the *whether* (bias −4 at start), `q = W rms(h)` made unitary the *what
place* (the trunk's magnitudes cannot change the place it names), `o` zero at start (growth's rule: exact at step 0),
zero read with nothing placed — the gate and the query are what trains, the lookup they route to is algebra. Wired
flag-gated: `CubbyModel(spatial=)`, `forward(tokens, positions, placed)` / `step(token, state, position, placed)` with
`state["scene"]`, `train_base.py` `CB_SPATIAL=1` (`meta["spatial"] = {scale, seed, at}`, absent at 0, every earlier
checkpoint's meta unchanged), `move_model` carries it; **the data**: `validation/make_spatial_shard.py` — villages (8–32
things of six kinds, one or two named regions, a 24×24×3 box) listed along the Hilbert curve with every name token
carrying its cell in a `.pos.i16` side file, then 6–12 questions in the loop's shapes answered from `Scene`'s exact
reading (position, `<direction> neighbour`, nearest `<kind>`, `n <direction> of`, region, and `What is at here?` with
`here` a placed token: the question itself is a place), documents padded to `--doc` tokens and listed in `_align.json`
so `train_base.Corpus` never cuts a village (it reads the side file, starts every window on a document, and hands
`forward_batch` a `Tokens` batch — ids plus places — that the link trains on; a corpus without places behaves exactly as
before); `--mode text` writes the control arm's shard, the same villages with cells as words and nothing placed;
`standin/ask.py` `AskLoop(scene=)` routes a spatial question to the scene's exact reading (the loop then fetches, gates,
stores with provenance `scene`, walks and verifies: "What is the position of the east neighbour of well?" → `11 7 0`,
verified, fetching only the hop that was missing; pinned in `test_scene.py`). The decode step binds, accumulates and
reads what the forward does, and the scene is causal (`test_spatial.py`); the shard's answers are the scene's, exactly
the name tokens and the asked `here` carry places, and the corpus/link path runs end to end with gradient at the gate
(`tests/spatial/test_shard.py`). **BlockWorld's surface, read for this** (`cubby-blockworld`): the world is `VoxelGrid`
cells `(x, y, z)` with z = 0 the floor and `FACES` the six neighbours — the scene's lattice and compass; `orders.py`
gives villagers tasks `at` a cell (`order(world, vid, task, at=)`), which is where a placed plan lands; Cubby v0 already
encodes cells compositionally for its own map (`agents/cubby_v0/vsa.py`, one operator Δ per move) and keeps its hut as
placements relative to where he stood (`hut.py`) — the same algebra, so his map and the trunk's scene can share a space.
**Gates (pre-registered):** (1) input — a base continued with `CB_SPATIAL=1` on the `spatial` shard
(`make_spatial_shard.py --villages 20000`, ~10M tokens; `CB_MIX_JSON {"spatial": w}` beside the text) answers the
held-out villages' questions at ≥ 90% where the same base continued on the `spatial_text` control (cells as words, no
link) sits below 60%; (2) route — the gate `g` opens (> 0.5) on ≥ 90% of spatial questions and stays shut (< 0.1) on ≥
95% of the held text, with the text held-out loss within 0.01 of the control's; (3) serving — through `AskLoop(scene=)`,
a village of 32 things: every spatial question the loop's grammar parses verified by the VM with 0 wrong spoken, and a
plan "place X n east of Y" executed through `orders.py` landing on the cell the scene computed; (4) the curve — the
450M's beyond-window recall (H-E20 gate 2) on scenes serialized along the curve at least its recall on the same scenes
in row-major order. **Kill:** gate 1 below 75%, or gate 2's text loss worse by more than 0.02 (the link costs the trunk
on text; keep the host side only). **Cost, stated:** the link has no grilly2 form yet (`export_base.py` refuses a
`spatial` meta; serve through torch until ported); FFTs on DirectML are untested — the link trains on CUDA. Status:
**OPEN — built and pinned (the encoding, the curve, the scene, the host route, the trunk's route mid-trunk, the shard
and its control, the corpus path); nothing trained yet — gates 1–2 run on the next base continuation (`CB_SPATIAL=1`,
both shards built); the Unity placement path (gate 3) is the next piece.** **First read at the 450M (2026-10-20,
`notebooks/spatial_facts_450m.ipynb` run 1: `base450m_cont_stable_end` continued 2,808 steps = 0.37B tokens an arm on an
RTX PRO 6000, the two loci-token shards at ~4.6% each; `link` = `CB_SPATIAL=1 CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1`,
the link added to the trained trunk with `o = 0`; `text` = the same villages with cells as words, at the link arm's step
count; `validation/exp_hypertoken_recall.py`, every held-out question teacher-forced):** villages -- `link` **0.793**
exact with the address on the question, **0.700** with the questions' places stripped (the trunk's own query, taught by
the route loss, finds the address), against `text` **0.995**: **gate 1 fails** -- in-window, cells written as numbers
are answered almost perfectly by attention, and the link is worse there at this budget (it is not killed by the letter:
0.793 is above 0.75; it is beaten). **Gate 2 passes** on the plain-text sources: held-out loss 2.2318 (`link`) against
2.2286 (`text`), +0.003, every source within 0.017 (`_mix`, which weights each arm's own loci-token shards in, reads
2.1662 / 2.1442); `instruct` and `program` read 0.10-0.13 lower on `link` because that arm still drew Drive's duplicate
copies of them, held-out tails included (fixed since: the Corpus skips such copies). What it says: the places channel
costs the trunk nothing on text and its route is learnable at scale, but where attention can see the words, the words
win; the case for places is what attention cannot see -- beyond the window, or more things than a window holds -- which
run 2 tests (H-K10). Throughput: 54.6k tok/s, 29.5% MFU (`link`, micro 16); run 2 at micro 32: 59.6k tok/s, 32.2%. Where
a step goes (run 2's `link`, its trace re-read from the device kernels alone -- the first `profile_base.py` counted
every kernel twice, once as itself and once in the op that launched it, and filed the attention kernels under matmul;
fixed): GEMMs **42%** of device time, running at ~377 TFLOPS, **75% of the card's peak**; attention 15% (dense-masked
memory-efficient kernels: no flex on this card, the full 1024 x 1024 computed for a 512 band); the MinGRU scan and its
backward 13%; other elementwise 22%; the vocabulary's softmax and loss 3%, optimizer 2%, the link's FFTs 1%; the GPU 99%
busy. MFU is GEMM speed x GEMM share: the GEMMs are fast, there are too few of them; the speed is in the 58% that is not
GEMM. The 128k vocabulary is not 2.8% (as first written here) but ~15% of the step: its head is 29% of the GEMM FLOPs.
**Run 3 on the G4 (`validation/speed_check.py`, under the profiler):** torch 57.1k tok/s, + the fused MinGRU kernel
63.7k (+11.5%; the scan 72 -> 18 ms a step), + fused MinGRU + flex attention at 99 KB tiles 71.8k (+25.7%; attention 84
-> 26 ms); the banded flash kernel failed there on its compile-time stride (fixed since, `e1185e6`). In the run: ~75.0k
tok/s, **40.6% MFU** (run 2: 59.6k, 32.2%). **Gate 3, real villages (order 0034, 2026-10-23; counts only in
`validation/logs/exp_s12_scene_files_2026-10-22.json`, the villages and their logs kept in the private BlockWorld
repo):** 60 villages (20 seeds × 3 days) exported from BlockWorld's own village sim, 3,600 questions fixed before any
village existed. **Host** (`AskLoop(scene=)` and the VM): **0 wrong spoken in 3,600**; it routes 4 of 7 shapes
(position, one-step neighbour, nearest of a kind, region: every answer right, every "nothing" refused), none of "n steps
away", "what is at x y z", "where is X". **Act:** 20 plans placed through the world's own order, **20 of 20 standing on
the computed cell**, 22 of 22 unplaceable targets refused before any order. **Trunk** (run 5, link + copy, on the placed
shapes it was trained on): things **0.006**, villager names **0.000**, "nothing" 0.972 -- **below the pre-registered
kill line (0.75): the synthetic villages did not transfer.** The same villages with every thing renamed to a shard word
read 0.581 (`diag_s12_names_2026-10-22.json`): mostly the words (the trunk opens an unseen village thing with "nothing",
and the copy does not supply the first token), partly the places (0.58, not the shard's 0.95). So: the host's sense of
place works on a real world and acts on it exactly; the trunk's does not yet, and the next step is a private shard of
exported villages (order 0035). **Order 0035, A and B (2026-10-23; counts in `exp_s12_scene_files_2026-10-23.json` and
`diag_s12_copy_2026-10-23.json`):** the host now routes all seven shapes on the same villages, 1,800 of 1,800 per view
and **0 wrong**. **The copy never fired on places:** its gate stays shut on village documents (λ 0.00 at all 359 asking
tokens of run 5's own shard), so run 5's 0.953 there was the read alone. The copy opens only in the far facts' shape:
the lead token asks, and the answer is written in its listed form. Put 0034's villages in that shape, untrained, and the
first token is right for things 0.726 and names 0.358. Most misses are another piece of the same name, which shares the
cell. So C goes on with its shard in the far facts' shape, each piece of a name at its own address. **Order 0035 C
(2026-10-23; counts in `exp_s12_copy_villages_run5c_2026-10-23.json`, the shard and its logs private):** the copy alone
(run 5's trunk frozen: 0 of 326 tensors moved; 1,027 trainable parameters, 410 steps, 4.99 h on the local CPU) on 3,350
exported-village documents from 24 seeds outside 0034's, in the space form (names written and answered as " name", the
":" of "Answer:" asking, each piece of a name at its own address). On 0034's 60 villages in that form: things **0.845**,
names **0.817**, "nothing" **0.910** (run 5 untrained in the same form: 0.818, 0.775, 0.609). The lines (0.75, 0.60) are
met, mostly by the shape; the training taught the gate to shut on an empty cell (its mean there 0.402 -> 0.094). The far
facts fell 0.9456 -> 0.9271 (within the 0.02 line by 0.0015; a rerun with the far facts at twice the share is to show
whether that cost is needed). D was not needed, and the kill was not reached. Wrong answers are still spoken (35 empty
cells named a neighbour's piece; 48 things or names denied), so the host's exact routes stay the speaking path. **The
rerun, C2 (2026-10-24; `train_spatial_link_5c2_2026-10-24.log`, `exp_hypertoken_recall_run5c2_cpu_2026-10-24.json`,
`exp_s12_copy_villages_run5c2_2026-10-24.json`, `diag_frozen_check_run5c2_2026-10-24.json`):** the far facts at twice
the villages' share, otherwise as 5c (410 steps, the copy alone; 0 of 326 trunk tensors moved). The far facts read
**0.9341** exact (132 wrong; 5c 0.9271, 146; run 5 before C 0.9456, 109). The space form reads things 0.845, names
0.825, "nothing" 0.908 (5c: 0.845, 0.817, 0.910), and run 5's own villages are unchanged at 0.9526. Below the 0.94 that
was set to keep it, so 5c stands: doubling the far facts' share recovers about a third of the cost, not all of it.
