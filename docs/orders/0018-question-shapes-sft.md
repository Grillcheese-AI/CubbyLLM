# 0018 — Question shapes, EN-US / EN-UK / FR-FR / FR-QC: an SFT set for the emitter, no external model
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("we need to SFT a dataset of question forms and shapes in
both French and English (UK, USA)") · hypothesis: H-E15 (the 450M emitter), with orders 0012 (question shape) and 0017
(claims) · **suggested after 0012**

## Why
The 450M emitter holds the programs seat (H-E15). It misses on **wording**: "Where was Marie Curie born?" planned
`birth`, a relation word the store does not hold. Two-hop and other shapes were not held either. People ask in
many shapes:
- wh-questions in any tense, "in 1990";
- yes/no, tag ("…, wasn't he?") and leading ("…, right?") questions;
- embedded ("Do you know who…"), imperative ("Name…") and clefted ("C'est qui qui…") questions;
- follow-ups ("And his wife?") and bare search queries ("napoleon birthplace").

They do it in four varieties Cubby must serve: US and UK English, France and Quebec French. A set built from the
worlds' facts × a per-relation lexicon × these shapes, with the plan as the target, teaches the emitter to read them
all into the same plan. Nick's rule: no external model writes any of it.

- **What it brings:** however someone asks, in English or French, from Lévis or London, Cubby reads the same
  question underneath.
- **Advantage over conventional models:** a chat model guesses at intent. Here every phrasing maps to one explicit
  plan the host can check, and a leading phrasing cannot change it.
- **In plain words:** ask your way, in your own English or French, and Cubby still understands exactly what you
  mean.

## Build
- **The lexicon** (`standin/data/question_lexicon/`, one file per variety):
  - per relation: the wh-word by answer type, the verb forms (active, passive, noun), the tense auxiliaries
    (was / is / will be; était / est / sera), the prepositions;
  - seeds: hdc's `relation_template_mapping.csv` (EN) and `wikidata_properties_en_fr.json` (FR labels), plus
    hand-written variants;
  - Nick reviews the **FR-QC** list (the "-tu" particle: "Il était-tu…?", "Tu sais-tu c'est qui qui…?";
    "C'est quoi…?"; "pis…?").
- **Shapes,** each in the four varieties:
  - wh: past / present / future, with and without "in {year}";
  - yes/no (inversion, and declarative with a "?");
  - tag: "wasn't he?", "isn't it?", UK "innit", "n'est-ce pas ?", QC "hein ?" / "non ?";
  - leading ("…, right?"), negative ("Wasn't…?", "C'est pas…?");
  - embedded ("Do you know / Can you tell me…", "Sais-tu / Peux-tu me dire…", QC "Tu sais-tu…");
  - imperative ("Name / Tell me…", "Dis-moi / Nomme…");
  - clefted ("Qui est-ce qui…", "C'est qui qui…");
  - elliptical follow-ups (given the previous turn: "And his wife?", "Pis sa femme ?");
  - search-query style (lowercase, no "?", keywords);
  - spelling and format variants (colour/color, "5 May 1821" / "May 5, 1821", missing accents and elisions).
- **Targets:** the plan the emitter already emits (H-E15's slots), plus a question **kind** (wh, yes/no or claim),
  a **time constraint** (0012) and the claim fields (0017). A leading or tag form has the same plan as its neutral
  twin.
- **Data** (`standin/data/build_question_shapes_sft.py`): facts from the worlds (the wiki world, hdc once 0014
  lands), split by entity. **Held out:**
  - 2 shapes per variety;
  - a set of relations;
  - every evaluation set: WebQuestions and NQ dev/test, GPQA, SEA-HELM.
- **Training:** an emitter LoRA on `base450m_cont` (as `v13e`), the v12e recipe, a Colab notebook for Cowork to
  run (~1 h on the G4, ~$1).

## Acceptance (pre-registered)
- **Held-out shapes:** plan exact ≥ **0.90** in each variety. **Held-out relations:** ≥ **0.80**.
- **Human questions** (never trained):
  - NQ / WebQuestions dev (EN-US);
  - 100 EN-UK questions, hand-written;
  - **100 FR-QC questions Nick writes himself**, plus 100 FR-FR.

  Through the ask loop, spoken-and-correct beats v12e's by **≥ 10 points**, with **0 wrong spoken**.
- **Invariance:** a neutral question and its tag and leading twins give the same plan, 100% on the held-out set.
- Tests: the builder's records parse; no held-out entity, shape or evaluation question in training. Full
  `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. **No external model** in the data (Nick). Evaluation sets never enter training.

## Out of scope
- The talk seat's wording (order 0010). Answering claims: the verdicts are order 0017; here only the plan.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
