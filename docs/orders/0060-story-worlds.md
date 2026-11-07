# 0060 — Story worlds: roleplay on a world state, where every action is checked and the story never contradicts itself
status: open (queue: after 0059)
written: 2026-11-07 by Cowork · approved: 2026-11-07 by Nick ("yes", after his roleplay example: a player shows an iron key to the innkeeper Eldrin) · hypothesis: MoWM (separate worlds), with 0055's two clocks and 0059's conversation state

## Why
Nick's example: a player slides an iron key across a tavern table and asks Eldrin about the symbol carved into it.
Behind the reply sit a check that the player has the key, a change to the inventory, a change to Eldrin's trust, and
Eldrin's line in character. The idea is to keep the story's world as records, the same way Cubby keeps facts: a
player's action is checked against them, and a change is committed only after the check passes.

**What it brings, in plain words:** Cubby can run a story where characters remember what happened, objects stay where
they were left, and nobody contradicts what was established.
- **Over a conventional model:** a chat model keeps the story only in its context window, so it drifts: a lost key
  reappears, a dead character walks back in, an innkeeper knows a secret nobody told him.

## Build
0. **Licences first, under 0010's rule** (addendum 3):
   - every licence is recorded in a manifest before a row is used;
   - non-commercial sets go to research runs only;
   - model-written roleplay sets are excluded, as inputs and as targets.
1. **One world per story.** Each story is a MoWM world with its own store. The store holds the characters, places,
   objects, who holds what, who is where, who knows what, the relations between characters, and each character's
   state.
   - The world is seeded from a story setup, written by Nick or by rule.
   - **Isolation:** a story fact never answers a real-world question, and a real-world fact enters a story only when
     the story's setup allows it (a historical setting, for example).
2. **Actions as programs.** A player's turn is parsed into an action program (move, take, give, show, ask, open,
   attack and so on). Its prerequisites are checked exactly against the world:
   - the player holds the item;
   - both characters are in the same place;
   - the character is alive and awake;
   - the door is not locked.

   If they pass, the changes are committed as dated changes (0055's two clocks), each with its turn. If they fail,
   the narrator says why, inside the story ("You reach for the key, but your pocket is empty"), and nothing changes.
3. **Characters speak from what they know.** The 0010 talk voice speaks as the character, grounded on that character's
   own knowledge in the world. A character can reveal only what it holds: Eldrin can explain the symbol only if the
   world says Eldrin knows it.
4. **Invented facts become canon.** A narrator has to invent, and Eldrin naming "the weavers of the lower district"
   is new lore.
   - An invented fact is a proposal, checked against the established facts with exact rules: one value where only
     one is allowed, no contradiction of the timeline, no disjoint classes.
   - If it passes, it is committed with the source "invented at turn N", and the story must stay consistent with it
     from then on.
   - If it fails, the line is redrafted without it.
5. **Exact checks and soft checks, labelled.** Prerequisites, continuity and who-knows-what are exact checks against
   the records. Tone and genre ("no anachronism in a medieval fantasy") are similarity scores, recorded as soft checks
   and never presented as certified.
6. **Character states change by rule.** Trust, suspicion and mood change through appraisal rules, from what the player
   did and what the character values, using the existing affect machinery. The model never picks the number. Each
   change is stored with its old value, its new value and its turn, and can be exported in EmotionML (0020's
   addendum).
7. **Rewind and replay.** Pinning the world to a turn gives the exact state at that turn, so a session can be replayed
   or rewound.
8. **The record:** each turn is kept in the conversation record format 0059 settles. Roleplay adds the action, its
   checks (each marked exact or soft), the state changes and any invented facts.
9. **Data:**
   - **LIGHT** (Urbanek et al., EMNLP 2019): human crowdworkers playing fantasy characters, with locations, objects
     and actions, which is the world-state supervision;
   - **CRD3** (Rameshkumar and Bailey, ACL 2020): transcripts of the Critical Role Dungeons & Dragons show, for the
     narrator's register;
   - both licences checked first.

## Acceptance (pre-registered)
- **A continuity battery:** 100 scripted sessions of 10 to 20 turns, written by rule, each with planted traps:
  - using an item already given away;
  - speaking to a dead character;
  - asking a character about something it never learned;
  - an invented fact that contradicts established lore;
  - a real-world question asked inside a story, and a story question asked outside it.
- **Hard lines:**
  - every trap that an exact check covers is caught, with 0 continuity breaks among them;
  - 0 leaks between a story world and the real-world store, in either direction.
- **Reported:**
  - invented facts committed, and invented facts rejected for contradiction;
  - the soft-check scores, labelled as soft;
  - rewind exactness: the state at a pinned turn equals the recorded state.
- **Nick's blind A/B:** 20 sessions against the current stack, judged for story quality and consistency. A comparison
  with a frontier chat model is a later step, through OpenRouter, and Nick's call.
- **Kill:** any leak between a story world and the real-world store. Story worlds are switched off until it is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **No external model** on the serving path or in the data.
- **The existing content, identity and voice rules hold.** This order adds no new unlock.
- Training notebooks are Nick's to run on Colab. Claude Code never starts a paid run.
- No local path in a committed file.

## Out of scope
- Several players in one story.
- Images and voice.
- Connecting to a game engine. That is a later step, once this record format is settled.
- Emotional support. That is 0010's addenda, with the crisis route first.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
