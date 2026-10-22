---
name: gutenberg-plays
version: 0.1.0
description: "Who the characters of public-domain plays are to each other, inside each play: fathers, mothers, wives, husbands, brothers, sisters, sons and daughters."
examples:
  - In Romeo and Juliet, who is Juliet's mother?
  - In Titus Andronicus, who is Lavinia's father?
  - In Macbeth, who is Malcolm's father?
  - In Romeo and Juliet, who is Tybalt's cousin?
  - In Antony and Cleopatra, who is Octavia's brother?
  - In She Stoops to Conquer, who is Tony Lumpkin's mother?
  - In Volpone, who is Celia's husband?
  - In An Enemy of the People, who is Petra's father?
  - In Troilus and Cressida, who is Hector's father?
  - In The Comedy of Errors, who is Adriana's husband?
not:
  - what is the capital of Canada?
  - convert 3 feet to metres
  - is there a storm warning in Lévis?
  - who is the father of Queen Elizabeth II?
  - when was Shakespeare born?
  - who wrote Hamlet?
  - where is the Globe Theatre?
  - what is the population of London?
  - who is the president of France?
  - how many plays did Shakespeare write?
kinds: [world]
audience: public
licence: Public domain (Project Gutenberg texts; the Gutenberg licence for the files)
requires: {plugins: [], base: null}
permissions: {network: [], write: true, gpu: false}
world: {route: fiction, data_env: CUBBY_PLAYS, glob: "[0-9]*.txt"}
entry: {}
---

# gutenberg-plays

The public-domain plays of order 0020 as a fiction world pack: each play its own world, the in-fiction facts read from
its text (who is whose father, mother, spouse, sibling, child), answering only a question that names the play, as
order 0015's fiction worlds do. A question that does not name a play gets nothing from it; a fact the play does not
state gets nothing either (a refusal).

The texts are not in this folder: `CUBBY_PLAYS` names the local folder of Gutenberg plays (`<id>.txt` with
`<id>_metadata.json`, the title read from the metadata); the pack is ingested once into the plugin's data folder.
