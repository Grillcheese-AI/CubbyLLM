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
  - who is Hamlet's mother in Hamlet?
  - In King Lear, who are Lear's daughters?
  - in Othello, who is Desdemona's father?
  - who is Ophelia's brother in Hamlet?
  - In The Merchant of Venice, who is Jessica's father?
  - who is Prospero's daughter in The Tempest?
  - In Much Ado About Nothing, who is Hero's father?
  - who is Lady Macbeth's husband?
  - In A Doll's House, who is Nora's husband?
  - In The Importance of Being Earnest, who is Gwendolen's mother?
  - who is Cordelia's father in King Lear?
  - In Julius Caesar, who is Brutus's wife?
  - in Romeo and Juliet, who is Romeo's father?
  - In Hedda Gabler, who is Hedda's husband?
  - who are the sons of Titus in Titus Andronicus?
  - dans Roméo et Juliette, qui est la mère de Juliette ?
  - dans Hamlet, qui est le père d'Ophélie ?
  - dans Le Roi Lear, qui sont les filles de Lear ?
  - dans Othello, qui est le mari de Desdémone ?
  - dans Macbeth, qui est la femme de Macbeth ?
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
  - who played Hamlet in the 1948 film?
  - when was Romeo and Juliet first performed?
  - who was Shakespeare's wife?
  - how many children did Shakespeare have?
  - what is the plot of Macbeth?
  - who was the real Macbeth?
  - who is the mother of King Charles III?
  - who wrote A Doll's House?
  - what is the most famous line in Hamlet?
  - where is Verona?
  - is Othello a tragedy or a comedy?
  - who is the father of Prince William?
  - what does 'wherefore art thou' mean?
  - how long is Hamlet?
  - qui a écrit Roméo et Juliette ?
  - quand Shakespeare est-il mort ?
  - qui est le père du roi Charles III ?
  - où se trouve le théâtre du Globe ?
  - de quoi parle Hamlet ?
  - qui était la femme de Molière ?
kinds: [world]
audience: public
licence: Public domain (Project Gutenberg texts; the Gutenberg licence for the files)
requires: {plugins: [], base: null}
permissions: {network: [], write: true, gpu: false}
world: {route: fiction, data_env: CUBBY_PLAYS, glob: "[0-9]*.txt"}
entry: {}
---

# gutenberg-plays

Public-domain plays as a fiction world pack: each play its own world, answering only a question that names the play, as
order 0015's fiction worlds do. Its facts are what each play's list of persons states, read as stated and in-fiction
("JULIET, daughter to Capulet" -> Juliet is the daughter of Capulet; "GERTRUDE, ... Hamlet's mother" -> Gertrude is the
mother of Hamlet), plus any kin its prose states. A relation the play does not state gets nothing (a refusal): the
reader never infers a mother from a father's wife.

The texts are not in this folder: `CUBBY_PLAYS` names the local folder of Project Gutenberg's modern-spelling editions
(`<id>.txt`, each keeping the Project Gutenberg licence text, with `<id>_metadata.json` for the title). Not Folger's
texts: they are CC BY-NC. The pack is ingested once into the plugin's data folder.
