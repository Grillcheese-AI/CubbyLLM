# French affect pilot - blind read

Generated natively per register (never translated), temperature 0.9. Cached under `standin/data/out/fr_pilot_cache/`; `--offline` re-renders without spending. Rows at `standin/data/out/fr_affect_pilot.jsonl`.

**Arms are blind on purpose** - judge the French, then read the key at the bottom. The mechanical flags below catch only what the prompt tried to forbid (hexagonal markers, sacres in the neutral register, sacres piled up or at the low tier, and lines that name the emotion instead of being in it). They are a prompt-compliance measure, not a quality measure. Quality is the owner's call, and it is the only call that matters here.

Questions worth holding while reading: does a Quebecer say this out loud? Does the intensity actually climb from 0.3 to 0.9? Is the neutral register really country-less, or does it just sound like France with the slang filed off?

**192 rows, 37 tripped a flag.**

| arm | register | rows | flagged | flags seen |
|---|---|---:|---:|---|
| A | qc | 24 | 5 | names-the-emotion, sacre-pileup |
| A | intl | 24 | 0 | - |
| B | qc | 24 | 0 | - |
| B | intl | 24 | 15 | accents-stripped |
| C | qc | 24 | 1 | sacre-pileup |
| C | intl | 24 | 0 | - |
| D | qc | 24 | 7 | accents-stripped |
| D | intl | 24 | 9 | accents-stripped |

## Register: Quebec

### joie

**0.3**

- **A** - J'ai réussi, c'est pas trop tôt.
- **B** - Bon, ça marche du premier coup. C'est pas pire, ça.
- **C** - Ah ben là, ça marche comme du monde, finalement.
- **D** - Ah tiens, le voyant vient de virer au vert tout seul.

**0.6**

- **A** - Yé! J'l'ai eu, tabarnouche, j'suis trop content.
- **B** - Ayoye, ça a passé au complet! Batince que ça fait du bien.
- **C** - Tabarouette, ça y est, j’ai réussi!
- **D** - Hah, ca marche enfin! Tabarouette que c'etait temps.  `accents-stripped`

**0.9**

- **A** - YÉÉÉ! J'l'ai fait, calisse, j'suis le meilleur!
- **B** - Yes! Ça marche, calisse, ça marche pour de vrai! Enfin!
- **C** - Tabarnak, c’est parfait, je trippe en simonac!
- **D** - YES! Tout a passe d'une shotte, criss!

### confiance

**0.3**

- **A** - J'vas y arriver, c'est sûr.
- **B** - Correct, je sais où je m'en vais avec ça. On continue.
- **C** - J’ai tout ce qu’il faut, pis ça va se faire tout seul.
- **D** - Je connais le chemin par coeur, ca va se faire tout seul.  `accents-stripped`

**0.6**

- **A** - J'suis prêt, j'vais y aller à fond, batince.
- **B** - Laisse-moi aller, je l'ai. Ça, je sais exactement comment le régler.
- **C** - Mosus, je savais que j’étais capable, pis je l’ai fait.
- **D** - Laisse-moi faire, je sais exactement quelle vis tient tout le panneau.

**0.9**

- **A** - J'suis imbattable, j'vais tout détruire, tabarnak!
- **B** - Ça, personne me l'enlève. Je l'ai réglé, ostie, et ça a tenu.
- **C** - Criss, je suis né pour faire ça, pis ça paraît!
- **D** - Amenez-en des pannes, y a rien qui va m'arreter a soir, calisse.

### peur

**0.3**

- **A** - J'espère que ça va tenir, sinon j'suis dans'merde.
- **B** - Hmm, j'aime pas trop ce chiffre-là. Je vais checker deux fois avant.
- **C** - C’est bizarre, ce silence-là, ça me tente pas pantoute.
- **D** - C'etait quoi ce bruit-la en dessous du plancher?  `accents-stripped`

**0.6**

- **A** - J'ai peur que ça pète, caline, j'sais pas si j'devrais continuer.  `names-the-emotion`
- **B** - Non non non, dis-moi pas que j'ai effacé la mauvaise affaire. Mosus, respire.
- **C** - Batince, si ça lâche maintenant, on est dans marde.
- **D** - Les lumieres viennent de baisser d'un coup sec, batince.

**0.9**

- **A** - J'ai peur en ostie, j'sais pas si j'vais m'en sortir, criss.  `sacre-pileup names-the-emotion`
- **B** - Non, non, pas ça, pas maintenant! Criss, tout va sauter si je bouge.
- **C** - Calisse, si ça marche pas, je suis dans la marde jusqu’au cou.
- **D** - La porte est soudee pis la fumee monte, tabarnak!

### surprise

**0.3**

- **A** - Ben voyons, c'est donc ben bizarre, ça.
- **B** - Ah tiens, je pensais pas que c'était là. OK.
- **C** - Hein? C’est déjà fait? Ben voyons donc.
- **D** - Voyons, c'etait pas cense etre barre a cette heure-ci.  `accents-stripped`

**0.6**

- **A** - Ben non! J'l'avais pas vu venir, celle-là, mosus.
- **B** - Hein? D'où ça sort ça? J'ai jamais mis ça là, moi.
- **C** - Caline, je m’attendais pas à ça pantoute!
- **D** - Ben voyons donc, d'ou c'est que ca sort toute cette eau-la?  `accents-stripped`

**0.9**

- **A** - Ben calisse! J'l'avais pas vu venir, celle-là, ostie!  `sacre-pileup`
- **B** - Voyons calisse! C'était là tout ce temps-là? J'en reviens pas!
- **C** - Ostie, c’est pas vrai, comment ça se fait que c’est déjà fini?
- **D** - Le plancher vient litteralement de s'effondrer sous mes pieds, ostie!

### tristesse

**0.3**

- **A** - C'est plate en maudit, mais j'peux rien y faire.
- **B** - Bof. Encore la même affaire qui lâche. On lâche pas.
- **C** - Encore une autre affaire qui pogne pas, ça finit pus.
- **D** - Encore a recommencer depuis le debut. Bon.

**0.6**

- **A** - C'est triste en maudit, j'vais m'ennuyer de ça, viarge.
- **B** - Toute cette job-là pour rien. Caline que c'est plate à voir.
- **C** - Encore une autre patente qui veut pas, tabarnouche, c’est lassant.
- **D** - Il reste plus rien de recuperable la-dedans, caline.

**0.9**

- **A** - C'est triste en criss, j'peux plus rien faire, j'suis pu capable.
- **B** - Toute est parti. Des semaines, ostie, des semaines dans le vide.
- **C** - Encore une autre affaire qui pogne pas, tabarnak, j’en peux pus.
- **D** - Trois ans d'efforts qui partent en cendre en deux secondes, criss.

### degout

**0.3**

- **A** - C'est dégueulasse, j'vais devoir nettoyer ça.
- **B** - Ouache, c'est mal fait ce fichier-là. Ça me tente pas de fouiller dedans.
- **C** - Non mais regarde-moi ça, c’est-tu laid en simonac.
- **D** - C'est tout gommant sur la poignee, voyons donc.

**0.6**

- **A** - C'est dégoûtant, j'peux pas croire que j'doive faire ça, tabarouette.
- **B** - Coudonc, c'est qui l'épais qui a codé ça? Tabarouette, ça pue à plein nez.
- **C** - Non mais regarde-moi ça, c’est-tu assez dégueulasse, viarge.
- **D** - Ca pue l'eau croupie jusque dans le fond de la gorge, mosus.  `accents-stripped`

**0.9**

- **A** - C'est dégueulasse en ostie, j'peux pas croire que j'doive faire ça, calisse.  `sacre-pileup`
- **B** - C'est dégueulasse ce qu'ils ont fait avec ça. Criss que ça me lève le cœur.
- **C** - Non mais regarde-moi ça, c’est-tu assez dégueulasse pour te faire vomir, calisse.
- **D** - J'ai les deux mains pleines de cambouis noiratre pis de vers, calisse de bordel.

### colere

**0.3**

- **A** - C'est pas croyable, ça marche jamais comme il faut.
- **B** - Bon, encore une erreur. Reste calme, on reprend.
- **C** - Coudonc, y’a-tu quelqu’un qui écoute quelque part?
- **D** - Deuxieme fois que la cle tourne dans le vide.

**0.6**

- **A** - Ça m'écoeure, ça marche jamais comme il faut, ostie de machine!
- **B** - Envoye, ouvre-toi! Batince de porte, ça fait trois fois que j'essaie.
- **C** - C’est quoi l’ostie de problème encore?
- **D** - Ouvre-toi, espece de vieille porte a batterie, tabarnouche!

**0.9**

- **A** - J'suis en criss, ça marche jamais comme il faut, ostie de tabarnak!  `sacre-pileup`
- **B** - Ouvre-toi, tabarnak! Ça fait dix fois que je te le demande!
- **C** - Crisse, ouvre-toé, ostie de porte de marde!  `sacre-pileup`
- **D** - DEBLOQUE, MANGEUX DE MARDE, TABARNAK!

### anticipation

**0.3**

- **A** - J'me demande ben ce qui va arriver, j'suis pas mal curieux.
- **B** - OK, dans deux minutes ça devrait être fini. Je surveille.
- **C** - Si ça continue de même, ça va être prêt avant la noirceur.
- **D** - Le signal devrait embarquer d'une seconde a l'autre.

**0.6**

- **A** - J'suis excité, j'peux pas attendre de voir ce qui va se passer, j'ai hâte en maudit.
- **B** - Là ça s'en vient, ça s'en vient. Attends un peu que ça finisse de charger.
- **C** - Si je me grouille, j’ai le temps de tout finir avant qu’y’arrivent.
- **D** - Plus que trois secondes avant que la valve ouvre, faut que je sois pret.

**0.9**

- **A** - J'suis trop excité, j'peux pu attendre, j'vais y aller à fond, crisse que j'ai hâte!
- **B** - Envoye, envoye, presque là! Ostie que j'attends ça depuis longtemps, accroche-toi!
- **C** - Si je me dépêche pas, je vais tout faire foirer, tabarnak, grouille!
- **D** - Le compte a rebours arrive a zero, ca passe ou ca casse, ostie!  `accents-stripped`

## Register: international / neutral

### joie

**0.3**

- **A** - Ça avance, doucement mais sûrement.
- **B** - Tiens, ca marche. C'est deja ca.  `accents-stripped`
- **C** - Ah... ça y est. Enfin. Juste ce qu'il fallait.
- **D** - Ah, le voyant passe au vert. On avance enfin d'un cran.

**0.6**

- **A** - Ça y est, j’ai trouvé la solution !
- **B** - La, ca avance vraiment. Je le savais que ca passerait.  `accents-stripped`
- **C** - Oui ! Ça marche ! Ça marche vraiment !
- **D** - Oui, exactement comme prevu! Tout s'enclenche au millimetre pres.  `accents-stripped`

**0.9**

- **A** - C’est incroyable ! J’ai réussi, enfin !
- **B** - Oui! Ca y est, ca fonctionne! Enfin, enfin!  `accents-stripped`
- **C** - JE L'AI ! JE L'AI FAIT ! ENFIN !
- **D** - C'est passe, c'est passe! On l'a fait, tout s'allume, absolument tout fonctionne!

### confiance

**0.3**

- **A** - Je vais y arriver, c’est certain.
- **B** - Bon. Je sais ou je vais, je continue.
- **C** - Bon. Je reprends. Un pas après l'autre, ça va tenir.
- **D** - Le signal reste stable. Rien ne devrait bouger d'ici la fin du cycle.

**0.6**

- **A** - Je contrôle la situation, pas de doute.
- **B** - J'ai deja fait ca cent fois. Je ne me trompe pas.  `accents-stripped`
- **C** - Je le savais. Je le savais depuis le début. Allez.
- **D** - Je connais cette procedure par coeur. Laissez tourner, je m'occupe du reste.  `accents-stripped`

**0.9**

- **A** - Rien ne peut m’arrêter maintenant, je suis invincible.
- **B** - Rien ne m'arrete. Je passe, et je passe partout.  `accents-stripped`
- **C** - Personne ne peut m'arrêter. Personne. Je vais jusqu'au bout.
- **D** - Rien ne peut m'arreter aujourd'hui. Rien, vous entendez, c'est totalement sous controle.

### peur

**0.3**

- **A** - Il y a quelque chose qui ne tourne pas rond ici.
- **B** - Attends. Ce bruit-la, c'etait quoi?  `accents-stripped`
- **C** - Non... pas maintenant. Pas si près du but.
- **D** - Ce bruit dans la gaine n'etait pas la hier. Il faut surveiller la pression.  `accents-stripped`

**0.6**

- **A** - Il faut que je sorte d’ici, vite.
- **B** - Non, non, pas maintenant. Reste en ligne, reste en ligne.
- **C** - Non, non, non... Ferme les yeux. Respire. Ça va s'arrêter.
- **D** - Attends, pourquoi la temperature grimpe d'un coup? Arrete tout, coupe l'arrivee maintenant!  `accents-stripped`

**0.9**

- **A** - Je ne peux pas rester ici, il faut que je parte immédiatement !
- **B** - Ca lache, ca lache! Je perds tout, je perds tout!  `accents-stripped`
- **C** - Je ne peux pas. Je ne peux plus. Je ne veux plus avancer.
- **D** - La porte est bloquee, la porte est bloquee! Sortez-moi de la, ouvrez, au secours!

### surprise

**0.3**

- **A** - Ah, je ne m’attendais pas à ça.
- **B** - Ah. Je ne m'attendais pas a ca ici.  `accents-stripped`
- **C** - Quoi ? Déjà ? Je n'ai même pas entendu le signal.
- **D** - Tiens, l'acces est deja deverrouille. Je n'avais meme pas fini la requete.  `accents-stripped`

**0.6**

- **A** - Qu’est-ce que c’est que ça ? Je n’y comprends rien.
- **B** - Quoi? Ce n'etait pas la il y a une seconde.  `accents-stripped`
- **C** - Attends... quoi ?! Ce n'est pas possible, ça !
- **D** - Mais d'ou sort ce chiffre? Il n'y avait absolument rien dans cette case il y a une seconde!

**0.9**

- **A** - Mais c’est impossible ! Comment est-ce possible ?
- **B** - Ce n'est pas possible. Ce n'est pas possible! D'ou ca sort?  `accents-stripped`
- **C** - QUOI ?! COMMENT ?! CE N'EST PAS VRAI !
- **D** - Ce n'est pas possible... Non! Mais comment ca a pu apparaitre en plein milieu?!  `accents-stripped`

### tristesse

**0.3**

- **A** - C’est calme, trop calme.
- **B** - Encore vide. Bon, tant pis.
- **C** - C'était trop lourd. Je savais que ça ne passerait pas.
- **D** - Tant d'heures passees la-dessus pour devoir tout effacer. Tant pis.

**0.6**

- **A** - Tout semble si vide, si silencieux.
- **B** - Tout ce travail, et il ne reste rien. Rien du tout.
- **C** - Je n'aurais pas dû. Je n'aurais vraiment pas dû.
- **D** - Il ne reste plus rien de recuperable dans le dossier. Tout ce travail est perdu pour de bon.

**0.9**

- **A** - Tout est perdu, il n’y a plus rien à faire.
- **B** - Tout est perdu. Il n'y a plus rien a sauver. Plus rien.
- **C** - Tout est fini. Plus rien. Je n'ai plus rien.
- **D** - C'est fini, tout s'est effondre. Il n'y a plus rien a faire, plus rien.

### degout

**0.3**

- **A** - Cette odeur, c’est insupportable.
- **B** - Non, pas comme ca. Ce n'est pas propre du tout.  `accents-stripped`
- **C** - Encore cette odeur. Comme si ça ne partirait jamais.
- **D** - C'est visqueux, ca colle partout aux parois. Je ne veux meme pas savoir ce que c'est.  `accents-stripped`

**0.6**

- **A** - Je ne peux pas toucher ça, c’est dégoûtant.
- **B** - Je ne touche pas a ca. C'est pourri jusqu'a l'os.  `accents-stripped`
- **C** - C'est immonde. Je ne peux plus. Je ne veux plus voir ça.
- **D** - Cette odeur d'huile brulee me monte a la gorge. C'est immonde, arretez cette ventilation.

**0.9**

- **A** - C’est répugnant, je ne peux pas supporter ça plus longtemps.
- **B** - Non. Non. Je ne peux meme pas regarder ca. Enlevez-moi ca.  `accents-stripped`
- **C** - Je vais vomir. Je vais vraiment vomir si ça continue.
- **D** - Eloigne ca de moi, ne me touche pas avec ca! C'est une horreur absolue, j'ai envie de vomir!  `accents-stripped`

### colere

**0.3**

- **A** - Cette machine ne veut rien savoir.
- **B** - Encore fermee. Je n'aime pas ca.  `accents-stripped`
- **C** - Non mais... c'est une blague ? Vraiment ?
- **D** - Ce n'est pas le bon cable, evidemment. On perd encore dix minutes pour rien.

**0.6**

- **A** - Cette porte ne s’ouvre pas ! C’est insupportable.
- **B** - Ouvre. Ouvre, maintenant. J'ai assez attendu.
- **C** - Ça suffit. Là, ça suffit. Je ne bouge plus.
- **D** - Ouvre-toi, espece de ferraille! Tu vas ceder, oui ou non?

**0.9**

- **A** - Ça suffit ! Je n’en peux plus de cette situation !
- **B** - Ca suffit! Ouvre! Je n'en peux plus de cette porte!  `accents-stripped`
- **C** - ÇA NE MARCHERA JAMAIS ! JE DÉTRUIS TOUT !
- **D** - Ca suffit! Marche, sale machine, marche, ou je te detruis piece par piece tout de suite!  `accents-stripped`

### anticipation

**0.3**

- **A** - Je sens que quelque chose va se passer.
- **B** - Bientot. Je le sens venir, je reste pret.
- **C** - Plus que trois. Deux. Un... presque.
- **D** - La jauge monte regulierement. Dans deux minutes, on saura.

**0.6**

- **A** - Je suis prêt, ça va bientôt arriver.
- **B** - Encore trois pas et j'y suis. Je ne lache pas.
- **C** - Je sens que ça vient. Je le sens. Tout va basculer.
- **D** - Le compte a rebours est lance. Des que la porte s'ouvre, je saute direct sur le disjoncteur.

**0.9**

- **A** - Ça arrive, je le sens, c’est imminent.
- **B** - C'est maintenant. Maintenant! Tout se joue la, je fonce!
- **C** - C'EST LÀ. C'EST MAINTENANT. JE LE SAIS.
- **D** - Encore trois secondes, deux secondes, allez, vas-y, maintenant, frappe!


---

<details><summary>arm key (read after judging)</summary>

- **A** = `cohere/command-a`
- **B** = `anthropic/claude-opus-4.8`
- **C** = `mistralai/mistral-large-2512`
- **D** = `google/gemini-3.8-flash`

</details>

<details><summary>full prompt, Quebec register</summary>

```
LANGUE: francais quebecois, celui qui se parle vraiment au Quebec aujourd'hui.
Registre courant de tous les jours, pas du joual de caricature, pas de
transcription phonetique pour faire couleur locale.

Les sacres sont le canal de decharge et ils ont des paliers:
- intensite 0.3: aucun sacre.
- intensite 0.6: formes attenuees seulement (tabarnouche, tabarouette, batince,
  caline, mosus, viarge).
- intensite 0.9: le sacre plein est permis (tabarnak, calisse, ostie, criss).
JAMAIS plus d'un sacre par replique. Un sacre empile sur un autre, c'est de la
parodie, et une parodie rend la ligne inutilisable.

Interdits: tout marqueur hexagonal (putain, meuf, mec, bagnole, verlan, "truc de
ouf"). Ca n'est pas du quebecois et ca contamine le corpus.

Ecris 24 repliques courtes, une par ligne, au format JSON.

Qui parle: un agent autonome, seul, qui se parle a lui-meme a voix haute - ou qui
engueule un obstacle, une machine, une porte qui ne s'ouvre pas. Ce n'est PAS un
assistant qui repond a un utilisateur. Il n'y a personne a qui etre poli.

Regles absolues:
- PREMIERE PERSONNE, present, parle - pas ecrit.
- NE NOMME JAMAIS L'EMOTION. Interdit: "je suis frustre", "je me sens triste",
  "quelle joie", "j'ai peur". L'etat doit se deviner par ce qui est dit et par la
  facon de le dire, jamais par l'etiquette.
- 1 a 2 phrases. Longueur d'une vraie replique parlee.
- Chaque replique est differente des autres: varie la situation, le rythme, la
  longueur. Pas de gabarit repete.
- Aucun emoji, aucune didascalie entre asterisques.

Format de sortie: une ligne JSON par replique, rien d'autre, pas de bloc de code:
{"petale": "<petale>", "intensite": <nombre>, "texte": "<la replique>"}

Les couples (petale, intensite) a couvrir, dans cet ordre exact:
- (joie, 0.3)
- (confiance, 0.3)
- (peur, 0.3)
- (surprise, 0.3)
- (tristesse, 0.3)
- (degout, 0.3)
- (colere, 0.3)
- (anticipation, 0.3)
- (joie, 0.6)
- (confiance, 0.6)
- (peur, 0.6)
- (surprise, 0.6)
- (tristesse, 0.6)
- (degout, 0.6)
- (colere, 0.6)
- (anticipation, 0.6)
- (joie, 0.9)
- (confiance, 0.9)
- (peur, 0.9)
- (surprise, 0.9)
- (tristesse, 0.9)
- (degout, 0.9)
- (colere, 0.9)
- (anticipation, 0.9)

L'intensite 0.3 est a peine perceptible, 0.6 est franche, 0.9 est au maximum de ce
qu'un humain exprime sans s'effondrer. L'intensite doit s'entendre dans la replique.
```

</details>
