# Cubby, expliqué simplement

*Une description en langage courant pour les gens qui ne sont pas ingénieurs. Version française de
`CUBBY_EXPLAINER.md` (2026-09-04). Les lignes marquées « à venir » décrivent la conception, pas quelque chose que
vous pouvez utiliser aujourd'hui.*

## Ce qu'est Cubby

Cubby est une petite assistante IA conçue par Grillcheese Research Lab qui tourne sur votre propre ordinateur, ne
répond que quand elle peut vérifier sa réponse, et dit « je ne sais pas encore » quand elle ne le peut pas. La plupart
des assistants sont un seul gros modèle qui devine le prochain mot. Cubby est un petit modèle attaché à un
vérificateur : le modèle propose, un vérificateur distinct, la machine virtuelle CubeLang, exécute la proposition
comme un petit programme et ne laisse sortir la réponse que si les faits derrière tiennent. Tout ce qu'elle affirme,
vous pouvez le retracer jusqu'aux faits qu'elle a utilisés.

## La liste, en mots simples

| ce que dit la conception | ce que ça veut dire pour vous |
|---|---|
| petite et économe en énergie | Elle tourne sur un PC de jeu ordinaire, pas dans un centre de données. Votre facture d'électricité s'en aperçoit, la planète moins. |
| dit qu'elle ne sait pas au lieu d'inventer | Quand les faits ne sont pas là, vous obtenez « je ne sais pas encore » plutôt qu'une réponse fausse dite avec assurance. |
| des pensées ancrées et vérifiables | Une réponse factuelle n'est pas une suite de mots probables. C'est un petit programme qui a tourné et qui a été vérifié, et vous pouvez lire les faits et les étapes qu'il a utilisés. |
| s'ajuste elle-même grâce à CubeLang | Ses étapes de raisonnement sont écrites dans un langage fait pour être vérifié : une mauvaise étape est attrapée par la machine, pas par chance. |
| pas de réentraînement pour gagner une compétence | Une nouvelle compétence est un programme, pas un nouveau modèle. Cubby peut l'écrire, le faire certifier et le garder. |
| séparation des responsabilités | Une partie parle, une partie raisonne, des spécialistes s'occupent de tâches précises. Quand une partie apprend du nouveau, les autres n'oublient pas ce qu'elles savaient. |
| des mondes ancrés dans un monde global | Chaque tâche a sa propre mémoire de travail de faits, tous rattachés à un ensemble commun. Ce qu'elle apprend à un endroit reste cohérent partout. |
| conversations chiffrées | Ce que vous lui dites est stocké chiffré sur votre machine. Rien n'est envoyé nulle part. |
| programmable, autonome, appels d'outils via ToolForge | Elle peut décider d'utiliser un outil, en demander un quand aucun ne convient, et chaque outil qu'elle utilise est certifié avant de tourner. |
| entraînée à l'empathie, avec mesure | Elle lit la température émotionnelle de ce que vous écrivez et ajuste son ton. Elle ne prétend pas avoir des sentiments. |
| chaînes et graphes de pensée gérés par la machine | Les réponses en plusieurs étapes sont parcourues étape par étape par le vérificateur, en millisecondes, pas en écrivant des pages de « réflexion ». Une chaîne cassée s'arrête au lieu de finir sur une devinette. |
| s'améliorer à partir du passé et d'Internet — *à venir* | Apprendre de son propre historique est construit ; surveiller le Web pour les tendances et les menaces est conçu, derrière une permission explicite, et pas encore activé. |
| cortex visuel en direct, webcam — *à venir* | Voir par une caméra existe dans la lignée du laboratoire et ne fait pas partie de ce que vous obtenez aujourd'hui. |

## Que peut-elle faire pour moi que les autres assistants ne peuvent pas ?

**Elle vous dit quand elle ne sait pas, et propose d'aller voir.** Demandez-lui quelque chose pour lequel elle n'a aucun
fait et vous obtenez la même courte phrase à chaque fois, dans votre langue, plutôt qu'une réponse inventée. Sur son
propre jeu de test, elle a répondu correctement à 96 % des questions et faussement à 0 %, le reste étant « je ne sais
pas encore ». Quand vous lui avez permis de chercher, la phrase continue : « Je peux chercher pour toi si tu veux. »
Dites oui et elle cherche, confronte ce qu'elle a trouvé à ce qu'elle sait déjà, garde les faits avec leur source, et
répond. La prochaine fois que quelqu'un pose la question, elle sait, et peut dire d'où elle l'a appris. C'est ainsi que
ses connaissances grandissent : pas en devinant, en allant voir avec votre permission et en gardant le reçu.

**Elle se souvient de ce que vous lui dites, et vous pouvez vérifier qu'elle l'a fait.** Dites « retiens que le wifi du
chalet est bluebird42 » et elle peut y répondre l'instant d'après. Dites-lui quelque chose qui contredit un fait antérieur
et elle pointe le fait antérieur au lieu de l'écraser en silence.

**Elle montre son travail.** Chaque réponse factuelle vient avec les faits qu'elle a utilisés et les étapes qu'elle a
suivies. Si vous n'êtes pas d'accord, vous voyez exactement où.

**Elle réfléchit à peu de frais sur les problèmes difficiles.** Les autres assistants raisonnent en écrivant des
centaines de mots de « réflexion » à chaque fois que vous posez une question, et vous payez chaque mot, en temps et en
électricité. Cubby écrit un court programme une fois, le vérificateur l'exécute en millisecondes, et un programme qui a
marché est gardé et réexécuté sans aucun mot : un raccourci qu'elle a inventé dans le labyrinthe lui économise quatre
coups chaque fois qu'elle l'utilise, et une recherche en plusieurs étapes qui serait une page de texte pour un autre
modèle est ici une marche de cinquante millisecondes. Plus le raisonnement est difficile, plus la différence est grande.

**Elle reste sur votre machine, et vos conversations restent à vous.** Pas de compte, pas d'infonuagique, pas de
journaux d'utilisation sur le serveur de quelqu'un d'autre. Ce qu'elle stocke de vos conversations est chiffré avec une
clé qui n'existe que sur votre ordinateur.

**Elle parle votre français.** Elle a été entraînée sur les expressions québécoises et sur la façon dont les gens d'ici
textent vraiment, pas seulement sur le français des manuels, et elle répond dans la langue où vous écrivez.

**Elle lit l'ambiance.** Elle perçoit si vous êtes frustré, inquiet ou content et ajuste son ton, sans prétendre à des
sentiments à elle. Si vous écrivez « rien ne marche aujourd'hui », vous obtenez une Cubby plus calme et plus brève.

**Elle apprend de nouvelles compétences sans être réentraînée.** Demandez quelque chose pour lequel elle n'a pas
d'outil, une minuterie, une conversion d'unités, un lancer de dés, et elle peut demander à sa propre fabrique d'outils
d'en construire un. Le nouvel outil est écrit comme un programme, testé par le vérificateur, et gardé pour la prochaine
fois. Pas de mise à jour, pas de téléchargement, pas d'attente d'une nouvelle version.

**Elle joue, et vous pouvez la regarder penser.** Dans cubby-man, son jeu de labyrinthe, elle apprend la carte à partir
des mouvements qu'on lui refuse, écrit ses propres raccourcis, les fait certifier, pose des pièges, a peur, se repose
quand elle est fatiguée, et raconte ce qu'elle fait dans ses propres mots. C'est le même cerveau que celui avec lequel
vous clavardez.

**Elle refuse ce qu'il faut refuser.** Elle reconnaît les tentatives de la manipuler ou de lui faire enfreindre ses
règles, et tout outil qu'elle pourrait atteindre est éteint par défaut tant que vous ne l'allumez pas.

**À venir :** demandez-lui « quoi de neuf sur les Canadiens ce soir » et elle va le chercher, par un outil que vous avez
explicitement autorisé, avec la source attachée. Le mécanisme est construit et en cours d'entraînement ; l'interrupteur
est encore éteint.

## Des cas d'usage réels

Le fil conducteur : Cubby n'arrive pas avec un ensemble fixe de capacités. Elle arrive avec une façon d'en acquérir. Un
besoin apparaît dans votre environnement, Cubby écrit le programme, le vérificateur certifie le programme sur vos
propres exemples, et à partir de là c'est une compétence qu'elle garde. Avec le bon programme, vous pouvez lui faire
faire ce dont vous avez besoin, et le programme est quelque chose qu'elle apprend à écrire en vivant avec vous. Chaque
cas ci-dessous dit ce qui tourne aujourd'hui et ce qui est l'étape suivante, pour que personne ne prenne un plan pour
une fonctionnalité.

**Pour un étudiant.** Cubby devient un partenaire d'étude qui n'invente jamais une citation. Donnez-lui vos notes de
cours et elle les garde comme des faits avec une source : « quand le traité de Paris a-t-il été signé » revient avec la
date et la page d'où elle vient, et « que veut dire cette expression » obtient une réponse dans votre français, québécois
inclus. Demandez au-delà de ce que couvrent les notes et elle dit qu'elle ne sait pas encore, puis propose d'aller voir,
et ce qu'elle trouve rejoint les notes avec sa source. Elle vous interroge à partir des faits qu'elle détient, pas de
choses inventées, et elle montre chaque étape d'une réponse en plusieurs étapes pour que vous voyiez où votre propre
raisonnement a dérapé. *Aujourd'hui : les faits avec provenance, le « je ne sais pas », les étapes, les deux langues.
Ensuite : la boucle chercher-et-apprendre.*

**Pour des alertes médicales.** Une règle comme « signaler l'échantillon si la lecture dépasse le seuil » est
exactement le genre de petit programme de décision que Cubby écrit déjà et fait certifier, et chacune de ces décisions
est signée et gardée dans un registre avec la lecture, le seuil, le verdict et le vérificateur qui a tranché. Une alerte
ne vient donc jamais d'une devinette, une valeur jamais mesurée ne peut pas être prononcée, et une contradiction entre
deux lectures arrête la chaîne au lieu d'en choisir une. Quand quelqu'un demande une posologie pour laquelle elle n'a
aucun fait, la réponse est la même courte phrase, pas un nombre plausible. *Aujourd'hui : les programmes de décision
certifiés, le registre signé, la barrière aux contradictions, le « je ne sais pas ». Ensuite : le module capteur. Et pour
être clair, rien de tout cela n'est un dispositif médical tant qu'il n'a pas été validé comme tel.*

**Pour des environnements sécurisés.** Cubby tourne sur la machine dans la pièce, sans infonuagique, sans compte et
sans trafic sortant. Ce qu'elle stocke des conversations est chiffré avec une clé qui ne quitte jamais cette machine.
Chaque outil qu'elle pourrait atteindre est éteint par défaut et s'allume une capacité à la fois, de sorte qu'un
programme qu'elle écrit peut composer les outils qu'elle a mais ne peut jamais s'en accorder un nouveau. Chaque décision
qu'elle certifie est signée et consignée, ce qui donne à un auditeur une trace plutôt qu'une transcription, et elle
reconnaît les tentatives de la manipuler pour enfreindre ses règles avant qu'elles n'atteignent quoi que ce soit. Le
raisonnement se fait dans un vérificateur qui exécute des programmes, pas dans un modèle qu'on peut convaincre.
*Aujourd'hui : local, chiffré au repos, le registre signé, la détection de manipulation, les outils éteints par défaut.
Ensuite : les interrupteurs de capacité par outil.*

**Pour les jeux vidéo.** cubby-man est la démonstration que vous pouvez regarder : un personnage qui part sans carte,
apprend le labyrinthe à partir des mouvements qu'on lui refuse, invente ses propres raccourcis et les fait certifier
avant de leur faire confiance, pose des pièges, a peur quand un fantôme est proche, se repose quand il est fatigué, et
dit ce qu'il pense dans ses propres mots. Rien de ce qu'il apprend n'est scénarisé, et rien de ce qu'il apprend n'est
jeté : un mouvement qui ne rapporte plus est retiré, pas effacé, de sorte que le personnage que vous rencontrez au
niveau sept est façonné par les niveaux un à six. Le même cerveau, déposé dans un autre jeu, c'est un personnage qui se
souvient de vous entre les sessions, qui a des humeurs venant de son propre état, et qui développe des compétences au
fil d'une partie au lieu de choisir dans un menu d'animations. *Aujourd'hui : tout cela, dans le labyrinthe. Ensuite : le
même moteur derrière un personnage dans un jeu qui n'est pas le labyrinthe.*

**Pour la fabrication, la soudure par exemple.** Une cellule robotisée typique répète une trajectoire fixe et n'apprend
qu'après coup que quelque chose a mal tourné. Cubby raisonne déjà en trois dimensions : le labyrinthe est une pile de
niveaux à six directions, et elle déduit où sont les choses, quelles cellules sont des culs-de-sac et ce que coûte un
chemin, sous forme de programmes vérifiés sur cet espace. Une trajectoire de soudure est le même genre d'objet que l'un de
ses mouvements de labyrinthe, un programme certifié avant de tourner, et un contrôle de défaut est le même genre d'objet
que ses décisions à seuil, une règle qui signale une lecture, signée et gardée. Ajoutez le capteur, et la règle peut
tourner avant que le joint soit fait plutôt qu'après. *Aujourd'hui : le raisonnement 3D et les programmes de mouvement
et de décision certifiés. Ensuite : le cortex de vision et de capteurs, qui existe dans la lignée du laboratoire et n'est
pas encore branché.*

Cinq pièces très différentes, un seul mécanisme : voir le besoin, écrire le programme, le certifier, le garder. C'est ce
que veut dire en pratique « avec le bon programme, vous pouvez lui faire faire ce dont vous avez besoin », et pourquoi le
bon programme est quelque chose que Cubby apprend à écrire pour vous en évoluant dans votre environnement.

## Ce qu'elle n'est pas, pour l'instant

Aujourd'hui, Cubby n'est pas un oracle omniscient, pas un moteur de recherche, et pas un agent de programmation qui
réécrit vos fichiers. C'est un petit modèle qui pense grand, construit pour avoir raison ou pour le dire.

## Où ça s'en va

L'objectif n'est pas de rattraper les gros modèles sur ce qu'ils font. C'est de les battre sur ce qu'ils ne peuvent pas
faire correctement, et ne pourront jamais, à cause de la façon dont ils sont construits. Un modèle qui prédit le prochain
mot ne peut pas se vérifier, ne peut pas dire « je ne sais pas » de façon fiable, ne peut pas apprendre un fait nouveau ce
soir sans être réentraîné, ne peut pas vous montrer une trace que vous pourriez auditer, et oublie ses anciennes
compétences quand on lui en enseigne de nouvelles. Ce ne sont pas des bogues à corriger dans la prochaine version. C'est
la forme même de la chose.

Cubby est construite dans l'autre sens, à partir du vérificateur : une pensée est un programme qui tourne et qui est
vérifié, un fait est gardé avec sa source et un certificat signé, une nouvelle compétence est un programme écrit et
certifié plutôt qu'un modèle réentraîné, et les parties qui parlent, raisonnent et se spécialisent sont séparées pour
que l'apprentissage de l'une n'efface jamais l'autre. Chaque pièce ci-dessus est petite aujourd'hui. Le pari, c'est
qu'une assistante qui n'est jamais sûre d'elle à tort, qui apprend pendant que vous l'utilisez, et qui peut montrer son
travail, finit par être plus utile qu'une qui en sait plus et à qui on ne peut rien confier.
