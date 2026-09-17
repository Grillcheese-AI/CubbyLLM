# Typographic intensity vs the model's intensity

697 labelled rows, 495 non-neutral. No spend - computed from text already on disk.

**70 of 697 messages carry any marking at all (10.0%).**

| marker | messages | rate |
|---|---:|---:|
| any `!` | 9 | 1.3% |
| `!!` or longer | 0 | 0.0% |
| emoji | 59 | 8.5% |
| `??` / `?!` | 0 | 0.0% |
| streeetched word | 4 | 0.6% |
| ALL-CAPS word | 21 | 3.0% |

## 1. Do the model's labels track the marking?

The labeller saw this text, so agreement here is NOT independent evidence - it only says whether the model is reading the most obvious signal available to it.

Spearman rho (labelled intensity vs typographic score) = **+0.288**

- vs `bang` alone: **+0.137**
- vs `emoji` alone: **+0.247**
- vs `caps` alone: **+0.100**

## 2. The decisive test: which scale separates on BURST?

`burst` is the one channel the labeller never saw that separated at all (model intensity: +0.029, p=0.036). Whichever scale separates it better is the better measure of arousal.

**model intensity** - burst>=2 mean 0.367 (n=84) vs burst==0 mean 0.339 (n=325), diff **+0.029**, p = **0.0356**

**typographic score** - burst>=2 mean 0.012 (n=84) vs burst==0 mean 0.059 (n=325), diff **-0.047**, p = **0.0018**

## 3. Spread

The pilot's giveaway was a median of 0.350 in every single group - a model anchoring on the middle. A usable scale has to actually spread.

- **model intensity**: median 0.350, sd 0.111, range 0.15-0.80, distinct values 21
- **typographic score**: median 0.000, sd 0.123, range 0.00-0.70, distinct values 8

## 4. The most marked messages

- `0.70` (joy/0.6) [PEER2]: Super 😊 

Il font un beau couple 😁

Tu as écrit un beau message sur Facebook 👍
- `0.66` (joy/0.7) [PEER2]: Tu été super Nic ! Bravo ! 🙂
- `0.55` (joy/0.35) [PEER2]: Bon matin 🙂 Sabin fait l'inventaire de l'entrepôt aujourd'hui 😉
- `0.55` (trust/0.35) [PEER2]: Bon matin 🙂 Je serai en réunion ce am. je vais faire les cours IA ce pm 😉
- `0.52` (joy/0.45) [PEER]: super! ca avance 🙂
- `0.49` (disgust/0.75) [SELF]: OUUUUUUUCHHHHHHHHHH je viens de voir ca dans la bd... des mot de passe sauver sans etre encrypter .... bonjour
- `0.33` (joy/0.35) [PEER]: on va continuer ici 🙂
- `0.33` (trust/0.3) [PEER]: ouais peut pas etre parfait 😉
- `0.33` (trust/0.45) [SELF]: c pas ben dur de le faire autrement 😉 on le fais deja
- `0.33` (anticipation/0.35) [SELF]: c sur que ca reglerais probablement linterface chiante 😋
- `0.33` (joy/0.25) [SELF]: pcq moi jai juste le vieux de sportpat qui doit pu mnarcher 😋
- `0.33` (joy/0.3) [PEER]: non il doit plus marcher 🤣
- `0.33` (joy/0.45) [SELF]: c pas bete y ont toute le seo qui vont avec 😋
- `0.33` (joy/0.45) [SELF]: nenon juste fini de changer les morceaux 😋
- `0.33` (joy/0.5) [PEER]: ca revient au meme🤣
