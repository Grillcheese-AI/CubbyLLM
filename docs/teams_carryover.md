# Carryover: does a late report predict a thin next day?

6758 `[SELF]` messages across **162 active days**. A day counts as a report night if it contains a message at 20:00 local or later of 15+ words.

**6 report nights**, **153 ordinary evenings**, each paired with its next active day.

This is the only test here that crosses a night, and so the only one that bears on the premise the whole architecture rests on: state integrates and decays rather than resetting. A per-turn model reads tomorrow's terseness as calm because it has nowhere to keep last night.

## The morning after

| metric | after a report night | after an ordinary evening | p |
|---|---:|---:|---:|
| messages that day | 61 | 41 | 0.358 |
| median gap (s) | 26 | 9015 | 0.268 |
| first message (local hour) | 10.3 | 10.3 | 1.000 |
| median words | 6.8 | 8.1 | 0.480 |
| share 1-5 words | 40.0% | 38.3% | 0.874 |

## The report nights themselves


- **14** 23:00, 28 words: _ca lair detre juste ca... si ta acces au champ dans ladmin tu peux lajouter la et selon ce que jai compris ca _
  next active day 15: 32 messages, started 09:00, median gap 23s

- **79** 21:00, 16 words: _boon cetait mon dernier update a l'api ca lavais faite planter le dashboard va marcher asteur_
  next active day 80: 31 messages, started 10:00, median gap 35s

- **104** 23:00, 33 words: _la size de la font va changer comment on montre les chose entre autre... oui on peut les changer mais si on a _
  next active day 105: 48 messages, started 10:00, median gap 17s

- **126** 22:00, 16 words: _pas de limites pis pas de frais 😉 100% notre code qui roule sur notre serveur_
  next active day 127: 7 messages, started 11:00, median gap 36s

- **145** 20:00, 36 words: _fac ca nous coute 8 cenne par produit le 4.58 ca devais etre une grosse query jai faite pour voir si ca allais_
  next active day 146: 77 messages, started 15:00, median gap 29s

- **147** 21:00, 48 words: _y a fallu je cherche avec lui lautre jour pour trouver ou c que la table etait liee ... dhabitude c dans la de_
  next active day 148: 173 messages, started 07:00, median gap 19s

## Reading this

Volume and cadence are where fatigue showed up within a day - length did not move. So a real carryover effect should look the same: fewer messages and longer gaps, with word count roughly flat. Word count dropping instead would contradict the day-curve result and is more likely noise at this n.
