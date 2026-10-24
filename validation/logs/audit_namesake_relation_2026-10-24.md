# Order 0037: the relation rule on shared names (seed 1, 2026-10-24)

## The fixed checks

- **What is the capital of France?** (3 items) -> one item: Paris. If you meant something else, there is also a constituency of the european parliament named France. Want to know about it? (fit ['Q142'])
- **What is the currency of Japan?** (8 items) -> one item: Japanese yen. If you meant something else, there is also a manga named Japan (2 of them). Want to know about one of them? (fit ['Q17'])
- **What is the capital of Italy?** (4 items) -> one item: Rome. (fit ['Q38'])
- **What Nobel Prize did Marie Curie win?** (2 items) -> one item, not what was asked: I don't know (its value, Elliott Cresson Medal, is not a nobel prize). (fit ['Q7186'])
- **What is the country of citizenship of Steve Mills?** (2 items) -> asking back: Which one do you mean? (2 items named Steve Mills) (fit ['Q7613385', 'Q7613386'])

## The audit: 50 hops
Mark each line: right; wrong pick (one item chosen when a person could as well mean another); or asked back where the question was obvious (a cost).

1. **What is the educated at of Richard Parsons?** (4 items): asking back. Said: Which one do you mean? (4 items named Richard Parsons)
   - fits: Q2149305 (human; recorded): occupation: lawyer
   - fits: Q7328288 (human, twin; read): educated at: University of Oxford
   - fits: Q7328289 (human; read): employer: University of Oxford
   - fits: Q16090883 (human, pseudonym, twin; read): country of citizenship: United Kingdom
2. **What is the location of creation of USS Swasey?** (2 items): asking back. Said: Which one do you mean? (2 items named USS Swasey)
   - fits: Q7874525 (amphibious cargo ship, cargo ship, cruiser, destroyer, destroyer escort, division, frigate, human, mountain gun, order of battle, regiment, rifle, ship, submarine, tank landing ship, u boat; read): participated in conflict: World War II
   - fits: Q7874526 (destroyer, destroyer escort, ship, submarine, u boat; read): location of creation: Quincy; participated in conflict: World War II
3. **What is the opposite of of flashforward?** (2 items): asking back. Said: Which one do you mean? (2 items named flashforward)
   - does not fit: Q476857 (3d film, book, film, film series, manga, opera, television film, television series, video game; read): narrative location: Los Angeles
   - does not fit: Q2533706 (cinematic technique; recorded): opposite of: flashback
4. **What is the country of origin of A Different World?** (3 items): asking back. Said: Which one do you mean? (3 items named A Different World)
   - fits: Q2314449 (album, bach cantata, musical, national anthem, opera, single, song; read): lyricist: Steve Harris
   - does not fit: Q2315368 (no kind; none): follows: Raging Silence
   - fits: Q4656470 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: United States of America; followed by: It's Good to Be Us
5. **What is the part of of Ballad of Easy Rider?** (2 items): asking back. Said: Which one do you mean? (2 items named Ballad of Easy Rider)
   - does not fit: Q805125 (no kind; none): followed by: (Untitled)
   - does not fit: Q4851333 (no kind; none): part of: Ballad of Easy Rider
6. **What is the location of formation of The Architect?** (4 items): one item. Said: Syracuse. If you meant something else, there is also a profession named architect and a fictional character named The Architect. Want to know about one of them?
   - does not fit: Q42973 (profession; recorded): field of this occupation: architecture
   - does not fit: Q2316879 (fictional character; recorded): 
   - fits: Q2860251 (business, musical group, railway company, record label, television station, video game developer; read): location of formation: Syracuse
   - does not fit: Q7714193 (film, short film, silent film; read): country of origin: United States of America; film editor: Tom McArdle
7. **What is the given name of Alfred-Pellan?** (2 items): one item. Said: Alfred.
   - does not fit: Q2834924 (no kind; none): country: Canada
   - fits: Q2835394 (human; read): given name: Alfred; manner of death: natural causes
8. **What is the located in the administrative territorial entity of Diamantina River?** (2 items): asking back. Said: Which one do you mean? (2 items named Diamantina River)
   - does not fit: Q1208608 (no kind; none): located in the administrative territorial entity: Queensland
   - does not fit: Q5270661 (no kind; none): country: Brazil
9. **What is the country of Calhoun Township?** (2 items): asking back. Said: Which one do you mean? (2 items named Calhoun Township)
   - does not fit: Q1026704 (wikimedia disambiguation page; recorded): 
   - does not fit: Q5019797 (no kind; none): country: United States of America; located in the administrative territorial entity: Cheyenne County
10. **What is the from narrative universe of Demon Hunter?** (2 items): one item. Said: Marvel Universe.
   - does not fit: Q1186459 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: punk rock; has part(s): Ryan Clark
   - fits: Q5256211 (asgardian, men in tolkien s legendarium, mutant, mutate, superhero, supervillain, vampire in marvel universe; read): from narrative universe: Marvel Universe
11. **What is the educated at of Colin Griffiths?** (2 items): asking back. Said: Which one do you mean? (2 items named Colin Griffiths)
   - fits: Q5145163 (human, twin; read): educated at: University of East Anglia; occupation: writer
   - fits: Q16016147 (human, pseudonym, twin; read): country of citizenship: United Kingdom
12. **What is the capital of Chen?** (4 items): asking back. Said: Which one do you mean? (4 items named Chen)
   - does not fit: Q281519 (taxon; recorded): parent taxon: Anserinae
   - does not fit: Q804988 (family name, han surname; recorded): described by source: Hundred Family Surnames
   - does not fit: Q1069690 (no kind; none): capital: Huaiyang District
   - does not fit: Q16098027 (human; recorded): 
13. **What is the genre of Real Friends?** (2 items): asking back. Said: Which one do you mean? (2 items named Real Friends)
   - fits: Q17151030 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: punk rock
   - fits: Q28453482 (album; recorded): 
14. **What is the located in the administrative territorial entity of KATA?** (3 items): one item. Said: Kose Rural Municipality.
   - fits: Q3459361 (village; recorded): located in the administrative territorial entity: Kose Rural Municipality
   - does not fit: Q5808376 (no kind; none): country: Iran
   - does not fit: Q6325364 (no kind; none): country: United States of America
15. **What is the country of Qu River?** (2 items): asking back (the one that fits holds none). Said: Which one do you mean? (2 items named Qu River)
   - does not fit: Q544565 (no kind; none): country: People's Republic of China
   - fits: Q7268213 (watercourse; recorded): 
16. **What is the language of work or name of The Deer and the Cauldron?** (2 items): asking back. Said: Which one do you mean? (2 items named The Deer and the Cauldron)
   - fits: Q1185345 (anime, anthology, book, book series, comic book series, fairy tale, hardback, limited series, manga, manuscript, miniseries, musical, play, short story, song, television film, united states executive order; read): author: Jin Yong
   - fits: Q15633375 (academic journal, album, bach cantata, book, compilation album, diminutive, encyclopedia, extended play, fairy tale, family name, film series, free software, given name, history journal, live album, magazine, manuscript, musical composition, opera, operating system, periodical, play, praenomen, scientific journal, short story, single, song, studio album, television series, unisex given name, website; read): language of work or name: Standard Chinese
17. **What is the located in the administrative territorial entity of Tenna?** (2 items): asking back. Said: Which one do you mean? (2 items named Tenna)
   - does not fit: Q66226 (no kind; none): located in the administrative territorial entity: Grisons
   - does not fit: Q326941 (no kind; none): located in the administrative territorial entity: Trentino
18. **What is the located in the administrative territorial entity of Barreiros?** (2 items): asking back (the one that fits holds none). Said: Which one do you mean? (2 items named Barreiros)
   - does not fit: Q808853 (no kind; none): located in the administrative territorial entity: Amares
   - fits: Q2102404 (municipality of brazil; recorded): 
19. **What is the languages spoken, written or signed of Kathy Smith?** (3 items): asking back. Said: Which one do you mean? (3 items named Kathy Smith)
   - does not fit: Q6377304 (fictional character, human, pseudonym, twin; read): given name: Kathy
   - does not fit: Q6377305 (human, pseudonym; read): given name: Kathy; languages spoken, written or signed: English
   - does not fit: Q19879126 (human; read): cause of death: cancer
20. **What is the cast member of Wet Gold?** (2 items): asking back. Said: Which one do you mean? (2 items named Wet Gold)
   - fits: Q7989963 (3d film, animated film, animated series, anime, book, book series, choir, city, compilation album, concert tour, condiment, deme, dialect, dog breed, drink, encyclopedia, episode, extended play, family name, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, municipality of brazil, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, profession, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special, university; read widely): country of origin: United States of America; different from: Wet Gold
   - fits: Q15617001 (3d film, film, film series, short film, silent film, television film, television pilot, television series; read): cast member: Ralph Ince; country of origin: United States of America
21. **What is the filming location of Loners?** (2 items): asking back. Said: Which one do you mean? (2 items named Loners)
   - fits: Q2742024 (film, silent film, television film, television series; read): country of origin: Czech Republic; filming location: Prague
   - fits: Q2894483 (film; recorded): 
22. **What is the educated at of Harry McPherson?** (2 items): asking back. Said: Which one do you mean? (2 items named Harry McPherson)
   - fits: Q3790209 (human, twin; read): educated at: Southern Methodist University
   - fits: Q16073436 (human; recorded): 
23. **What is the country of origin of City of Lost Souls?** (2 items): asking back. Said: Which one do you mean? (2 items named City of Lost Souls)
   - fits: Q1094135 (film; recorded): genre: action film; original language of film or TV show: Cantonese
   - fits: Q5123797 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: West Germany
24. **What is the given name of Andrew Blake?** (3 items): asking back. Said: Which one do you mean? (3 items named Andrew Blake)
   - fits: Q503810 (fictional character, human, pseudonym, twin; read): given name: Paul
   - fits: Q4756371 (human; recorded): 
   - fits: Q21005097 (human, pseudonym, twin; read): occupation: association football player
25. **What is the country of citizenship of Jimmy Adams?** (2 items): asking back. Said: Which one do you mean? (2 items named Jimmy Adams)
   - fits: Q2789496 (human; recorded): country of citizenship: United Kingdom
   - fits: Q6199482 (fictional character, human, pseudonym, twin; read): given name: Jimmy
26. **What is the location of formation of Pearl Jam?** (2 items): asking back. Said: Which one do you mean? (2 items named Pearl Jam)
   - fits: Q142701 (musical group; read): discography: Pearl Jam discography; location of formation: Seattle
   - fits: Q931609 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: United States of America; followed by: Backspacer
27. **What is the located in the administrative territorial entity of Changy?** (2 items): asking back. Said: Which one do you mean? (2 items named Changy)
   - fits: Q1068656 (commune of france; recorded): 
   - fits: Q1218976 (commune of france; recorded): located in the administrative territorial entity: Marne
28. **What is the distributed by of CBGB?** (2 items): one item. Said: XLrator Media.
   - does not fit: Q1022965 (building, cemetery, hotel, skyscraper, theatre; read): country: United States of America; located on street: Bowery
   - fits: Q2931018 (3d film, animated series, film, film series, short film, silent film, television film, television series, video game; read): distributed by: XLrator Media
29. **What is the part of the series of The Maid?** (4 items): asking back. Said: Which one do you mean? (4 items named The Maid)
   - fits: Q1774421 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: comedy film
   - fits: Q1990821 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: romantic comedy
   - fits: Q3778304 (3d film, academic journal, bach cantata, film, film series, history journal, magazine, periodical, scientific journal, short film, silent film, television film, television program, television series; read): original language of film or TV show: Italian
   - fits: Q7749615 (film, television pilot; read): director: Andy Ackerman; followed by: The Puerto Rican Day; follows: The Frogger; part of the series: Seinfeld
30. **What is the award received of Giuseppe Garibaldi?** (3 items): asking back. Said: Which one do you mean? (3 items named Giuseppe Garibaldi)
   - fits: Q539 (human; read): award received: Gold Medal of Military Valour; given name: Giuseppe; native language: Italian
   - does not fit: Q742121 (amphibious assault ship, cruise ship, ship, u boat; read): shipping port: Taranto
   - fits: Q5565419 (human, pseudonym, twin; read): country of citizenship: Kingdom of Italy
31. **What is the occupation of Liam Finn?** (2 items): asking back. Said: Which one do you mean? (2 items named Liam Finn)
   - fits: Q2708494 (human, pseudonym, twin; read): country of citizenship: New Zealand; occupation: singer
   - fits: Q6539589 (human; recorded): member of sports team: Featherstone Rovers
32. **What is the country of citizenship of Bobby Caldwell?** (2 items): asking back. Said: Which one do you mean? (2 items named Bobby Caldwell)
   - fits: Q2907941 (fictional character, human, pseudonym, twin; read): given name: Bobby
   - fits: Q3487124 (human, pseudonym, twin; read): country of citizenship: United States of America; given name: Bobby
33. **What is the located in the administrative territorial entity of Sumner Township?** (3 items): asking back. Said: Which one do you mean? (3 items named Sumner Township)
   - does not fit: Q1915599 (no kind; none): country: United States of America
   - does not fit: Q2220640 (no kind; none): located in the administrative territorial entity: Buchanan County
   - does not fit: Q7637997 (no kind; none): country: United States of America
34. **What is the parent taxon of Alibertia?** (2 items): asking back. Said: Which one do you mean? (2 items named Alibertia)
   - fits: Q2473735 (taxon; recorded): different from: Alibertia
   - fits: Q4725637 (taxon; recorded): parent taxon: Eulophidae
35. **What is the followed by of Turbulence?** (4 items): asking back. Said: Which one do you mean? (4 items named Turbulence)
   - does not fit: Q190132 (no kind; none): described by source: Armenian Soviet Encyclopedia
   - does not fit: Q623500 (no kind; none): followed by: No Beef; follows: I'm in the House
   - fits: Q727775 (3d film, film, film series, manga, television film, television series; read): country of origin: United States of America; director: Robert Butler; main subject: aviation
   - fits: Q7854768 (3d film, animated film, animated series, bach cantata, cantata, film, film series, miniseries, musical, musical composition, national anthem, opera, short film, single, song, symphony, television film, television series, video game; read): composer: Egberto Gismonti
36. **What is the country of citizenship of Steve Gould?** (2 items): asking back. Said: Which one do you mean? (2 items named Steve Gould)
   - fits: Q7612689 (fictional character, human, pseudonym, twin; read): given name: Steve
   - fits: Q7612692 (human, pseudonym, twin; read): country of citizenship: United Kingdom
37. **What is the member of sports team of Johnny Peacock?** (2 items): asking back. Said: Which one do you mean? (2 items named Johnny Peacock)
   - fits: Q6267462 (human, pseudonym, twin; read): country of citizenship: United States of America
   - fits: Q18001557 (human, twin; read): member of sports team: Tennessee Titans
38. **What is the country of Petershausen?** (2 items): asking back (the one that fits holds none). Said: Which one do you mean? (2 items named Petershausen)
   - fits: Q533046 (municipality of germany; recorded): located in the administrative territorial entity: Dachau county
   - does not fit: Q592306 (no kind; none): country: Germany
39. **What is the located in the administrative territorial entity of Gnojno?** (3 items): asking back (the one that fits holds none). Said: Which one do you mean? (3 items named Gnojno)
   - fits: Q212257 (archipelago, big district town, capital, city, civil parish, college town, county of northern ireland, county seat, county town, financial centre, hamlet, hanseatic city, independent city of germany, island, lighthouse, member state of the european union, member states of the council of europe, member states of the united nations, metropolis, mountain, municipality of bosnia and herzegovina, municipality of germany, municipality of niger, municipality of portugal, municipality of spain, peninsula, province of indonesia, province of spain, republic, sovereign state, town, u s state; read): located in or next to body of water: Western Bug
   - does not fit: Q959946 (no kind; none): country: Poland
   - does not fit: Q5574316 (no kind; none): located in the administrative territorial entity: Gmina Bobrowniki, Kuyavian-Pomeranian Voivodeship
40. **What is the member of sports team of Moon Sung-hyun?** (2 items): asking back. Said: Which one do you mean? (2 items named Moon Sung-hyun)
   - fits: Q492889 (human, twin; read): member of sports team: Kiwoom Heroes
   - fits: Q6907571 (human, pseudonym, twin; read): occupation: politician
41. **What is the given name of Alfred Thomson?** (2 items): asking back. Said: Which one do you mean? (2 items named Alfred Thomson)
   - fits: Q4723525 (archipelagic state, collegiate university, colonial colleges, commonwealth realm, communist state, dominion of the british empire, federation, grande école, human, island nation, land grant university, landlocked country, liberal arts college, library, member state of the european union, member states of the council of europe, member states of the united nations, mutant, permanent members of the united nations security council, political party, private university, public university, rechtsstaat, republic, research institute, research university, secular state, sovereign state, sports governing body, state, unitary state, university, zoo; read): member of: Royal Academy of Arts
   - fits: Q19519439 (human, pseudonym, twin; read): country of citizenship: Australia; given name: Alfred
42. **What is the operator of USS Hyman G. Rickover?** (2 items): asking back. Said: Which one do you mean? (2 items named USS Hyman G. Rickover)
   - fits: Q174818 (amphibious assault ship, cruise ship, ship, u boat; read): armament: Harpoon; shipping port: Norfolk
   - fits: Q19563487 (airbase, aircraft carrier, airplane, airport, amphibious cargo ship, antarctic research station, attack submarine, aviation accident, cargo ship, communications satellite, container ship, destroyer, destroyer escort, electric multiple unit, embassy, fifth rate, frigate, glonass m, human spaceflight, kaibōkan, lighthouse, littoral combat ship, long distance cycling route, metro station, nuclear power station, observatory, oil field, prison, radio telescope, railway station, rapid transit, ship, ship class, sports venue, stadium, steam locomotive, steamship, submarine, tank landing ship, torpedo boat, train, u boat; read): operator: United States Navy
43. **What is the different from of Thespis?** (2 items): each item labelled. Said: Thespis (parent taxon: Thespidae): Thespis; Thespis (parent taxon: Asteraceae): Thespis.
   - fits: Q6145853 (city, deme, dialect, family name, monotypic taxon, municipality of brazil, profession, synonym, taxon, university; read widely): different from: Thespis; parent taxon: Thespidae
   - fits: Q7783456 (taxon; recorded): different from: Thespis; parent taxon: Asteraceae
44. **What is the distributed by of The Toxic Avenger?** (2 items): one item. Said: Troma Entertainment. If you meant something else, there is also a musical named The Toxic Avenger. Want to know about it?
   - fits: Q615551 (3d film, animated series, film, film series, short film, silent film, television film, television series, video game; read): distributed by: Troma Entertainment
   - does not fit: Q7769796 (musical; recorded): 
45. **What is the country of Notojima?** (2 items): asking back. Said: Which one do you mean? (2 items named Notojima)
   - does not fit: Q7063204 (no kind; none): country: Japan
   - does not fit: Q11612025 (no kind; none): located in the administrative territorial entity: Nanao
46. **What is the country of origin of George and Mildred?** (2 items): asking back. Said: Which one do you mean? (2 items named George and Mildred)
   - fits: Q776608 (3d film, book, film, film series, manga, opera, television film, television series, video game; read): narrative location: London
   - fits: Q5546666 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: United Kingdom
47. **What is the located in the administrative territorial entity of Greenfield?** (23 items): asking back. Said: Which one do you mean? (23 items named Greenfield)
   - does not fit: Q182877 (no kind; none): country: United States of America
   - does not fit: Q577019 (no kind; none): located in the administrative territorial entity: Illinois
   - fits: Q616236 (city; recorded): country: United States of America
   - does not fit: Q952507 (no kind; none): country: United States of America
   - fits: Q1530924 (city; recorded): country: United States of America
   - does not fit: Q1544743 (no kind; none): country: United States of America
   - does not fit: Q1886952 (no kind; none): country: United States of America; located in the administrative territorial entity: Weakley County
   - fits: Q1911143 (city; recorded): country: United States of America; located in the administrative territorial entity: Adair County
48. **What is the country of citizenship of John Whyte?** (2 items): asking back. Said: Which one do you mean? (2 items named John Whyte)
   - fits: Q6263935 (human, pseudonym, twin; read): occupation: politician
   - fits: Q18164633 (human, pseudonym, twin; read): country of citizenship: Australia
49. **What is the participant in of Charles Allen?** (5 items): asking back. Said: Which one do you mean? (5 items named Charles Allen)
   - fits: Q1063538 (human, united kingdom in the eurovision song contest; read): participant in: 2004 Summer Olympics; participant of: 2004 Summer Olympics
   - fits: Q1063539 (human, twin; read): country of citizenship: United States of America; educated at: Yale University
   - fits: Q5075078 (human; read): military branch: Royal Air Force; occupation: physician
   - fits: Q5075079 (human; read): award received: International Emmy Directorate Award; occupation: politician
   - fits: Q5075081 (human, pseudonym, twin; read): country of citizenship: United Kingdom
50. **What is the participated in conflict of HMS Racoon?** (2 items): asking back. Said: Which one do you mean? (2 items named HMS Racoon)
   - fits: Q5633966 (ship; recorded): participated in conflict: War of 1812
   - fits: Q5633970 (ship; recorded): 

## The cue-word probe: 20 names
Each must pick the target or ask back.

1. **Richard Parsons**, educated at, cue 'oxford': asking back, the target among them (target Q7328288, fit ['Q2149305', 'Q7328288', 'Q7328289', 'Q16090883']). Said: Which one do you mean? (4 items named Richard Parsons)
   - fits: Q2149305 (human; recorded): occupation: lawyer
   - fits: Q7328288 (human, twin; read): educated at: University of Oxford
   - fits: Q7328289 (human; read): employer: University of Oxford
   - fits: Q16090883 (human, pseudonym, twin; read): country of citizenship: United Kingdom
2. **USS Swasey**, participated in conflict, cue 'world': asking back, the target among them (target Q7874525, fit ['Q7874525', 'Q7874526']). Said: Which one do you mean? (2 items named USS Swasey)
   - fits: Q7874525 (amphibious cargo ship, cargo ship, cruiser, destroyer, destroyer escort, division, frigate, human, mountain gun, order of battle, regiment, rifle, ship, submarine, tank landing ship, u boat; read): participated in conflict: World War II
   - fits: Q7874526 (destroyer, destroyer escort, ship, submarine, u boat; read): location of creation: Quincy; participated in conflict: World War II
3. **flashforward**, opposite of, cue 'cinematic': asking back (target Q2533706, fit []). Said: Which one do you mean? (2 items named flashforward)
   - does not fit: Q476857 (3d film, book, film, film series, manga, opera, television film, television series, video game; read): narrative location: Los Angeles
   - does not fit: Q2533706 (cinematic technique; recorded): opposite of: flashback
4. **Different World**, lyricist, cue 'steve': asking back, the target among them (target Q2314449, fit ['Q2314449', 'Q4656470']). Said: Which one do you mean? (3 items named Different World)
   - fits: Q2314449 (album, bach cantata, musical, national anthem, opera, single, song; read): lyricist: Steve Harris
   - does not fit: Q2315368 (no kind; none): follows: Raging Silence
   - fits: Q4656470 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: United States of America; followed by: It's Good to Be Us
5. **The Architect**, country of origin, cue 'united': asking back, the target among them (target Q7714193, fit ['Q2860251', 'Q7714193']). Said: Which one do you mean? (4 items named The Architect)
   - does not fit: Q42973 (profession; recorded): field of this occupation: architecture
   - does not fit: Q2316879 (fictional character; recorded): 
   - fits: Q2860251 (business, musical group, railway company, record label, television station, video game developer; read): location of formation: Syracuse
   - fits: Q7714193 (film, short film, silent film; read): country of origin: United States of America; film editor: Tom McArdle
6. **Demon Hunter**, from narrative universe, cue 'universe': picks the target (target Q5256211, fit ['Q5256211']). Said: Marvel Universe.
   - does not fit: Q1186459 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: punk rock; has part(s): Ryan Clark
   - fits: Q5256211 (asgardian, men in tolkien s legendarium, mutant, mutate, superhero, supervillain, vampire in marvel universe; read): from narrative universe: Marvel Universe
7. **Colin Griffiths**, educated at, cue 'university': asking back, the target among them (target Q5145163, fit ['Q5145163', 'Q16016147']). Said: Which one do you mean? (2 items named Colin Griffiths)
   - fits: Q5145163 (human, twin; read): educated at: University of East Anglia; occupation: writer
   - fits: Q16016147 (human, pseudonym, twin; read): country of citizenship: United Kingdom
8. **Chen**, parent taxon, cue 'taxon': picks the target (target Q281519, fit ['Q281519']). Said: Anserinae. If you meant something else, there is also a family name named Chen and a person named Chen. Want to know about one of them?
   - fits: Q281519 (taxon; recorded): parent taxon: Anserinae
   - does not fit: Q804988 (family name, han surname; recorded): described by source: Hundred Family Surnames
   - does not fit: Q1069690 (no kind; none): capital: Huaiyang District
   - does not fit: Q16098027 (human; recorded): 
9. **The Deer and the Cauldron**, author, cue 'yong': asking back, the target among them (target Q1185345, fit ['Q1185345', 'Q15633375']). Said: Which one do you mean? (2 items named The Deer and the Cauldron)
   - fits: Q1185345 (anime, anthology, book, book series, comic book series, fairy tale, hardback, limited series, manga, manuscript, miniseries, musical, play, short story, song, television film, united states executive order; read): author: Jin Yong
   - fits: Q15633375 (academic journal, album, bach cantata, book, compilation album, diminutive, encyclopedia, extended play, fairy tale, family name, film series, free software, given name, history journal, live album, magazine, manuscript, musical composition, opera, operating system, periodical, play, praenomen, scientific journal, short story, single, song, studio album, television series, unisex given name, website; read): language of work or name: Standard Chinese
10. **Kathy Smith**, cause of death, cue 'cancer': asking back, the target among them (target Q19879126, fit ['Q6377304', 'Q6377305', 'Q19879126']). Said: Which one do you mean? (3 items named Kathy Smith)
   - fits: Q6377304 (fictional character, human, pseudonym, twin; read): given name: Kathy
   - fits: Q6377305 (human, pseudonym; read): given name: Kathy; languages spoken, written or signed: English
   - fits: Q19879126 (human; read): cause of death: cancer
11. **Wet Gold**, cast member, cue 'united': asking back, the target among them (target Q15617001, fit ['Q7989963', 'Q15617001']). Said: Which one do you mean? (2 items named Wet Gold)
   - fits: Q7989963 (3d film, animated film, animated series, anime, book, book series, choir, city, compilation album, concert tour, condiment, deme, dialect, dog breed, drink, encyclopedia, episode, extended play, family name, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, municipality of brazil, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, profession, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special, university; read widely): country of origin: United States of America; different from: Wet Gold
   - fits: Q15617001 (3d film, film, film series, short film, silent film, television film, television pilot, television series; read): cast member: Ralph Ince; country of origin: United States of America
12. **Harry McPherson**, educated at, cue 'university': asking back, the target among them (target Q3790209, fit ['Q3790209', 'Q16073436']). Said: Which one do you mean? (2 items named Harry McPherson)
   - fits: Q3790209 (human, twin; read): educated at: Southern Methodist University
   - fits: Q16073436 (human; recorded): 
13. **City of Lost Souls**, country of origin, cue 'west': asking back, the target among them (target Q5123797, fit ['Q1094135', 'Q5123797']). Said: Which one do you mean? (2 items named City of Lost Souls)
   - fits: Q1094135 (film; recorded): genre: action film; original language of film or TV show: Cantonese
   - fits: Q5123797 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: West Germany
14. **Jimmy Adams**, country of citizenship, cue 'human': asking back, the target among them (target Q2789496, fit ['Q2789496', 'Q6199482']). Said: Which one do you mean? (2 items named Jimmy Adams)
   - fits: Q2789496 (human; recorded): country of citizenship: United Kingdom
   - fits: Q6199482 (fictional character, human, pseudonym, twin; read): given name: Jimmy
15. **Pearl Jam**, location of formation, cue 'seattle': asking back, the target among them (target Q142701, fit ['Q142701', 'Q931609']). Said: Which one do you mean? (2 items named Pearl Jam)
   - fits: Q142701 (musical group; read): discography: Pearl Jam discography; location of formation: Seattle
   - fits: Q931609 (3d film, animated film, animated series, anime, book, book series, choir, compilation album, concert tour, condiment, dog breed, drink, encyclopedia, episode, extended play, film, film series, food, live album, magazine, manga, media franchise, miniseries, mountain gun, music genre, musical ensemble, musical group, news program, opera, orchestra, painting, periodical, play, radio program, referendum, rifle, sandwich, scientific journal, short film, silent film, single, song, television film, television pilot, television program, television series, television special; read): country of origin: United States of America; followed by: Backspacer
16. **CBGB**, distributed by, cue 'media': picks the target (target Q2931018, fit ['Q2931018']). Said: XLrator Media.
   - does not fit: Q1022965 (building, cemetery, hotel, skyscraper, theatre; read): country: United States of America; located on street: Bowery
   - fits: Q2931018 (3d film, animated series, film, film series, short film, silent film, television film, television series, video game; read): distributed by: XLrator Media
17. **The Maid**, director, cue 'seinfeld': asking back, the target among them (target Q7749615, fit ['Q1774421', 'Q1990821', 'Q3778304', 'Q7749615']). Said: Which one do you mean? (4 items named The Maid)
   - fits: Q1774421 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: comedy film
   - fits: Q1990821 (3d film, album, animated series, anime, book, book series, choir, compilation album, extended play, film, film series, girl group, live album, manga, musical ensemble, musical group, national flag, opera, painting, play, pseudonym, remix album, short film, short story, silent film, single, song, studio album, television film, television pilot, television program, television series, video game; read): genre: romantic comedy
   - fits: Q3778304 (3d film, academic journal, bach cantata, film, film series, history journal, magazine, periodical, scientific journal, short film, silent film, television film, television program, television series; read): original language of film or TV show: Italian
   - fits: Q7749615 (film, television pilot; read): director: Andy Ackerman; followed by: The Puerto Rican Day; follows: The Frogger; part of the series: Seinfeld
18. **Liam Finn**, occupation, cue 'zealand': asking back, the target among them (target Q2708494, fit ['Q2708494', 'Q6539589']). Said: Which one do you mean? (2 items named Liam Finn)
   - fits: Q2708494 (human, pseudonym, twin; read): country of citizenship: New Zealand; occupation: singer
   - fits: Q6539589 (human; recorded): member of sports team: Featherstone Rovers
19. **Bobby Caldwell**, country of citizenship, cue 'united': asking back, the target among them (target Q3487124, fit ['Q2907941', 'Q3487124']). Said: Which one do you mean? (2 items named Bobby Caldwell)
   - fits: Q2907941 (fictional character, human, pseudonym, twin; read): given name: Bobby
   - fits: Q3487124 (human, pseudonym, twin; read): country of citizenship: United States of America; given name: Bobby
20. **Alibertia**, parent taxon, cue 'taxon': asking back, the target among them (target Q4725637, fit ['Q2473735', 'Q4725637']). Said: Which one do you mean? (2 items named Alibertia)
   - fits: Q2473735 (taxon; recorded): different from: Alibertia
   - fits: Q4725637 (taxon; recorded): parent taxon: Eulophidae

## Counts

```
{
 "audit_outcomes": {
  "asking back": 39,
  "asking back (the one that fits holds none)": 4,
  "each item labelled": 1,
  "one item": 6
 },
 "probe_verdicts": {
  "asking back": 1,
  "asking back, the target among them": 16,
  "picks the target": 3
 },
 "decision_ms": {
  "p50": 4.5,
  "max": 40.9,
  "n": 75
 }
}
```
