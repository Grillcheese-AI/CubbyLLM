# Order 0055 step 3 (a): the relation rule on the Wikidata store's shared names (seed 1, 2026-11-10-class)

## The fixed checks

- **What is the capital of France?** (50 items) -> one item: Paris. If you meant something else, there is also a star trek location named France. Want to know about it? (fit ['Q142'])
- **What is the currency of Japan?** (33 items) -> one item: yen. (fit ['Q17'])
- **What is the capital of Italy?** (36 items) -> one item: Rome. If you meant something else, there is also a star trek location named Italy. Want to know about it? (fit ['Q38'])
- **What Nobel Prize did Marie Curie win?** (18 items) -> one item: Nobel Prize in Chemistry, Nobel Prize in Physics. If you meant something else, there is also a charitable organization named Marie Curie and a comic strip named Marie Curie. Want to know about one of them? (fit ['Q7186'])
- **What is the country of citizenship of Steve Mills?** (10 items) -> asking back: Which one do you mean? (10 items named Steve Mills) (fit ['Q7613383', 'Q7613385', 'Q7613386', 'Q8870613', 'Q23617386', 'Q50330345', 'Q100984733', 'Q106099569', 'Q133610806'])

## The audit: 50 hops
Mark each line: right; wrong pick (one item chosen when a person could as well mean another); or asked back where the question was obvious (a cost).

1. **What is the educated at of Richard Parsons?** (20 items): asking back. Said: Which one do you mean? (20 items named Richard Parsons)
   - fits: Q2149305 (human; recorded): place of birth: Brooklyn; sex or gender: male; country of citizenship: United Kingdom; country of citizenship: United States
   - fits: Q7328287 (human; recorded): sex or gender: male; father: Richard Parsons, 1st Viscount Rosse; mother: Elizabeth Hamilton; spouse: Frances Claxton
   - fits: Q7328288 (human; recorded): sex or gender: male; country of citizenship: United Kingdom; educated at: Ulverston Victoria High School; educated at: University of Oxford
   - fits: Q7328289 (human; recorded): sex or gender: male; educated at: Durham School; educated at: Magdalen College; occupation: Anglican priest
   - fits: Q7328290 (human; recorded): sex or gender: male; country of citizenship: Australia; occupation: explorer; family name: Parsons
   - fits: Q16090883 (human; recorded): sex or gender: male; country of citizenship: United Kingdom; educated at: Brasenose College; educated at: Bembridge School
   - fits: Q18757051 (human; recorded): sex or gender: male; country of citizenship: Kingdom of England; occupation: ecclesiastical judge; occupation: antiquarian
   - fits: Q21466855 (human; recorded): sex or gender: male; occupation: painter; date of birth: 1925; date of death: 2000
2. **What is the location of creation of USS Swasey?** (2 items): each item labelled. Said: USS Swasey (instance: destroyer escort): Houston; USS Swasey (instance: shipwreck): Quincy.
   - fits: Q7874525 (destroyer escort; recorded): operator: United States Navy; named after: Charles Swasey; manufacturer: Brown Shipbuilding; vessel class: Edsall-class destroyer escort
   - fits: Q7874526 (destroyer, shipwreck; recorded): country: United Kingdom; located in the administrative territorial entity: United Kingdom; operator: Royal Navy; operator: Royal Navy
3. **What is the opposite of of flashforward?** (4 items): one item. Said: flashback. If you meant something else, there is also a television series named FlashForward and a literary work named Flashforward. Want to know about one of them?
   - does not fit: Q476857 (television series; recorded): director: David S. Goyer; composer: Ramin Djawadi; genre: drama television series; genre: science fiction television program
   - does not fit: Q2346621 (literary work; recorded): author: Robert J. Sawyer; publisher: Tor Books; genre: science fiction; language of work or name: English
   - fits: Q2533706 (cinematic technique, narrative technique; recorded): subclass of: scene; opposite of: flashback; different from: spoiler
   - does not fit: Q131868451 (version edition or translation; recorded): publisher: Tor Books; place of publication: United States; language of work or name: English; distribution format: hardcover
4. **What is the country of origin of A Different World?** (5 items): each item labelled. Said: Different World (genre: heavy metal): I do not know; Different World (genre: dance music): I do not know.
   - fits: Q2314449 (single; recorded): composer: Steve Harris; genre: heavy metal; follows: The Reincarnation of Benjamin Breeg; followed by: El Dorado
   - does not fit: Q2315368 (album; recorded): genre: Burlington; genre: arena rock; genre: hard rock; genre: Collector's Guide Publishing
   - fits: Q59576383 (single; recorded): genre: dance music; follows: Diamond Heart; followed by: Lost Control; performer: K-391
   - does not fit: Q59617920 (album; recorded): genre: electronic dance music; producer: Jenna Andrews; producer: Steve Aoki; producer: Mood Melodies
   - does not fit: Q109286761 (audio track; recorded): performer: Alan Walker; performer: K-391; performer: CORSAK; performer: Sofia Carson
5. **What is the part of of Ballad of Easy Rider?** (2 items): each item labelled. Said: Ballad of Easy Rider (genre: rock music): The Byrds' albums in chronological order; Ballad of Easy Rider (genre: folk rock): Ballad of Easy Rider.
   - fits: Q805125 (album; recorded): genre: rock music; producer: Terry Melcher; performer: The Byrds; record label: Columbia Records
   - fits: Q4851333 (single; recorded): genre: folk rock; follows: Wasn't Born to Follow; followed by: Jesus Is Just Alright; performer: Roger McGuinn
6. **What is the location of formation of The Architect?** (32 items): each item labelled. Said: Architect (instance: rock band): Syracuse; Architect (instance: business): I do not know.
   - fits: Q2860251 (rock band; recorded): genre: progressive metal; record label: Black Market Activities; country of origin: United States; inception: 2004
   - fits: Q11285528 (business; recorded): country: Japan; headquarters location: Japan
7. **What is the located in the administrative territorial entity of Diamantina River?** (2 items): each item labelled. Said: Diamantina River (country: Australia): Queensland; Diamantina River (country: Brazil): Goiás.
   - fits (placed): Q1208608 (river; recorded): country: Australia; continent: Australian continent; located in the administrative territorial entity: Queensland; named after: Diamantina Bowen
   - fits (placed): Q5270661 (river; recorded): country: Brazil; located in the administrative territorial entity: Goiás; mouth of the watercourse: Araguaia River; elevation above sea level: 420 metre
8. **What is the country of Calhoun Township?** (4 items): one answer, every item agrees. Said: United States. (4 items named Calhoun Township agree)
   - fits (placed): Q2218297 (township of iowa; recorded): country: United States; located in the administrative territorial entity: Calhoun County; population: 123; number of households: 51
   - fits (placed): Q5019797 (township of kansas; recorded): country: United States; located in the administrative territorial entity: Cheyenne County; population: 41; number of households: 3
   - fits (placed): Q9037091 (civil township, township of iowa; recorded): country: United States; located in the administrative territorial entity: Harrison County; population: 358; number of households: 82
   - fits (placed): Q10835456 (township of arkansas; recorded): country: United States; located in the administrative territorial entity: Prairie County
9. **What is the from narrative universe of Demon Hunter?** (14 items): asking back. Said: Which one do you mean? (14 items named Demon Hunter)
   - does not fit: Q1186459 (musical group; recorded): country: United States; genre: metalcore; genre: alternative metal; genre: groove metal
   - does not fit: Q1756223 (album; recorded): genre: metalcore; followed by: Summer of Darkness; producer: Aaron Sprinkle; performer: Demon Hunter
   - does not fit: Q2895087 (film; recorded): director: Gilberto de Anda; screenwriter: Gilberto de Anda; composer: Marco Flores; genre: horror film
   - fits: Q5256211 (comics character; recorded): sex or gender: male; creator: David Anthony Kraft; creator: Rich Buckler; from narrative universe: Earth-616
   - does not fit: Q5256212 (film; recorded): director: Scott Ziehl; composer: Jon Lee; genre: horror film; cast member: Sean Patrick Flanery
   - fits: Q58795262 (fictional profession; recorded): subclass of: hunter; said to be the same as: Demon hunter; topic's main category: Category:Fictional demon hunters; enemy: fictional demon
   - does not fit: Q109062277 (video game series; recorded): 
   - fits: Q113624913 (fictional profession; recorded): field of work: demonology; subclass of: hunter; subclass of: warrior; said to be the same as: demon hunter
10. **What is the educated at of Colin Griffiths?** (4 items): asking back. Said: Which one do you mean? (4 items named Colin Griffiths)
   - fits: Q5145163 (human; recorded): place of birth: Kent; sex or gender: male; country of citizenship: United Kingdom; educated at: University of East Anglia
   - fits: Q16016147 (human; recorded): place of birth: Upminster; sex or gender: male; country of citizenship: United Kingdom; occupation: cricketer
   - fits: Q124867295 (human; recorded): sex or gender: male; country of citizenship: Australia; occupation: architect; date of birth: 1936
   - fits: Q134473850 (human; recorded): sex or gender: male; occupation: guitarist; member of: U-J3RK5; family name: Griffiths
11. **What is the capital of Chen?** (29 items): one item. Said: Huaiyang District.
   - fits: Q1069690 (ancient chinese state, twelve vassals; recorded): continent: Asia; capital: Huaiyang District; dissolved, abolished or demolished date: 478 BC; topic's main category: Category:Chen (state)
12. **What is the genre of Real Friends?** (7 items): asking back. Said: Which one do you mean? (7 items named Real Friends)
   - fits: Q17151030 (musical group; recorded): genre: punk rock; record label: Fearless Records; discography: Real Friends discography; country of origin: United States
   - fits: Q22905752 (single; recorded): genre: hip-hop; producer: Boi-1da; performer: Kanye West; performer: Ty Dolla Sign
   - fits: Q28453482 (album; recorded): genre: soul; follows: Highly Distinct; followed by: Whatever; performer: The Friends of Distinction
   - fits: Q45642682 (single; recorded): genre: pop music; follows: Sangria Wine; performer: Camila Cabello; record label: Epic Records
   - does not fit: Q66011342 (television series episode; recorded): follows: Locket & Key; followed by: Tunnel of Shadows; part of the series: Lego Friends: Girls on a Mission; title: Real Friends
   - fits: Q85796217 (album; recorded): genre: country music; performer: Chris Janson; record label: Warner Bros. Records
   - fits: Q136328192 (audio track; recorded): performer: Kanye West; title: Real Friends; duration: 252 second
13. **What is the located in the administrative territorial entity of KATA?** (28 items): nothing read. Said: I don't know.
14. **What is the country of Qu River?** (3 items): one answer, every item agrees. Said: People's Republic of China. (3 items named Qu River agree)
   - fits (placed): Q544565 (river, watercourse; recorded): country: People's Republic of China; located in the administrative territorial entity: Sichuan; located in the administrative territorial entity: Chongqing; mouth of the watercourse: Jialing River
   - fits (placed): Q7268213 (watercourse; recorded): country: People's Republic of China; Commons category: Qu River (Nanpan River); tributary: Lian River; described by source: Gujin Tushu Jicheng
   - fits (placed): Q15895848 (river; recorded): country: People's Republic of China; located in the administrative territorial entity: Huaihua; located in the administrative territorial entity: Jingzhou Miao and Dong Autonomous County; located in the administrative territorial entity: Hunan
15. **What is the language of work or name of The Deer and the Cauldron?** (2 items): each item labelled. Said: The Deer and the Cauldron (country of origin: Hong Kong): Chinese; The Deer and the Cauldron (country of origin: People's Republic of China): Standard Chinese.
   - fits: Q1185345 (written work; recorded): author: Jin Yong; publisher: Ming Pao; genre: wuxia; genre: historical prose literature
   - fits: Q15633375 (television series; recorded): cast member: Han Dong; language of work or name: Standard Chinese; original broadcaster: Anhui Television; country of origin: People's Republic of China
16. **What is the located in the administrative territorial entity of Tenna?** (11 items): asking back. Said: Which one do you mean? (11 items named Tenna)
   - fits (placed): Q66226 (former municipality of switzerland, village; recorded): country: Switzerland; located in the administrative territorial entity: Canton of the Grisons; postal code: 7106; Commons category: Tenna
   - fits (placed): Q326941 (commune of italy; recorded): country: Italy; capital: Tenna; shares border with: Pergine Valsugana; shares border with: Caldonazzo
   - fits (placed): Q1145289 (japanese era name; recorded): country: Japan; follows: Enpō; followed by: Jōkyō; start time: 9 November 1681
   - fits (placed): Q3983464 (river; recorded): country: Italy; Commons category: Tenna (river); mouth of the watercourse: Adriatic Sea; length: 70 kilometre
   - fits (placed): Q7699953 (human settlement; recorded): country: Sri Lanka; located in the administrative territorial entity: Central Province; located in time zone: UTC+05:30
   - fits (placed): Q11224398 (island; recorded): country: Norway; located in the administrative territorial entity: Herøy Municipality; area: 3.5 square kilometre
   - fits (placed): Q30031045 (municipality seat; recorded): country: Italy; located in the administrative territorial entity: Tenna; located in time zone: UTC+01:00; located in time zone: UTC+02:00
   - fits (placed): Q31540182 (mountain; recorded): country: Norway; located in the administrative territorial entity: Volda Municipality
17. **What is the located in the administrative territorial entity of Barreiros?** (24 items): asking back. Said: Which one do you mean? (21 items named Barreiros)
   - fits (placed): Q367720 (automobile manufacturer, brand; recorded): country: Spain; founded by: Eduardo Barreiros; headquarters location: Villaverde; Commons category: Barreiros vehicles
   - fits (placed): Q799367 (former freguesia of portugal; recorded): country: Portugal; located in the administrative territorial entity: Sonim e Barreiros; located in time zone: Western European Time; elevation above sea level: 455 metre
   - fits (placed): Q808853 (freguesia of portugal; recorded): country: Portugal; located in the administrative territorial entity: Amares; elevation above sea level: 74 metre; area: 2.99 square kilometre
   - fits (placed): Q1634820 (municipality of galicia; recorded): head of government: Ana Ermida Igrexas; country: Spain; capital: San Miguel de Reinante, Barreiros; shares border with: Lourenzá
   - fits (placed): Q2102404 (municipality of brazil; recorded): country: Brazil; shares border with: São José da Coroa Grande; shares border with: Maragogi; shares border with: Tamandaré
   - fits (placed): Q12384188 (settlement in galicia; recorded): country: Spain; located in the administrative territorial entity: Rioaveso, Vilalba; on focus list of Wikimedia project: Wikiproject Toponymy
   - fits (placed): Q20541823 (abandoned village, settlement in galicia; recorded): country: Spain; located in the administrative territorial entity: As Chás, Maceda; on focus list of Wikimedia project: Wikiproject Toponymy
   - fits (placed): Q20541870 (settlement in galicia; recorded): country: Spain; on focus list of Wikimedia project: Wikiproject Toponymy
18. **What is the languages spoken, written or signed of Kathy Smith?** (7 items): asking back. Said: Which one do you mean? (7 items named Kathy Smith)
   - fits: Q3066300 (human; recorded): place of birth: Maidenhead; sex or gender: female; country of citizenship: United Kingdom; occupation: archer
   - fits: Q6377304 (human; recorded): sex or gender: female; country of citizenship: Australia; occupation: film director; employer: University of Southern California
   - fits: Q6377305 (human; recorded): place of birth: Arizona; sex or gender: female; country of citizenship: United States; child: Kate Grace
   - fits: Q19879126 (human; recorded): place of birth: Sheffield; sex or gender: female; country of citizenship: Australia; country of citizenship: United Kingdom
   - fits: Q100320057 (human; recorded): sex or gender: female; country of citizenship: United States; occupation: politician; family name: Smith
   - fits: Q116879195 (human; recorded): educated at: University of Mary Hardin–Baylor; field of work: nursing science; occupation: academic; occupation: researcher
   - fits: Q120603796 (human; recorded): place of birth: Chicago; sex or gender: female; languages spoken, written or signed: English; languages spoken, written or signed: Czech
19. **What is the cast member of Wet Gold?** (2 items): each item labelled. Said: Wet Gold (genre: treasure hunt film): Brian Kerwin, Brooke Shields, Burgess Meredith, Thomas Byrd; Wet Gold (genre: adventure film): Ralph Ince.
   - fits: Q7989963 (film; recorded): director: Dick Lowry; screenwriter: David Sherwin; composer: Sylvester Levay; genre: treasure hunt film
   - fits: Q15617001 (film; recorded): director: Ralph Ince; screenwriter: John Ernest Williamson; genre: adventure film; genre: silent film
20. **What is the filming location of Loners?** (5 items): asking back. Said: Which one do you mean? (5 items named Loners)
   - fits: Q2742024 (film; recorded): director: David Ondříček; screenwriter: Petr Zelenka; screenwriter: Olga Dabrowská; screenwriter: David Ondříček
   - fits: Q2894483 (film; recorded): director: Renen Schorr; screenwriter: Moshe Zonder; screenwriter: Renen Schorr; composer: Shem Tov Levi
   - does not fit: Q5144153 (fictional organization; recorded): from narrative universe: Earth-616
   - fits: Q23302084 (film; recorded): director: Sutton Roley; composer: Fred Karger; genre: drama film; cast member: Dean Stockwell
   - fits: Q65045205 (film; recorded): distribution format: video on demand; publication date: 2019
21. **What is the educated at of Harry McPherson?** (2 items): each item labelled. Said: Harry McPherson (occupation: lawyer): Southern Methodist University; Harry McPherson (occupation: Australian rules football player): I do not know.
   - fits: Q3790209 (human; recorded): place of birth: Tyler; place of death: Bethesda; sex or gender: male; country of citizenship: United States
   - fits: Q16073436 (human; recorded): place of birth: Longwood; place of death: Malvern; sex or gender: male; member of sports team: St Kilda Football Club
22. **What is the country of origin of City of Lost Souls?** (4 items): each item labelled. Said: The City of Lost Souls (genre: action film): Japan; City of Lost Souls (genre: LGBT-related film): Germany; City of Lost Souls (genre: adventure fiction): United States.
   - fits: Q1094135 (film; recorded): director: Takashi Miike; composer: Kōji Endō; genre: action film; genre: gangster film
   - fits: Q5123797 (film; recorded): director: Rosa von Praunheim; screenwriter: Rosa von Praunheim; composer: Holger Münzer; genre: LGBT-related film
   - fits: Q5770821 (literary work; recorded): author: Cassandra Clare; publisher: Simon & Schuster; genre: adventure fiction; genre: fantasy
   - does not fit: Q60377320 (version edition or translation; recorded): author: Cassandra Clare; publisher: Margaret K. McElderry Books; language of work or name: English; distribution format: hardcover
23. **What is the given name of Andrew Blake?** (28 items): nothing read. Said: I don't know.
24. **What is the country of citizenship of Jimmy Adams?** (6 items): asking back. Said: Which one do you mean? (6 items named Jimmy Adams)
   - fits: Q2789496 (human; recorded): place of birth: Troon; place of death: London; sex or gender: male; country of citizenship: United Kingdom
   - fits: Q3763931 (human; recorded): place of birth: Port Maria La luz; sex or gender: male; country of citizenship: South Africa; occupation: cricketer
   - fits: Q6199479 (human; recorded): place of birth: Stoke-on-Trent; place of death: Stoke-on-Trent; sex or gender: male; country of citizenship: United Kingdom
   - fits: Q6199482 (human; recorded): place of birth: Norfolk; sex or gender: male; country of citizenship: United States; occupation: engineer
   - fits: Q75475333 (human; recorded): sex or gender: male; father: Harold Adams; mother: Coco Lequerre; family name: Adams
   - fits: Q137150557 (human; recorded): sex or gender: male; date of birth: 1929; family name: Adams; given name: Jimmy
25. **What is the location of formation of Pearl Jam?** (2 items): one item. Said: Seattle. If you meant something else, there is also an album named Pearl Jam. Want to know about it?
   - fits: Q142701 (rock band; recorded): genre: grunge; genre: hard rock; award received: Rock and Roll Hall of Fame; award received: MTV Video Music Award for Best Group
   - does not fit: Q931609 (album; recorded): genre: hard rock; genre: alternative rock; genre: grunge; follows: Riot Act
26. **What is the located in the administrative territorial entity of Changy?** (7 items): asking back. Said: Which one do you mean? (7 items named Changy)
   - fits (placed): Q210441 (commune of france; recorded): country: France; shares border with: Marcilly-la-Gueurce; shares border with: Charolles; shares border with: Lugny-lès-Charolles
   - fits (placed): Q1068656 (commune of france; recorded): country: France; shares border with: Saint-Bonnet-des-Quarts; shares border with: Ambierle; shares border with: La Pacaudière
   - fits (placed): Q1218976 (commune of france; recorded): country: France; shares border with: Outrepont; shares border with: Saint-Quentin-les-Marais; shares border with: Bassuet
   - fits (placed): Q2956090 (commune of france; recorded): country: France; located in the administrative territorial entity: arrondissement of Montargis; located in time zone: UTC+02:00; located in time zone: UTC+01:00
   - fits (placed): Q34791782 (human settlement; recorded): country: France
   - fits (placed): Q109572248 (commune of france; recorded): country: France; located in the administrative territorial entity: Cher; dissolved, abolished or demolished date: 1823
   - fits (placed): Q109573949 (commune of france; recorded): country: France; located in the administrative territorial entity: Nièvre; dissolved, abolished or demolished date: 1836
27. **What is the distributed by of CBGB?** (2 items): one item. Said: Disney+, Netflix, XLrator Media. If you meant something else, there is also a music venue named CBGB. Want to know about it?
   - does not fit: Q1022965 (music venue, nightclub; recorded): country: United States; founded by: Hilly Kristal; owned by: Hilly Kristal; located in the administrative territorial entity: Manhattan
   - fits: Q2931018 (film; recorded): director: Randall Miller; screenwriter: Randall Miller; composer: Cliff Eidelman; genre: biographical film
28. **What is the part of the series of The Maid?** (38 items): asking back. Said: Which one do you mean? (5 items named The Maid)
   - fits: Q1990821 (television film; recorded): director: Ian Toynton; screenwriter: Timothy Prager; composer: Didier Vasseur; genre: romantic comedy
   - fits: Q7749615 (television series episode; recorded): director: Andy Ackerman; follows: The Frogger; followed by: The Puerto Rican Day; part of the series: Seinfeld
   - fits: Q114289268 (literary work; recorded): author: Nita Prose; genre: crime fiction; language of work or name: English; publication date: 2022
   - fits: Q114477888 (television series episode; recorded): follows: The Director; followed by: Hand Model; part of the series: My Wife and Kids; publication date: 5 May 2004
   - fits: Q135073847 (video game; recorded): genre: action game; genre: adventure video game; genre: action-adventure game; platform: Microsoft Windows
29. **What is the award received of Giuseppe Garibaldi?** (18 items): asking back. Said: Which one do you mean? (18 items named Giuseppe Garibaldi)
   - fits: Q539 (human; recorded): place of birth: Nice; place of death: Caprera; sex or gender: male; father: Domenico Garibaldi
   - does not fit: Q742121 (anti submarine warfare carrier; recorded): operator: Italian Navy; named after: Giuseppe Garibaldi; manufacturer: Fincantieri; Commons category: Giuseppe Garibaldi (C 551)
   - does not fit: Q1133711 (armored cruiser; recorded): operator: Royal Italian Navy; named after: Giuseppe Garibaldi; manufacturer: Ansaldo; vessel class: Giuseppe Garibaldi-class armored cruiser
   - does not fit: Q2261834 (light cruiser; recorded): operator: Royal Italian Navy; operator: Italian Navy; named after: Giuseppe Garibaldi; followed by: Andrea Doria-class cruiser
   - does not fit: Q3758350 (hospital ship; recorded): operator: Royal Italian Navy; named after: Giuseppe Garibaldi; manufacturer: Regio Cantiere di Castellammare di Stabia; Commons category: Garibaldi (ship, 1860)
   - fits: Q5565419 (human; recorded): place of birth: Cipressa; place of death: Province of Imperia; sex or gender: male; country of citizenship: Kingdom of Italy
   - fits: Q15292979 (human; recorded): sex or gender: male; father: Menotti Garibaldi; mother: Italia Bidischini dall'Oglio; family name: Garibaldi
   - fits: Q15292987 (human; recorded): sex or gender: male; father: Menotti Garibaldi; mother: Italia Bidischini dall'Oglio; family name: Garibaldi
30. **What is the occupation of Liam Finn?** (2 items): each item labelled. Said: Liam Finn (country of citizenship: New Zealand): singer, singer-songwriter; Liam Finn (country of citizenship: Ireland): rugby league player.
   - fits: Q2708494 (human; recorded): place of birth: Melbourne; sex or gender: male; father: Neil Finn; country of citizenship: New Zealand
   - fits: Q6539589 (human; recorded): place of birth: Yorkshire; sex or gender: male; country of citizenship: Ireland; member of sports team: Wakefield Trinity
31. **What is the country of citizenship of Bobby Caldwell?** (2 items): one answer, every item agrees. Said: United States. (2 items named Bobby Caldwell agree)
   - fits: Q2907941 (human; recorded): place of birth: Manhattan; place of death: Manhattan; sex or gender: male; country of citizenship: United States
   - fits: Q3487124 (human; recorded): place of birth: Florida; sex or gender: male; country of citizenship: United States; occupation: musician
32. **What is the located in the administrative territorial entity of Sumner Township?** (14 items): asking back. Said: Which one do you mean? (14 items named Sumner Township)
   - fits (placed): Q615480 (township of kansas; recorded): country: United States; located in the administrative territorial entity: Phillips County; population: 35; number of households: 0
   - fits (placed): Q1915599 (township of minnesota; recorded): country: United States; located in the administrative territorial entity: Fillmore County; population: 482; number of households: 162
   - fits (placed): Q2220640 (township of iowa; recorded): country: United States; located in the administrative territorial entity: Buchanan County; Commons category: Sumner Township, Buchanan County, Iowa; population: 3335
   - fits (placed): Q5397574 (township of michigan; recorded): country: United States; located in the administrative territorial entity: Gratiot County; population: 1863; number of households: 826
   - fits (placed): Q5400616 (township of iowa; recorded): country: United States; located in the administrative territorial entity: Iowa County; population: 399; number of households: 199
   - fits (placed): Q6031928 (township in the united states; recorded): country: United States; located in the administrative territorial entity: Guilford County
   - fits (placed): Q6031931 (township in the united states; recorded): country: United States; located in the administrative territorial entity: Spink County; population: 10; number of households: 9
   - fits (placed): Q7637997 (township of illinois; recorded): country: United States; located in the administrative territorial entity: Kankakee County; inception: 8 October 1855; population: 837
33. **What is the parent taxon of Alibertia?** (2 items): asking back. Said: Which one do you mean? (2 items named Alibertia)
   - fits: Q2473735 (taxon; recorded): taxon rank: genus; named after: Jean-Louis Alibert; parent taxon: Cordiereae; taxon name: Alibertia
   - fits: Q4725637 (taxon; recorded): taxon rank: genus; parent taxon: Eulophidae; taxon name: Alibertia; homonymous taxon: Alibertia
34. **What is the followed by of Turbulence?** (32 items): asking back. Said: Which one do you mean? (10 items named Turbulence)
   - fits: Q623500 (single; recorded): genre: crunk; genre: big room house; follows: I'm in the House; follows: Hey
   - fits: Q2191062 (album; recorded): follows: III; performer: Brother Ape; publication date: 2009
   - fits: Q3542269 (album; recorded): performer: Michel Portal; record label: Harmonia Mundi; publication date: 1987
   - fits: Q3542271 (album; recorded): genre: instrumental rock; genre: progressive rock; follows: The Steve Howe Album; followed by: The Grand Scheme of Things
   - fits: Q7854770 (album; recorded): follows: Aviator; performer: Aviator; record label: Harvest; publication date: 1980
   - does not fit: Q7854771 (no kind; none): country: United States
   - fits: Q50845899 (television series episode; recorded): director: Kevin Fair; follows: Infamous; followed by: Hex; part of the series: Smallville
   - fits: Q114461979 (television series episode; recorded): follows: Control; followed by: Family Matters; part of the series: The Night Shift; publication date: 20 July 2017
35. **What is the country of citizenship of Steve Gould?** (2 items): each item labelled. Said: Steve Gould (occupation: curler): Canada; Steve Gould (occupation: singer): United Kingdom.
   - fits: Q7612689 (human; recorded): place of birth: Winnipeg; sex or gender: male; country of citizenship: Canada; occupation: curler
   - fits: Q7612692 (human; recorded): sex or gender: male; country of citizenship: United Kingdom; occupation: singer; occupation: guitarist
36. **What is the member of sports team of Johnny Peacock?** (2 items): each item labelled. Said: Johnny Peacock (occupation: baseball player): Boston Red Sox, Minneapolis Millers, Philadelphia Phillies; Johnny Peacock (occupation: American football player): Houston Cougars football, Tennessee Titans.
   - fits: Q6267462 (human; recorded): place of birth: Fremont; place of death: Wilson; sex or gender: male; country of citizenship: United States
   - fits: Q18001557 (human; recorded): place of birth: Austin; sex or gender: male; country of citizenship: United States; member of sports team: Tennessee Titans
37. **What is the country of Petershausen?** (4 items): asking back. Said: Which one do you mean? (4 items named Petershausen)
   - fits (placed): Q533046 (municipality without town privileges in germany; recorded): head of government: Marcel Fath; country: Germany; shares border with: Weichs; shares border with: Vierkirchen
   - fits (placed): Q592306 (ortsteil, quarter; recorded): country: Germany; located in the administrative territorial entity: Constance; Commons category: Petershausen (Konstanz); located in time zone: UTC+01:00
   - fits (placed): Q32221839 (municipality seat, ortsteil, parish village; recorded): country: Germany; located in the administrative territorial entity: Petershausen; postal code: 85238; population: 2718
   - fits (placed): Q50675411 (railway station; recorded): transport network: Munich S-Bahn; country: Germany; connecting line: Munich–Treuchtlingen railway; connecting line: Munich-Laim Pbf–Petershausen (Oberbay) railway
38. **What is the located in the administrative territorial entity of Gnojno?** (8 items): asking back. Said: Which one do you mean? (8 items named Gnojno)
   - fits (placed): Q212257 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Konstantynów; located in or next to body of water: Western Bug; postal code: 21-543
   - fits (placed): Q212638 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Gnojno; Commons category: Gnojno, Świętokrzyskie Voivodeship; located in time zone: UTC+02:00
   - fits (placed): Q553519 (rural municipality of poland; recorded): country: Poland; continent: Europe; capital: Gnojno; shares border with: Gmina Pierzchnica
   - fits (placed): Q959946 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Inowrocław; located in time zone: UTC+01:00; located in time zone: UTC+02:00
   - fits (placed): Q5574316 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Bobrowniki; Commons category: Gnojno, powiat lipnowski; licence plate code: CLI
   - fits (placed): Q5574318 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Pułtusk; postal code: 06-100; located in time zone: UTC+01:00
   - fits (placed): Q5574319 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Działdowo; located in time zone: UTC+01:00; located in time zone: UTC+02:00
   - fits (placed): Q5574320 (village of poland; recorded): country: Poland; located in the administrative territorial entity: Gmina Kutno; postal code: 99-300; licence plate code: EKU
39. **What is the member of sports team of Moon Sung-hyun?** (2 items): each item labelled. Said: Moon Sung-hyun (occupation: baseball player): Kiwoom Heroes; Moon Sung-Hyun (occupation: politician): I do not know.
   - fits: Q492889 (human; recorded): place of birth: Seoul; sex or gender: male; country of citizenship: South Korea; member of sports team: Kiwoom Heroes
   - fits: Q6907571 (human; recorded): sex or gender: male; country of citizenship: South Korea; educated at: Jinju High School; educated at: Seoul National University
40. **What is the given name of Alfred Thomson?** (3 items): asking back. Said: Which one do you mean? (3 items named Alfred Thomson)
   - fits: Q4723525 (human; recorded): place of birth: Bengaluru; place of death: Chelsea; sex or gender: male; country of citizenship: British Raj
   - fits: Q19519439 (human; recorded): place of death: Paddington; sex or gender: male; country of citizenship: Australia; occupation: cricketer
   - fits: Q127008072 (human; recorded): sex or gender: male; educated at: Imperial University of Dorpat; described by source: Album academicum universitatis tartuensis
41. **What is the operator of USS Hyman G. Rickover?** (2 items): one answer, every item agrees. Said: United States Navy. (2 items named USS Hyman G. Rickover agree)
   - fits: Q174818 (attack submarine, nuclear submarine; recorded): operator: United States Navy; manufacturer: General Dynamics; vessel class: Los Angeles-class submarine; Commons category: USS Hyman G. Rickover (SSN-709)
   - fits: Q19563487 (nuclear powered attack submarine; recorded): operator: United States Navy; named after: Hyman G. Rickover; manufacturer: General Dynamics Electric Boat; vessel class: Virginia-class submarine
42. **What is the different from of Thespis?** (5 items): asking back. Said: Which one do you mean? (4 items named Thespis)
   - fits: Q117176 (human; recorded): place of birth: Ikarion; sex or gender: male; country of citizenship: Classical Athens; occupation: poet
   - fits: Q972539 (dramatico musical work; recorded): composer: Arthur Sullivan; librettist: W. S. Gilbert; genre: comic opera; genre: féerie
   - fits: Q6145853 (taxon; recorded): taxon rank: genus; parent taxon: Thespidae; taxon name: Thespis; Commons category: Thespis (genus)
   - fits: Q6769679 (taxon; recorded): taxon rank: genus; parent taxon: Lagenophorinae; parent taxon: Asteraceae; taxon name: Thespis
43. **What is the distributed by of The Toxic Avenger?** (7 items): asking back. Said: Which one do you mean? (7 items named The Toxic Avenger)
   - fits: Q615551 (film; recorded): director: Michael Herz; director: Lloyd Kaufman; screenwriter: Lloyd Kaufman; genre: action film
   - does not fit: Q3523058 (human; recorded): place of birth: France; sex or gender: male; country of citizenship: France; occupation: disc jockey
   - does not fit: Q3989664 (character, film character; recorded): sex or gender: male; present in work: The Toxic Avenger Part II; present in work: The Toxic Avenger; present in work: The Toxic Avenger Part III: The Last Temptation of Toxie
   - does not fit: Q7769796 (dramatico musical work; recorded): composer: David Bryan; librettist: Joe DiPietro; lyricist: Joe DiPietro; form of creative work: musical
   - fits: Q7769798 (film series; recorded): has part(s): The Toxic Avenger; has part(s): Citizen Toxie: The Toxic Avenger IV; has part(s): The Toxic Avenger Part II; has part(s): The Toxic Avenger Part III: The Last Temptation of Toxie
   - fits: Q64755318 (media franchise; recorded): topic's main category: Category:The Toxic Avenger (franchise)
   - fits: Q107404778 (film; recorded): director: Macon Blair; screenwriter: Macon Blair; composer: Brooke Blair; composer: Will Blair
44. **What is the country of Notojima?** (2 items): one answer, every item agrees. Said: Japan. (2 items named Notojima agree)
   - fits (placed): Q7063204 (dissolved municipality of japan; recorded): country: Japan; located in the administrative territorial entity: Kashima district; inception: 1 February 1955; dissolved, abolished or demolished date: 1 October 2004
   - fits (placed): Q11612025 (island; recorded): country: Japan; located in the administrative territorial entity: Nanao; Commons category: Notojima; highest point: Yomuratukayama
45. **What is the country of origin of George and Mildred?** (2 items): one answer, every item agrees. Said: United Kingdom. (2 items named George and Mildred agree)
   - fits: Q776608 (television series; recorded): genre: sitcom; follows: Man About the House; cast member: Avril Elgar; cast member: Roy Kinnear
   - fits: Q5546666 (film; recorded): director: Peter Frazer-Jones; screenwriter: Dick Sharples; composer: Les Reed; genre: comedy film
46. **What is the located in the administrative territorial entity of Greenfield?** (69 items): nothing read. Said: I don't know.
47. **What is the country of citizenship of John Whyte?** (10 items): asking back. Said: Which one do you mean? (10 items named John Whyte)
   - fits: Q6263935 (human; recorded): place of birth: Dunfermline; place of death: Leeds; sex or gender: male; country of citizenship: Canada
   - fits: Q18164633 (human; recorded): sex or gender: male; country of citizenship: Australia; date of birth: 1826; date of death: 16 February 1902
   - fits: Q18219677 (human; recorded): place of birth: Easdale; sex or gender: male; country of citizenship: United Kingdom of Great Britain and Ireland; date of death: 1913
   - fits: Q50992423 (human; recorded): sex or gender: male; spouse: Tom Wilson Weinberg; educated at: Swarthmore College; educated at: University of Pennsylvania
   - fits: Q75341757 (human; recorded): sex or gender: male; child: Theodore Whyte; date of birth: 14 May 1806; date of death: 24 July 1889
   - fits: Q75608708 (human; recorded): sex or gender: male; child: Louisa Whyte; family name: Whyte; given name: John
   - fits: Q76317327 (human; recorded): sex or gender: male; child: Alexander Whyte; date of birth: 1812; date of death: 1871
   - fits: Q94594251 (human; recorded): sex or gender: male; occupation: researcher; given name: John
48. **What is the participant in of Charles Allen?** (30 items): nothing read. Said: I don't know.
49. **What is the participated in conflict of HMS Racoon?** (6 items): asking back. Said: Which one do you mean? (6 items named HMS Racoon)
   - fits: Q5633965 (sloop of war; recorded): operator: Royal Navy; designed by: John Randall; vessel class: Diligence-class brig-sloop; service entry: 1795
   - fits: Q5633966 (sloop of war; recorded): operator: Royal Navy; vessel class: Cormorant-class sloop-of-war; country of origin: United Kingdom of Great Britain and Ireland; participated in conflict: War of 1812
   - does not fit: Q5633970 (torpedo cruiser; recorded): operator: Royal Navy; manufacturer: HMNB Devonport; vessel class: Archer-class torpedo cruiser; Commons category: HMS Racoon (ship, 1887)
   - fits: Q11861024 (destroyer; recorded): operator: Royal Navy; vessel class: Beagle-class destroyer; significant event: ship launching; different from: HMS Racoon
   - fits: Q22570343 (brig sloop; recorded): operator: Royal Navy; country of origin: Kingdom of Great Britain; inception: 1782; significant event: ship launching
   - fits: Q85765324 (corvette, sailing ship; recorded): operator: Royal Navy; vessel class: Pearl-class corvette; Commons category: HMS Racoon (ship, 1857); significant event: ship launching
50. **What is the language of work or name of Traveler?** (43 items): asking back. Said: Which one do you mean? (25 items named Traveler)
   - fits: Q493191 (literary work; recorded): author: John Twelve Hawks; genre: science fiction; part of the series: Fourth Realm trilogy; language of work or name: English
   - fits: Q734223 (television series; recorded): composer: Trevor Morris; genre: drama television series; cast member: Logan Marshall-Green; cast member: Aaron Stanford
   - fits: Q7770002 (magazine; recorded): publisher: Shanghai; country of origin: People's Republic of China; inception: 2000
   - fits: Q7835975 (album; recorded): genre: blues; follows: Fuse; followed by: Limelight; performer: Colin James
   - fits: Q7835976 (album; recorded): genre: ambient music; performer: Steve Roach; record label: Domino Recording Company; part of: Steve Roach's albums in chronological order
   - fits: Q7835977 (album; recorded): genre: J-pop; genre: pop rock; performer: Hitomi; record label: Avex Trax
   - fits: Q7835978 (album; recorded): performer: Trey Anastasio; record label: Rubber Jungle Records; part of: Trey Anastasio's albums in chronological order; publication date: 16 October 2012
   - fits: Q9361370 (album; recorded): performer: Hard; publication date: 2008

## The cue-word probe: 20 names
Each must pick the target or ask back.

1. **Richard Parsons**, country of citizenship, cue 'male': asking back, the target among them (target Q7328290, fit ['Q2149305', 'Q7328287', 'Q7328288', 'Q7328289', 'Q7328290', 'Q16090883', 'Q18757051', 'Q21466855', 'Q21524713', 'Q44606086', 'Q65046276', 'Q75255222', 'Q75255312', 'Q75255324', 'Q75437721', 'Q76090772', 'Q86633580', 'Q92534386', 'Q102342311']). Said: Which one do you mean? (20 items named Richard Parsons)
   - fits: Q2149305 (human; recorded): place of birth: Brooklyn; sex or gender: male; country of citizenship: United Kingdom; country of citizenship: United States
   - fits: Q7328287 (human; recorded): sex or gender: male; father: Richard Parsons, 1st Viscount Rosse; mother: Elizabeth Hamilton; spouse: Frances Claxton
   - fits: Q7328288 (human; recorded): sex or gender: male; country of citizenship: United Kingdom; educated at: Ulverston Victoria High School; educated at: University of Oxford
   - fits: Q7328289 (human; recorded): sex or gender: male; educated at: Durham School; educated at: Magdalen College; occupation: Anglican priest
   - fits: Q7328290 (human; recorded): sex or gender: male; country of citizenship: Australia; occupation: explorer; family name: Parsons
   - fits: Q16090883 (human; recorded): sex or gender: male; country of citizenship: United Kingdom; educated at: Brasenose College; educated at: Bembridge School
   - fits: Q18757051 (human; recorded): sex or gender: male; country of citizenship: Kingdom of England; occupation: ecclesiastical judge; occupation: antiquarian
   - fits: Q21466855 (human; recorded): sex or gender: male; occupation: painter; date of birth: 1925; date of death: 2000
2. **USS Swasey**, manufacturer, cue 'clemson': picks the target (target Q7874526, fit ['Q7874526']). Said: Bethlehem Shipbuilding Corporation. If you meant something else, there is also a destroyer escort named USS Swasey. Want to know about it?
   - ruled out by 'clemson': Q7874525 (destroyer escort; recorded): operator: United States Navy; named after: Charles Swasey; manufacturer: Brown Shipbuilding; vessel class: Edsall-class destroyer escort
   - fits: Q7874526 (destroyer, shipwreck; recorded): country: United Kingdom; located in the administrative territorial entity: United Kingdom; operator: Royal Navy; operator: Royal Navy
3. **Flashforward**, publisher, cue 'translation': each item labelled, the target among them (target Q131868451, fit ['Q2346621', 'Q131868451']). Said: Flashforward (instance: literary work): Tor Books; Flashforward (instance: version, edition or translation): Tor Books.
   - does not fit: Q476857 (television series; recorded): director: David S. Goyer; composer: Ramin Djawadi; genre: drama television series; genre: science fiction television program
   - fits: Q2346621 (literary work; recorded): author: Robert J. Sawyer; publisher: Tor Books; genre: science fiction; language of work or name: English
   - does not fit: Q2533706 (cinematic technique, narrative technique; recorded): subclass of: scene; opposite of: flashback; different from: spoiler
   - fits: Q131868451 (version edition or translation; recorded): publisher: Tor Books; place of publication: United States; language of work or name: English; distribution format: hardcover
4. **Different World**, producer, cue 'arena': asking back, the target among them (target Q2315368, fit ['Q2314449', 'Q2315368', 'Q59576383', 'Q59617920', 'Q109286761']). Said: Which one do you mean? (5 items named Different World)
   - fits: Q2314449 (single; recorded): composer: Steve Harris; genre: heavy metal; follows: The Reincarnation of Benjamin Breeg; followed by: El Dorado
   - fits: Q2315368 (album; recorded): genre: Burlington; genre: arena rock; genre: hard rock; genre: Collector's Guide Publishing
   - fits: Q59576383 (single; recorded): genre: dance music; follows: Diamond Heart; followed by: Lost Control; performer: K-391
   - fits: Q59617920 (album; recorded): genre: electronic dance music; producer: Jenna Andrews; producer: Steve Aoki; producer: Mood Melodies
   - fits: Q109286761 (audio track; recorded): performer: Alan Walker; performer: K-391; performer: CORSAK; performer: Sofia Carson
5. **Ballad of Easy Rider**, part of, cue 'november': each item labelled, the target among them (target Q805125, fit ['Q805125', 'Q4851333']). Said: Ballad of Easy Rider (genre: rock music): The Byrds' albums in chronological order; Ballad of Easy Rider (genre: folk rock): Ballad of Easy Rider.
   - fits: Q805125 (album; recorded): genre: rock music; producer: Terry Melcher; performer: The Byrds; record label: Columbia Records
   - fits: Q4851333 (single; recorded): genre: folk rock; follows: Wasn't Born to Follow; followed by: Jesus Is Just Alright; performer: Roger McGuinn
6. **Diamantina River**, named after, cue 'warburton': asking back (target Q1208608, fit []). Said: Which one do you mean? (2 items named Diamantina River)
   - does not fit: Q1208608 (river; recorded): country: Australia; continent: Australian continent; located in the administrative territorial entity: Queensland; named after: Diamantina Bowen
   - does not fit: Q5270661 (river; recorded): country: Brazil; located in the administrative territorial entity: Goiás; mouth of the watercourse: Araguaia River; elevation above sea level: 420 metre
7. **Calhoun Township**, located in the administrative territorial entity, cue 'states': asking back, the target among them (target Q9037091, fit ['Q2218297', 'Q5019797', 'Q9037091', 'Q10835456']). Said: Which one do you mean? (4 items named Calhoun Township)
   - fits (placed): Q2218297 (township of iowa; recorded): country: United States; located in the administrative territorial entity: Calhoun County; population: 123; number of households: 51
   - fits (placed): Q5019797 (township of kansas; recorded): country: United States; located in the administrative territorial entity: Cheyenne County; population: 41; number of households: 3
   - fits (placed): Q9037091 (civil township, township of iowa; recorded): country: United States; located in the administrative territorial entity: Harrison County; population: 358; number of households: 82
   - fits (placed): Q10835456 (township of arkansas; recorded): country: United States; located in the administrative territorial entity: Prairie County
8. **Demon Hunter**, game mode, cue 'adventure': asking back, the target among them (target Q124064797, fit ['Q109062277', 'Q124064797', 'Q135032719']). Said: Which one do you mean? (14 items named Demon Hunter)
   - does not fit: Q1186459 (musical group; recorded): country: United States; genre: metalcore; genre: alternative metal; genre: groove metal
   - does not fit: Q1756223 (album; recorded): genre: metalcore; followed by: Summer of Darkness; producer: Aaron Sprinkle; performer: Demon Hunter
   - does not fit: Q2895087 (film; recorded): director: Gilberto de Anda; screenwriter: Gilberto de Anda; composer: Marco Flores; genre: horror film
   - does not fit: Q5256211 (comics character; recorded): sex or gender: male; creator: David Anthony Kraft; creator: Rich Buckler; from narrative universe: Earth-616
   - does not fit: Q5256212 (film; recorded): director: Scott Ziehl; composer: Jon Lee; genre: horror film; cast member: Sean Patrick Flanery
   - does not fit: Q58795262 (fictional profession; recorded): subclass of: hunter; said to be the same as: Demon hunter; topic's main category: Category:Fictional demon hunters; enemy: fictional demon
   - fits: Q109062277 (video game series; recorded): 
   - does not fit: Q113624913 (fictional profession; recorded): field of work: demonology; subclass of: hunter; subclass of: warrior; said to be the same as: demon hunter
9. **Colin Griffiths**, country of citizenship, cue 'anglia': asking back, the target among them (target Q5145163, fit ['Q5145163', 'Q16016147', 'Q124867295', 'Q134473850']). Said: Which one do you mean? (4 items named Colin Griffiths)
   - fits: Q5145163 (human; recorded): place of birth: Kent; sex or gender: male; country of citizenship: United Kingdom; educated at: University of East Anglia
   - fits: Q16016147 (human; recorded): place of birth: Upminster; sex or gender: male; country of citizenship: United Kingdom; occupation: cricketer
   - fits: Q124867295 (human; recorded): sex or gender: male; country of citizenship: Australia; occupation: architect; date of birth: 1936
   - fits: Q134473850 (human; recorded): sex or gender: male; occupation: guitarist; member of: U-J3RK5; family name: Griffiths
10. **Real Friends**, performer, cue 'album': asking back, the target among them (target Q85796217, fit ['Q22905752', 'Q28453482', 'Q45642682', 'Q85796217', 'Q136328192']). Said: Which one do you mean? (7 items named Real Friends)
   - does not fit: Q17151030 (musical group; recorded): genre: punk rock; record label: Fearless Records; discography: Real Friends discography; country of origin: United States
   - fits: Q22905752 (single; recorded): genre: hip-hop; producer: Boi-1da; performer: Kanye West; performer: Ty Dolla Sign
   - fits: Q28453482 (album; recorded): genre: soul; follows: Highly Distinct; followed by: Whatever; performer: The Friends of Distinction
   - fits: Q45642682 (single; recorded): genre: pop music; follows: Sangria Wine; performer: Camila Cabello; record label: Epic Records
   - does not fit: Q66011342 (television series episode; recorded): follows: Locket & Key; followed by: Tunnel of Shadows; part of the series: Lego Friends: Girls on a Mission; title: Real Friends
   - fits: Q85796217 (album; recorded): genre: country music; performer: Chris Janson; record label: Warner Bros. Records
   - fits: Q136328192 (audio track; recorded): performer: Kanye West; title: Real Friends; duration: 252 second
11. **Qu River**, located in the administrative territorial entity, cue 'dong': asking back, the target among them (target Q15895848, fit ['Q544565', 'Q7268213', 'Q15895848']). Said: Which one do you mean? (3 items named Qu River)
   - fits (placed): Q544565 (river, watercourse; recorded): country: People's Republic of China; located in the administrative territorial entity: Sichuan; located in the administrative territorial entity: Chongqing; mouth of the watercourse: Jialing River
   - fits (placed): Q7268213 (watercourse; recorded): country: People's Republic of China; Commons category: Qu River (Nanpan River); tributary: Lian River; described by source: Gujin Tushu Jicheng
   - fits (placed): Q15895848 (river; recorded): country: People's Republic of China; located in the administrative territorial entity: Huaihua; located in the administrative territorial entity: Jingzhou Miao and Dong Autonomous County; located in the administrative territorial entity: Hunan
12. **The Deer and the Cauldron**, genre, cue 'written': each item labelled, the target among them (target Q1185345, fit ['Q1185345', 'Q15633375']). Said: The Deer and the Cauldron (country of origin: Hong Kong): historical prose literature, wuxia; The Deer and the Cauldron (country of origin: People's Republic of China): I do not know.
   - fits: Q1185345 (written work; recorded): author: Jin Yong; publisher: Ming Pao; genre: wuxia; genre: historical prose literature
   - fits: Q15633375 (television series; recorded): cast member: Han Dong; language of work or name: Standard Chinese; original broadcaster: Anhui Television; country of origin: People's Republic of China
13. **Tenna**, country, cue 'volda': asking back, the target among them (target Q31540182, fit ['Q66226', 'Q3983464', 'Q7699953', 'Q11224398', 'Q31540182']). Said: Which one do you mean? (11 items named Tenna)
   - fits (placed): Q66226 (former municipality of switzerland, village; recorded): country: Switzerland; located in the administrative territorial entity: Canton of the Grisons; postal code: 7106; Commons category: Tenna
   - fits (placed): Q326941 (commune of italy; recorded): country: Italy; capital: Tenna; shares border with: Pergine Valsugana; shares border with: Caldonazzo
   - fits (placed): Q1145289 (japanese era name; recorded): country: Japan; follows: Enpō; followed by: Jōkyō; start time: 9 November 1681
   - fits (placed): Q3983464 (river; recorded): country: Italy; Commons category: Tenna (river); mouth of the watercourse: Adriatic Sea; length: 70 kilometre
   - fits (placed): Q7699953 (human settlement; recorded): country: Sri Lanka; located in the administrative territorial entity: Central Province; located in time zone: UTC+05:30
   - fits (placed): Q11224398 (island; recorded): country: Norway; located in the administrative territorial entity: Herøy Municipality; area: 3.5 square kilometre
   - fits (placed): Q30031045 (municipality seat; recorded): country: Italy; located in the administrative territorial entity: Tenna; located in time zone: UTC+01:00; located in time zone: UTC+02:00
   - fits (placed): Q31540182 (mountain; recorded): country: Norway; located in the administrative territorial entity: Volda Municipality
14. **Barreiros**, Soundex, cue 'index': picks the target (target Q36894757, fit ['Q36894757']). Said: B662. If you meant something else, there is also a brand named Barreiros and a former freguesia of portugal named Barreiros. Want to know about one of them?
   - does not fit: Q367720 (automobile manufacturer, brand; recorded): country: Spain; founded by: Eduardo Barreiros; headquarters location: Villaverde; Commons category: Barreiros vehicles
   - does not fit: Q799367 (former freguesia of portugal; recorded): country: Portugal; located in the administrative territorial entity: Sonim e Barreiros; located in time zone: Western European Time; elevation above sea level: 455 metre
   - does not fit: Q808853 (freguesia of portugal; recorded): country: Portugal; located in the administrative territorial entity: Amares; elevation above sea level: 74 metre; area: 2.99 square kilometre
   - does not fit: Q1634820 (municipality of galicia; recorded): head of government: Ana Ermida Igrexas; country: Spain; capital: San Miguel de Reinante, Barreiros; shares border with: Lourenzá
   - does not fit: Q2102404 (municipality of brazil; recorded): country: Brazil; shares border with: São José da Coroa Grande; shares border with: Maragogi; shares border with: Tamandaré
   - does not fit: Q12384188 (settlement in galicia; recorded): country: Spain; located in the administrative territorial entity: Rioaveso, Vilalba; on focus list of Wikimedia project: Wikiproject Toponymy
   - does not fit: Q20541823 (abandoned village, settlement in galicia; recorded): country: Spain; located in the administrative territorial entity: As Chás, Maceda; on focus list of Wikimedia project: Wikiproject Toponymy
   - does not fit: Q20541870 (settlement in galicia; recorded): country: Spain; on focus list of Wikimedia project: Wikiproject Toponymy
15. **Kathy Smith**, place of birth, cue 'female': asking back, the target among them (target Q120603796, fit ['Q3066300', 'Q6377304', 'Q6377305', 'Q19879126', 'Q100320057', 'Q116879195', 'Q120603796']). Said: Which one do you mean? (7 items named Kathy Smith)
   - fits: Q3066300 (human; recorded): place of birth: Maidenhead; sex or gender: female; country of citizenship: United Kingdom; occupation: archer
   - fits: Q6377304 (human; recorded): sex or gender: female; country of citizenship: Australia; occupation: film director; employer: University of Southern California
   - fits: Q6377305 (human; recorded): place of birth: Arizona; sex or gender: female; country of citizenship: United States; child: Kate Grace
   - fits: Q19879126 (human; recorded): place of birth: Sheffield; sex or gender: female; country of citizenship: Australia; country of citizenship: United Kingdom
   - fits: Q100320057 (human; recorded): sex or gender: female; country of citizenship: United States; occupation: politician; family name: Smith
   - fits: Q116879195 (human; recorded): educated at: University of Mary Hardin–Baylor; field of work: nursing science; occupation: academic; occupation: researcher
   - fits: Q120603796 (human; recorded): place of birth: Chicago; sex or gender: female; languages spoken, written or signed: English; languages spoken, written or signed: Czech
16. **Wet Gold**, title, cue 'july': each item labelled, the target among them (target Q7989963, fit ['Q7989963', 'Q15617001']). Said: Wet Gold (genre: treasure hunt film): Wet Gold; Wet Gold (genre: adventure film): Wet Gold.
   - fits: Q7989963 (film; recorded): director: Dick Lowry; screenwriter: David Sherwin; composer: Sylvester Levay; genre: treasure hunt film
   - fits: Q15617001 (film; recorded): director: Ralph Ince; screenwriter: John Ernest Williamson; genre: adventure film; genre: silent film
17. **Loners**, distribution format, cue 'video': asking back, the target among them (target Q65045205, fit ['Q2742024', 'Q2894483', 'Q23302084', 'Q65045205']). Said: Which one do you mean? (5 items named Loners)
   - fits: Q2742024 (film; recorded): director: David Ondříček; screenwriter: Petr Zelenka; screenwriter: Olga Dabrowská; screenwriter: David Ondříček
   - fits: Q2894483 (film; recorded): director: Renen Schorr; screenwriter: Moshe Zonder; screenwriter: Renen Schorr; composer: Shem Tov Levi
   - does not fit: Q5144153 (fictional organization; recorded): from narrative universe: Earth-616
   - fits: Q23302084 (film; recorded): director: Sutton Roley; composer: Fred Karger; genre: drama film; cast member: Dean Stockwell
   - fits: Q65045205 (film; recorded): distribution format: video on demand; publication date: 2019
18. **Harry McPherson**, Commons category, cue 'male': each item labelled, the target among them (target Q3790209, fit ['Q3790209', 'Q16073436']). Said: Harry McPherson (occupation: lawyer): I do not know; Harry McPherson (occupation: Australian rules football player): I do not know.
   - fits: Q3790209 (human; recorded): place of birth: Tyler; place of death: Bethesda; sex or gender: male; country of citizenship: United States
   - fits: Q16073436 (human; recorded): place of birth: Longwood; place of death: Malvern; sex or gender: male; member of sports team: St Kilda Football Club
19. **City of Lost Souls**, publisher, cue 'york': each item labelled, the target among them (target Q5770821, fit ['Q5770821', 'Q60377320']). Said: City of Lost Souls (instance: literary work): Simon & Schuster; City of Lost Souls (instance: version, edition or translation): Margaret K. McElderry Books.
   - does not fit: Q1094135 (film; recorded): director: Takashi Miike; composer: Kōji Endō; genre: action film; genre: gangster film
   - does not fit: Q5123797 (film; recorded): director: Rosa von Praunheim; screenwriter: Rosa von Praunheim; composer: Holger Münzer; genre: LGBT-related film
   - fits: Q5770821 (literary work; recorded): author: Cassandra Clare; publisher: Simon & Schuster; genre: adventure fiction; genre: fantasy
   - fits: Q60377320 (version edition or translation; recorded): author: Cassandra Clare; publisher: Margaret K. McElderry Books; language of work or name: English; distribution format: hardcover
20. **Jimmy Adams**, sex or gender, cue 'male': asking back, the target among them (target Q75475333, fit ['Q2789496', 'Q3763931', 'Q6199479', 'Q6199482', 'Q75475333', 'Q137150557']). Said: Which one do you mean? (6 items named Jimmy Adams)
   - fits: Q2789496 (human; recorded): place of birth: Troon; place of death: London; sex or gender: male; country of citizenship: United Kingdom
   - fits: Q3763931 (human; recorded): place of birth: Port Maria La luz; sex or gender: male; country of citizenship: South Africa; occupation: cricketer
   - fits: Q6199479 (human; recorded): place of birth: Stoke-on-Trent; place of death: Stoke-on-Trent; sex or gender: male; country of citizenship: United Kingdom
   - fits: Q6199482 (human; recorded): place of birth: Norfolk; sex or gender: male; country of citizenship: United States; occupation: engineer
   - fits: Q75475333 (human; recorded): sex or gender: male; father: Harold Adams; mother: Coco Lequerre; family name: Adams
   - fits: Q137150557 (human; recorded): sex or gender: male; date of birth: 1929; family name: Adams; given name: Jimmy

## Counts

```
{
 "audit_outcomes": {
  "asking back": 23,
  "each item labelled": 13,
  "nothing read": 4,
  "one answer, every item agrees": 6,
  "one item": 4
 },
 "probe_verdicts": {
  "asking back": 1,
  "asking back, the target among them": 11,
  "each item labelled, the target among them": 6,
  "picks the target": 2
 },
 "crowded": 9,
 "decision_ms": {
  "p50": 38.2,
  "max": 230.9,
  "n": 75
 }
}
```
