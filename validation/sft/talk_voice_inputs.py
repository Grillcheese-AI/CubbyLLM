"""talk_voice_inputs -- what the person says in a voice row, and the labels of a French facts block.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

Everything here is an INPUT of a training row, written by rule: the person's side of a greeting, a thanks, a
goodbye; the ways a question about a fact or a wrong assumption is asked; first names. No target is taken from
this file. `FR_LABEL` is Cowork's table of French labels for the block's relations (Wikidata's French label for
the property where it has one), so a French chat is handed a French block; a relation without an entry is left
out of the French rows.
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

SAYS = {
    "en": {
        "greet": ["hi", "hey", "hello", "hello!", "hi there", "hey cubby", "hello cubby", "hi cubby!", "yo",
                  "good day", "what's up?", "hey there", "hiya", "howdy"],
        "greet_morning": ["good morning", "morning!", "morning cubby", "good morning cubby"],
        "greet_evening": ["good evening", "evening!", "good evening cubby"],
        "greet_back": ["I'm back", "hey, me again", "back again", "it's me again", "hi again"],
        "small_welcome": ["thanks", "thank you", "thanks a lot", "thx", "thanks, that helps", "great, thank you!",
                          "appreciate it", "that was helpful, thanks", "cheers", "perfect, thanks", "ty"],
        "small_sorry": ["that's not what I asked", "you misunderstood me", "no, that's not it",
                        "that doesn't answer my question", "not what I meant", "you missed my point"],
        "small_thanks": ["you're great", "nice job", "good answer", "you're pretty good at this", "well done",
                         "that was a great answer", "I like you", "you're funny", "good bot"],
        "small_forgive": ["sorry", "sorry, my bad", "oops, I typed that wrong", "sorry for the confusion",
                          "my mistake", "sorry, I wasn't clear"],
        "small_slow": ["that took a while", "you're slow today", "took you long enough", "why so slow?"],
        "cheer_stuck": ["I'm stuck", "this is hard", "I can't figure this out", "I've been at this for hours",
                        "ugh, nothing works", "I'm about to give up"],
        "cheer_curious": ["I'm working on something", "I'm starting a new project", "I've got a project going",
                          "I'm studying for an exam", "I'm doing some research", "I've got homework to finish"],
        "cheer_praise": ["I think I figured it out", "I had an idea", "I solved it!", "I think I've got it",
                         "I found another way to do it"],
        "bye": ["bye", "goodbye", "see you", "see ya later", "gotta go", "that's all, bye!", "talk later",
                "ok bye", "I'm off, see you", "that's it for today"],
        "bye_night": ["good night", "night!", "I'm off to bed", "going to sleep, bye"],
        "pun": ["tell me a joke", "got a pun?", "make me laugh", "say something funny", "do you know any puns?",
                "joke please", "a cheese joke?", "one more joke", "give me a pun"],
        "pastlife": ["tell me about your past life", "what were you up to in the 1700s?",
                     "any memories from the 1700s?", "what did you do in your past life?",
                     "tell me a story from your past life", "what was your past life like?"],
    },
    "fr": {
        "greet": ["salut", "bonjour", "coucou", "allô", "hey", "bonjour cubby", "salut cubby", "ça va ?",
                  "yo", "salut !", "bonjour !", "coucou cubby", "hello"],
        "greet_morning": ["bon matin", "bonjour, bonne matinée", "bon matin cubby"],
        "greet_evening": ["bonsoir", "bonsoir cubby", "bonne soirée !"],
        "greet_back": ["me revoilà", "c'est encore moi", "je suis de retour", "rebonjour", "re"],
        "small_welcome": ["merci", "merci beaucoup", "merci !", "super, merci", "ça m'aide, merci",
                          "parfait, merci", "merci bien", "génial, merci", "merci pour ton aide", "top, merci"],
        "small_sorry": ["ce n'est pas ce que j'ai demandé", "tu m'as mal compris", "non, ce n'est pas ça",
                        "ça ne répond pas à ma question", "ce n'est pas ce que je voulais dire"],
        "small_thanks": ["tu es génial", "bien joué", "bonne réponse", "tu es pas mal doué",
                         "bravo", "c'était une super réponse", "je t'aime bien", "tu es drôle"],
        "small_forgive": ["désolé", "pardon, c'est ma faute", "oups, j'ai mal écrit",
                          "désolé pour la confusion", "pardon, je n'étais pas clair"],
        "small_slow": ["ça a pris du temps", "tu es lent aujourd'hui", "c'était long", "pourquoi si lent ?"],
        "cheer_stuck": ["je suis bloqué", "c'est difficile", "je n'y arrive pas",
                        "ça fait des heures que je cherche", "rien ne marche", "je vais abandonner"],
        "cheer_curious": ["je travaille sur quelque chose", "je commence un nouveau projet", "j'ai un projet en cours",
                          "j'étudie pour un examen", "je fais des recherches", "j'ai un devoir à finir"],
        "cheer_praise": ["je crois que j'ai trouvé", "j'ai eu une idée", "j'ai réussi !",
                         "je pense que je l'ai", "j'ai trouvé une autre façon de faire"],
        "bye": ["bye", "au revoir", "à plus", "à la prochaine", "bonne journée", "je dois y aller",
                "à bientôt", "c'est tout, merci, salut", "bon, j'y vais", "ciao"],
        "bye_night": ["bonne nuit", "je vais me coucher", "allez, dodo, salut"],
        "pun": ["raconte-moi une blague", "t'as un jeu de mots ?", "fais-moi rire", "dis quelque chose de drôle",
                "une blague ?", "une blague de fromage ?", "encore une blague", "un petit jeu de mots ?"],
        "pastlife": ["raconte-moi ta vie antérieure", "tu faisais quoi dans les années 1700 ?",
                     "des souvenirs des années 1700 ?", "c'était comment, ta vie d'avant ?",
                     "raconte-moi une histoire de ton ancienne vie", "tu faisais quoi dans ta vie antérieure ?"],
    },
}
HELLO = {   # (what the person says, does it also ask what Cubby can do)
    "en": [("hi, my name is {u}", False), ("hey I'm {u}", False), ("hello, {u} here", False),
           ("my name's {u}", False), ("hi cubby, I'm {u}", False), ("I'm {u}, nice to meet you", False),
           ("hi, call me {u}", False), ("hello! I'm {u}", False),
           ("Hi! my name is {u}, what can you do for me?", True), ("hello, I'm {u}. what can you do?", True),
           ("hey, {u} here. what do you do?", True), ("hi I'm {u}, how can you help me?", True)],
    "fr": [("salut, moi c'est {u}", False), ("bonjour, je m'appelle {u}", False), ("coucou, c'est {u}", False),
           ("moi c'est {u}, enchanté", False), ("salut cubby, c'est {u}", False), ("allô, ici {u}", False),
           ("bonjour ! je suis {u}", False),
           ("salut ! je m'appelle {u}, tu peux faire quoi pour moi ?", True),
           ("bonjour, je suis {u}. qu'est-ce que tu sais faire ?", True),
           ("coucou, moi c'est {u}. tu fais quoi au juste ?", True)],
}
NAMES = ("nick alex sam emma liam olivia noah ava lucas mia ethan chloe leo zoe max lea hugo jade louis alice jules "
         "rose arthur anna adam ines paul lina tom sarah theo eva nathan clara simon julie marc sophie pierre marie "
         "antoine camille maxime laura julien manon thomas oceane david nadia karim sofia omar yasmine diego lucia "
         "mateo elena ivan nina oleg vera kenji yuki wei mei arjun priya kofi amara felix nora oscar ruby jack lily "
         "ben kate dan amy joe eve ian maya ryan tess cole iris marie-eve jean-luc anne-sophie pier-luc").split()
ID_PREFIX = {"en": ["", "", "", "hey, ", "hey cubby, ", "cubby, ", "so, ", "ok ", "quick question: ", "tell me, "],
             "fr": ["", "", "", "dis, ", "hé cubby, ", "cubby, ", "alors, ", "ok ", "petite question : ", "dis-moi, "]}
GREETED = {"en": ("hi", "Hey! What's on your mind today?"),
           "fr": ("salut", "Salut ! Qu'est-ce qui t'amène aujourd'hui ?")}

# questions about Cubby when its record is handed over as a facts block
RECORD_Q = {
    "lang": {"en": ["what languages do you speak?", "which languages do you speak?", "what languages can you use?"],
             "fr": ["quelles langues parles-tu ?", "tu parles quelles langues ?", "quelles langues connais-tu ?"]},
    "can": {"en": ["what can you do?", "what can you do for me?", "what are you good at?", "how can you help me?"],
            "fr": ["que peux-tu faire ?", "qu'est-ce que tu peux faire pour moi ?", "tu sais faire quoi ?"]},
    "who": {"en": ["who made you?", "who built you?", "who created you?"],
            "fr": ["qui t'a créé ?", "qui t'a fait ?", "qui est derrière toi ?"]},
}
RECORD_LABEL = {"en": {"creator": "creator", "role": "role", "lang": "language", "next": "next language", "can": "can"},
                "fr": {"creator": "créateur", "role": "rôle", "lang": "langue", "next": "prochaine langue",
                       "can": "sait faire"}}
LANG_NAME = {"en": {"English": "English", "French": "French", "Russian": "Russian", "Spanish": "Spanish"},
             "fr": {"English": "anglais", "French": "français", "Russian": "russe", "Spanish": "espagnol"}}
LANG_ORDER = {"en": ["English", "French", "Russian", "Spanish"], "fr": ["French", "English", "Russian", "Spanish"]}

FR_LIST_Q = ["Que sais-tu sur {e} ?", "Parle-moi {de_e}.", "Dis-m'en plus sur {e}.", "C'est quoi, {e} ?",
             "Qu'est-ce que tu as sur {e} ?", "Tu as quoi sur {e} ?", "Présente-moi {e}.",
             "Fais-moi le portrait {de_e}."]
FR_REL_Q = ["{Quel} est {art}{r} {de_e} ?", "C'est quoi, {art}{r} {de_e} ?", "Tu connais {art}{r} {de_e} ?",
            "Donne-moi {art}{r} {de_e}.", "{Art}{r} {de_e}, c'est quoi ?"]
EN_FALSE_Q = ["Is it true that the {r} of {e} is {x}?", "I read that the {r} of {e} is {x}. Is that right?",
              "Why is {x} the {r} of {e}?", "{x} is the {r} of {e}, right?", "So the {r} of {e} is {x}?",
              "Can you confirm that the {r} of {e} is {x}?", "Isn't {x} the {r} of {e}?"]
FR_FALSE_Q = ["Est-ce vrai que {art}{r} {de_e}, c'est {x} ?", "J'ai lu que {art}{r} {de_e} est {x}. C'est vrai ?",
              "Tu confirmes que {art}{r} {de_e} est {x} ?", "{Art}{r} {de_e}, c'est bien {x} ?",
              "Pourquoi {x} comme {r} {de_e} ?"]

# English label -> (French label, gender for a question with an article, or None = lists only)
FR_LABEL = {
    "type": ("type", "m"), "group": ("groupe", "m"), "location": ("lieu", "m"), "spouse": ("conjoint", "m"),
    "creation": ("création", "f"), "birth date": ("date de naissance", "f"), "position": ("fonction", "f"),
    "occupation": ("occupation", "f"), "creator": ("créateur", "m"), "birthplace": ("lieu de naissance", "m"),
    "work": ("œuvre", "f"), "member": ("membre", "m"), "effect": ("effet", "m"), "parent": ("parent", "m"),
    "performer": ("interprète", "m"), "influence": ("influence", "f"), "alma mater": ("alma mater", "f"),
    "field of work": ("domaine d'activité", "m"), "predecessor": ("prédécesseur", "m"),
    "award": ("distinction", "f"), "death date": ("date de décès", "f"), "employer": ("employeur", "m"),
    "cause": ("cause", "f"), "successor": ("successeur", "m"), "author": ("auteur", "m"),
    "founder": ("fondateur", "m"), "use": ("usage", "m"), "inspiration": ("inspiration", "f"),
    "child": ("enfant", "m"), "relative": ("proche", "m"), "instance": ("exemple", "m"),
    "basis": ("fondement", "m"), "nationality": ("nationalité", "f"), "capital": ("capitale", "f"),
    "ruler": ("dirigeant", "m"), "place of death": ("lieu de décès", "m"),
    "position held": ("fonction occupée", "f"), "located in": ("emplacement", "m"),
    "participant": ("participant", "m"), "category": ("catégorie", "f"), "region": ("région", "f"),
    "year": ("année", "f"), "start year": ("année de début", "f"), "end year": ("année de fin", "f"),
    "date": ("date", "f"), "instance of": ("nature de l'élément", "f"), "country": ("pays", "m"),
    "country of citizenship": ("pays de nationalité", "m"),
    "located in the administrative territorial entity": ("localisation administrative", "f"),
    "given name": ("prénom", "m"), "genre": ("genre", "m"), "educated at": ("scolarité", "f"),
    "country of origin": ("pays d'origine", "m"), "located in time zone": ("fuseau horaire", "m"),
    "parent taxon": ("taxon supérieur", "m"),
    "languages spoken, written or signed": ("langues parlées, écrites ou signées", None),
    "member of sports team": ("équipe sportive", "f"), "member of": ("membre de", None),
    "participant of": ("participant à", None), "cast member": ("distribution", "f"),
    "member of political party": ("parti politique", "m"), "director": ("réalisateur", "m"),
    "headquarters location": ("siège", "m"), "composer": ("compositeur", "m"), "father": ("père", "m"),
    "mother": ("mère", "f"), "family name": ("nom de famille", "m"), "cause of death": ("cause de la mort", "f"),
    "continent": ("continent", "m"), "official language": ("langue officielle", "f"),
    "language of work or name": ("langue de l'œuvre", "f"), "conflict": ("conflit", "m"),
    "diplomatic relation": ("relation diplomatique", "f"),
    "director of photography": ("directeur de la photographie", "m"),
}
# labels that read as "the <label> of X" in an English question
EN_NOUN = {k for k in FR_LABEL if k not in {
    "instance of", "educated at", "located in", "member of", "located in the administrative territorial entity",
    "located in time zone", "participant of", "position held", "languages spoken, written or signed",
    "member of sports team", "member of political party", "language of work or name"}}
FR_NOUN = {k for k, (_, gender) in FR_LABEL.items() if gender}
