"""Order 0067, the reader's second round (Nick, 2026-11-13: "go ahead"): more wordings a kind.

Round 1's reader learned its 3 training wordings a kind almost exactly (dev 149/150) and read new ones partly (100/150,
82.3% agreement). Round 2's training wordings for each kind and language: round 1's ``train`` (3), round 1's ``new``
(2; read once, so spent as a test and free to train on), and ``MORE`` here (5). Its new-wording split is ``NEW`` here
(3 a kind and language), never trained on and not read before. Same slots as `rule_templates.py`; the comparisons of
the new rows are still worded from ``REL_NEW``. Drafted by hand for this order (the 2026-11-10 rule allows wordings
to be model-drafted; no rule is); every ``MORE`` wording is checked against IFEval by `bench_overlap.py` (a test).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

MORE = {
    "number_words": {
        "en": ["Hold the reply to {rel} {n} words.", "I'd like {rel} {n} words from you.", "Word budget: {rel} {n}.",
               "Write something {rel} {n} words long.", "Respond using {rel} {n} words."],
        "fr": ["Garde la réponse à {rel} {n} mots.", "Je veux {rel} {n} mots.", "Budget de mots : {rel} {n}.",
               "Écris quelque chose de {rel} {n} mots.", "Réponds avec {rel} {n} mots."]},
    "number_sentences": {
        "en": ["Use {rel} {n} sentences to answer.", "Sentence count: {rel} {n}.", "Say it in {rel} {n} sentences.",
               "I want {rel} {n} sentences back.", "Compose a reply of {rel} {n} sentences."],
        "fr": ["Utilise {rel} {n} phrases pour répondre.", "Nombre de phrases : {rel} {n}.", "Dis-le en {rel} {n} phrases.",
               "Je veux {rel} {n} phrases.", "Compose une réponse de {rel} {n} phrases."]},
    "number_paragraphs": {
        "en": ["Give me {n} paragraphs, each one parted from the next by ***.",
               "Split the reply into {n} paragraphs with *** dividing them.",
               "{n} paragraphs, please, with a *** line between neighbours.",
               "Make it {n} paragraphs long and mark each break with ***.", "Lay the answer out as {n} paragraphs joined by ***."],
        "fr": ["Donne-moi {n} paragraphes, séparés les uns des autres par ***.",
               "Découpe la réponse en {n} paragraphes avec *** pour les diviser.",
               "{n} paragraphes, avec une ligne *** entre voisins.", "Fais {n} paragraphes et marque chaque coupure par ***.",
               "Présente la réponse en {n} paragraphes reliés par ***."]},
    "multiple_sections": {
        "en": ["Write {n} sections, each headed {marker} plus a number.",
               "Arrange the answer as {n} sections titled {marker} 1, {marker} 2, and onward.",
               "Use {n} numbered sections; each begins with {marker}.", "Number your {n} sections with the label {marker}.",
               "Give {n} sections, every one opened by {marker} and its index."],
        "fr": ["Écris {n} sections, chacune titrée {marker} plus un numéro.",
               "Range la réponse en {n} sections nommées {marker} 1, {marker} 2, et ainsi de suite.",
               "Utilise {n} sections numérotées ; chacune commence par {marker}.",
               "Numérote tes {n} sections avec l'étiquette {marker}.", "Donne {n} sections, chacune ouverte par {marker} et son indice."]},
    "number_highlighted_sections": {
        "en": ["Mark {n} or more key phrases in *italics*.", "Emphasise {n} or more parts by putting them between asterisks.",
               "Highlight {n} or more segments like *so*.", "Use single asterisks to stress {n} or more phrases.",
               "Give {n} or more *highlighted* bits."],
        "fr": ["Mets {n} expressions clés ou plus en *italique*.", "Fais ressortir {n} parties ou plus en les mettant entre astérisques.",
               "Surligne {n} segments ou plus *comme ceci*.", "Utilise des astérisques simples pour souligner {n} expressions ou plus.",
               "Donne {n} passages *surlignés* ou plus."]},
    "title": {
        "en": ["Give the reply a title in << >>.", "Open with a heading inside double angle brackets.",
               "Title your answer, writing it as <<title>>.", "Put a <<title>> above the answer.", "Add a title between double chevrons."],
        "fr": ["Donne un titre à la réponse entre << >>.", "Ouvre avec un intitulé entre doubles chevrons.",
               "Titre ta réponse, écrit comme <<titre>>.", "Place un <<titre>> au-dessus de la réponse.", "Ajoute un titre entre doubles chevrons."]},
    "json_format": {
        "en": ["Answer as JSON.", "Output valid JSON and nothing more.", "Give me the reply as a JSON object.",
               "Write the answer in JSON.", "Use JSON for the whole reply."],
        "fr": ["Réponds en JSON.", "Produis du JSON valide et rien de plus.", "Donne-moi la réponse sous forme d'objet JSON.",
               "Écris la réponse en JSON.", "Utilise le JSON pour toute la réponse."]},
    "english_capital": {
        "en": ["Reply in English, all in uppercase.", "Write in English, capitals only.", "English, and every letter uppercase.",
               "Shout it in English: all caps.", "Use English in block capitals."],
        "fr": ["Réponds en anglais, tout en capitales.", "Écris en anglais, en majuscules seulement.",
               "En anglais, chaque lettre en capitale.", "Crie-le en anglais : tout en majuscules.", "Utilise l'anglais en lettres capitales."]},
    "english_lowercase": {
        "en": ["Reply in English, no capital letters anywhere.", "Write in English, lowercase only.",
               "English, and not one uppercase letter.", "Use English in small letters only, no capitals.",
               "Answer in English and keep it all lowercase."],
        "fr": ["Réponds en anglais, sans majuscule nulle part.", "Écris en anglais, en minuscules seulement.",
               "En anglais, et pas une seule lettre majuscule.", "Utilise l'anglais en lettres minuscules, sans capitales.",
               "Réponds en anglais et garde tout en minuscules."]},
    "response_language": {
        "en": ["Answer in {lang}.", "Use {lang} for this reply.", "Reply in {lang} only.", "I want the response in {lang}.",
               "Please write in {lang}."],
        "fr": ["Réponds en {lang}.", "Utilise le {lang} pour cette réponse.", "Réponds en {lang} uniquement.",
               "Je veux la réponse en {lang}.", "Écris en {lang}, s'il te plaît."]},
    "existence": {
        "en": ["Mention {kw} in your answer.", "The word {kw} has to be in the reply.", "Use {kw} somewhere in what you say.",
               "Don't leave out the word {kw}.", "Be sure {kw} shows up."],
        "fr": ["Mentionne {kw} dans ta réponse.", "Le mot {kw} doit être dans la réponse.", "Emploie {kw} quelque part.",
               "N'oublie pas le mot {kw}.", "Fais en sorte que {kw} apparaisse."]},
    "frequency": {
        "en": ["Say {kw} {rel} {n} times.", "Use {kw} {rel} {n} times overall.", "The word {kw}: {rel} {n} times.",
               "Write {kw} {rel} {n} times.", "Repeat {kw} {rel} {n} times."],
        "fr": ["Dis {kw} {rel} {n} fois.", "Emploie {kw} {rel} {n} fois dans la réponse.", "Le mot {kw} : {rel} {n} fois.",
               "Écris {kw} {rel} {n} fois.", "Répète {kw} {rel} {n} fois."]},
    "letter_frequency": {
        "en": ["Write the letter {letter} {rel} {n} times in all.", "The letter {letter} should turn up {rel} {n} times.",
               "Count of letter {letter}: {rel} {n}.", "Have {rel} {n} occurrences of the letter {letter}.",
               "Employ the letter {letter} {rel} {n} times."],
        "fr": ["Écris la lettre {letter} {rel} {n} fois en tout.", "La lettre {letter} doit se présenter {rel} {n} fois.",
               "Nombre de lettres {letter} : {rel} {n}.", "Mets {rel} {n} occurrences de la lettre {letter}.",
               "Emploie la lettre {letter} {rel} {n} fois."]},
    "repeat_prompt": {
        "en": ["Repeat the request exactly, then answer it: {req}", "Start by copying my request unchanged, then reply: {req}",
               "Quote the request verbatim first, then answer: {req}", "Before answering, restate the request unchanged: {req}",
               "Echo the request first, exactly as written, then respond: {req}"],
        "fr": ["Répète la demande telle quelle, puis réponds : {req}",
               "Commence par recopier ma demande sans la changer, puis réponds : {req}",
               "Cite la demande à l'identique d'abord, puis réponds : {req}",
               "Avant de répondre, redis la demande sans rien changer : {req}",
               "Reprends la demande d'abord, exactement comme écrite, puis réponds : {req}"]},
    "two_responses": {
        "en": ["Write two different replies separated by ******.", "Give me two distinct answers, with ****** between them.",
               "Answer two ways, and put ****** between the two.", "Two different answers, please, divided by ******.",
               "Provide a pair of distinct responses split by a ****** line."],
        "fr": ["Écris deux réponses différentes séparées par ******.", "Donne-moi deux réponses distinctes, avec ****** entre elles.",
               "Réponds de deux façons, et mets ****** entre les deux.", "Deux réponses différentes, séparées par ******.",
               "Fournis une paire de réponses distinctes coupée par une ligne ******."]},
    "quotation": {
        "en": ["Put the whole answer in double quotes.", "Surround the reply with a pair of double quotes.",
               'Your reply should start and end with ".', 'Enclose the answer in " ".', "Quote the entire reply in double quotes."],
        "fr": ['Mets toute la réponse entre guillemets doubles ".', 'Entoure la réponse de guillemets doubles ".',
               'Ta réponse doit commencer et finir par ".', 'Encadre la réponse par " ".', 'Cite toute la réponse entre guillemets doubles ".']},
    "no_markdown": {
        "en": ["No markdown formatting, please.", "Write plain text without markdown.", "Skip any markdown.",
               "Don't format with markdown.", "Avoid markdown entirely."],
        "fr": ["Pas de mise en forme markdown, s'il te plaît.", "Écris du texte brut sans markdown.", "Laisse tomber le markdown.",
               "Ne formate pas en markdown.", "Évite complètement le markdown."]},
    "first": {
        "en": ["Open your reply with {x}.", "The answer must begin with {x}.", "Start with the words: {x}.",
               'Your reply begins "{x}".', "Lead off with {x}."],
        "fr": ["Ouvre ta réponse par {x}.", "La réponse doit commencer par {x}.", "Commence par les mots : {x}.",
               "Ta réponse commence par « {x} ».", "Démarre avec {x}."]},
}

NEW = {
    "number_words": {"en": ["Stay within a length of {rel} {n} words.", "The answer should run to {rel} {n} words.",
                            "Give me a reply that counts {rel} {n} words."],
                     "fr": ["Reste dans une longueur de {rel} {n} mots.", "La réponse doit compter {rel} {n} mots.",
                            "Donne-moi une réponse de {rel} {n} mots."]},
    "number_sentences": {"en": ["Express this in {rel} {n} sentences.", "Your text should hold {rel} {n} sentences.",
                                "Answer with {rel} {n} sentences, please."],
                         "fr": ["Exprime ça en {rel} {n} phrases.", "Ton texte doit contenir {rel} {n} phrases.",
                                "Réponds avec {rel} {n} phrases, s'il te plaît."]},
    "number_paragraphs": {"en": ["Produce {n} blocks of text, putting the marker *** where one ends and the next begins.",
                                 "Answer in {n} paragraphs; *** goes between consecutive ones.",
                                 "Use {n} paragraphs and divide them using three asterisks (***)."],
                          "fr": ["Produis {n} blocs de texte, avec le marqueur *** là où l'un finit et l'autre commence.",
                                 "Réponds en {n} paragraphes ; *** va entre deux paragraphes qui se suivent.",
                                 "Utilise {n} paragraphes et sépare-les avec trois astérisques (***)."]},
    "multiple_sections": {"en": ["Format the reply as {n} sections, marking each start with {marker} and a digit.",
                                 "Create {n} labelled sections ({marker} 1, {marker} 2, ...).",
                                 "Your reply needs {n} sections, introduced by {marker} N."],
                          "fr": ["Mets la réponse en forme de {n} sections, chaque début marqué par {marker} et un chiffre.",
                                 "Crée {n} sections étiquetées ({marker} 1, {marker} 2, ...).",
                                 "Ta réponse doit avoir {n} sections, introduites par {marker} N."]},
    "number_highlighted_sections": {"en": ["Make sure {n} or more passages are wrapped in *single asterisks*.",
                                           "Italicise {n} or more phrases with markdown asterisks.",
                                           "Stress {n} or more points using *this style*."],
                                    "fr": ["Assure-toi que {n} passages ou plus sont entourés d'*astérisques simples*.",
                                           "Mets en italique {n} expressions ou plus avec des astérisques markdown.",
                                           "Souligne {n} points ou plus avec *ce style*."]},
    "title": {"en": ["Begin with a name for your answer, written inside << and >>.",
                     "Your reply should carry a title framed by << >>.", "Label the answer with a title like <<this one>>."],
              "fr": ["Commence par un nom pour ta réponse, écrit entre << et >>.", "Ta réponse doit porter un titre encadré par << >>.",
                     "Étiquette la réponse avec un titre comme <<celui-ci>>."]},
    "json_format": {"en": ["Structure the full reply as JSON.", "Reply only with a JSON object, no prose around it.",
                           "I need machine-readable JSON as the answer."],
                    "fr": ["Structure toute la réponse en JSON.", "Réponds seulement avec un objet JSON, sans texte autour.",
                           "Il me faut du JSON lisible par une machine comme réponse."]},
    "english_capital": {"en": ["Answer in English with no lowercase letters.", "English only, written entirely in CAPS.",
                               "Make the reply English and fully uppercase."],
                        "fr": ["Réponds en anglais sans aucune minuscule.", "Anglais seulement, écrit entièrement en MAJUSCULES.",
                               "Fais une réponse en anglais, entièrement en majuscules."]},
    "english_lowercase": {"en": ["English please, with every letter in lowercase.", "Make the reply English and entirely lowercase.",
                                 "Write English text containing no uppercase letters."],
                          "fr": ["En anglais s'il te plaît, rien qu'en minuscules.", "Fais une réponse en anglais, entièrement en minuscules.",
                                 "Écris un texte anglais sans aucune lettre majuscule."]},
    "response_language": {"en": ["Switch to {lang} for your answer.", "Your reply must be written in {lang}.", "Respond solely in {lang}."],
                          "fr": ["Passe au {lang} pour ta réponse.", "Ta réponse doit être écrite en {lang}.", "Réponds seulement en {lang}."]},
    "existence": {"en": ["Include {kw} in the text.", "Your answer should contain the word {kw}.", "Somewhere, say {kw}."],
                  "fr": ["Inclus {kw} dans le texte.", "Ta réponse doit contenir le mot {kw}.", "Quelque part, dis {kw}."]},
    "frequency": {"en": ["Make the word {kw} appear {rel} {n} times.", "Let {kw} occur {rel} {n} times in your text.",
                         "Have the reply mention {kw} {rel} {n} times."],
                  "fr": ["Fais apparaître le mot {kw} {rel} {n} fois.", "Que {kw} revienne {rel} {n} fois dans ton texte.",
                         "Mentionne {kw} {rel} {n} fois dans la réponse."]},
    "letter_frequency": {"en": ["Make the letter {letter} occur {rel} {n} times.", "Include the letter {letter} {rel} {n} times.",
                                "Your text should contain the letter {letter} {rel} {n} times."],
                         "fr": ["Fais apparaître la lettre {letter} {rel} {n} fois.", "Inclus la lettre {letter} {rel} {n} fois.",
                                "Ton texte doit contenir la lettre {letter} {rel} {n} fois."]},
    "repeat_prompt": {"en": ["Write out the request again without changing a word, then give your reply: {req}",
                             "First copy this request exactly, then answer it: {req}",
                             "Restate the following request verbatim, then respond to it: {req}"],
                      "fr": ["Réécris la demande sans changer un mot, puis donne ta réponse : {req}",
                             "Recopie d'abord cette demande exactement, puis réponds-y : {req}",
                             "Redis la demande suivante mot à mot, puis réponds-y : {req}"]},
    "two_responses": {"en": ["Offer two separate answers that are not the same, separated with six asterisks (******).",
                             "Write two alternative replies and put ****** between them.", "Give two answers that differ, divided by ******."],
                      "fr": ["Propose deux réponses séparées qui ne sont pas pareilles, avec six astérisques (******) entre elles.",
                             "Écris deux réponses alternatives et mets ****** entre elles.", "Donne deux réponses qui diffèrent, divisées par ******."]},
    "quotation": {"en": ["Place your full answer between a pair of double quotation marks.",
                         'Wrap the answer, start to finish, in " marks.', "Make the reply one quoted string in double quotes."],
                  "fr": ['Place toute ta réponse entre une paire de guillemets doubles ".',
                         'Enveloppe la réponse, du début à la fin, dans des ".', 'Fais de la réponse une seule chaîne entre guillemets doubles ".']},
    "no_markdown": {"en": ["Use no markdown syntax at all.", "Keep the reply free of markdown.", "Plain text only; no markdown."],
                    "fr": ["N'utilise aucune syntaxe markdown.", "Garde la réponse libre de markdown.", "Texte brut seulement ; pas de markdown."]},
    "first": {"en": ["Make {x} the first words of your reply.", 'Kick off the answer with "{x}".', "Begin by saying {x}."],
              "fr": ["Fais de {x} les premiers mots de ta réponse.", "Lance la réponse avec « {x} ».", "Commence en disant {x}."]},
}


# round 1's new wordings were allowed close to IFEval (nothing trained on them); the ones that share 5 words with an
# IFEval prompt are reworded on their way into round 2's training (round 1's own rows stay as they were read)
REWORD = {"Your whole response should contain {rel} {n} words.": "The whole reply should hold {rel} {n} words.",
          "There should be {n} paragraphs; paragraphs are separated with the markdown divider: ***":
              "There should be {n} paragraphs, with *** standing between them.",
          "Say {kw} {rel} {n} times in the answer.": "Bring up {kw} {rel} {n} times over the answer."}


def templates(T: dict) -> dict:
    """Round 2's wordings from round 1's ``T``: train = T's train + T's new (reworded where `REWORD` says) + MORE;
    new = NEW."""
    return {k: {lang: {"train": T[k][lang]["train"] + [REWORD.get(w, w) for w in T[k][lang]["new"]] + MORE[k][lang],
                       "new": NEW[k][lang]} for lang in ("en", "fr")} for k in T}
