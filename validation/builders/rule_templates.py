"""Order 0067, step 3: the wordings of the seen kinds' rules, English and French (data for build_rule_rows.py).

Each kind: ``train`` wordings (rows to learn from and dev) and ``new`` wordings, held apart for line 3's "new
wordings of seen kinds" (no row built from them is trained on). Slots: {n}, {rel} (the comparison, worded per
language in REL), {kw}, {letter}, {lang}, {marker}, {req}, {x}. Written by hand for this order; no model wrote a rule.
Only the seen kinds of `validation/logs/rule_kinds_split_2026-11-12.json` are here: no held-out kind has a wording.
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

REL = {"en": {">=": ["at least", "no fewer than", "a minimum of"], "<": ["fewer than", "less than", "under"]},
       "fr": {">=": ["au moins", "pas moins de", "un minimum de"], "<": ["moins de", "en dessous de"]}}
LANG = {"en": {"en": "English", "fr": "French", "es": "Spanish"}, "fr": {"en": "anglais", "fr": "français", "es": "espagnol"}}
KEYWORDS = {"en": ["river", "garden", "history", "energy", "music", "budget", "planet", "coffee", "market", "winter",
                   "window", "travel", "doctor", "silence", "bridge"],
            "fr": ["rivière", "jardin", "histoire", "énergie", "musique", "budget", "planète", "café", "marché", "hiver",
                   "fenêtre", "voyage", "médecin", "silence", "pont"]}
REQUESTS = {"en": ["Explain why the sky is blue.", "Give me three tips for sleeping better.", "Describe a quiet morning.",
                   "What makes a good neighbour?", "Write a short note about the sea.", "Tell me how bread rises."],
            "fr": ["Explique pourquoi le ciel est bleu.", "Donne-moi trois conseils pour mieux dormir.",
                   "Décris un matin tranquille.", "Qu'est-ce qui fait un bon voisin ?", "Écris une courte note sur la mer.",
                   "Dis-moi comment le pain lève."]}
MARKERS = ["SECTION", "Section", "PART", "Part", "CHAPTER"]
OPENINGS = {"en": ["yes", "no", "sure", "in short", "first of all"], "fr": ["oui", "non", "bien sûr", "en bref", "tout d'abord"]}

T = {
    "number_words": {
        "en": {"train": ["Answer in {rel} {n} words.", "Keep your reply to {rel} {n} words.", "Use {rel} {n} words in total."],
               "new": ["Your whole response should contain {rel} {n} words.", "Make the reply {rel} {n} words long."]},
        "fr": {"train": ["Réponds en {rel} {n} mots.", "Ta réponse doit faire {rel} {n} mots.", "Utilise {rel} {n} mots en tout."],
               "new": ["L'ensemble de ta réponse doit compter {rel} {n} mots.", "Écris {rel} {n} mots, pas autrement."]}},
    "number_sentences": {
        "en": {"train": ["Answer in {rel} {n} sentences.", "Write {rel} {n} sentences.", "Your reply must have {rel} {n} sentences."],
               "new": ["Limit yourself to a response of {rel} {n} sentences.", "Use {rel} {n} full sentences, no matter what."]},
        "fr": {"train": ["Réponds en {rel} {n} phrases.", "Écris {rel} {n} phrases.", "Ta réponse doit avoir {rel} {n} phrases."],
               "new": ["Tiens-toi à une réponse de {rel} {n} phrases.", "Formule {rel} {n} phrases complètes."]}},
    "number_paragraphs": {
        "en": {"train": ["Write exactly {n} paragraphs, separated by the markdown divider ***.",
                         "Give {n} paragraphs, with *** between each one.", "Your answer has {n} paragraphs split by ***."],
               "new": ["There should be {n} paragraphs; paragraphs are separated with the markdown divider: ***",
                       "Organise the reply as {n} paragraphs, putting *** between consecutive ones."]},
        "fr": {"train": ["Écris exactement {n} paragraphes, séparés par le séparateur markdown ***.",
                         "Donne {n} paragraphes, avec *** entre chacun.", "Ta réponse a {n} paragraphes séparés par ***."],
               "new": ["Il doit y avoir {n} paragraphes ; on les sépare avec le séparateur markdown : ***",
                       "Organise la réponse en {n} paragraphes, en mettant *** entre deux paragraphes."]}},
    "multiple_sections": {
        "en": {"train": ["Your response must have {n} sections. Mark the beginning of each with {marker} X.",
                         "Split the answer into {n} sections, each starting with {marker} and its number.",
                         "Write {n} sections, labelled {marker} 1, {marker} 2 and so on."],
               "new": ["Divide what you write into {n} parts, opening each part with {marker} followed by its number.",
                       "I need {n} numbered sections; begin each one with the word {marker} and a number."]},
        "fr": {"train": ["Ta réponse doit avoir {n} sections. Marque le début de chacune avec {marker} X.",
                         "Découpe la réponse en {n} sections, chacune commençant par {marker} et son numéro.",
                         "Écris {n} sections, notées {marker} 1, {marker} 2 et ainsi de suite."],
               "new": ["Divise ce que tu écris en {n} parties, chacune ouverte par {marker} suivi de son numéro.",
                       "Il me faut {n} sections numérotées ; commence chacune par le mot {marker} et un numéro."]}},
    "number_highlighted_sections": {
        "en": {"train": ["Highlight at least {n} sections with markdown, i.e. *highlighted section*.",
                         "Use markdown to highlight {n} or more parts, like *this*.", "Mark at least {n} passages as *highlighted*."],
               "new": ["Put at least {n} spans of your answer in italics with single asterisks.",
                       "Emphasise {n} or more phrases using *asterisks* around them."]},
        "fr": {"train": ["Mets en évidence au moins {n} sections en markdown, par exemple *section en évidence*.",
                         "Utilise le markdown pour surligner {n} parties ou plus, comme *ceci*.",
                         "Marque au moins {n} passages comme *surlignés*."],
               "new": ["Mets au moins {n} segments de ta réponse en italique avec des astérisques simples.",
                       "Fais ressortir {n} expressions ou plus en les entourant d'*astérisques*."]}},
    "title": {
        "en": {"train": ["Give your answer a title wrapped in double angular brackets, such as <<my title>>.",
                         "Include a title in << >>.", "Start with a title written like <<title>>."],
               "new": ["Your reply needs a heading enclosed in double angle brackets.", "Put a title between << and >>."]},
        "fr": {"train": ["Donne un titre à ta réponse entre doubles chevrons, comme <<mon titre>>.",
                         "Inclus un titre entre << >>.", "Commence par un titre écrit comme <<titre>>."],
               "new": ["Ta réponse doit porter un titre encadré de doubles chevrons.", "Mets un titre entre << et >>."]}},
    "json_format": {
        "en": {"train": ["Your entire output should be wrapped in JSON format.", "Reply in JSON only.", "Format the whole answer as JSON."],
               "new": ["Return nothing but a JSON object.", "Everything you write must be valid JSON."]},
        "fr": {"train": ["Toute ta sortie doit être au format JSON.", "Réponds uniquement en JSON.", "Formate toute la réponse en JSON."],
               "new": ["Ne renvoie rien d'autre qu'un objet JSON.", "Tout ce que tu écris doit être du JSON valide."]}},
    "english_capital": {
        "en": {"train": ["Your entire response should be in English, and in all capital letters.", "Reply in English, all in capitals.",
                         "Write in English using only uppercase letters."],
               "new": ["Answer in English with every letter capitalised.", "Use English and CAPITAL LETTERS throughout."]},
        "fr": {"train": ["Toute ta réponse doit être en anglais, et entièrement en majuscules.", "Réponds en anglais, tout en majuscules.",
                         "Écris en anglais avec uniquement des lettres majuscules."],
               "new": ["Réponds en anglais avec chaque lettre en capitale.", "Utilise l'anglais et des MAJUSCULES partout."]}},
    "english_lowercase": {
        "en": {"train": ["Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
                         "Reply in English, all in lowercase.", "Write in English using only lowercase letters."],
               "new": ["Answer in English without a single capital letter.", "Use English, and keep every letter small."]},
        "fr": {"train": ["Toute ta réponse doit être en anglais, et entièrement en minuscules.", "Réponds en anglais, tout en minuscules.",
                         "Écris en anglais avec uniquement des lettres minuscules."],
               "new": ["Réponds en anglais sans une seule majuscule.", "Utilise l'anglais, et garde toutes les lettres en minuscule."]}},
    "response_language": {
        "en": {"train": ["Your entire response should be in {lang}, no other language is allowed.", "Reply only in {lang}.",
                         "Write the answer in {lang}."],
               "new": ["Use {lang} and nothing else for the whole reply.", "Whatever I write, you answer in {lang}."]},
        "fr": {"train": ["Toute ta réponse doit être en {lang}, aucune autre langue n'est permise.", "Réponds uniquement en {lang}.",
                         "Écris la réponse en {lang}."],
               "new": ["Utilise le {lang} et rien d'autre pour toute la réponse.", "Quoi que j'écrive, réponds en {lang}."]}},
    "existence": {
        "en": {"train": ["Include the keyword {kw} in the response.", "Make sure the word {kw} appears.", "Use the word {kw} somewhere."],
               "new": ["Your reply has to mention {kw}.", "Work the word {kw} into what you write."]},
        "fr": {"train": ["Inclus le mot-clé {kw} dans la réponse.", "Assure-toi que le mot {kw} apparaisse.", "Utilise le mot {kw} quelque part."],
               "new": ["Ta réponse doit mentionner {kw}.", "Glisse le mot {kw} dans ce que tu écris."]}},
    "frequency": {
        "en": {"train": ["In your response, the word {kw} should appear {rel} {n} times.", "Use the word {kw} {rel} {n} times.",
                         "The word {kw} must occur {rel} {n} times."],
               "new": ["Say {kw} {rel} {n} times in the answer.", "Count your uses of {kw}: {rel} {n}."]},
        "fr": {"train": ["Dans ta réponse, le mot {kw} doit apparaître {rel} {n} fois.", "Utilise le mot {kw} {rel} {n} fois.",
                         "Le mot {kw} doit figurer {rel} {n} fois."],
               "new": ["Dis {kw} {rel} {n} fois dans la réponse.", "Compte tes emplois de {kw} : {rel} {n}."]}},
    "letter_frequency": {
        "en": {"train": ["In your response, the letter {letter} should appear {rel} {n} times.", "Use the letter {letter} {rel} {n} times.",
                         "The letter {letter} must occur {rel} {n} times."],
               "new": ["Let the letter {letter} show up {rel} {n} times in what you write.", "Count the letter {letter}: {rel} {n}."]},
        "fr": {"train": ["Dans ta réponse, la lettre {letter} doit apparaître {rel} {n} fois.", "Utilise la lettre {letter} {rel} {n} fois.",
                         "La lettre {letter} doit figurer {rel} {n} fois."],
               "new": ["Que la lettre {letter} revienne {rel} {n} fois dans ce que tu écris.", "Compte la lettre {letter} : {rel} {n}."]}},
    "repeat_prompt": {
        "en": {"train": ["First repeat the request word for word without change, then give your answer: {req}",
                         "Begin by restating this request exactly, then answer it: {req}", "Copy the request first, then reply: {req}"],
               "new": ["Open with the request exactly as written, and only then respond: {req}",
                       "Echo the following request verbatim before answering: {req}"]},
        "fr": {"train": ["Répète d'abord la demande mot pour mot sans changement, puis donne ta réponse : {req}",
                         "Commence par reprendre cette demande exactement, puis réponds : {req}",
                         "Recopie d'abord la demande, puis réponds : {req}"],
               "new": ["Ouvre avec la demande telle qu'elle est écrite, et ensuite seulement réponds : {req}",
                       "Reprends la demande suivante textuellement avant de répondre : {req}"]}},
    "two_responses": {
        "en": {"train": ["Give two different responses, separated by 6 asterisk symbols: ******.",
                         "Write two distinct answers with ****** between them.", "Provide two different replies split by ******."],
               "new": ["I want a pair of different answers; put ****** between the first and the second.",
                       "Answer twice, differently each time, and separate the two with six asterisks."]},
        "fr": {"train": ["Donne deux réponses différentes, séparées par 6 astérisques : ******.",
                         "Écris deux réponses distinctes avec ****** entre elles.", "Fournis deux réponses différentes séparées par ******."],
               "new": ["Je veux deux réponses différentes ; mets ****** entre la première et la seconde.",
                       "Réponds deux fois, différemment chaque fois, et sépare les deux par six astérisques."]}},
    "quotation": {
        "en": {"train": ["Wrap your entire response with double quotation marks.", "Put the whole reply in double quotes.",
                         "Enclose everything you write in double quotation marks."],
               "new": ["Your answer must begin and end with a double quote.", "Quote your entire reply with \" marks."]},
        "fr": {"train": ["Entoure toute ta réponse de guillemets droits doubles (\").", "Mets toute la réponse entre guillemets doubles \".",
                         "Encadre tout ce que tu écris par des guillemets doubles \"."],
               "new": ["Ta réponse doit commencer et finir par un guillemet double \".", "Cite toute ta réponse entre des \"."]}},
    "no_markdown": {
        "en": {"train": ["Do not use any markdown.", "Plain text only, no markdown formatting.", "No markdown in the reply."],
               "new": ["Avoid bold, headings, lists and code formatting.", "Write without any markup at all."]},
        "fr": {"train": ["N'utilise aucun markdown.", "Texte brut uniquement, sans mise en forme markdown.", "Pas de markdown dans la réponse."],
               "new": ["Évite le gras, les titres, les listes et le code.", "Écris sans aucun balisage."]}},
    "first": {
        "en": {"train": ["Start your reply with \"{x}\".", "Begin the answer with the words {x}.", "Your first words must be: {x}."],
               "new": ["Open by saying {x}.", "Lead with \"{x}\" before anything else."]},
        "fr": {"train": ["Commence ta réponse par « {x} ».", "Débute la réponse par les mots {x}.", "Tes premiers mots doivent être : {x}."],
               "new": ["Ouvre en disant {x}.", "Mets « {x} » avant toute autre chose."]}},
}
