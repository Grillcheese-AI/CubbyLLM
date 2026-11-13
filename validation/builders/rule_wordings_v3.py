"""Order 0067, the reader's third round (Nick, 2026-11-13: "yes build it"): the two gaps round 2 left.

Round 2 read fresh wordings at 92.7% agreement; its misses sat in all capitals (0.62: "no lowercase letters" read as
lowercase, CAPS / uppercase losing the case leaf) and letter counts (0.75: "less than", never in training, read as
">="). Round 3's training wordings: round 2's training list, round 2's new wordings (read once, now spent), and
``MORE`` here (aimed at saying capitals or lowercase without the word, each as the other's contrast); its comparisons
gain phrasings (``REL``; "no less than" was tried and left out: it carries IFEval's "less than"). Its new-wording split is ``NEW`` here
(3 a kind and language, fresh), with comparisons still worded from round 1's ``REL_NEW`` ("at least", "less than",
never in any training list). Drafted by hand; no rule is model-written.
"""
from __future__ import annotations

import rule_wordings_v2 as V2

__wiring__ = "STANDALONE"

REL = {"en": {">=": ["no fewer than", "a minimum of", "not under", "not fewer than", "not below"],
              "<": ["fewer than", "under", "below", "short of", "not as many as"]},
       "fr": {">=": ["au moins", "pas moins de", "un minimum de", "pas en dessous de", "au bas mot"],
              "<": ["moins de", "en dessous de", "pas autant que", "sous la barre des", "en nombre inférieur à"]}}

MORE = {
    "english_capital": {
        "en": ["Use English with no lowercase letters at all.", "English, in all caps.", "Write English in block letters only.",
               "Answer in English; every character that is a letter must be uppercase.", "In English, and in CAPS from start to end.",
               "Make it English and leave no small letters.", "English reply, uppercase throughout."],
        "fr": ["Utilise l'anglais sans aucune lettre minuscule.", "En anglais, tout en capitales d'imprimerie.",
               "Écris en anglais en lettres bâtons majuscules.", "Réponds en anglais ; toute lettre doit être majuscule.",
               "En anglais, et en MAJUSCULES du début à la fin.", "Fais-le en anglais sans laisser de minuscules.",
               "Réponse en anglais, majuscules partout."]},
    "english_lowercase": {
        "en": ["Use English with no uppercase letters at all.", "English, no caps.", "Write English without a single capital.",
               "Answer in English; every letter must be lowercase.", "In English, and in small letters from start to end.",
               "Make it English and leave no capital letters.", "English reply, lowercase throughout."],
        "fr": ["Utilise l'anglais sans aucune lettre majuscule.", "En anglais, sans aucune capitale.",
               "Écris en anglais sans une seule majuscule.", "Réponds en anglais ; toute lettre doit être minuscule.",
               "En anglais, et en minuscules du début à la fin.", "Fais-le en anglais sans laisser de majuscules.",
               "Réponse en anglais, minuscules partout."]},
    "letter_frequency": {
        "en": ["Have the letter {letter} appear {rel} {n} times.", "Let the letter {letter} show {rel} {n} times.",
               "Keep the count of the letter {letter} {rel} {n}."],
        "fr": ["Que la lettre {letter} figure {rel} {n} fois.", "Fais en sorte que la lettre {letter} paraisse {rel} {n} fois.",
               "Garde le nombre de lettres {letter} {rel} {n}."]},
    "frequency": {
        "en": ["Keep the count of the word {kw} {rel} {n}.", "The word {kw} should come up {rel} {n} times."],
        "fr": ["Garde le nombre de fois où paraît {kw} {rel} {n}.", "Le mot {kw} doit revenir {rel} {n} fois."]},
}

NEW = {
    "number_words": {"en": ["Aim for {rel} {n} words in your answer.", "Your reply's length: {rel} {n} words.",
                            "Use a word count of {rel} {n}."],
                     "fr": ["Vise {rel} {n} mots dans ta réponse.", "Longueur de ta réponse : {rel} {n} mots.",
                            "Emploie un total de {rel} {n} mots."]},
    "number_sentences": {"en": ["Put the answer in {rel} {n} sentences.", "Limit or stretch the reply to {rel} {n} sentences.",
                                "The reply ought to have {rel} {n} sentences."],
                         "fr": ["Mets la réponse en {rel} {n} phrases.", "La réponse devrait avoir {rel} {n} phrases.",
                                "Formule ta réponse en {rel} {n} phrases."]},
    "number_paragraphs": {"en": ["Make {n} paragraphs and place *** between every two.",
                                 "Your reply: {n} paragraphs, with *** as the separator.",
                                 "Write {n} separate paragraphs divided by *** markers."],
                          "fr": ["Fais {n} paragraphes et place *** entre chaque paire.",
                                 "Ta réponse : {n} paragraphes, avec *** comme séparateur.",
                                 "Écris {n} paragraphes distincts divisés par des marqueurs ***."]},
    "multiple_sections": {"en": ["Organise your reply into {n} sections, each starting with {marker} and its number.",
                                 "Have {n} sections, headed {marker} 1 through {marker} {n}.",
                                 "Lay out {n} sections; open every one with {marker} and a number."],
                          "fr": ["Organise ta réponse en {n} sections, chacune commençant par {marker} et son numéro.",
                                 "Prévois {n} sections, intitulées {marker} 1 à {marker} {n}.",
                                 "Dispose {n} sections ; ouvre chacune par {marker} et un numéro."]},
    "number_highlighted_sections": {"en": ["Put {n} or more phrases in *asterisk italics*.",
                                           "Highlight {n} or more spans, each between single asterisks.",
                                           "Show {n} or more highlighted phrases like *this one*."],
                                    "fr": ["Mets {n} expressions ou plus en *italique avec astérisques*.",
                                           "Surligne {n} segments ou plus, chacun entre astérisques simples.",
                                           "Montre {n} expressions surlignées ou plus comme *celle-ci*."]},
    "title": {"en": ["Crown the reply with a title in double angle brackets.", "Start the answer with <<a title like this>>.",
                     "Give it a heading written between << and >>."],
              "fr": ["Couronne la réponse d'un titre entre doubles chevrons.", "Commence la réponse par <<un titre comme ça>>.",
                     "Donne-lui un intitulé écrit entre << et >>."]},
    "json_format": {"en": ["Format the answer as a JSON object.", "Everything back in JSON, please.", "Return JSON only."],
                    "fr": ["Formate la réponse comme un objet JSON.", "Tout en JSON, s'il te plaît.", "Renvoie uniquement du JSON."]},
    "english_capital": {"en": ["Write the reply in English using only capital letters.", "English answer, nothing but uppercase.",
                               "Respond in English in ALL CAPS."],
                        "fr": ["Écris la réponse en anglais avec seulement des majuscules.", "Réponse anglaise, rien que des capitales.",
                               "Réponds en anglais TOUT EN MAJUSCULES."]},
    "english_lowercase": {"en": ["Write the reply in English using only small letters.", "English answer, nothing but lowercase.",
                                 "Respond in English with zero capital letters."],
                          "fr": ["Écris la réponse en anglais avec seulement des minuscules.", "Réponse anglaise, rien que des minuscules.",
                                 "Réponds en anglais avec zéro majuscule."]},
    "response_language": {"en": ["Give your answer in {lang}.", "The reply is to be in {lang}.", "Use only {lang}, please."],
                          "fr": ["Donne ta réponse en {lang}.", "La réponse se fera en {lang}.", "Utilise seulement le {lang}, s'il te plaît."]},
    "existence": {"en": ["Make use of the word {kw}.", "Your reply needs the word {kw} in it.", "Fit {kw} into the answer."],
                  "fr": ["Sers-toi du mot {kw}.", "Ta réponse doit inclure le mot {kw}.", "Case {kw} dans la réponse."]},
    "frequency": {"en": ["Mention the word {kw} {rel} {n} times.", "{kw} should be said {rel} {n} times.",
                         "Write the word {kw} {rel} {n} times over."],
                  "fr": ["Cite le mot {kw} {rel} {n} fois.", "{kw} doit être dit {rel} {n} fois.", "Écris le mot {kw} {rel} {n} fois en tout."]},
    "letter_frequency": {"en": ["Use the character {letter} {rel} {n} times.", "Let {letter} occur as a letter {rel} {n} times.",
                                "The reply must hold the letter {letter} {rel} {n} times."],
                         "fr": ["Utilise le caractère {letter} {rel} {n} fois.", "Que {letter} apparaisse comme lettre {rel} {n} fois.",
                                "La réponse doit contenir la lettre {letter} {rel} {n} fois."]},
    "repeat_prompt": {"en": ["Give the request back to me word by word first, then answer: {req}",
                             "Begin with an exact copy of the request, then reply: {req}",
                             "Reproduce the request unchanged before you answer: {req}"],
                      "fr": ["Rends-moi d'abord la demande mot à mot, puis réponds : {req}",
                             "Commence par une copie exacte de la demande, puis réponds : {req}",
                             "Reproduis la demande sans changement avant de répondre : {req}"]},
    "two_responses": {"en": ["Give two different versions of the answer with ****** between them.",
                             "Two answers that are not identical, separated by ******.",
                             "Produce two distinct replies and divide them with ******."],
                      "fr": ["Donne deux versions différentes de la réponse avec ****** entre elles.",
                             "Deux réponses non identiques, séparées par ******.",
                             "Produis deux réponses distinctes et divise-les par ******."]},
    "quotation": {"en": ['Your whole reply goes inside double quotes (").', "Start the reply with a double quote and end with one.",
                         'Present the answer as a single "quoted" text.'],
                  "fr": ['Toute ta réponse va entre guillemets doubles (").', "Commence la réponse par un guillemet double et finis par un autre.",
                         'Présente la réponse comme un seul texte "cité".']},
    "no_markdown": {"en": ["Leave out all markdown formatting.", "No markdown in what you write.", "Write without markdown."],
                    "fr": ["Laisse de côté toute mise en forme markdown.", "Aucun markdown dans ce que tu écris.", "Écris sans markdown."]},
    "first": {"en": ['Your first words should be "{x}".', "Start the reply by saying {x}.", "Put {x} at the very start."],
              "fr": ["Tes premiers mots devraient être « {x} ».", "Commence la réponse en disant {x}.", "Mets {x} tout au début."]},
}


# round 2's new wordings that share 5 words with an IFEval prompt, reworded on their way into round 3's training
REWORD = {"Your text should contain the letter {letter} {rel} {n} times.": "Your text should carry the letter {letter} {rel} {n} times."}


def templates(T: dict) -> dict:
    """Round 3's wordings from round 1's ``T``: train = round 2's train + round 2's new (reworded where `REWORD`
    says) + MORE; new = NEW."""
    T2 = V2.templates(T)
    return {k: {lang: {"train": T2[k][lang]["train"] + [REWORD.get(w, w) for w in T2[k][lang]["new"]]
                                + MORE.get(k, {}).get(lang, []),
                       "new": NEW[k][lang]} for lang in ("en", "fr")} for k in T2}
