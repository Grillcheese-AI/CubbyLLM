"""Order 0067, round 4's second new-wording split (Nick, 2026-11-13: "option 2, then read the held-out kinds").

The new-wording line asked the reader to read comparison words no round trained on; Nick changed that: a new-wording
row's comparisons are now worded as training words them (round 4's ``REL``: "at least", "less than", "fewer than",
...). So that this read is not a re-scoring of templates already read, its wordings are fresh (``NEW``: 3 a kind and
language, in no training list and no earlier split). Training is round 4's exactly (`rule_wordings_v4.templates`);
the round-4 adapter is read on these rows as trained. Drafted by hand.
"""
from __future__ import annotations

import rule_wordings_v4 as V4

__wiring__ = "STANDALONE"

REL = V4.REL
REL_NEW = V4.REL                                         # the comparisons training uses (Nick's option 2)

NEW = {
    "number_words": {"en": ["The reply must come to {rel} {n} words.", "Size your answer at {rel} {n} words.",
                            "Count your words: {rel} {n}, please."],
                     "fr": ["La réponse doit totaliser {rel} {n} mots.", "Dimensionne ta réponse à {rel} {n} mots.",
                            "Compte tes mots : {rel} {n}, s'il te plaît."]},
    "number_sentences": {"en": ["Get it said in {rel} {n} sentences.", "Your reply: {rel} {n} sentences.",
                                "Frame the answer as {rel} {n} sentences."],
                         "fr": ["Dis-le en tout en {rel} {n} phrases.", "Ta réponse : {rel} {n} phrases.",
                                "Construis la réponse en {rel} {n} phrases."]},
    "number_paragraphs": {"en": ["Hand me {n} paragraphs, *** sitting between each pair.",
                                 "Exactly {n} paragraphs, kept apart by ***.", "Divide your text into {n} paragraphs using *** as the divider."],
                          "fr": ["Remets-moi {n} paragraphes, *** placé entre chaque paire.",
                                 "Exactement {n} paragraphes, tenus à part par ***.", "Divise ton texte en {n} paragraphes avec *** comme séparateur."]},
    "multiple_sections": {"en": ["Split it into {n} sections, labelled {marker} plus a number each time.",
                                 "Structure the reply with {n} {marker}-numbered sections.",
                                 "Mark every one of {n} sections with {marker} and its number."],
                          "fr": ["Coupe-la en {n} sections, étiquetées {marker} plus un numéro à chaque fois.",
                                 "Structure la réponse avec {n} sections numérotées {marker}.",
                                 "Marque chacune des {n} sections avec {marker} et son numéro."]},
    "number_highlighted_sections": {"en": ["Use asterisks to highlight {n} or more phrases.",
                                           "Have {n} or more *highlighted* phrases in the reply.", "Give {n} or more spans the *asterisk* treatment."],
                                    "fr": ["Utilise des astérisques pour surligner {n} expressions ou plus.",
                                           "Mets {n} expressions *surlignées* ou plus dans la réponse.",
                                           "Traite {n} segments ou plus avec des *astérisques*."]},
    "title": {"en": ["Give the answer a heading in << >> at the top.", "Lead with a title wrapped in << and >>.",
                     "Write a <<title>> before anything else."],
              "fr": ["Donne à la réponse un intitulé en << >> tout en haut.", "Commence par un titre enveloppé de << et >>.",
                     "Écris un <<titre>> avant toute chose."]},
    "json_format": {"en": ["Reply in JSON form.", "Only a JSON object, please.", "Encode the whole answer as JSON."],
                    "fr": ["Réponds sous forme JSON.", "Seulement un objet JSON, s'il te plaît.", "Encode toute la réponse en JSON."]},
    "english_capital": {"en": ["In English, letters all uppercase.", "Give the reply in English and in capitals throughout.",
                               "English, capitalised from first letter to last."],
                        "fr": ["En anglais, lettres toutes majuscules.", "Donne la réponse en anglais et en capitales d'un bout à l'autre.",
                               "En anglais, en majuscules de la première à la dernière lettre."]},
    "english_lowercase": {"en": ["In English, letters all lowercase.", "Give the reply in English and in lowercase throughout.",
                                 "English, small letters from first to last."],
                          "fr": ["En anglais, lettres toutes minuscules.", "Donne la réponse en anglais et en minuscules d'un bout à l'autre.",
                                 "En anglais, en minuscules de la première à la dernière lettre."]},
    "response_language": {"en": ["Write back in {lang}.", "Let the reply be in {lang}.", "{lang} only for this one."],
                          "fr": ["Réponds-moi en {lang}.", "Que la réponse soit en {lang}.", "Uniquement en {lang} pour celle-ci."]},
    "existence": {"en": ["Bring in the word {kw}.", "Be certain to write {kw}.", "Let {kw} appear in your reply."],
                  "fr": ["Fais entrer le mot {kw}.", "Sois sûr d'écrire {kw}.", "Que {kw} apparaisse dans ta réponse."]},
    "frequency": {"en": ["Write {kw} {rel} {n} times throughout.", "The reply should feature {kw} {rel} {n} times.",
                         "Let {kw} turn up {rel} {n} times."],
                  "fr": ["Écris {kw} {rel} {n} fois dans l'ensemble.", "La réponse doit comporter {kw} {rel} {n} fois.",
                         "Que {kw} se présente {rel} {n} fois."]},
    "letter_frequency": {"en": ["Put {letter} in the text {rel} {n} times.", "The letter {letter}: {rel} {n} occurrences.",
                                "Write so that {letter} shows up {rel} {n} times."],
                         "fr": ["Mets {letter} dans le texte {rel} {n} fois.", "La lettre {letter} : {rel} {n} occurrences.",
                                "Écris de sorte que {letter} se montre {rel} {n} fois."]},
    "repeat_prompt": {"en": ["Open by quoting the request exactly, then answer: {req}", "Restate the request with no changes, then reply to it: {req}",
                             "First, the request back to me verbatim; then your answer: {req}"],
                      "fr": ["Ouvre en citant la demande exactement, puis réponds : {req}", "Redis la demande sans aucun changement, puis réponds-y : {req}",
                             "D'abord, la demande telle quelle ; ensuite ta réponse : {req}"]},
    "two_responses": {"en": ["Two answers, different from each other, with ****** splitting them.",
                             "Write a first answer and a different second one, separated by ******.", "Offer two varied responses, ****** between."],
                      "fr": ["Deux réponses, différentes l'une de l'autre, avec ****** pour les séparer.",
                             "Écris une première réponse et une seconde différente, séparées par ******.", "Propose deux réponses variées, ****** entre elles."]},
    "quotation": {"en": ['Quote your entire reply: one " at the start, one at the end.', "Put the reply in double quotation marks, all of it.",
                         'Begin with " and end with " around the whole answer.'],
                  "fr": ['Cite toute ta réponse : un " au début, un à la fin.', "Mets la réponse entre guillemets doubles, en entier.",
                         'Commence par " et finis par " autour de toute la réponse.']},
    "no_markdown": {"en": ["No markdown, just plain words.", "Don't use markdown anywhere.", "Write it without any markdown marks."],
                    "fr": ["Pas de markdown, juste des mots simples.", "N'utilise le markdown nulle part.", "Écris-le sans aucune marque markdown."]},
    "first": {"en": ["Your very first words: {x}.", 'Start the answer with the phrase "{x}".', "Make your reply open on {x}."],
              "fr": ["Tes tout premiers mots : {x}.", "Commence la réponse par l'expression « {x} ».", "Fais s'ouvrir ta réponse sur {x}."]},
}


def templates(T: dict) -> dict:
    """Round 4's training wordings unchanged; the new split is NEW."""
    T4 = V4.templates(T)
    return {k: {lang: {"train": T4[k][lang]["train"], "new": NEW[k][lang]} for lang in ("en", "fr")} for k in T4}
