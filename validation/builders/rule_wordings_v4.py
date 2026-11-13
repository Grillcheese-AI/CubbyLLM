"""Order 0067, the reader's fourth round (Nick, 2026-11-13: "yes allow them, build round 4").

Rounds 2 and 3 left the comparisons as the gap: "less than", kept out of training because IFEval uses it, was read as
">=". Nick allows everyday comparisons in training wordings; the 5-word check (`bench_overlap.py`) still keeps
IFEval's sentences out. Round 4's training: round 3's training list, round 3's new wordings (read once, now spent),
and comparisons ``REL`` = round 3's plus "at least" / "less than". Its new-wording split is ``NEW`` here (fresh, 3 a
kind and language), its comparisons worded from ``REL_NEW``, never in any training list: in English "at a minimum" /
"not any fewer than" for ">=" and "lower than" / "beneath" for "<"; in French round 1's "au minimum" / "en deçà de".
Drafted by hand; no rule is model-written.
"""
from __future__ import annotations

import rule_wordings_v3 as V3

__wiring__ = "STANDALONE"

REL = {"en": {">=": V3.REL["en"][">="] + ["at least"], "<": V3.REL["en"]["<"] + ["less than"]},
       "fr": V3.REL["fr"]}
REL_NEW = {"en": {">=": ["at a minimum", "not any fewer than"], "<": ["lower than", "beneath"]},
           "fr": {">=": ["au minimum"], "<": ["en deçà de"]}}

NEW = {
    "number_words": {"en": ["Your answer may run {rel} {n} words.", "Write a reply {rel} {n} words in length.",
                            "Keep this to {rel} {n} words."],
                     "fr": ["Ta réponse peut faire {rel} {n} mots.", "Écris une réponse longue de {rel} {n} mots.",
                            "Tiens ça à {rel} {n} mots."]},
    "number_sentences": {"en": ["Cover it in {rel} {n} sentences.", "Deliver {rel} {n} sentences in all.",
                                "Respond in a total of {rel} {n} sentences."],
                         "fr": ["Traite-le en {rel} {n} phrases.", "Livre {rel} {n} phrases en tout.",
                                "Réponds en un total de {rel} {n} phrases."]},
    "number_paragraphs": {"en": ["Break your answer into {n} paragraphs, using *** as the break.",
                                 "I'd like {n} paragraphs, separated one from another by ***.",
                                 "Shape the reply as {n} paragraphs with *** marking each boundary."],
                          "fr": ["Coupe ta réponse en {n} paragraphes, avec *** comme coupure.",
                                 "J'aimerais {n} paragraphes, séparés l'un de l'autre par ***.",
                                 "Donne à la réponse la forme de {n} paragraphes, *** marquant chaque limite."]},
    "multiple_sections": {"en": ["Present {n} sections, each opening with {marker} followed by a number.",
                                 "Give your reply {n} parts headed {marker} 1, {marker} 2 and so forth.",
                                 "Mark out {n} sections with the heading {marker} and a number."],
                          "fr": ["Présente {n} sections, chacune ouverte par {marker} suivi d'un numéro.",
                                 "Donne à ta réponse {n} parties titrées {marker} 1, {marker} 2 et la suite.",
                                 "Délimite {n} sections avec l'en-tête {marker} et un numéro."]},
    "number_highlighted_sections": {"en": ["Set off {n} or more phrases between single asterisks.",
                                           "Wrap {n} or more passages in *asterisks* to highlight them.",
                                           "Pick out {n} or more phrases in *italic asterisks*."],
                                    "fr": ["Détache {n} expressions ou plus entre astérisques simples.",
                                           "Entoure {n} passages ou plus d'*astérisques* pour les surligner.",
                                           "Fais ressortir {n} expressions ou plus en *italique entre astérisques*."]},
    "title": {"en": ["Head your answer with a title between << and >>.", "Name your reply with a <<title>> first.",
                     "Open with a title inside a pair of double angle brackets."],
              "fr": ["Mets en tête de ta réponse un titre entre << et >>.", "Nomme ta réponse par un <<titre>> d'abord.",
                     "Ouvre avec un titre dans une paire de doubles chevrons."]},
    "json_format": {"en": ["Answer with a JSON object only.", "Make the entire reply valid JSON.", "Send back JSON, nothing else."],
                    "fr": ["Réponds avec un objet JSON seulement.", "Fais de toute la réponse du JSON valide.",
                           "Renvoie du JSON, rien d'autre."]},
    "english_capital": {"en": ["Answer in English, uppercase letters only.", "English, with every letter in capitals.",
                               "Write your English reply in all-capital letters."],
                        "fr": ["Réponds en anglais, en lettres majuscules uniquement.", "En anglais, toutes les lettres en capitales.",
                               "Écris ta réponse anglaise entièrement en lettres capitales."]},
    "english_lowercase": {"en": ["Answer in English, lowercase letters only.", "English, with every letter in lowercase.",
                                 "Write your English reply in all-small letters, no capitals."],
                          "fr": ["Réponds en anglais, en lettres minuscules uniquement.", "En anglais, toutes les lettres en minuscules.",
                                 "Écris ta réponse anglaise entièrement en minuscules, sans capitale."]},
    "response_language": {"en": ["Reply using {lang}.", "Your answer has to be in {lang}.", "Stick to {lang} for the reply."],
                          "fr": ["Réponds en utilisant le {lang}.", "Ta réponse doit être en {lang}.", "Tiens-t'en au {lang} pour la réponse."]},
    "existence": {"en": ["Use the word {kw} at some point.", "Work {kw} into your reply.", "Your reply should feature {kw}."],
                  "fr": ["Emploie le mot {kw} à un moment.", "Intègre {kw} à ta réponse.", "Ta réponse devrait comporter {kw}."]},
    "frequency": {"en": ["The word {kw} is to appear {rel} {n} times.", "Put {kw} into the reply {rel} {n} times.",
                         "Use the word {kw} {rel} {n} times in what you write."],
                  "fr": ["Le mot {kw} doit paraître {rel} {n} fois.", "Mets {kw} dans la réponse {rel} {n} fois.",
                         "Emploie le mot {kw} {rel} {n} fois dans ce que tu écris."]},
    "letter_frequency": {"en": ["The letter {letter} is to appear {rel} {n} times.", "Write {letter} as a letter {rel} {n} times.",
                                "Have {rel} {n} instances of the letter {letter}."],
                         "fr": ["La lettre {letter} doit paraître {rel} {n} fois.", "Écris {letter} comme lettre {rel} {n} fois.",
                                "Mets {rel} {n} exemplaires de la lettre {letter}."]},
    "repeat_prompt": {"en": ["Copy my request out first, exactly, and then answer it: {req}",
                             "Lead with the request as written, word for word, then respond: {req}",
                             "Say the request back to me unaltered, then reply: {req}"],
                      "fr": ["Recopie ma demande d'abord, exactement, puis réponds-y : {req}",
                             "Commence par la demande telle qu'écrite, mot pour mot, puis réponds : {req}",
                             "Redis-moi la demande sans la modifier, puis réponds : {req}"]},
    "two_responses": {"en": ["Write two unlike answers and separate them with ******.", "Give a pair of answers that differ, split by ******.",
                             "Two separate, different responses, with ****** in the middle."],
                      "fr": ["Écris deux réponses dissemblables et sépare-les par ******.", "Donne une paire de réponses qui diffèrent, coupées par ******.",
                             "Deux réponses séparées et différentes, avec ****** au milieu."]},
    "quotation": {"en": ['Keep the whole reply inside double quote marks (").', 'Open and close your answer with ".',
                         "Enclose your full reply in double quotes."],
                  "fr": ['Garde toute la réponse entre guillemets doubles (").', 'Ouvre et ferme ta réponse par ".',
                         "Encadre toute ta réponse de guillemets doubles."]},
    "no_markdown": {"en": ["Format nothing with markdown.", "Keep markdown out of the answer.", "The reply should have no markdown."],
                    "fr": ["Ne mets aucune mise en forme markdown.", "Garde le markdown hors de la réponse.", "La réponse ne doit pas avoir de markdown."]},
    "first": {"en": ["Have your reply start with {x}.", 'Start off with "{x}".', "Your answer opens with {x}."],
              "fr": ["Fais commencer ta réponse par {x}.", "Débute avec « {x} ».", "Ta réponse s'ouvre par {x}."]},
}


# round 3's new wordings that share 5 words with an IFEval prompt, reworded on their way into round 4's training
REWORD = {"Crown the reply with a title in double angle brackets.": "Crown the reply with a title set in double angle brackets."}


def templates(T: dict) -> dict:
    """Round 4's wordings from round 1's ``T``: train = round 3's train + round 3's new (reworded where `REWORD`
    says); new = NEW. A train/dev row whose numbers make it meet a benchmark prompt is drawn again by the builder."""
    T3 = V3.templates(T)
    return {k: {lang: {"train": T3[k][lang]["train"] + [REWORD.get(w, w) for w in T3[k][lang]["new"]], "new": NEW[k][lang]}
                for lang in ("en", "fr")} for k in T3}
