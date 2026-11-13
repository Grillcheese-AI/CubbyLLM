"""Order 0067, the second reader's step 3: the sentences of a long instruction that are not checks, by hand, English
and French, each with its kind for the split:
- ``task``: the job the answer does with a subject (Nick's task words, `cubbyllm/speech/task_verbs.py`): "Explain the
  water cycle.";
- ``process``: how to go about it, nothing the reply shows: "Read the question carefully first.";
- ``none``: not an instruction: "This is for our team newsletter.";
- ``asked``: an instruction no measurement covers (round 5's tone phrases and round 6's structure phrases; the
  held-out ones only in held-out rows: `real_rows._phrases`).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

TOPICS = [  # (EN, FR with its article)
    ("the water cycle", "le cycle de l'eau"), ("the history of the bicycle", "l'histoire du vélo"),
    ("the rules of chess", "les règles des échecs"), ("the life of a star", "la vie d'une étoile"),
    ("the causes of the French Revolution", "les causes de la Révolution française"),
    ("the way vaccines work", "le fonctionnement des vaccins"), ("the way bees make honey", "la fabrication du miel"),
    ("the tides", "les marées"), ("the rise of bread dough", "la levée de la pâte à pain"),
    ("the first moon landing", "le premier alunissage"), ("the invention of printing", "l'invention de l'imprimerie"),
    ("the migration of birds", "la migration des oiseaux"), ("the seasons", "les saisons"),
    ("the uses of copper", "les usages du cuivre"), ("the Roman roads", "les voies romaines"),
]
TASKS = {  # task -> (EN template, FR template); {t} the topic, {u} a second topic
    "explain": ("Explain {t}.", "Explique {t}."), "describe": ("Describe {t}.", "Décris {t}."),
    "summarise": ("Summarise {t}.", "Résume {t}."), "outline": ("Outline {t}.", "Donne les grandes lignes de {t}."),
    "compare": ("Compare {t} and {u}.", "Compare {t} et {u}."), "discuss": ("Discuss {t}.", "Discute de {t}."),
    "analyse": ("Analyse {t}.", "Analyse {t}."), "illustrate": ("Illustrate {t} with examples.", "Illustre {t} par des exemples."),
    "evaluate": ("Evaluate {t}.", "Évalue {t}."), "narrate": ("Narrate {t}.", "Raconte {t}."),
}
PROCESS = {
    "en": ["Read the question carefully first.", "Take a moment to plan before you write.",
           "Check your answer before you send it.", "Think it over before answering.", "Draft it first, then revise it.",
           "Look at it from a reader's side before you start."],
    "fr": ["Lis d'abord la question attentivement.", "Prends un moment pour planifier avant d'écrire.",
           "Vérifie ta réponse avant de l'envoyer.", "Réfléchis-y avant de répondre.", "Fais un brouillon, puis révise-le.",
           "Mets-toi à la place du lecteur avant de commencer."],
}
CONTEXT = {
    "en": ["This is for our team newsletter.", "I'm preparing a talk for my class.", "Thanks in advance!",
           "My teacher asked us to write about this.", "I have to hand this in tomorrow.", "It's for a quiz night.",
           "My little brother keeps asking me about it.", "We discussed it at lunch today."],
    "fr": ["C'est pour la lettre d'information de l'équipe.", "Je prépare un exposé pour ma classe.", "Merci d'avance !",
           "Mon professeur nous a demandé d'écrire là-dessus.", "Je dois le rendre demain.", "C'est pour une soirée quiz.",
           "Mon petit frère me pose sans cesse la question.", "On en a parlé ce midi."],
}


def _fr_de(text: str) -> str:
    """French "de" before an article: "de le" -> "du", "de les" -> "des"."""
    return text.replace("de le ", "du ").replace("de les ", "des ")


def task_sentence(rng, lang: str) -> tuple[str, str]:
    """(sentence, task key)."""
    key = rng.choice(sorted(TASKS))
    a, b = rng.sample(TOPICS, 2)
    t, u = (a[0], b[0]) if lang == "en" else (a[1], b[1])
    s = TASKS[key][0 if lang == "en" else 1].format(t=t, u=u)
    return (s if lang == "en" else _fr_de(s)), key


__all__ = ["CONTEXT", "PROCESS", "TASKS", "TOPICS", "task_sentence"]
