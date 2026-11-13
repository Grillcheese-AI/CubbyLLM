"""Order 0067, the reader's sixth round: instructions on a reply's content or structure that no reply-world measurement
covers, by hand, English and French. Their answer is ``asked`` (kept for the talk seat, never checked).

Round 5's `asked_rows.PHRASES` are tone and style; round 5's IFBench read showed the reader turning structural
instructions it cannot measure into the nearest count. These teach that a structural instruction, even one with a
number in it ("give three examples", "the five senses"), is not a count the reply world keeps.

Kept clear of the test kinds: none of IFBench's 58 kinds (``IFBENCH_KINDS``: alphabet, palindromes, stop-word ratios,
keyword positions, punctuation counts, ...) and none of IFEval's 9 held-out kinds (no commas, an end phrase, a
postscript, placeholders, a fixed answer, bullet lists, forbidden words, a paragraph's first word, capitalised words).
A seeded quarter of the keys is held out whole (``real_asked_test``).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

STRUCT = [
    ("dialogue", ["Write it as a dialogue between two people.", "Frame the answer as a conversation."],
     ["Écris-le sous forme de dialogue entre deux personnes.", "Présente la réponse comme une conversation."]),
    ("everyday_example", ["Include an example from everyday life.", "Illustrate it with an everyday situation."],
     ["Donne un exemple de la vie quotidienne.", "Illustre-le avec une situation de tous les jours."]),
    ("one_line_summary", ["Add a one-line summary after your answer.", "Sum it all up in a single closing line."],
     ["Ajoute un résumé d'une ligne après ta réponse.", "Résume le tout en une seule ligne finale."]),
    ("rhetorical", ["Include a rhetorical question.", "Slip in a rhetorical question somewhere."],
     ["Glisse une question rhétorique.", "Ajoute une question rhétorique quelque part."]),
    ("analogy", ["Explain it with an analogy.", "Use an analogy to make it clear."],
     ["Explique-le avec une analogie.", "Sers-toi d'une analogie pour que ce soit clair."]),
    ("step_by_step", ["Explain it step by step.", "Walk me through it one step at a time."],
     ["Explique étape par étape.", "Guide-moi une étape à la fois."]),
    ("weigh", ["Weigh the advantages against the drawbacks.", "Lay out what speaks for it and against it."],
     ["Pèse les avantages et les inconvénients.", "Expose ce qui parle pour et contre."]),
    ("counterargument", ["Address one counterargument.", "Answer the strongest objection."],
     ["Réponds à un contre-argument.", "Réponds à l'objection la plus forte."]),
    ("no_adverbs", ["Avoid adverbs.", "Keep adverbs out of it."], ["Évite les adverbes.", "Laisse les adverbes de côté."]),
    ("few_adjectives", ["Use as few adjectives as possible.", "Go easy on adjectives."],
     ["Utilise le moins d'adjectifs possible.", "Sois économe en adjectifs."]),
    ("historical_event", ["Mention a historical event.", "Tie it to something that happened in history."],
     ["Mentionne un événement historique.", "Relie-le à quelque chose qui s'est passé dans l'histoire."]),
    ("simple_words", ["Stick to simple, everyday words.", "Choose plain words a child would know."],
     ["N'emploie que des mots simples et courants.", "Choisis des mots qu'un enfant connaîtrait."]),
    ("define_terms", ["Define every technical term you use.", "Explain any jargon the first time it appears."],
     ["Définis chaque terme technique que tu emploies.", "Explique tout jargon dès sa première apparition."]),
    ("chronological", ["Present the events in chronological order.", "Go through it in the order things happened."],
     ["Présente les événements dans l'ordre chronologique.", "Suis l'ordre dans lequel les choses se sont passées."]),
    ("moral", ["Add a moral, like a fable.", "Draw a lesson from it, as a fable would."],
     ["Ajoute une morale, comme dans une fable.", "Tire-en une leçon, comme le ferait une fable."]),
    ("familiar_comparison", ["Compare it to something familiar.", "Relate it to something people already know."],
     ["Compare-le à quelque chose de familier.", "Rapproche-le de quelque chose que les gens connaissent déjà."]),
    ("active_verbs", ["Use strong, active verbs.", "Prefer lively, active verbs."],
     ["Emploie des verbes forts et actifs.", "Préfère des verbes vifs et actifs."]),
    ("five_senses", ["Appeal to the five senses.", "Describe what all five senses would notice."],
     ["Fais appel aux cinq sens.", "Décris ce que les cinq sens remarqueraient."]),
    ("three_examples", ["Give three examples.", "Back it up with three examples."],
     ["Donne trois exemples.", "Appuie-le avec trois exemples."]),
    ("two_reasons", ["Give two reasons.", "Offer a couple of reasons."], ["Donne deux raisons.", "Avance deux raisons."]),
    ("cause_effect", ["Explain the cause before the effect.", "Start from the cause, then the consequences."],
     ["Explique la cause avant l'effet.", "Pars de la cause, puis des conséquences."]),
    ("concrete_numbers", ["Back your claims with concrete figures.", "Support it with real numbers where you can."],
     ["Appuie tes affirmations sur des chiffres concrets.", "Étaye-le avec de vrais chiffres quand tu peux."]),
    ("open_question", ["Leave the reader with something to think about.", "Close on a thought that lingers."],
     ["Laisse le lecteur avec matière à réflexion.", "Conclus sur une pensée qui reste."]),
    ("first_person", ["Tell it from your own point of view.", "Speak from personal experience."],
     ["Raconte-le de ton propre point de vue.", "Parle d'expérience personnelle."]),
]

IFBENCH_KINDS = (
    "count:conjunctions count:keywords_multiple count:numbers count:person_names count:pronouns count:punctuation "
    "count:unique_word_count count:word_count_range count:words_japanese custom:character_reverse custom:csv_city "
    "custom:csv_quotes custom:csv_special_character custom:date_format_list custom:european_capitals_sort "
    "custom:mcq_count_length custom:multiples custom:reverse_newline custom:sentence_alphabet custom:word_reverse "
    "format:emoji format:line_indent format:list format:newline format:no_bullets_bullets format:no_whitespace "
    "format:options format:output_template format:parentheses format:quote_unquote format:quotes format:sub-bullets "
    "format:thesis format:title_case ratio:overlap ratio:sentence_balance ratio:sentence_type ratio:sentence_words "
    "ratio:stop_words repeat:repeat_change repeat:repeat_simple repeat:repeat_span sentence:alliteration_increment "
    "sentence:increment sentence:keyword words:alphabet words:consonants words:keywords_specific_position "
    "words:last_first words:no_consecutive words:odd_even_syllables words:palindrome words:paragraph_last_first "
    "words:prime_lengths words:repeats words:start_verb words:vowel words:words_position").split()

__all__ = ["IFBENCH_KINDS", "STRUCT"]
