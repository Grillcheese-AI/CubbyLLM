"""Order 0067: the task an instruction sets (compare, explain, summarise ...) is found in English and French, a longer
cue wins over the shorter one inside it, and a clause with no task word sets none."""
from __future__ import annotations

from cubbyllm.speech import task_verbs as T


def test_every_task_has_cues_in_both_languages_and_its_own_gloss():
    assert len(T.TASKS) == 22
    for key, (en, fr, gloss, shape) in T.TASKS.items():
        assert en and fr and gloss and shape, key


def test_tasks_are_found_in_order():
    got = [t["task"] for t in T.tasks_in("Compare the two plans, then explain why one is cheaper. Summarise at the end.")]
    assert got == ["compare", "explain", "summarise"]
    assert [t["task"] for t in T.tasks_in("Dans quelle mesure ce plan est-il réaliste ? Résume-le.")] == \
        ["to_what_extent", "summarise"]


def test_the_longer_cue_wins():
    assert [t["task"] for t in T.tasks_in("Critically evaluate the claim.")] == ["critically_evaluate"]


def test_french_imperatives_repeats_and_language():
    assert [t["task"] for t in T.tasks_in("Explique-moi pourquoi.")] == ["explain"]
    assert [t["task"] for t in T.tasks_in("Compare-les, puis résume-le.")] == ["compare", "summarise"]
    assert T.tasks_in("resume writing tips", lang="en") == []
    assert [t["task"] for t in T.tasks_in("Resume ce texte.", lang="fr")] == ["summarise"]


def test_no_task_word_no_task():
    assert T.tasks_in("Answer in fewer than 50 words.") == []
