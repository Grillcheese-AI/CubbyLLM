"""The voice builder's pure parts (order 0010, Cowork's addendum): the pick, the block, a filled template."""
from validation.sft.talk_voice_pick import pick, stray_capitals, subtype, why_dropped
from validation.sft.talk_voice_text import (fill_frame, fill_list, frame_one, grouped, parse, render, render_items,
                                            to_french)


def _row(cat, text, lang="en", flags=()):
    return {"cat": cat, "lang": lang, "text": text, "flags": list(flags), "model": "lab/model"}


def test_a_name_or_a_place_inside_a_line_is_dropped():
    assert stray_capitals("Here's what I have: {facts} What next?") == []          # the word after a slot opens a sentence
    assert stray_capitals("Hi {user}! I'm Cubby.") == []
    assert stray_capitals("My horse was named Fact.") == ["Fact"]
    assert why_dropped(_row("pun", "Full of holes, like Swiss cheese.")) == "a name inside the line"
    assert why_dropped(_row("greet", "Bonjour du québec !", "fr")) == "a place"


def test_a_claim_to_be_right_or_an_offer_cubby_cannot_keep_is_dropped():
    assert why_dropped(_row("frame", "That one I know for sure: {fact}")) == "claims to be right"
    assert why_dropped(_row("about", "Happy to explain! I can do this: {abilities}")) == "offers what Cubby does not do yet"
    assert why_dropped(_row("correct", "That is not correct about {subject}. {facts}")) is None   # a correction may say so
    assert why_dropped(_row("greet", "Bonjour, comment allez-vous ?", "fr")) == "says vous"
    assert why_dropped(_row("frame", "Here you go: {fact}")) is None


def test_a_social_line_goes_to_the_turn_it_answers():
    assert subtype(_row("small", "You're welcome! Glad that helped.")) == "small_welcome"
    assert subtype(_row("small", "Sorry about that. Let's untangle it.")) == "small_sorry"
    assert subtype(_row("small", "No need to apologize, truly.")) == "small_forgive"
    assert subtype(_row("small", "Ooh, that's an interesting question.")) is None
    assert subtype(_row("greet", "Good morning! What's first?")) == "greet_morning"
    assert subtype(_row("greet", "Welcome back, friend.")) == "greet_back"


def test_pick_keeps_the_kind_and_counts_what_it_drops():
    kept, dropped = pick([_row("frame", "Here you go: {fact}"), _row("frame", "x {fact}", flags=["number"]),
                          {**_row("greet", "Hello there!"), "model": "other/model"}], {"other"})
    assert [r["kind"] for r in kept] == ["frame"]
    assert dropped == {"flagged by the panel script": 1, "provider left out": 1}


BLOCK = [("Nutan", "birth date", "1936-06-04"), ("Nutan", "spouse", "Rajnish Bahl"), ("Nutan", "award", "A"),
         ("Nutan", "award", "B")]


def test_the_block_round_trips_and_carries_the_premise_line():
    prompt = render(BLOCK, "Who is Nutan?")
    assert parse(prompt) == (BLOCK, "Who is Nutan?")
    assert render(BLOCK, "q", "contradicted").startswith("Premise: contradicted\nFacts:\n- Nutan — birth date: 1936-06-04")
    assert parse("Question: hi\nAnswer:") is None
    assert to_french(BLOCK)[0] == ("Nutan", "date de naissance", "1936-06-04")
    assert to_french([("X", "no such label", "y")]) is None


def test_one_fact_is_shown_inline_and_several_as_a_numbered_list():
    assert render_items([("spouse", ["Rajnish Bahl"])], "en") == ("**Spouse:** Rajnish Bahl", False)
    text, is_list = render_items(grouped(BLOCK, "Nutan"), "en")
    assert is_list and text.split("\n") == ["1. **Birth date:** 1936-06-04", "2. **Spouse:** Rajnish Bahl",
                                             "3. **Award:** A and B"]
    assert render_items([("distinction", ["A", "B"])], "fr")[0] == "**Distinction :** A et B"
    out = fill_list("Here's what I have, {facts} Anything else?", "facts", text, True, "en")
    assert out.startswith("Here's what I have:\n1. ") and out.endswith("\n\nAnything else?")
    assert fill_list("About it. {facts} There.", "facts", "**Spouse:** X", False, "en") == "About it. **Spouse:** X. There."


def test_a_frame_takes_a_sentence_or_a_label_and_value():
    assert fill_frame("My notes say, {fact}", "The successor is X.") == "My notes say, the successor is X."
    assert fill_frame("Here you go: {fact} Enjoy!", "Nutan was born in 1936.") == "Here you go: Nutan was born in 1936."
    assert frame_one("Voici : {fact}", "pays", ["Canada"], "fr") == "Voici : **Pays :** Canada."
    assert frame_one("D'après mes fiches, {fact}", "pays", ["Canada"], "fr") == "D'après mes fiches, pays : Canada."
    assert frame_one("Mes notes disent que {fact}", "pays", ["Canada"], "fr") is None
