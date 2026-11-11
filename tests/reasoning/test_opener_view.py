"""Order 0010 change a: the host's guard reads a reply through the opener view under CB_OPENER_VIEW=1, and only then."""
from cubbyllm.reasoning import ask_claims
from cubbyllm.reasoning.opener_view import INTERJECTIONS, common_words, guard_text, opener_view

FACTS = ["Paris is the capital of France", "France capital: Paris"]


def test_off_the_host_reads_the_raw_draft_and_refuses_a_framed_reply(monkeypatch):
    monkeypatch.delenv("CB_OPENER_VIEW", raising=False)
    draft = "Here is what I have: Paris."
    assert guard_text(draft) == draft
    said, why = ask_claims.talk_reply(["Paris"], draft, FACTS, "France", [])
    assert said is None and why.startswith("ungrounded")


def test_on_a_framed_reply_is_spoken_and_a_name_outside_the_facts_still_refused(monkeypatch):
    monkeypatch.setenv("CB_OPENER_VIEW", "1")
    said, why = ask_claims.talk_reply(["Paris"], "Here is what I have: Paris.", FACTS, "France", [])
    assert (said, why) == ("Here is what I have: Paris.", "spoken")
    said, why = ask_claims.talk_reply(["Paris"], "Here is what I have: Paris, says Napoleon.", FACTS, "France", [])
    assert said is None and "Napoleon" in why                       # a name mid-sentence keeps its capital
    said, why = ask_claims.talk_reply(["Paris"], "Hmm. Lyon, I think. Paris.", FACTS, "France", [])
    assert said is None and "Lyon" in why                           # a name opening a sentence is not a common word
    said, why = ask_claims.talk_reply(["Paris"], "Here is what I have: Lyon.", FACTS, "France", [])
    assert said is None                                              # the value check still reads the draft


def test_the_view_lowers_only_forced_capitals():
    common = common_words()
    assert INTERJECTIONS <= common and "here" in common
    assert opener_view("Hmm, here it is. I think so.") == "hmm, here it is. i think so."
    assert opener_view("1. Here is one\n2. Paris") == "\nhere is one\n\nParis"     # enumerators go, a name stays
    assert opener_view("Bonjour ! Voici Paris.").startswith("bonjour")
    assert opener_view("Paris is here.") == "Paris is here."


def test_the_builder_view_is_the_hosts_with_its_own_words():
    from validation.sft import build_chat_sft as b
    text = "Here is what I have:\n1. The answer\n2. I think so"
    assert b.chat_view(text, {"the", "here"}) == opener_view(text, {"the", "here"})
    assert b.COMMON.exists() and set(b.COMMON.read_text(encoding="utf-8").split()) | INTERJECTIONS == common_words()
