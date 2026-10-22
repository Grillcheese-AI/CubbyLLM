"""The books as a source (order 0012): standin/books.py, book_frames.py, book_source.py, and the ask loop's
`before` tier and attributed answers -- on synthetic paragraphs only (no book text enters the repo).

Pinned: the index reads twins once and a cleaned mirror's copy before the original, takes author / title /
era / region / topic from the names, joins the events file as one book per era, and a re-run adds only a
new book; paragraphs are blank-line blocks (wrapped lines joined, long lines one each, cut at `cap`);
mentions are capitalised name matches, common words never; the passage finder ranks the cue's paragraph
first, keeps an era, and falls back to the phrase for a name it does not hold; a frame fires only with
the entity as its subject; one author is attributed, two independent authors corroborate, two values for
a one-valued relation are held; in the loop a book's fact is asked before the source, an answer on one
author is spoken hedged with no source named and its citation in the record, a second source lifts it
to plain, a contradiction is never said; without books the loop is what it was.
Run: python -m pytest standin/tests/test_books.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import book_frames as bf  # noqa: E402
from book_source import BookSource, credited, independent  # noqa: E402
from books import BookIndex, Mentions, common_words, discover, meta, paragraphs  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import STORE, DictSource, ProvStore, run  # noqa: E402

from ask import AskLoop  # noqa: E402
from question_forms import TemplateEmitter  # noqa: E402
from test_ask import FakeEmitter  # noqa: E402

NAPOLEON = ("Napoleon Bonaparte was born on 15 August 1769 in Ajaccio, the son of a minor noble.\n\n"
            "Napoleon Bonaparte crowned himself emperor in Paris.\n\n"
            "Henry II ruled England from 1154 to 1189, holding lands from Scotland to the Pyrenees.\n")
NAMES = ["Napoleon Bonaparte", "Ajaccio", "Paris", "Rome", "Romulus", "Art", "Carthage", "England", "Henry II"]
KNOWN = ["Remus is the founded by of Alba Longa", "Pella is the place of birth of Alexander",   # the relations a world
         "1749 is the date of birth of Goethe", "Ine is the head of state of Wessex"]          # holds (Wikidata's labels)


def _book(root: pathlib.Path, rel: str, text: str) -> pathlib.Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _library(tmp_path):
    src, clean, other = tmp_path / "src", tmp_path / "clean", tmp_path / "hbc"
    _book(src, "MiddleAges/France/Alice Smith - Lives of Emperors [Retail].epub.txt", "the epub twin")
    _book(src, "MiddleAges/France/Alice Smith - Lives of Emperors [Retail].pdf.txt", "the pdf twin")
    _book(clean, "MiddleAges/France/Alice Smith - Lives of Emperors [Retail].txt", NAPOLEON)
    _book(src, "AncientClassical/Italy/Rome/Bob Jones - The City.pdf.txt",
          "Rome was founded by Romulus in 753 BC, on the Palatine.\n\nArt was everywhere in Rome, and art was cheap.\n")
    _book(other, "EPUB/The Corsican - Carol White/Carol White - The Cors_can (v5.0).epub.txt",
          "Napoleon Bonaparte was born in 1768 in Ajaccio.\n")
    _book(other, "EPUB/Corsica - Eve Black/Eve Black - Corsica (v5.0).epub.txt",
          "Napoleon Bonaparte was born in 1769 at Ajaccio, the second son of Carlo.\n")
    events = tmp_path / "historical_facts.jsonl"
    events.write_text(json.dumps({"title": "Founding of Rome", "text": "Rome was founded in 753 BC.", "year_start": -753}) + "\n",
                      encoding="utf-8")
    return [(src, clean), (other, None)], events


def test_the_index_reads_each_book_once_names_it_and_a_rerun_adds_only_a_new_book(tmp_path):
    roots, events = _library(tmp_path)
    found = {meta(*f)["author"]: f for f in discover(roots)}
    assert len(found) == 4 and found["Alice Smith"][1].name == "clean"           # twins once; the mirror's copy
    assert meta(*found["Alice Smith"]) == {"author": "Alice Smith", "title": "Lives of Emperors", "era": "MiddleAges",
                                           "region": "France", "topic": ""}
    assert meta(*found["Bob Jones"])["topic"] == "Rome" and meta(*found["Bob Jones"])["era"] == "AncientClassical"
    assert meta(*found["Carol White"]) == {"author": "Carol White", "title": "The Corsican", "era": "", "region": "", "topic": ""}
    idx = BookIndex(tmp_path / "books.sqlite")
    m = Mentions(NAMES)
    first = idx.build(roots, events, m)
    assert first["added"] == 5 and first["books"] == 5                            # four books and one era of events
    assert idx.build(roots, events, m)["added"] == 0
    _book(roots[1][0], "EPUB/Dan - Carthage/Dan Brown - Carthage (v5.0).epub.txt", "Carthage, an ancient city in modern Tunisia, fell.\n")
    again = idx.build(roots, events, m)
    assert again["added"] == 1 and again["skipped"] == 5 and again["books"] == 6


def test_paragraphs_join_wrapped_lines_keep_long_ones_cut_at_the_cap_and_drop_fragments():
    wrapped = "The siege of the city lasted many months and the de-\nfenders held out until the spring came.\n\n12\n\n"
    assert paragraphs(wrapped) == [(0, "The siege of the city lasted many months and the defenders held out until the spring came.")]
    long = ("A" * 10 + ". ") * 40
    ps = paragraphs(long + "\n" + long + "\n", cap=200)
    assert all(len(p) <= 200 for _o, p in ps) and ps[0][0] == 0 and len(ps) >= 4
    assert paragraphs("x" * 350 + "\n" + "y" * 350 + "\n") == [(0, "x" * 350), (351, "y" * 350)]


def test_mentions_are_capitalised_name_matches_longest_first_and_never_common_words():
    m = Mentions(NAMES + ["Napoleon"], common=common_words(["art was cheap and art was good. Art fades."], NAMES))
    assert m("Napoleon Bonaparte left Ajaccio. Art was cheap.") == {"napoleon bonaparte", "ajaccio"}
    assert m("in paris and rome") == set()                                       # lowercase: not a mention


def test_the_passage_finder_ranks_the_cues_paragraph_first_keeps_an_era_and_falls_back_to_the_phrase(tmp_path):
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    ps = idx.passages("Napoleon Bonaparte", words=["born"], k=5)
    assert ps[0].text.startswith("Napoleon Bonaparte was born") and {p.author for p in ps} == {"Alice Smith", "Carol White", "Eve Black"}
    assert len(ps) == 4 and "crowned" in ps[-1].text                             # the cue first, then the rest
    assert [p.author for p in idx.passages("Napoleon Bonaparte", words=["born"], era="MiddleAges")] == ["Alice Smith", "Alice Smith"]
    assert idx.passages("the son of a minor noble")[0].title == "Lives of Emperors"   # not a name: the phrase
    assert {p.era for p in idx.passages("Rome", words=["founded"])} == {"AncientClassical"}


def test_a_frame_fires_only_with_the_entity_as_its_subject():
    read = lambda s, e: [(t.rel, t.obj) for t, *_ in bf.read(s, e)]               # noqa: E731
    assert read("Napoleon Bonaparte was born on 15 August 1769 in Ajaccio, Corsica.", "Napoleon Bonaparte") == \
        [("date of birth", "1769-08-15"), ("place of birth", "Ajaccio")]
    assert read("Bonaparte was born in 1769, in Ajaccio.", "Napoleon Bonaparte")[0] == ("date of birth", "1769")
    assert read("His father Carlo Bonaparte was born in 1746.", "Napoleon Bonaparte") == []     # another Bonaparte
    assert read("He was born in 1769.", "Napoleon Bonaparte") == []                             # a pronoun is not followed
    assert read("Rome was founded in 753 BC by Romulus.", "Rome") == [("founded by", "Romulus"), ("inception", "753 BC")]
    assert read("Henry II ruled England from 1154 to 1189.", "Henry II") == [("reign", "1154-1189")]
    realm = bf.read("Henry II ruled England from 1154 to 1189.", "England")[0]                  # the realm's ruler, dated
    assert realm[::2] == (bf.Triple(obj="Henry II", rel="head of state", subj="England"), {"start": "1154", "end": "1189"})
    assert bf.read("Napoleon Bonaparte was born in 1769 in Ajaccio.", "Napoleon Bonaparte")[1][2] == {"point": "1769"}
    assert read("Paris, the capital of France, was besieged.", "France") == [("capital", "Paris")]
    assert read("The Battle of Hastings in 1066 changed England.", "Battle of Hastings") == [("point in time", "1066")]
    assert read("Napoleon Bonaparte was born in 1,500 days of rain.", "Napoleon Bonaparte") == []


def test_one_author_attributes_two_independent_authors_corroborate_and_two_values_are_held(tmp_path):
    assert credited("Michelle Hayward, Lesley-Gail Atkinson & Michael Cinquino") == \
        {"michelle hayward", "lesley gail atkinson", "michael cinquino"}
    assert independent([{"author": "Alice Smith"}, {"author": "Carol White"}])
    assert not independent([{"author": "Alice Smith, Carol White"}, {"author": "Carol White"}])
    assert not independent([{"author": ""}, {"author": "Carol White"}])
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    src = BookSource(idx)
    got = {(t.rel, t.obj) for t in src.facts("Napoleon Bonaparte", relations=["place of birth"])}
    assert got == {("place of birth", "Ajaccio")}
    k = normalize("Ajaccio is the place of birth of Napoleon Bonaparte")
    assert k not in src.attributed and k not in src.held                         # Alice Smith and Carol White: plain
    assert {w["author"] for w in src.citations[k]} == {"Alice Smith", "Carol White", "Eve Black"}
    got = {t.obj for t in src.facts("Napoleon Bonaparte", relations=["date of birth"])}
    assert got == {"1769-08-15", "1768"}                                         # 1769 agrees with 1769-08-15: one value
    iso = normalize("1769-08-15 is the date of birth of Napoleon Bonaparte")
    assert {w["author"] for w in src.citations[iso]} == {"Alice Smith", "Eve Black"}
    for v in ("1769-08-15", "1768"):                                             # Carol White says 1768: held, both
        assert normalize(f"{v} is the date of birth of Napoleon Bonaparte") in src.held
    src.facts("Rome", relations=["founded by"])
    assert normalize("Romulus is the founded by of Rome") in src.attributed       # Bob Jones alone
    assert src.provenance[normalize("Romulus is the founded by of Rome")] == "book:bob jones"


def exact_vm():
    """`faithful_vm` at a one-hop chain's exact recall (the VM's own 1.0; the ask loop's 1-hop threshold)."""
    inner = faithful_vm([])

    def run(source, fn):
        r = inner(source, fn)
        return {**r, "similarity": 1.0} if r.get("similarity") is not None else r
    return run


class TimedStore(ProvStore):
    """A store that keeps the dates its sources stated beside the facts, as `worlds.FactStore` does."""
    def __init__(self, texts):
        self.times = {}; super().__init__(texts)


def _loop(tmp_path, source, books=True, tense=None, emitter=None):
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    world = TimedStore(STORE)
    for r in ("founded by", "place of birth", "date of birth", "head of state"):    # declared, as a Wikidata Triple is
        world.index.declare_relation(r)
    for f in KNOWN:
        world.add(f)
    return AskLoop(emitter or TemplateEmitter(), world=world, source=source, lexicon=False, run_fn=exact_vm(),
                   books=BookSource(idx) if books else None, tense=tense)


def test_in_the_loop_one_author_is_said_hedged_with_the_citation_on_record_and_a_second_source_says_it_flat(tmp_path):
    loop = _loop(tmp_path, DictSource({}))
    rec = loop.ask("Who founded Rome?")
    assert rec["verified"] and rec["answer"] == "Romulus" and rec["tier"] == "attributed"
    assert rec["reply"] == "One account says Romulus." and "Bob" not in rec["reply"] and "City" not in rec["reply"]
    (cite,) = rec["evidence"]["Romulus is the founded by of Rome"]
    assert all(w in cite["sentence"] for w in ("Rome", "Romulus", "founded")) and cite["author"] == "Bob Jones"
    loop2 = _loop(tmp_path / "b", DictSource({"rome": ["Romulus is the founded by of Rome"]}))
    rec2 = loop2.ask("Who was the founder of Rome?")
    assert rec2["verified"] and "tier" not in rec2 and rec2["answer"] == "Romulus"     # a second witness: plain
    assert loop2.world.provenance["Romulus is the founded by of Rome"] == "book:bob jones+test"
    rec3 = loop2.ask("Where was Napoleon Bonaparte born?")
    assert rec3["verified"] and "tier" not in rec3 and rec3["answer"] == "Ajaccio"   # two authors: plain
    rec4 = loop2.ask("When was Napoleon Bonaparte born?")
    assert not rec4["verified"] and rec4["answer"] is None                         # the books disagree: never said


def test_the_tense_and_the_year_asked_must_be_the_facts(tmp_path):
    loop = _loop(tmp_path, DictSource({}))
    assert loop.tense                                                             # on with the books
    rec = loop.ask("Who ruled England?")
    assert rec["verified"] and rec["answer"] == "Henry II" and rec["tier"] == "attributed"   # ended: was
    rec = loop.ask("Who rules England?")
    assert not rec["verified"] and rec["reason"] == "time_mismatch" and rec["answer"] is None
    assert "ended in 1189" in rec["refused"]["why"] and rec["refused"]["answer"] == "Henry II"
    assert loop.ask("Who was the ruler of England in 1160?")["verified"]          # the year inside the reign
    rec = loop.ask("Who was the ruler of England in 1200?")
    assert not rec["verified"] and "1200" in rec["refused"]["why"]
    loop2 = _loop(tmp_path / "b", DictSource({}))
    assert loop2.ask("Where was Napoleon Bonaparte born?")["verified"]
    rec = loop2.ask("Where is Napoleon Bonaparte born?")                          # a past event asked in the present
    assert not rec["verified"] and rec["reason"] == "time_mismatch"
    off = _loop(tmp_path / "c", DictSource({}), tense=False)
    assert off.ask("Who rules England?")["verified"]                              # the check is the switch's


def test_before_and_the_attributed_tier_change_nothing_when_unused(tmp_path):
    canonical = FakeEmitter({"Who founded Rome?": ("Rome", ["founded by"])})        # a plan in the store's label
    loop = _loop(tmp_path, DictSource({"rome": ["Romulus is the founded by of Rome"]}), books=False, emitter=canonical)
    rec = loop.ask("Who founded Rome?")
    assert rec["verified"] and "tier" not in rec and "evidence" not in rec and loop.books is None and not loop.tense
    store, known = ProvStore(STORE), StoreRelations(STORE)
    a = run("What is the capital of the country of citizenship of Marie?", store, known,
            DictSource({"marie": ["canada is the country of citizenship of marie"]}))
    store2, known2 = ProvStore(STORE), StoreRelations(STORE)
    b = run("What is the capital of the country of citizenship of Marie?", store2, known2,
            DictSource({"marie": ["canada is the country of citizenship of marie"]}), before=[])
    assert (a.result.reason, a.entities, a.fetched, a.attributed) == (b.result.reason, b.entities, b.fetched, b.attributed)
    assert a.attributed == []


class Synonyms:
    """A wording resolver (the lexicon's shape): 'place of birth' is the world's 'birthplace'."""
    name = "lexicon"

    def relations(self, text):
        return ["birthplace"] if normalize(text) == "place of birth" else []


def test_a_book_fact_is_walked_under_its_own_label_when_a_resolver_had_moved_the_plan_to_the_worlds(tmp_path):
    """The lexicon rewrites 'place of birth' to the world's 'birthplace' (the world holds it); the books bring
    'Ajaccio is the place of birth of ...'; the walk goes back to the proposed wording -- also when the source
    below has no wording table of its own (2026-10-22, the bench: the reset waited on the source's table)."""
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    world = TimedStore(STORE)
    world.index.declare_relation("birthplace")
    world.add("Stagira is the birthplace of Aristotle")                          # the world's wording, not Wikidata's
    wikidata_label = FakeEmitter({"Where was Napoleon Bonaparte born?": ("Napoleon Bonaparte", ["place of birth"])})
    loop = AskLoop(wikidata_label, world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm(),
                   books=BookSource(idx))
    loop.resolvers.insert(0, Synonyms())
    rec = loop.ask("Where was Napoleon Bonaparte born?")
    assert ["place of birth", "birthplace"] in rec["aliased"]                     # the plan was moved to the world's
    assert rec["verified"] and rec["answer"] == "Ajaccio" and rec["plan"] == ["place of birth"]


class WrongSynonyms:
    """The lexicon's real readings of a question's own words (2026-10-22): 'die' -> 'dice', 'born' -> 'max born'."""
    name = "lexicon"

    def __init__(self, *_a, **_k):
        pass

    def relations(self, text):
        return {"die": ["dice"], "born": ["max born"]}.get(normalize(text), [])


def test_the_question_forms_name_a_wording_before_the_lexicon(tmp_path, monkeypatch):
    """The first resolver to name a wording decides. With the lexicon first, 'When was Ada Lovelace born?' asked
    'max born', which nothing holds, and the books' date was learned and never walked."""
    import cubbyllm.reasoning.lexicon as lx
    monkeypatch.setattr(lx, "Lexicon", WrongSynonyms)
    roots, events = _library(tmp_path)
    idx = BookIndex(tmp_path / "books.sqlite")
    idx.build(roots, events, Mentions(NAMES))
    loop = AskLoop(TemplateEmitter(), world=TimedStore(STORE), source=DictSource({}), lexicon=True, run_fn=exact_vm(),
                   books=BookSource(idx))
    assert type(loop.resolvers[0]).__name__ == "Forms" and isinstance(loop.resolvers[1], WrongSynonyms)
    rec = loop.ask("Where was Napoleon Bonaparte born?")
    assert rec["verified"] and rec["answer"] == "Ajaccio"


def test_a_place_the_pdf_broke_is_joined_and_a_heading_run_into_it_is_no_place():
    """The bulk pass's capitals (2026-10-22): 'Mem-phis' and 'The-bes' are the PDF's line ends; 'BICULTURAL
    KINGDOMAlexandria' is a heading run into a name; 'Horn-Bad Meinberg' is hyphenated on purpose."""
    read = lambda s, e: [(t.rel, t.obj) for t, _s, _w in bf.read(s, e)]                    # noqa: E731
    assert read("Mem-phis, the capital of Egypt, grew.", "Egypt") == [("capital", "Memphis")]
    assert read("BICULTURAL KINGDOMAlexandria, the capital of Egypt, grew.", "Egypt") == []
    assert read("Horn-Bad Meinberg, the capital of Lippe, grew.", "Lippe") == [("capital", "Horn-Bad Meinberg")]
