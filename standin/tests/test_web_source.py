"""standin/web_source.py -- the web behind Wikidata (2026-09-24). No network: a fake search backend and
fake pages. Pinned: a page is a witness, not a fact -- one site's triple is HELD (the latent tier) and
never carries an answer; two independent sites lift it; one site twice, or Wikipedia and its mirror, is
still one witness; the frames read only sentences about the entity; the local reader may only write a
value the passage says; robots.txt is honoured; a Wikipedia hit hands over to Wikidata only when ONE
item carries the relation; the tally carries across days; and the loop asks the web only after the
primary source has nothing more.
Run: python -m pytest standin/tests/test_web_source.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning.learn import learn_and_answer  # noqa: E402
from cubbyllm.reasoning.plan_verify import StoreRelations  # noqa: E402
from cubbyllm.reasoning.planner import Triple, normalize  # noqa: E402
from cubbyllm.reasoning.sleep import trusted_source  # noqa: E402
from lfm_source import LfmReader  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from web_source import Hit, WebSource, frame_facts, jsonld_facts, site_of  # noqa: E402


class FakeBackend:
    name = "fake"

    def __init__(self, hits): self.hits = hits; self.calls = []

    def search(self, q, n=8):
        self.calls.append(q)
        return list(self.hits)[:n]


def pages(by_url, robots=None):
    """A fake transport: url -> html; robots.txt per scheme://host."""
    def http(url, headers=None, timeout_s=15, limit=0):
        if url.endswith("/robots.txt"):
            host = url[: -len("/robots.txt")]
            return ("text/plain", robots[host].encode()) if robots and host in robots else None
        if url in by_url:
            return ("text/html; charset=utf-8", by_url[url].encode("utf-8"))
        return None
    return http


def person_page(name, **props):
    ld = {"@context": "https://schema.org", "@type": "Person", "name": name, **props}
    return f'<html lang="en"><head><script type="application/ld+json">{json.dumps(ld)}</script></head><body><p>{name}.</p></body></html>'


def web(hits, by_url=None, tmp=None, **kw):
    return WebSource(FakeBackend(hits), cache_dir=tmp, http=pages(by_url or {}, kw.pop("robots", None)), **kw)


def test_a_site_is_a_registrable_domain_and_the_wikipedia_family_is_one_witness():
    assert site_of("https://en.wikipedia.org/wiki/X") == site_of("https://www.wikiwand.com/en/X") == "wikipedia.org"
    assert site_of("https://news.bbc.co.uk/a") == "bbc.co.uk" and site_of("https://www.britannica.com/b") == "britannica.com"
    assert site_of("https://web.archive.org/web/2020/https://x.com") is None      # an archive repeats another site


def test_one_site_is_held_two_independent_sites_attest(tmp_path):
    a = "https://a.example.com/marie"; b = "https://b.example.org/marie"; a2 = "https://a.example.com/other"
    by = {u: person_page("Marie", nationality="Canada") for u in (a, b, a2)}
    one = web([Hit(a, "Marie", ""), Hit(a2, "Marie", "")], by, tmp_path / "one")
    t = one.facts("Marie", relations=["country of citizenship"])
    k = normalize("Canada is the country of citizenship of Marie")
    assert [(x.obj, x.rel) for x in t] == [("Canada", "country of citizenship")]
    assert k in one.held and one.provenance[k] == "web:example.com"             # two pages, one site
    two = web([Hit(a, "Marie", ""), Hit(b, "Marie", "")], by, tmp_path / "two")
    two.facts("Marie", relations=["country of citizenship"])
    assert k not in two.held and two.provenance[k] == "web:example.com+web:example.org"
    assert "1 attested" in two.last["how"]


def test_the_tally_carries_a_first_site_to_the_second_on_a_later_day(tmp_path):
    a = "https://a.one.com/m"; b = "https://b.two.com/m"
    by = {a: person_page("Marie", nationality="Canada"), b: person_page("Marie", nationality="Canada")}
    k = normalize("Canada is the country of citizenship of Marie")
    d1 = web([Hit(a, "Marie", "")], by, tmp_path)
    d1.facts("Marie"); assert k in d1.held
    d2 = web([Hit(b, "Marie", "")], by, tmp_path, ttl_s=0)                       # a later day: the cache expired, the tally did not
    d2.facts("Marie"); assert k not in d2.held and d2.provenance[k] == "web:one.com+web:two.com"


def test_frames_read_only_sentences_about_the_entity():
    got = frame_facts("Albert Einstein", "Albert Einstein (14 March 1879 – 18 April 1955) was a physicist. "
                                         "Albert Einstein's friend was born in Berlin.")
    rels = {(t.rel, t.obj) for t, _s in got}
    assert ("date of birth", "1879-03-14") in rels and ("date of death", "1955-04-18") in rels
    assert not any(r == "place of birth" for r, _o in rels)                      # the possessive is not about him


def test_jsonld_reads_the_node_named_as_the_entity_and_nothing_else():
    blocks = [json.dumps({"@graph": [{"@type": "Person", "name": "Ada Lovelace", "birthDate": "1815-12-10",
                                      "birthPlace": {"@type": "Place", "name": "London"}},
                                     {"@type": "Person", "name": "Lord Byron", "birthDate": "1788-01-22"}]})]
    got = {(t.rel, t.obj) for t in jsonld_facts("Ada Lovelace", blocks)}
    assert got == {("date of birth", "1815-12-10"), ("place of birth", "London")}


class FakeEm:
    def __init__(self, text): self.text = text
    def complete(self, prompt, **kw): return self.text


class NoAliases:
    def relations(self, text): return []


def test_the_local_reader_may_only_write_what_the_passage_says(tmp_path):
    url = "https://c.site.net/ada"
    page = "<html><body><p>Ada Lovelace, the mathematician, came into the world in London in 1815.</p></body></html>"   # no frame reads this
    good = LfmReader("x.gguf", aliases=NoAliases(), cache_dir=None, emitter=FakeEm("London is the place of birth of Ada Lovelace\n"))
    bad = LfmReader("x.gguf", aliases=NoAliases(), cache_dir=None, emitter=FakeEm("Paris is the place of birth of Ada Lovelace\n"))
    w = web([Hit(url, "Ada", "")], {url: page}, tmp_path / "g", reader=good)
    assert [(t.obj, t.rel) for t in w.facts("Ada Lovelace", relations=["place of birth"])] == [("London", "place of birth")]
    assert w.last["facts"][normalize("London is the place of birth of Ada Lovelace")]["readers"] == ["lfm"]
    w2 = web([Hit(url, "Ada", "")], {url: page}, tmp_path / "b", reader=bad)
    assert w2.facts("Ada Lovelace", relations=["place of birth"]) == [] and bad.last["ungrounded"] == 1


def test_robots_are_honoured_and_the_snippet_still_counts(tmp_path):
    url = "https://d.closed.com/p"
    w = web([Hit(url, "Ada", "Ada Lovelace (10 December 1815 – 27 November 1852) was a mathematician.")],
            {url: person_page("Ada Lovelace", birthPlace="London")}, tmp_path,
            robots={"https://d.closed.com": "User-agent: *\nDisallow: /\n"})
    got = {(t.rel, t.obj) for t in w.facts("Ada Lovelace")}
    assert w.last["robots_skipped"] == 1 and w.last["pages"] == 0
    assert ("date of birth", "1815-12-10") in got and ("place of birth", "London") not in got


class FakeWikidata:
    def __init__(self, by_qid): self.by = by_qid; self.times = {}
    def facts(self, entity, relations=None, via=None, qid=None):
        return [Triple(obj=o, rel=r, subj=entity) for r, o in self.by.get(qid, [])]


def wiki_api(qids):
    import urllib.parse
    def http(url, headers=None, timeout_s=15, limit=0):
        if "api.php" not in url:
            return None
        title = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["titles"][0]
        if title in qids:
            return ("application/json", json.dumps({"query": {"pages": {"1": {"pageprops": {"wikibase_item": qids[title]}}}}}).encode())
        return None
    return http


def test_a_wikipedia_hit_hands_over_to_wikidata_only_when_one_item_carries_the_relation(tmp_path):
    hits = [Hit("https://en.wikipedia.org/wiki/Marie_A", "Marie A", ""), Hit("https://en.wikipedia.org/wiki/Marie_B", "Marie B", "")]
    one = WebSource(FakeBackend(hits), cache_dir=tmp_path / "1", http=wiki_api({"Marie A": "Q1", "Marie B": "Q2"}),
                    wikidata=FakeWikidata({"Q1": [("country of citizenship", "Canada")], "Q2": [("occupation", "painter")]}))
    got = one.facts("Marie", relations=["country of citizenship"])
    k = normalize("Canada is the country of citizenship of Marie")
    assert [t.obj for t in got] == ["Canada"] and one.provenance[k] == "wikidata (via web)" and k not in one.held
    both = WebSource(FakeBackend(hits), cache_dir=tmp_path / "2", http=wiki_api({"Marie A": "Q1", "Marie B": "Q2"}),
                     wikidata=FakeWikidata({"Q1": [("country of citizenship", "Canada")], "Q2": [("country of citizenship", "France")]}))
    assert both.facts("Marie", relations=["country of citizenship"]) == []          # the ranking does not pick the item
    assert both.last["ambiguous_items"] == ["Q1", "Q2"]


# ── the loop: the web after the primary source, and never on one site's word ──────────────────────────
STORE = ["ottawa is the capital of canada", "paris is the capital of france", "france is the country of citizenship of jean"]
Q = "What is the capital of the country of citizenship of Marie?"


class ProvStore(LookupStore):
    def __init__(self, texts):
        super().__init__(texts); self.provenance = {}; self.times = {}


def loop(web_source, primary=None):
    store = ProvStore(STORE)
    r = learn_and_answer(Q, lambda q, k: [], faithful_vm([]), store=store, known=StoreRelations(STORE),
                         source=primary or DictSource({}), tau_vm=0.5, fallbacks=[web_source])
    return r, store


def test_one_site_never_carries_an_answer_two_do(tmp_path):
    a = "https://a.one.com/m"; b = "https://b.two.com/m"
    by = {a: person_page("Marie", nationality="Canada"), b: person_page("Marie", nationality="Canada")}
    r1, s1 = loop(web([Hit(a, "Marie", "")], by, tmp_path / "1"))
    assert not r1.result.verified and r1.result.reason == "latent_only"
    assert normalize(r1.result.refused["answer"]) == "ottawa"                       # the would-be answer, on record
    assert s1.provenance["Canada is the country of citizenship of Marie"] == "web:one.com (latent)"
    r2, s2 = loop(web([Hit(a, "Marie", ""), Hit(b, "Marie", "")], by, tmp_path / "2"))
    assert r2.result.verified and normalize(r2.result.answer) == "ottawa"
    assert s2.provenance["Canada is the country of citizenship of Marie"] == "web:one.com+web:two.com"


def test_the_web_is_asked_only_after_the_primary_source_has_nothing(tmp_path):
    a = "https://a.one.com/m"
    w = web([Hit(a, "Marie", "")], {a: person_page("Marie", nationality="Canada")}, tmp_path)
    r, _s = loop(w, primary=DictSource({"marie": ["canada is the country of citizenship of marie"]}))
    assert r.result.verified and w.backend.calls == []                              # Wikidata answered: no search
    r2, _s2 = loop(w)
    assert w.backend.calls == ['"Marie" country of citizenship', "Marie nationality"]   # the stalled entity, the hop's label, then as a person types it


def test_sleep_keeps_a_web_fact_only_at_two_sites():
    t = {"wikidata", "web"}
    assert trusted_source("web:a.com+web:b.org", t) and trusted_source("wikidata (via web)", t)
    assert not trusted_source("web:a.com", t) and not trusted_source("web:a.com+web:b.org (latent)", t)
    assert not trusted_source("web:a.com+web:b.org", {"wikidata"})


def test_the_ask_loop_hands_the_web_to_the_walk_and_the_record_names_the_sites(tmp_path):
    from ask import AskLoop
    from test_ask import FakeEmitter
    a = "https://a.one.com/m"; b = "https://b.two.com/m"
    by = {a: person_page("Marie", nationality="Canada"), b: person_page("Marie", nationality="Canada")}
    w = web([Hit(a, "Marie", ""), Hit(b, "Marie", "")], by, tmp_path)
    lp = AskLoop(FakeEmitter({Q: ("marie", ["country of citizenship", "capital"])}), world=ProvStore(STORE),
                 source=DictSource({}), lexicon=False, run_fn=faithful_vm([]), web=w)
    rec = lp.ask(Q)
    assert rec["verified"] and normalize(rec["answer"]) == "ottawa"
    assert [l["source"] for l in rec["learned"]] == ["web:one.com+web:two.com"]
    fetches = [n for n in rec["graph"]["nodes"] if n["kind"] == "fetch"]
    assert [n["source"] for n in fetches] == ["test", "web"]                       # Wikidata's stand-in first, then the web


def test_a_page_s_own_metadata_is_not_a_fact_about_its_subject():
    """Live, 2026-09-24: Wikipedia's JSON-LD is an Article named 'Ada Lovelace' authored by 'Contributors
    to Wikimedia projects' -- read as a Person, that is the author of Ada Lovelace."""
    blocks = [json.dumps({"@type": "Article", "name": "Ada Lovelace", "author": {"name": "Contributors to Wikimedia projects"},
                          "datePublished": "2001-03-12"})]
    assert jsonld_facts("Ada Lovelace", blocks) == []


def test_a_value_must_be_of_the_relation_s_kind_and_dates_are_written_one_way():
    """Live, 2026-09-24: the reader wrote '1815-1852' as a date of birth and '112 languages' as a place
    of birth; both occur on the page, neither is one. '10 December 1815' is one, in another spelling."""
    from web_source import iso_date, plausible
    assert iso_date("10 December 1815") == iso_date("December 10, 1815") == iso_date("10 décembre 1815") == "1815-12-10"
    assert iso_date("1815-1852") is None and not plausible("date", "1815-1852")
    assert not plausible("name", "112 languages") and plausible("name", "London") and plausible("number", "1,234")


def test_one_date_in_two_spellings_is_grounded_and_a_born_on_sentence_is_read():
    from lfm_source import grounded
    passage = "Marie Curie, née Maria Sklodowska, was born in Warsaw on November 7, 1867, the daughter of a teacher."
    assert grounded("7 November 1867", passage) and grounded("1867-11-07", passage) and not grounded("1867-11-08", passage)
    got = {(t.rel, t.obj) for t, _s in frame_facts("Marie Curie", passage)}
    assert ("date of birth", "1867-11-07") in got and ("place of birth", "Warsaw") in got


def test_the_backends_read_their_documented_shapes_and_nothing_else():
    from web_source import BraveBackend, SearxngBackend
    brave_body = {"web": {"results": [{"title": "<strong>Ada</strong> Lovelace", "url": "https://x.org/ada",
                                        "description": "Ada Lovelace (<strong>born</strong> 10 December 1815)",
                                        "extra_snippets": ["She was a mathematician."]}]},
                  "summarizer": {"key": "an answer box is never read"}}
    seen = {}

    def http(url, headers=None, **kw):
        seen["url"], seen["headers"] = url, headers
        return ("application/json", json.dumps(brave_body).encode())
    hits = BraveBackend(key="k", http=http).search("Ada Lovelace", 5)
    assert seen["headers"]["X-Subscription-Token"] == "k" and "extra_snippets=true" in seen["url"]
    assert [(h.url, h.title, h.snippet, h.extra) for h in hits] == [
        ("https://x.org/ada", "Ada Lovelace", "Ada Lovelace (born 10 December 1815)", ["She was a mathematician."])]
    sx = SearxngBackend("http://127.0.0.1:8888", http=lambda url, headers=None, **kw: (
        "application/json", json.dumps({"results": [{"url": "https://y.com/a", "title": "A", "content": "c"}],
                                        "answers": ["never read"]}).encode()))
    assert [(h.url, h.snippet) for h in sx.search("A")] == [("https://y.com/a", "c")]
    assert BraveBackend(key="k", http=lambda *a, **k: None).search("x") is None      # down is not an answer


def test_the_key_is_read_from_the_environment_or_a_gitignored_env_file(tmp_path, monkeypatch):
    import web_source
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False); monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("OTHER=1\nBRAVE_SEARCH_API_KEY=\"from-file\"\n", encoding="utf-8")
    monkeypatch.setattr(web_source, "ENV_FILES", (tmp_path / "missing.env", f))
    assert web_source.env_key("BRAVE_SEARCH_API_KEY") == "from-file"
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "from-env")
    assert web_source.env_key("BRAVE_SEARCH_API_KEY") == "from-env"


def test_a_licensed_backend_s_searches_are_archived_and_each_site_keeps_its_evidence(tmp_path):
    a = "https://a.one.com/m"
    by = {a: person_page("Marie", nationality="Canada")}
    lic = FakeBackend([Hit(a, "Marie", "")]); lic.license = "test: storage permitted"
    w = WebSource(lic, cache_dir=tmp_path / "lic", http=pages(by))
    w.facts("Marie", relations=["country of citizenship"])
    arch = [json.loads(l) for l in (tmp_path / "lic" / "searches.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["q"] for r in arch] == ['"Marie" country of citizenship', "Marie nationality"] and arch[0]["hits"][0]["url"] == a
    ev = w.tally[normalize("Canada is the country of citizenship of Marie")]["one.com"]
    assert ev["url"] == a and ev["readers"] == ["jsonld"] and ev["t"] > 0
    plain = web([Hit(a, "Marie", "")], by, tmp_path / "plain")                    # no licence: nothing archived
    plain.facts("Marie")
    assert not (tmp_path / "plain" / "searches.jsonl").exists()


def test_brave_keeps_under_the_plan_s_rate():
    import time as _t
    from web_source import BraveBackend
    b = BraveBackend(key="k", http=lambda *a, **k: ("application/json", b'{"web": {"results": []}}'), rate_per_s=20)
    t0 = _t.monotonic()
    for _ in range(5):
        b.search("x")
    assert _t.monotonic() - t0 >= 4 / 20 - 0.01                                    # 5 calls, 4 gaps of 1/20 s


def _place_page(value):
    return person_page("Ada Lovelace", birthPlace=value)


def test_one_place_in_two_spellings_is_one_claim_and_two_places_are_two(tmp_path):
    """Live, 2026-09-24: 'London, England, UK' (one site) and 'London, England' (another) never agreed."""
    u1, u2, u3 = "https://a.one.com/x", "https://b.two.com/x", "https://c.three.com/x"
    w = web([Hit(u1, "", ""), Hit(u2, "", "")], {u1: _place_page("London, England, UK"), u2: _place_page("London, England")},
            tmp_path / "same")
    got = [(t.obj, t.rel) for t in w.facts("Ada Lovelace", relations=["place of birth"])]
    k = normalize("London, England is the place of birth of Ada Lovelace")
    assert got == [("London, England", "place of birth")] and k not in w.held
    assert w.provenance[k] == "web:one.com+web:two.com"
    x = web([Hit(u1, "", ""), Hit(u2, "", ""), Hit(u3, "", "")],
            {u1: _place_page("London, England"), u2: _place_page("London, Ontario"), u3: _place_page("London")}, tmp_path / "differ")
    got = {t.obj for t in x.facts("Ada Lovelace", relations=["place of birth"])}
    assert got == {"London, England", "London, Ontario", "London"}                  # they disagree: nothing merged
    assert all(_k in x.held for _k in [normalize(f"{o} is the place of birth of Ada Lovelace") for o in got])


def test_death_is_read_in_the_ways_the_web_writes_it():
    got = lambda s, e: {(t.rel, t.obj) for t, _s in frame_facts(e, s)}
    assert got("Gino Marchetti died on April 29, 2019, in Philadelphia.", "Gino Marchetti") == {
        ("date of death", "2019-04-29"), ("place of death", "Philadelphia")}
    assert got("Heinrich Kayser died October 14, 1940 in Bonn.", "Heinrich Kayser") == {
        ("date of death", "1940-10-14"), ("place of death", "Bonn")}
    assert got("Paul Krassner died at his home in Desert Hot Springs, California.", "Paul Krassner") == {
        ("place of death", "Desert Hot Springs, California")}


def _dated(value, key="deathDate"):
    return person_page("Dorothy Malone", **{key: value})


def test_a_finer_value_vouches_for_a_coarser_one_and_never_the_reverse(tmp_path):
    u1, u2 = "https://a.one.com/x", "https://b.two.com/x"
    w = web([Hit(u1, "", ""), Hit(u2, "", "")], {u1: _dated("2018"), u2: _dated("2018-01-19")}, tmp_path)
    got = [t.obj for t in w.facts("Dorothy Malone", relations=["date of death"]) if t.rel == "date of death"]
    assert got == ["2018"]                                   # the day is one site's; the year is both sites'
    assert w.provenance[normalize("2018 is the date of death of Dorothy Malone")] == "web:one.com+web:two.com"


def test_a_place_in_its_own_commas_vouches_for_the_region():
    w = WebSource(FakeBackend([]), cache_dir=None)
    assert w._finer("name", "Syracuse, New York", "New York") and not w._finer("name", "New York", "Syracuse, New York")
    assert not w._finer("name", "London, Ontario", "England")


def test_one_site_s_different_value_is_not_a_claim_unless_dissent_is_a_veto(tmp_path):
    us = [f"https://s{i}.site{i}.com/x" for i in range(3)]
    by = {us[0]: _dated("1935-04-28"), us[1]: _dated("1935-04-28"), us[2]: _dated("1935-04-01")}
    w = web([Hit(u, "", "") for u in us], by, tmp_path / "maj")
    got = [t.obj for t in w.facts("Dorothy Malone", relations=["date of death"]) if t.rel == "date of death"]
    assert got == ["1935-04-28"] and w.last["facts"][normalize("1935-04-28 is the date of death of Dorothy Malone")]["dissent"] == ["1935-04-01"]
    v = web([Hit(u, "", "") for u in us], by, tmp_path / "veto", dissent_blocks=True)
    got = sorted(t.obj for t in v.facts("Dorothy Malone", relations=["date of death"]) if t.rel == "date of death")
    assert got == ["1935-04-01", "1935-04-28"]              # both reach the store: the walk refuses on the split


def test_a_cut_off_snippet_is_read_only_where_it_is_whole():
    from web_source import untruncate
    assert untruncate("Tom Cruise was born on July 3, 1962 in Syracuse, New Yor…") == "Tom Cruise was born on July 3, 1962 in Syracuse"
    assert untruncate("A whole sentence.") == "A whole sentence."


def test_sites_that_copied_one_text_are_one_witness(tmp_path):
    """Live, 2026-09-24: four sites gave one death date in the same words -- an old Wikipedia lead."""
    same = "Agha Hashar Kashmiri (3 April 1879 – 28 April 1935) was an eminent Urdu poet, playwright and dramatist of his age."
    other = "Agha Hashar Kashmiri died on 28 April 1935 in Lahore after a long illness that kept him from the stage."
    hits = [Hit("https://a.one.com/x", "", same), Hit("https://b.two.com/x", "", same)]
    w = web(hits, {}, tmp_path / "copy")
    w.facts("Agha Hashar Kashmiri", relations=["date of death"])
    k = normalize("1935-04-28 is the date of death of Agha Hashar Kashmiri")
    assert k in w.held and w.last["facts"][k]["witnesses"] == 1 and w.last["facts"][k]["sites"] == ["one.com", "two.com"]
    w3 = web(hits + [Hit("https://c.three.com/x", "", other)], {}, tmp_path / "indep")
    w3.facts("Agha Hashar Kashmiri", relations=["date of death"])
    assert k not in w3.held and w3.last["facts"][k]["witnesses"] == 2


def test_the_live_wikipedia_article_speaks_for_its_family_not_its_mirrors(tmp_path):
    lead = {"type": "standard", "extract": "Agha Hashar Kashmiri (3 April 1879 – 1 April 1935) was an Urdu poet.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Agha_Hashar_Kashmiri"}}}
    mirror = "https://alchetron.com/Agha-Hashar-Kashmiri"

    def http(url, headers=None, **kw):
        if "rest_v1/page/summary" in url:
            return ("application/json", json.dumps(lead).encode())
        return None
    w = WebSource(FakeBackend([Hit(mirror, "", "Agha Hashar Kashmiri died on 28 April 1935 in Lahore, British India.")]),
                  cache_dir=tmp_path, http=http)
    w.facts("Agha Hashar Kashmiri", relations=["date of death"])
    facts = {k: f for k, f in w.last["facts"].items() if "date of death" in k}
    assert list(facts) == [normalize("1935-04-01 is the date of death of Agha Hashar Kashmiri")]
    assert facts[list(facts)[0]]["sites"] == ["wikipedia.org"]


def test_one_place_with_two_true_qualifiers_is_one_claim(tmp_path):
    u1, u2 = "https://a.one.com/x", "https://b.two.com/x"
    w = WebSource(FakeBackend([Hit(u1, "", ""), Hit(u2, "", "")]), cache_dir=tmp_path, containment=object(),
                  http=pages({u1: person_page("Kumar Sangakkara", birthPlace="Matale, Sri Lanka"),
                              u2: person_page("Kumar Sangakkara", birthPlace="Matale, Central Province")}))
    w.items_up = lambda place: [{"central province", "sri lanka"}] if _fold_(place) == "matale" else []
    got = [t.obj for t in w.facts("Kumar Sangakkara", relations=["place of birth"]) if t.rel == "place of birth"]
    assert got == ["Matale"] and normalize("Matale is the place of birth of Kumar Sangakkara") not in w.held


def _fold_(s):
    from web_source import _fold
    return _fold(s)


def test_a_place_does_not_run_on_into_the_next_sentence_s_name():
    got = {(t.rel, t.obj) for t, _s in frame_facts("Josephine Cochrane", "Josephine Cochrane was born in Ashtabula County, Ohio Josephine")}
    assert ("place of birth", "Ashtabula County, Ohio") in got
