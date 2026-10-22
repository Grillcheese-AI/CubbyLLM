"""cubbyllm/ opens no socket and names no key file (order 0031): each source fetches only through what it is handed,
says so when it was handed nothing, and the vault writes its key only where the server says."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_the_wiki_world_fetches_a_missing_export_only_with_the_fetch_it_is_handed(tmp_path, monkeypatch):
    from cubbyllm.worlds import wikikg as wk
    monkeypatch.setattr(wk, "DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="cubby_serve.sources.hub"):
        wk.ensure_data(("triplets",))
    got = []
    wk.ensure_data(("triplets",), fetch=lambda url, dst: got.append((url, dst)))
    assert got == [(wk.HUB_PARQUET.format(cfg="triplets"), str(tmp_path / "triplets.parquet"))]


def test_an_online_wikidata_source_without_a_fetch_says_so_and_offline_needs_none():
    from cubbyllm.sources.wikidata import WikidataSource
    with pytest.raises(RuntimeError, match="fetch="):
        WikidataSource(cache_dir=None, offline=False)._get({"action": "wbgetentities", "ids": "Q1"})
    assert WikidataSource(cache_dir=None, offline=True)._get({"action": "wbgetentities", "ids": "Q1"}) is None
    urls = []
    src = WikidataSource(cache_dir=None, sleep_s=0.0, fetch=lambda url: urls.append(url) or b'{"entities": {}}')
    assert src._get({"action": "wbgetentities", "ids": "Q1"}) == {"entities": {}} and src.calls == 1
    assert urls and urls[0].startswith("https://www.wikidata.org/") and "format=json" in urls[0]


def test_the_web_source_has_no_transport_of_its_own():
    from cubbyllm.sources.web import BraveBackend, _no_http, backend_from_env
    with pytest.raises(RuntimeError, match="http="):
        _no_http("https://example.org/")
    seen = []
    b = BraveBackend(key="k", http=lambda url, headers=None, **kw: seen.append(url) or ("application/json", b'{"web": {"results": []}}'))
    b.search("x", 3)
    assert seen and seen[0].startswith(BraveBackend.URL)
    assert backend_from_env("auto", None) is None or backend_from_env("auto", None).http is _no_http


def test_a_news_source_without_a_fetch_treats_its_feeds_as_down(tmp_path):
    from cubbyllm.sources.news.source import NewsSource
    src = NewsSource(feeds={"f": "http://test/f"}, cache_dir=tmp_path)
    assert src._raw("f") is None and "RuntimeError" in src.errors["f"] and "fetch=" in src.errors["f"]
    ok = NewsSource(feeds={"f": "http://test/f"}, cache_dir=tmp_path / "2", fetch=lambda url: b"<rss/>")
    assert ok._raw("f") == b"<rss/>" and ok.calls == 1


def test_a_news_archive_without_a_fetch_keeps_its_files_and_fetches_nothing(tmp_path):
    from cubbyllm.sources.news.archive import Archive
    a = Archive(tmp_path / "a")
    with pytest.raises(RuntimeError, match="without a fetch"):
        a.fetch("https://example.org/feed")
    assert a.robots("example.org") is None                   # no robots.txt to read is not a yes and not a no
    row = a.poll({"id": "x", "url": "https://example.org/feed", "status": "active"})
    assert "error" in row and "RuntimeError" in row["error"]
    assert not list((tmp_path / "a").glob("????-??-??.jsonl"))


def test_the_vault_writes_its_key_only_where_it_is_told(tmp_path, monkeypatch):
    from cubbyllm.speech import vault
    monkeypatch.delenv(vault.KEY_ENV, raising=False)
    monkeypatch.setattr(vault, "KEY_FILE", None)
    with pytest.raises(RuntimeError, match="use_key_file"):
        vault.host_key()
    vault.use_key_file(tmp_path / "k" / "host.key")
    k = vault.host_key()
    assert (tmp_path / "k" / "host.key").read_text(encoding="ascii").strip() == k.hex() and vault.host_key() == k
    blob = vault.encrypt_bytes(b"hello")
    assert vault.decrypt_bytes(blob) == b"hello"
