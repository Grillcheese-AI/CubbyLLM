"""validation/fact_store/ttl_count.py and dump_fetch.zenodo -- order 0055 step 1 on the WikiKGQA dump (Turtle), on a
dump written here in Wikidata's RDF layout (no network).

Pinned: the count cuts the stream at whole entities however the parts are cut; it counts what the JSON pass counts
(statements by rank, novalue apart from a qualifier's no value, time qualifiers, references by hash, labels, sitelinks,
scholarly articles, statements by their property's datatype); samples the same ids as the JSON pass; keeps the dump's
date and its redirects; and checks the parts against Zenodo's md5.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fact_store import dump_count as C  # noqa: E402
from fact_store import dump_fetch as F  # noqa: E402
from fact_store import ttl_count as T  # noqa: E402

HEAD = """@prefix wd: <http://www.wikidata.org/entity/> .
@prefix wds: <http://www.wikidata.org/entity/statement/> .

wikibase:Dump a schema:Dataset,
\t\towl:Ontology ;
\tschema:dateModified "2025-12-29T23:00:38Z"^^xsd:dateTime ;
\towl:imports <http://wikiba.se/ontology-1.0.owl> .
"""


def _item(i: int) -> str:
    q = f"Q{i}"
    scholarly = "wd:Q13442814" if i % 10 == 0 else "wd:Q5"
    rank = "wikibase:DeprecatedRank" if i % 7 == 0 else "wikibase:NormalRank"
    ref = hashlib.sha1(str(i % 3).encode()).hexdigest()                        # three references, shared
    return f"""
data:{q} a schema:Dataset ;
\tschema:about wd:{q} ;
\tschema:version "{1000 + i}"^^xsd:integer .

wd:{q} a wikibase:Item .

<https://en.wikipedia.org/wiki/X{i}> a schema:Article ;
\tschema:about wd:{q} ;
\tschema:isPartOf <https://en.wikipedia.org/> .

wd:{q} rdfs:label "x{i}"@en ;
\trdfs:label "x{i}"@de ;
\tskos:altLabel "a"@fr,
\t\t"b"@fr ;
\twdt:P31 {scholarly} .

wd:{q} p:P31 wds:{q}-a .

wds:{q}-a a wikibase:Statement, wikibase:BestRank ;
\twikibase:rank wikibase:NormalRank ;
\tps:P31 {scholarly} .

wd:{q} p:P39 wds:{q}-b .

wds:{q}-b a wikibase:Statement ;
\twikibase:rank {rank} ;
\tps:P39 wd:Q11696 ;
\tpq:P580 "2001-01-01T00:00:00Z"^^xsd:dateTime ;
\tpq:P642 _:b{i} ;
\ta wdno:P582 ;
\tprov:wasDerivedFrom wdref:{ref} .

wd:{q} p:P1448 wds:{q}-c .

wds:{q}-c a wikibase:Statement ;
\ta wdno:P1448 ;
\twikibase:rank wikibase:NormalRank .
"""


PROPS = """
data:P39 a schema:Dataset ;
\tschema:version "7"^^xsd:integer .

wd:P39 a wikibase:Property ;
\twikibase:propertyType wikibase:WikibaseItem .

data:P1448 a schema:Dataset ;
\tschema:version "8"^^xsd:integer .

wd:P1448 a wikibase:Property ;
\twikibase:propertyType wikibase:Monolingualtext .

data:Q99999 a schema:Dataset ;
\tschema:about wd:Q99999 ;
\tschema:version "9"^^xsd:integer .

wd:Q99999 owl:sameAs wd:Q3 .
"""


def _dump(n=60):
    text = HEAD + "".join(_item(i) for i in range(1, n + 1)) + PROPS
    return bz2.compress(text.encode())


def test_the_turtle_count_counts_what_the_json_pass_counts_however_the_parts_are_cut(tmp_path):
    blob = _dump()
    for cut in (97, 5000):
        d = tmp_path / f"p{cut}"
        d.mkdir()
        for k in range(0, len(blob), cut):
            (d / f"part_{k // cut:05d}").write_bytes(blob[k:k + cut])
        (d / "manifest.json").write_text(json.dumps({"md5": hashlib.md5(blob).hexdigest()}))
        res = C.run(str(d), str(tmp_path / f"o{cut}"), workers=2, fmt=T.Ttl)
        c = res["counts"]
        assert c["entities:item"] == 60 and c["entities:property"] == 2 and c["revisions"] == 63   # + the redirect's header
        assert c["statements:item"] == 180 and c["rank:deprecated"] == 8 and c["rank:normal"] == 172
        assert c["snak:value"] == 120 and c["snak:novalue"] == 60
        assert c["qualifier_snaks"] == 180 and c["qualifier_snaks:time"] == 120      # 60 P580, 60 P642, 60 no P582
        assert c["reference_blocks"] == 60 and res["references_distinct_est"] == 3
        assert c["labels:en"] == 60 and "labels:de" not in c and c["aliases:fr"] == 120 and c["sitelinks:enwiki"] == 60
        assert c["items:scholarly"] == 6 and c["statements:scholarly"] == 18
        assert res["statements_by_datatype"] == {"wikibase:WikibaseItem": 60, "wikibase:Monolingualtext": 60, "?": 60}
        assert res["hash_algo"] == "md5" and res["hash_ok"] is True and res["ended_at_close"] is True
        assert res["dump_date"] == "2025-12-29T23:00:38Z" and c["redirects"] == 1
        side = gzip.open(tmp_path / f"o{cut}" / "side.tsv.gz", "rt").read()
        assert "redirect\twd:Q99999\twd:Q3" in side
        got = gzip.open(tmp_path / f"o{cut}" / "sample_1pct.ttl.gz", "rb").read().split(b"\ndata:")
        ids = sorted(x.split(b" ", 1)[0].replace(b"data:", b"").decode() for x in got if x.strip())
        assert ids == sorted(f"Q{i}" for i in range(1, 61) if C.in_sample(f"Q{i}"))


def test_zenodo_reads_the_records_file_and_its_md5():
    rec = {"metadata": {"publication_date": "2026-09-08", "license": {"id": "cc-zero"}},
           "files": [{"key": "a.ttl.bz2", "size": 12, "checksum": "md5:abc", "links": {"self": "https://z/a/content"}}]}
    info = F.zenodo("1", "a.ttl.bz2", opener=lambda u, h=None, timeout=120: io.BytesIO(json.dumps(rec).encode()))
    assert info["md5"] == "abc" and info["size"] == 12 and info["url"] == "https://z/a/content"
    assert info["licence"] == "cc-zero"
