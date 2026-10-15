"""Build BibTeX entries from their source of record, never from memory.

    python tools/bib_from_ids.py paper/refs.seed paper/refs.bib

Each line of the seed file: `key | arxiv:ID | doi:DOI | title:TITLE` (one of the three), then `|`
and the title you mean to cite. The record is fetched (arXiv, Crossref, else a title search) and the
entry is written from it; a record whose title is not the one you meant is reported and left out.
"""
from __future__ import annotations

import sys
import time

from check_citations import first_family, from_arxiv, from_crossref, search, title_sim, words


def bibtex(key: str, rec: dict, aid: str | None) -> str:
    authors = " and ".join(rec["authors"]) or "Anonymous"
    f = {"title": "{" + rec["title"] + "}", "author": authors, "year": str(rec.get("year") or "")}
    if aid:
        f.update({"eprint": aid, "archivePrefix": "arXiv", "journal": f"arXiv preprint arXiv:{aid}"})
        etype = "article"
    else:
        if rec.get("venue"):
            f["journal"] = rec["venue"]
        if rec.get("id") and "/" in str(rec["id"]):
            f["doi"] = rec["id"]
        etype = "article"
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in f.items())
    return f"@{etype}{{{key},\n{body}\n}}\n"


def main(seed: str, out: str) -> int:
    entries, problems = [], []
    for line in open(seed, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, ref, meant = [x.strip() for x in line.split("|")[:3]]
        kind, _, val = ref.partition(":")
        aid = val if kind == "arxiv" else None
        rec = from_arxiv(val) if kind == "arxiv" else from_crossref(val) if kind == "doi" else search(val, "")
        time.sleep(0.5)
        if rec is None:
            problems.append(f"{key}: nothing found for {ref}")
            continue
        sim = title_sim(meant, rec["title"])
        if sim < 0.85:
            problems.append(f"{key}: {ref} is '{rec['title']}', not '{meant}' ({sim:.2f})")
            continue
        entries.append(bibtex(key, rec, aid))
        print(f"ok  {key:26} {rec['source']:16} {rec['title'][:70]} ({rec.get('year')})", flush=True)
    open(out, "w", encoding="utf-8").write("\n".join(entries))
    for p in problems:
        print("LEFT OUT", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
