"""Check every reference in a BibTeX file against its source of record before a paper leaves the lab.

    python tools/check_citations.py paper/refs.bib [--report report.md]

Each entry is resolved from what it carries — an arXiv id (eprint, or arXiv:XXXX in any field, or an
arxiv.org URL), else a DOI, else its title (Crossref, then Semantic Scholar) — and the record found
must match the entry: title (normalised, similarity >= 0.90), the first author's family name, and
the year (within one, for preprints that became proceedings). Anything that cannot be resolved, or
resolves to something else, fails: remove it or fix it from the record, never from memory.

Standard library only (urllib); exit status 1 when any entry fails.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "citation-checker/1.0 (mailto:research@example.org)"}
ARXIV_RE = re.compile(r"(?:arxiv[:/ ]|abs/|pdf/)?\b(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b", re.I)


# ---------------------------------------------------------------------------------------- BibTeX
def parse_bib(text: str) -> list[dict]:
    """A small BibTeX reader: @type{key, field = {value} | "value" | number, ...} with nested braces."""
    out, i = [], 0
    while True:
        at = text.find("@", i)
        if at < 0:
            return out
        brace = text.find("{", at)
        if brace < 0:
            return out
        etype = text[at + 1:brace].strip().lower()
        depth, j = 0, brace
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[brace + 1:j]
        i = j + 1
        if etype in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        e = {"type": etype, "key": key.strip()}
        k = 0
        while k < len(rest):
            m = re.match(r"\s*([A-Za-z_\-]+)\s*=\s*", rest[k:])
            if not m:
                break
            name = m.group(1).lower()
            k += m.end()
            if k < len(rest) and rest[k] == "{":
                d, s0 = 0, k
                while k < len(rest):
                    if rest[k] == "{":
                        d += 1
                    elif rest[k] == "}":
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                val = rest[s0 + 1:k]
                k += 1
            elif k < len(rest) and rest[k] == '"':
                s0 = k
                k = rest.index('"', k + 1)
                val = rest[s0 + 1:k]
                k += 1
            else:
                m2 = re.match(r"[^,]*", rest[k:])
                val = m2.group(0).strip()
                k += m2.end()
            e[name] = re.sub(r"\s+", " ", val).strip()
            m3 = re.match(r"\s*,", rest[k:])
            if m3:
                k += m3.end()
        out.append(e)


def words(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", s)
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def title_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


def first_family(authors: str) -> str:
    first = re.split(r"\s+and\s+", authors or "")[0].strip()
    if "," in first:
        return words(first.split(",")[0]).split()[-1] if words(first.split(",")[0]) else ""
    w = words(first).split()
    return w[-1] if w else ""


# ---------------------------------------------------------------------------------------- sources
def get(url: str, tries: int = 3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception as e:                                       # rate limits, timeouts: back off
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


def from_arxiv(aid: str) -> dict | None:
    raw = get("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns)
        if not title or title.strip().lower() == "error":
            return None
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
        year = int(entry.findtext("a:published", default="0", namespaces=ns)[:4] or 0)
        return {"source": "arXiv", "id": aid, "title": " ".join(title.split()), "authors": authors, "year": year}
    return None


def from_crossref(doi: str) -> dict | None:
    try:
        m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    except Exception:
        return None
    return crossref_record(m)


def crossref_record(m: dict) -> dict:
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "") for a in m.get("author", [])]
    year = 0
    for k in ("published-print", "published-online", "issued", "created"):
        dp = (m.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break
    return {"source": "Crossref", "id": m.get("DOI"), "title": " ".join((m.get("title") or [""])[0].split()),
            "authors": authors, "year": year, "venue": (m.get("container-title") or [""])[0]}


def search(title: str, author: str) -> dict | None:
    """By title: Crossref's bibliographic search, then Semantic Scholar's; the closest title wins."""
    best = None
    try:
        q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author}", "rows": 5})
        for m in json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]:
            r = crossref_record(m)
            if best is None or title_sim(title, r["title"]) > title_sim(title, best["title"]):
                best = r
    except Exception:
        pass
    if best is None or title_sim(title, best["title"]) < 0.9:
        try:
            q = urllib.parse.urlencode({"query": title, "limit": 5, "fields": "title,authors,year,externalIds,venue"})
            for p in json.loads(get("https://api.semanticscholar.org/graph/v1/paper/search?" + q)).get("data", []):
                r = {"source": "Semantic Scholar", "id": (p.get("externalIds") or {}).get("DOI") or p.get("paperId"),
                     "title": p.get("title") or "", "authors": [a.get("name", "") for a in p.get("authors", [])],
                     "year": p.get("year") or 0, "venue": p.get("venue") or ""}
                if best is None or title_sim(title, r["title"]) > title_sim(title, best["title"]):
                    best = r
        except Exception:
            pass
    return best


def arxiv_id(e: dict) -> str | None:
    if e.get("eprint") and (e.get("archiveprefix", "arxiv").lower() == "arxiv"):
        return e["eprint"].strip()
    for f in ("journal", "url", "note", "howpublished", "booktitle"):
        v = e.get(f, "")
        if "arxiv" in v.lower():
            m = ARXIV_RE.search(v.replace("arXiv preprint", ""))
            if m:
                return m.group(1)
    return None


# ---------------------------------------------------------------------------------------- check
def check(e: dict) -> dict:
    title, authors, year = e.get("title", ""), e.get("author", ""), int(re.sub(r"\D", "", e.get("year", "0")) or 0)
    rec, how = None, ""
    aid = arxiv_id(e)
    if aid:
        rec, how = from_arxiv(aid), f"arXiv:{aid}"
    if rec is None and e.get("doi"):
        rec, how = from_crossref(e["doi"]), f"doi:{e['doi']}"
    if rec is None and title:
        rec, how = search(title, first_family(authors)), "title search"
    out = {"key": e["key"], "how": how, "entry": {"title": title, "first_author": first_family(authors), "year": year}}
    if rec is None:
        return {**out, "status": "UNRESOLVED", "why": "no record found"}
    out["record"] = {k: rec.get(k) for k in ("source", "id", "title", "year", "venue")}
    out["record"]["first_authors"] = rec["authors"][:3]
    problems = []
    sim = title_sim(title, rec["title"])
    if sim < 0.90:
        problems.append(f"title differs ({sim:.2f})")
    fam = first_family(authors)
    rec_fams = [words(a).split()[-1] for a in rec["authors"] if words(a)]
    if fam and rec_fams and fam not in rec_fams[:1] and fam not in " ".join(rec_fams[:1]):
        problems.append(f"first author '{fam}' is not the record's first author '{rec_fams[0]}'")
    if year and rec.get("year") and abs(year - int(rec["year"])) > 1:
        problems.append(f"year {year} vs record {rec['year']}")
    return {**out, "status": "OK" if not problems else "MISMATCH", "why": "; ".join(problems)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    entries = parse_bib(open(a.bib, encoding="utf-8").read())
    keys = [e["key"] for e in entries]
    dup = {k for k in keys if keys.count(k) > 1}
    results = []
    for e in entries:
        r = check(e)
        results.append(r)
        print(f"{r['status']:10} {r['key']:28} {r['how']:26} {r.get('why', '')}", flush=True)
        time.sleep(0.4)                                             # be polite to the APIs
    bad = [r for r in results if r["status"] != "OK"]
    lines = [f"# Citation check: {a.bib}", "", f"{len(results)} entries, {len(results) - len(bad)} OK, {len(bad)} failed"
             + (f"; duplicate keys: {', '.join(sorted(dup))}" if dup else ""), "",
             "| key | status | resolved by | record title | why |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['key']} | {r['status']} | {r['how']} | {(r.get('record') or {}).get('title', '')} | {r.get('why', '')} |")
    if a.report:
        open(a.report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    print(f"\n{len(results) - len(bad)} of {len(results)} OK" + (f"; duplicates {sorted(dup)}" if dup else ""))
    return 1 if bad or dup else 0


if __name__ == "__main__":
    sys.exit(main())
