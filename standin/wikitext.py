"""wikitext — an OFFLINE Source for search-and-learn: Wikipedia-style articles (title + text) read
through a fixed table of sentence frames into template facts with provenance.

Wired: WIRED (stand-in serve stack; implements `cubbyllm.reasoning.learn.Source`). No model, no
network: a frame is a regular expression over the article's opening, and a fact it yields names
the corpus, the article and the sentence it came from. What the frames do not say, the source
does not say -- exp_r12 (2026-09-12) measured the ceiling on DBpedia's 4.6M abstracts: 72% open
with 'X is a/an ...', 2.9% carry '(born DATE)'. The frames below are the ones whose relation is a
property label the emitter names (date of birth / death, place of birth, inception, located in
the administrative territorial entity, ...), in English and in French.

Corpora: a BeIR-style parquet (`_id`, `title`, `text`) and/or jsonl files (`title`, `text`,
optional `lang`). A title index (normalized title -> where the article is) is built once per
corpus and cached beside the data as `<name>.titles.json`; DBpedia's 4.6M titles index in ~10 s
from the parquet's title column and load in ~3 s.
"""
from __future__ import annotations

import json
import pathlib
import re

from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

_MONTHS_EN = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTHS_FR = "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
_MEN = {m.lower(): i for i, m in enumerate(_MONTHS_EN.split("|"), 1)}
_MFR = {m: i for i, m in enumerate(_MONTHS_FR.split("|"), 1)}
_DATE_EN = rf"(?P<m>{_MONTHS_EN}) (?P<d>\d{{1,2}}), (?P<y>\d{{4}})"
_DATE_FR = rf"(?P<d>1er|\d{{1,2}}) (?P<m>{_MONTHS_FR}) (?P<y>\d{{4}})"


def _iso(d: str, m: str, y: str, months: dict) -> str:
    return f"{int(y):04d}-{months[m.lower()]:02d}-{int(d.replace('er', '')):02d}"


# (relation label, compiled frame, how to read the object from the match). The frames run on the
# opening of the article (its first ~600 characters); each may fire once.
FRAMES_EN = [
    ("date of birth", re.compile(rf"\(born {_DATE_EN}"), lambda m: _iso(m["d"], m["m"], m["y"], _MEN)),
    ("date of birth", re.compile(rf"\({_DATE_EN}\s*[–-]\s*(?:{_MONTHS_EN}) \d{{1,2}}, \d{{4}}\)"), lambda m: _iso(m["d"], m["m"], m["y"], _MEN)),
    ("date of death", re.compile(rf"\((?:{_MONTHS_EN}) \d{{1,2}}, \d{{4}}\s*[–-]\s*{_DATE_EN}\)"), lambda m: _iso(m["d"], m["m"], m["y"], _MEN)),
    ("place of birth", re.compile(r"\bborn (?:on [A-Z][a-z]+ \d{1,2}, \d{4},? )?in (?P<o>[A-Z][\w'-]+(?:,? [A-Z][\w'-]+){0,3})"), lambda m: m["o"]),
    ("inception", re.compile(r"\b(?:was |were )?(?:founded|established|formed|incorporated) (?:on [A-Z][a-z]+ \d{1,2}, )?in (?P<o>\d{4})\b"), lambda m: m["o"]),
    ("located in the administrative territorial entity", re.compile(r"\b(?:is|was) an? (?:[a-z-]+ ){0,3}(?:village|town|city|municipality|commune|census-designated place|district|county) (?:located )?in (?P<o>[A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3})"), lambda m: m["o"]),
    # a person is at least two capitalised tokens, not followed by a lowercase word: 'by British author
    # Jane Doe' must not read 'British' (exp_r11 wikitext run 1 admitted 'British is the author of The Roar')
    ("director", re.compile(r"\bfilm directed by (?P<o>[A-Z][\w'-]+(?: [A-Z][\w'-]+){1,3})(?! [a-z])"), lambda m: m["o"]),
    ("author", re.compile(r"\b(?:novel|book|play|poem) (?:written )?by (?P<o>[A-Z][\w'-]+(?: [A-Z][\w'-]+){1,3})(?! [a-z])"), lambda m: m["o"]),
    ("capital", re.compile(r"\b(?P<o>[A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3}) is (?:the|its) capital\b"), lambda m: m["o"]),
]
FRAMES_FR = [
    ("date of birth", re.compile(rf"\bnée? le {_DATE_FR}"), lambda m: _iso(m["d"], m["m"], m["y"], _MFR)),
    ("date of death", re.compile(rf"\b(?:mort|morte|décédée?) le {_DATE_FR}"), lambda m: _iso(m["d"], m["m"], m["y"], _MFR)),
    ("place of birth", re.compile(r"\bnée? (?:le \S+ \S+ \d{4} )?à (?P<o>[A-ZÀ-Ý][\w'’-]+(?:[ -][A-ZÀ-Ý][\w'’-]+){0,3})"), lambda m: m["o"]),
    ("inception", re.compile(r"\b(?:fondée?|créée?|établie?) en (?P<o>\d{4})\b"), lambda m: m["o"]),
    ("located in the administrative territorial entity", re.compile(r"\best une (?:commune|ville|village) (?:française |suisse |belge )?(?:située |situé )?dans (?:le |la |les |l')?(?:département (?:de |du |des |d')|canton (?:de |du |des |d')|province (?:de |du |des |d'))?(?P<o>[A-ZÀ-Ý][\w'’-]+(?:[ -][A-ZÀ-Ý][\w'’-]+){0,3})"), lambda m: m["o"]),
]
OPENING = 600

# the words a question uses for the relations the frames read -- what `relations(text)` answers
# (lever 4 for an offline source: 'born' names two labels; the store decides, the host refuses two)
TRIGGERS = {
    "date of birth": ["born", "birth", "birth date", "birthday", "date of birth", "né", "née", "naissance", "date de naissance"],
    "date of death": ["died", "death", "death date", "date of death", "mort", "morte", "décès", "décédé", "date de décès"],
    "place of birth": ["born", "birthplace", "place of birth", "né", "née", "naissance", "lieu de naissance"],
    "inception": ["founded", "established", "formed", "inception", "founding", "creation", "fondée", "fondé", "création"],
    "located in the administrative territorial entity": ["located", "located in", "district", "county", "municipality", "situé", "située", "département"],
    "director": ["directed", "director", "directed by", "réalisateur", "réalisé"],
    "author": ["author", "written by", "writer", "auteur", "écrit par"],
    "capital": ["capital", "capitale"],
}
_TRIGGER_INDEX: dict[str, list[str]] = {}
for _rel, _words in TRIGGERS.items():
    for _w in _words:
        _TRIGGER_INDEX.setdefault(normalize(_w), []).append(_rel)
# the value kind each frame reads, by construction: what `learn.narrow_by_ask` narrows a two-sense
# wording ('born' -> date of birth / place of birth) by when the question asks 'when' or 'who'
KIND = {"date of birth": "date", "date of death": "date", "inception": "date", "place of birth": "name",
        "located in the administrative territorial entity": "name", "director": "name", "author": "name", "capital": "name"}


def title_key(s: str) -> str:
    """Lowercased, punctuation-stripped, ARTICLES KEPT: 'roar' must not match 'The Roar'
    (exp_r11 wikitext run 1 stored a novel's author under a TV series' seed)."""
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def read_frames(title: str, text: str, lang: str = "en") -> list[tuple[Triple, str]]:
    """(Triple, the sentence it came from) for every frame that fires on the article's opening."""
    head = " ".join((text or "")[:OPENING].split())
    out: list[tuple[Triple, str]] = []
    seen: set[str] = set()
    for rel, rx, obj in (FRAMES_FR if lang.startswith("fr") else FRAMES_EN):
        m = rx.search(head)
        if not m or rel in seen:
            continue
        o = obj(m)
        if not o or normalize(o) == normalize(title):
            continue
        seen.add(rel)
        sentence = head[max(0, head.rfind(".", 0, m.start()) + 1): head.find(".", m.end()) + 1 or None].strip()
        out.append((Triple(obj=o, rel=rel, subj=title), sentence))
    return out


class WikiTextSource:
    """`facts(entity)`: the frames read off the article whose title is the entity (exact,
    normalized), from any of the corpora, first corpus that has it. `last` carries the
    provenance of the last call: corpus, title, and one sentence per fact."""
    name = "wikitext"

    def __init__(self, parquet: str | pathlib.Path | None = None, jsonl: list[str | pathlib.Path] | None = None) -> None:
        self.corpora: list[tuple[str, str, pathlib.Path, dict[str, tuple]]] = []   # (name, kind, path, title -> locator)
        self.last: dict = {}
        if parquet:
            p = pathlib.Path(parquet); self.corpora.append((p.stem, "parquet", p, self._parquet_index(p)))
        for j in jsonl or []:
            p = pathlib.Path(j); self.corpora.append((p.stem, "jsonl", p, self._jsonl_index(p)))

    # -- indexes, built once and cached ------------------------------------------------------
    @staticmethod
    def _cache(p: pathlib.Path) -> pathlib.Path:
        return p.with_suffix(p.suffix + ".titles2.json")

    def _parquet_index(self, p: pathlib.Path) -> dict[str, tuple]:
        c = self._cache(p)
        if c.exists():
            return {k: tuple(v) for k, v in json.loads(c.read_text(encoding="utf-8")).items()}
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p); idx: dict[str, tuple] = {}
        for g in range(pf.num_row_groups):
            for i, t in enumerate(pf.read_row_group(g, columns=["title"]).column("title").to_pylist()):
                idx.setdefault(title_key(t), (g, i))
        c.write_text(json.dumps(idx), encoding="utf-8")
        return idx

    def _jsonl_index(self, p: pathlib.Path) -> dict[str, tuple]:
        c = self._cache(p)
        if c.exists():
            return {k: tuple(v) for k, v in json.loads(c.read_text(encoding="utf-8")).items()}
        idx: dict[str, tuple] = {}
        with open(p, "rb") as fh:
            off = 0
            for line in fh:
                try:
                    r = json.loads(line)
                    idx.setdefault(title_key(r.get("title", "")), (off, r.get("lang", "en")))
                except ValueError:
                    pass
                off += len(line)
        c.write_text(json.dumps(idx), encoding="utf-8")
        return idx

    # -- article access ------------------------------------------------------------------------
    def article(self, entity: str) -> tuple[str, str, str, str] | None:
        """(corpus, title, text, lang) or None."""
        key = title_key(entity)
        for name, kind, path, idx in self.corpora:
            loc = idx.get(key)
            if loc is None:
                continue
            if kind == "parquet":
                import pyarrow.parquet as pq
                g, i = loc
                # a BeIR parquet has ~1M rows per row group: keep the last group read (one at a time)
                if getattr(self, "_grp", None) is None or self._grp[0] != (name, g):
                    tb = pq.ParquetFile(path).read_row_group(g, columns=["title", "text"])
                    self._grp = ((name, g), tb.column("title"), tb.column("text"))
                _k, titles, texts = self._grp
                return name, titles[i].as_py(), texts[i].as_py() or "", "en"
            off, lang = loc
            with open(path, "rb") as fh:
                fh.seek(off); r = json.loads(fh.readline())
            return name, r.get("title", ""), r.get("text", ""), "fr" if ".fr" in str(lang) or str(lang).startswith("fr") else "en"
        return None

    def relations(self, text: str) -> list[str]:
        """The relation labels whose trigger words include this wording ('born' -> date of
        birth, place of birth): the source names what its own frames can read, nothing else."""
        return list(_TRIGGER_INDEX.get(normalize(text), []))

    def kind(self, label: str) -> str | None:
        return KIND.get(normalize(label))

    def facts(self, entity: str) -> list[Triple]:
        self.last = {"entity": entity, "corpus": None, "title": None, "sentences": {}}
        art = self.article(entity)
        if art is None:
            return []
        corpus, title, text, lang = art
        self.last.update(corpus=corpus, title=title, lang=lang)
        out = []
        for t, sentence in read_frames(title, text, lang):
            out.append(t); self.last["sentences"][f"{t.rel}|{t.obj}"] = sentence
        return out
