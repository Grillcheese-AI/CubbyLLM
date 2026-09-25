"""The local model as a fact Source (2026-09-13, Nick: "lfm is probably already having a bunch of facts,
why don't we also use them?").

LFM2.5-2.6B -- the emitter's own base, a GGUF in the stand-in stack, local -- holds a long tail of
facts in its weights. `LfmSource` asks it, few-shot, for an entity's facts in the store's own form
(`X is the R of E`, R one of the property table's labels), parses the lines into Triples, and hands
them to the same gate every other Source feeds. Two things make it honest:

  * it is a PROPOSER of facts, never a judge: every line is parsed, resolved through the property table
    (a relation the table cannot name is dropped, not invented), gated (duplicate / contradiction /
    template) and stored with provenance `lfm (parametric)`; nothing it says is true by saying it;
  * its facts are LATENT: `latent = True` tells `learn_and_answer` that a verified chain resting on an
    LFM-only fact is not spoken -- it is refused as `latent_only` with the would-be answer on record --
    until a second source states the same fact (a later Wikidata / encyclopedia fetch that lands on the
    same string upgrades the provenance). Nick's rule from 2026-09-12: uncertain incoming content goes
    to a latent world to be verified later, never to the answer.

Wired: `LfmSource` is OFF the serving path (exp_r11 `--source lfm` is the measurement). `LfmReader`
(2026-09-24) is the web source's reader: the same model reading a passage for one relation, every value
it writes checked against the passage (`grounded`), credited to the passage's site.
"""
from __future__ import annotations

import hashlib, json, pathlib, re
from typing import Iterable

from cubbyllm.reasoning.planner import Triple, normalize, of_prefixes

CACHE = pathlib.Path(__file__).resolve().parent / "data" / "out" / "lfm_cache"

# the relations the prompt offers: common Wikidata labels for persons, places, works and organisations
RELATIONS = ("date of birth", "place of birth", "date of death", "place of death", "country of citizenship",
             "occupation", "educated at", "spouse", "father", "mother", "child", "employer", "award received",
             "member of", "position held", "instance of", "country", "located in the administrative territorial entity",
             "capital", "continent", "inception", "founded by", "headquarters location", "author", "genre",
             "performer", "publication date", "director", "cast member", "original language of film or TV show")

FEWSHOT = """Facts about Albert Einstein:
- 1879-03-14 is the date of birth of Albert Einstein
- Ulm is the place of birth of Albert Einstein
- 1955-04-18 is the date of death of Albert Einstein
- Princeton is the place of death of Albert Einstein
- physicist is the occupation of Albert Einstein
- Switzerland is the country of citizenship of Albert Einstein
- ETH Zurich is the educated at of Albert Einstein
- Nobel Prize in Physics is the award received of Albert Einstein

Facts about Bordeaux:
- France is the country of Bordeaux
- city is the instance of Bordeaux
- Gironde is the located in the administrative territorial entity of Bordeaux
- Europe is the continent of Bordeaux

Facts about Pride and Prejudice:
- Jane Austen is the author of Pride and Prejudice
- 1813 is the publication date of Pride and Prejudice
- novel is the instance of Pride and Prejudice
- English is the original language of film or TV show of Pride and Prejudice

Facts about {entity}:
- """

_LINE = re.compile(r"^\s*-?\s*(?P<obj>.+?)\s+is\s+the\s+(?P<body>.+?)\s*$")     # body = 'R of E', split where E is the entity
_BAD_OBJ = {"unknown", "n/a", "none", "not known", "?", "unclear", "not applicable", "not available"}


def parse_fact_lines(text: str, entity: str, label, max_facts: int, stats: dict | None = None) -> list[Triple]:
    """`X is the R of E` lines -> Triples about `entity`, R resolved by `label(rel) -> str | None` (None
    drops the line: the model does not get to invent relations). Shared by the recall source (`LfmSource`)
    and the reader (`LfmReader`); `stats` counts what was dropped and why."""
    st = stats if stats is not None else {}
    for k in ("lines", "parsed", "dropped_relation", "dropped_subject", "dropped_object"):
        st.setdefault(k, 0)
    out: list[Triple] = []; seen: set[tuple[str, str]] = set()
    ent = normalize(entity)
    for line in text.splitlines():
        if not line.strip():
            continue
        st["lines"] += 1
        m = _LINE.match(line)
        if not m:
            continue
        obj, body = m.group("obj").strip().strip('."'), m.group("body").strip().strip('."')
        rel = next((r for r in of_prefixes(body) if normalize(body[len(r) + 4:]) == ent), None)   # 'R of E': E is the entity
        if rel is None:                                     # the model drifted to another subject
            st["dropped_subject"] += 1; continue
        if not obj or normalize(obj) in _BAD_OBJ or "?" in obj or len(obj) > 80:
            st["dropped_object"] += 1; continue
        lab = label(rel)
        if lab is None:
            st["dropped_relation"] += 1; continue
        key = (normalize(lab), normalize(obj))
        if key in seen:
            continue
        seen.add(key); st["parsed"] += 1
        out.append(Triple(obj=obj, rel=lab, subj=entity))
        if len(out) >= max_facts:
            break
    return out


class LfmSource:
    """`facts(entity)` -> Triples the local model states about the entity, in the store's form; the
    wording questions (`relations` / `kind` / `wordings`) go to the property table like every Source."""

    name = "lfm"
    latent = True

    def __init__(self, gguf: str, aliases=None, relations: Iterable[str] = RELATIONS, max_new: int = 320,
                 cache_dir: pathlib.Path | str | None = CACHE, max_facts: int = 40) -> None:
        from emitter import LlamaCppEmitter
        from sources import PropertyAliases
        self.gguf = gguf
        self.em = LlamaCppEmitter(gguf, script_ban=False)      # a base model, few-shot: no chat render, no ban
        self.aliases = aliases if aliases is not None else PropertyAliases()
        self.relations_offered = tuple(relations)
        self.max_new, self.max_facts = int(max_new), int(max_facts)
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0                                          # model calls (the log's 'api calls' for this Source: 0 network)
        self.last: dict = {}
        self.model_calls = 0

    # -- the wording questions: the table's ---------------------------------------------------
    def relations(self, text: str) -> list[str]:
        return self.aliases.relations(text)

    def kind(self, label: str) -> str | None:
        return self.aliases.kind(label)

    def wordings(self, label: str) -> list[str]:
        return self.aliases.wordings(label)

    # -- the facts -----------------------------------------------------------------------------
    def prompt(self, entity: str) -> str:
        return FEWSHOT.replace("{entity}", entity)

    def _raw(self, entity: str) -> str:
        key = hashlib.sha256(f"{pathlib.Path(self.gguf).name}|{self.max_new}|{entity}".encode("utf-8")).hexdigest()[:24]
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.em.complete(self.prompt(entity), max_new_tokens=self.max_new, temperature=0.0, stop=["\n\n", "\nFacts about"])
        self.model_calls += 1
        if path:
            path.write_text(json.dumps({"entity": entity, "text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _label(self, rel: str) -> str | None:
        """The property table's label for the model's relation words: the label itself, or a wording that
        names exactly one property. Anything else is dropped -- the model does not get to invent relations."""
        r = normalize(rel)
        for lab in self.relations_offered:
            if normalize(lab) == r:
                return lab
        labs = self.aliases.relations(rel)
        return labs[0] if len(labs) == 1 else None

    def facts(self, entity: str) -> list[Triple]:
        self.last = {"entity": entity}
        return parse_fact_lines("- " + self._raw(entity), entity, self._label, self.max_facts, self.last)


# ── the reader (2026-09-24): the same model, reading a passage instead of recalling ───────────────
READ_FEWSHOT = """Write the facts the passage states about the subject for the relation asked, one per line, as
"<value> is the <relation> of <subject>". Write "- none" when the passage does not state it.

Passage: Marie Curie (7 November 1867 – 4 July 1934) was a Polish and naturalised-French physicist and chemist, born in Warsaw.
Subject: Marie Curie
Relation: place of birth
Facts:
- Warsaw is the place of birth of Marie Curie

Passage: The Louvre is the world's most-visited art museum, located in Paris, France.
Subject: The Louvre
Relation: inception
Facts:
- none

Passage: {passage}
Subject: {entity}
Relation: {relation}
Facts:
- """

_MONTHS = {m: i for i, m in enumerate(("january february march april may june july august september october "
                                         "november december").split(), 1)}
_MONTHS.update({m: i for i, m in enumerate(("janvier février mars avril mai juin juillet août septembre octobre "
                                              "novembre décembre").split(), 1)})
_ISO = re.compile(r"^(?P<y>\d{4})(?:-(?P<m>\d{2})(?:-(?P<d>\d{2}))?)?$")


def grounded(value: str, passage: str) -> bool:
    """Does the passage SAY this value? The value's words in order, or -- for an ISO date, which the
    reader writes in the store's form -- the year, the month (by name or number) and the day, all in
    the passage. What the passage does not say, the reader may not write."""
    p = normalize(passage)
    v = normalize(value)
    if v and f" {v} " in f" {p} ":
        return True
    parts = _date_parts(value)
    if parts is None:
        return False
    y, mo, d = parts
    words = set(p.split())
    if y not in words:
        return False
    if mo and not ({k for k, i in _MONTHS.items() if i == mo} & words or str(mo) in words):
        return False
    return not d or str(d) in words or f"{d:02d}" in words


_DMY = re.compile(r"^(?P<d>\d{1,2})(?:er)? (?P<m>[^\W\d_]+) (?P<y>\d{4})$")
_MDY = re.compile(r"^(?P<m>[^\W\d_]+) (?P<d>\d{1,2}),? (?P<y>\d{4})$")


def _date_parts(value: str) -> tuple[str, int | None, int | None] | None:
    """(year, month, day) of a date written as ISO, '7 November 1867' or 'November 7, 1867' -- the
    passage and the reader may spell one date two ways and it is still the date the passage says."""
    v = " ".join((value or "").split())
    m = _ISO.match(v)
    if m:
        return m["y"], int(m["m"]) if m["m"] else None, int(m["d"]) if m["d"] else None
    m = _DMY.match(v) or _MDY.match(v)
    if m and m["m"].lower() in _MONTHS:
        return m["y"], _MONTHS[m["m"].lower()], int(m["d"])
    return None


class LfmReader:
    """`read(entity, relation, passage)` -> the Triples the passage states for that relation, each one
    GROUNDED (its value occurs in the passage) -- a proposer that may only point at text it was shown.
    The web source (`standin/web_source.py`) credits a read to the passage's site, and a site is one
    witness among the two it needs; nothing read here is spoken on this reader's word."""

    name = "lfm-reader"

    def __init__(self, gguf: str, aliases=None, max_new: int = 96, cache_dir: pathlib.Path | str | None = CACHE,
                 n_gpu_layers: int = 0, emitter=None) -> None:
        self.gguf = gguf
        if emitter is None:
            from emitter import LlamaCppEmitter
            emitter = LlamaCppEmitter(gguf, script_ban=False, n_gpu_layers=n_gpu_layers)   # CPU by default: the GPU is the emitter's
        self.em = emitter
        if aliases is None:
            from sources import PropertyAliases
            aliases = PropertyAliases()
        self.aliases = aliases
        self.max_new = int(max_new)
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.model_calls = 0
        self.last: dict = {}

    def _label(self, rel: str) -> str | None:
        r = normalize(rel)
        for lab in RELATIONS:
            if normalize(lab) == r:
                return lab
        labs = self.aliases.relations(rel)
        return labs[0] if len(labs) == 1 else None

    def _raw(self, prompt: str) -> str:
        key = hashlib.sha256(f"read|{pathlib.Path(self.gguf).name}|{self.max_new}|{prompt}".encode("utf-8")).hexdigest()[:24]
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.em.complete(prompt, max_new_tokens=self.max_new, temperature=0.0, stop=["\n\n", "\nPassage:"])
        self.model_calls += 1
        if path:
            path.write_text(json.dumps({"prompt": prompt[-400:], "text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def read(self, entity: str, relation: str, passage: str) -> list[Triple]:
        passage = " ".join((passage or "").split())[:1200]
        prompt = READ_FEWSHOT.replace("{passage}", passage).replace("{entity}", entity).replace("{relation}", relation)
        st: dict = {}
        out = parse_fact_lines("- " + self._raw(prompt), entity, self._label, 4, st)
        want = normalize(self._label(relation) or relation)
        kept = [t for t in out if normalize(t.rel) == want and grounded(t.obj, passage)]
        st["ungrounded"] = sum(1 for t in out if normalize(t.rel) == want and not grounded(t.obj, passage))
        st["off_relation"] = sum(1 for t in out if normalize(t.rel) != want)
        self.last = st
        return kept
