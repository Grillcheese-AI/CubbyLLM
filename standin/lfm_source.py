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

Wired: OFF the serving path (exp_r11 `--source lfm` is the measurement); nothing imports this yet.
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
        self.last = {"entity": entity, "lines": 0, "parsed": 0, "dropped_relation": 0, "dropped_subject": 0, "dropped_object": 0}
        text = "- " + self._raw(entity)
        out: list[Triple] = []; seen: set[tuple[str, str]] = set()
        ent = normalize(entity)
        for line in text.splitlines():
            if not line.strip():
                continue
            self.last["lines"] += 1
            m = _LINE.match(line)
            if not m:
                continue
            obj, body = m.group("obj").strip().strip('."'), m.group("body").strip().strip('."')
            rel = next((r for r in of_prefixes(body) if normalize(body[len(r) + 4:]) == ent), None)   # 'R of E': E is the entity
            if rel is None:                                     # the model drifted to another subject
                self.last["dropped_subject"] += 1; continue
            if not obj or normalize(obj) in _BAD_OBJ or "?" in obj or len(obj) > 80:
                self.last["dropped_object"] += 1; continue
            label = self._label(rel)
            if label is None:
                self.last["dropped_relation"] += 1; continue
            key = (normalize(label), normalize(obj))
            if key in seen:
                continue
            seen.add(key); self.last["parsed"] += 1
            out.append(Triple(obj=obj, rel=label, subj=entity))
            if len(out) >= self.max_facts:
                break
        return out
