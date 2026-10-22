"""wikikg — AutomatedScientist/wikikg-trajectories as template facts, a serve world and chain questions.

Wired: WIRED (stand-in serve path via `--wiki`; nothing in cubbyllm/ imports this).

The dataset (MIT; screened in validation/exp_g4_wikikg_screen.py, H-G6): 365,923 (subject, relation, object)
triples from 45,416 Wikipedia articles, 2,902 UPPER_SNAKE relation types, 1.5M two-hop random-walk paths.
Every triple becomes ONE template fact — the shape `planner.parse_fact` reads and `TripleIndex` indexes — and,
where the relation has a clean inverse noun, a second fact reading the other way, so the walk can take either
direction of a hop by lookup alone:

    (S, R, O)  form "os":  "O is the NOUN of S"      e.g. (Desargues, BORN_ON, 1591)  -> "1591 is the birth date of Girard Desargues"
               form "so":  "S is the NOUN of O"      e.g. (Desargues, CREATOR_OF, Plane) -> "Girard Desargues is the creator of Desarguesian Plane"
               inverse:    the same triple read back  -> "Desarguesian Plane is the creation of Girard Desargues"

`RELATIONS` maps the relation types that carry ~95% of the triples; an unmapped type verbalizes generically
("O is the <lowercased words> of S", or "S is the X of O" for an X_OF name) so nothing is dropped silently.
Entities are de-camel-cased ("KeikoMatsuzaka" -> "Keiko Matsuzaka"); dates stay as written.

The world is LOOKUP + a lexical fallback: 359k facts would need 14.7 GB of cosine rows, the index takes ~284 MB
(exp_g4). Chain questions come from FUNCTIONAL hops only — the graph is many-valued (TIMELINE_EVENT alone is 21%
of triples) and a random-walk end is not a gold answer (162 of 300 VM-verified chains ended elsewhere, validly).
"""
from __future__ import annotations

import os
import random
import re

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .wikikg_relations import RELATIONS, _DATE, decamel, facts_for, hop_phrase, relation_entry  # noqa: F401

__wiring__ = "WIRED"

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "standin", "data"))
DATA_DIR = os.path.join(HERE, "out", "wikikg")                       # gitignored; the Hub's parquet export
TRIPLETS = os.path.join(DATA_DIR, "triplets.parquet")
PATHS = os.path.join(DATA_DIR, "paths.parquet")
HUB_PARQUET = "https://huggingface.co/api/datasets/AutomatedScientist/wikikg-trajectories/parquet/{cfg}/train/0.parquet"


def ensure_data(cfgs: tuple[str, ...] = ("triplets",), fetch=None) -> None:
    """The Hub's parquet export in DATA_DIR (triplets 8 MB, paths 189 MB). A missing file is fetched with
    ``fetch(url, dst)``, which the caller hands in (the server: ``cubby_serve.sources.hub.download``): this package
    opens no socket (order 0031). Without one, a missing file is an error that says how to fetch it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for cfg in cfgs:
        dst = os.path.join(DATA_DIR, f"{cfg}.parquet")
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            continue
        if fetch is None:
            raise FileNotFoundError(f"the wikikg export ({cfg}) is not cached in {DATA_DIR}: fetch it with "
                                    f"`python -m cubby_serve.sources.hub wikikg {cfg}`")
        fetch(HUB_PARQUET.format(cfg=cfg), dst)


def load_triples(path: str = TRIPLETS) -> list[tuple[str, str, str]]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["subject", "relation", "object"])
    return list(zip(t.column("subject").to_pylist(), t.column("relation").to_pylist(), t.column("object").to_pylist()))


def facts_from_triples(triples, inverse: bool = True) -> list[str]:
    """Deduped template facts, forward first then inverse readings, in triple order."""
    seen: set[str] = set()
    out: list[str] = []
    for s, r, o in triples:
        for f in facts_for(s, r, o, inverse=inverse):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def hygiene(triples, same_place=None):
    """The mount's mending (order 0015): `world_hygiene.hygiene` -- debris, self-references, spellings, same places."""
    from .hygiene import hygiene as _hygiene
    return _hygiene(triples, same_place)


def wiki_world(path: str | None = None, triples=None, inverse: bool = True, name: str = "wiki"):
    """A `FactStore` over the whole graph: lookup-first through its TripleIndex, IDF-overlap fallback (no cosine
    encoder at this size). ~20 s and ~0.5 GB for the full graph with inverses. Mounted through `hygiene`; its counts
    are the store's `hygiene`."""
    from .base import FactStore
    if triples is None:
        if path is None:
            ensure_data()
            path = TRIPLETS
        triples = load_triples(path)
    triples, counts = hygiene(triples)
    store = FactStore(facts_from_triples(triples, inverse=inverse), name=name)
    store.hygiene = counts
    return store


def chain_questions(paths, index, n: int, seed: int = 0, max_scan: int | None = None) -> list[dict]:
    """Two-hop chain questions from random-walk paths whose BOTH hops are functional in `index` (exactly one
    stored fact serves each), so the walk's end is the one gold answer. `paths` is an iterable of
    (entities, relations, directions). Returns {question, answer, facts, entities, relations}."""
    from cubbyllm.reasoning.planner import parse_question
    rng = random.Random(seed)
    rows = list(paths) if max_scan is None else list(paths)[:max_scan]
    rng.shuffle(rows)
    out: list[dict] = []
    for es, rs, ds in rows:
        if len(rs) != 2:
            continue
        phrases = [hop_phrase(r, d) for r, d in zip(rs, ds)]
        if any(p is None for p in phrases):
            continue
        q = f"What is the {phrases[1]} of the {phrases[0]} of {decamel(es[0])}?"
        plan = parse_question(q)
        if plan is None:
            continue
        entity, facts, ok = None, [], True
        for h in range(2):
            cands = index.hop(plan, h, entity)
            if len(cands) != 1:
                ok = False
                break
            f, t = cands[0]
            facts.append(f)
            entity = t.obj
        if not ok or entity is None:
            continue
        out.append({"question": q, "answer": entity, "facts": facts, "entities": [decamel(e) for e in es], "relations": list(rs)})
        if len(out) >= n:
            break
    return out
