"""CLUTRR loader: the certified kinship graph, not the prose.

Wired: STANDALONE (validation helper; never imported by cubbyllm/).

WHERE THE DATA IS
-----------------
Set `CUBBY_BENCH_ROOT` to the directory WO-2.6's benchmarks were pulled into, or
pass `--bench-root`. Under it this module expects::

    <root>/clutrr/<task>/CLUTRR_v1_<task>_<split>.json

as written by::

    hf download kendrivp/CLUTRR_v1_extracted --repo-type dataset \\
        --local-dir <root>/clutrr

No path is hard-coded here: `docs/WORK_ORDERS.md` rule, "No local drive paths in
committed files".

THE CONVENTION, AND WHY IT IS WRITTEN DOWN
------------------------------------------
CLUTRR writes a triple `(a, r, b)` to mean **"b is the r of a"**. The relation
belongs to the SECOND entity. Worked through on record `a44d478f`:

    story       "[Scott] and [Lewis] are brothers. [Jason] is father of their father"
    edge 0      ('Jason', 'grandson', 'Scott')
    genders     Jason:male, Scott:male, Lewis:male

Jason is the father of Scott's father, so Jason is Scott's **grandfather** — and
the edge is labelled `grandson`, which is what Scott is to Jason. Read it the
other way and every answer in the set inverts: the harness reports a wall of
WRONG that belongs to the loader, not to the model. `verify_convention()` below
re-derives this from `proof_state` on every load rather than trusting this
docstring, because a silent inversion is exactly the instrument that lies.

The project's fact template is `"OBJ is the REL of SUBJ"` (`planner.parse_fact`),
so an edge `(a, r, b)` becomes the fact `"b is the r of a"` directly.

NODE INDICES
------------
`story_edges` is a list of `(i, j)` index pairs and `edge_types` the parallel
relation list; `genders` gives the entity names **in node order**, which is how
an index becomes a name. Checked against `proof_state`'s leaves on load.

THE CHAIN
---------
`story_edges` for a solvable record is a path `0 -> 1 -> ... -> k`, and
`query_edge` is `(0, k)`. Walking it gives the hop sequence the VM would have to
execute. The question the VM CAN be asked is the nested one::

    Who is the {r_k} of the {r_{k-1}} of ... the {r_1} of {name(0)}?

whose gold answer is `name(k)`. CLUTRR's own question — *name the relation
between node 0 and node k* — is a different query type and needs a kinship
algebra the engine registry does not have; see WO-2.6.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re

__wiring__ = "STANDALONE"

TASKS = ("gen_train23_test2to10", "gen_train234_test2to10",
         "rob_train_clean_23_test_all_23", "rob_train_disc_23_test_all_23",
         "rob_train_irr_23_test_all_23", "rob_train_sup_23_test_all_23")
SPLITS = ("train", "validation", "test")
DEFAULT_TASK = "gen_train23_test2to10"

_TRIPLE = re.compile(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)'\)")


class ConventionError(RuntimeError):
    """The data does not read the way this module says it does. Stop, do not score."""


def bench_root(explicit: str | None = None) -> pathlib.Path:
    root = explicit or os.environ.get("CUBBY_BENCH_ROOT")
    if not root:
        raise RuntimeError(
            "set CUBBY_BENCH_ROOT (or pass --bench-root) to the directory the "
            "WO-2.6 benchmarks were pulled into -- see validation/clutrr.py")
    p = pathlib.Path(root)
    if not p.is_dir():
        raise RuntimeError(f"benchmark root does not exist: {p}")
    return p


def load_raw(split: str = "test", task: str = DEFAULT_TASK,
             root: str | None = None) -> list[dict]:
    p = bench_root(root) / "clutrr" / task / f"CLUTRR_v1_{task}_{split}.json"
    if not p.is_file():
        raise RuntimeError(f"missing {p} -- pull it with:\n"
                           f"  hf download kendrivp/CLUTRR_v1_extracted "
                           f"--repo-type dataset --local-dir <root>/clutrr")
    return json.loads(p.read_text(encoding="utf-8"))


def _names(rec: dict) -> list[str]:
    """Entity names in NODE order, from `genders` ('Name:gender, ...')."""
    return [g.split(":")[0].strip() for g in rec["genders"].split(",") if ":" in g]


def _proof_triples(rec: dict) -> set[tuple[str, str, str]]:
    """EVERY triple appearing anywhere in `proof_state`.

    `proof_state` is a derivation, not a flat leaf list: a list of
    `{conclusion: [premise, premise]}` steps whose conclusions feed later steps,
    so it holds the story's edges *and* the intermediate conclusions. Taking
    only the first step's premises (the first cut here) mistook an intermediate
    for a missing edge and the check fired on a correct reading -- kept as a
    comment because a convention check that cries wolf gets deleted, and then
    it is not there on the day the reading really is wrong.

    The story's edges must all appear in this set; the intermediates are extra.
    """
    return set(_TRIPLE.findall(rec.get("proof_state") or ""))


# Relation -> the gender of the person the relation NAMES. Used to pin the
# direction of `(a, r, b)` from the data instead of from a reading of the prose:
# under "b is the r of a" it is b's gender that must match, under the inverted
# reading it would be a's. Over a thousand records the two hypotheses do not
# tie. Relations whose gender is not determined are left out.
_REL_GENDER = {
    "sister": "female", "mother": "female", "daughter": "female", "aunt": "female",
    "grandmother": "female", "granddaughter": "female", "niece": "female",
    "wife": "female", "mother-in-law": "female", "daughter-in-law": "female",
    "brother": "male", "father": "male", "son": "male", "uncle": "male",
    "grandfather": "male", "grandson": "male", "nephew": "male", "husband": "male",
    "father-in-law": "male", "son-in-law": "male",
}


def _gender_map(rec: dict) -> dict[str, str]:
    out = {}
    for g in (rec.get("genders") or "").split(","):
        if ":" in g:
            name, sex = g.split(":", 1)
            out[name.strip()] = sex.strip().lower()
    return out


def chain(rec: dict) -> list[tuple[str, str, str]] | None:
    """The record's edges as `(a, rel, b)` with NAMES, in path order from node 0.

    None when the record's `story_edges` is not a simple path from the query's
    first entity to its second -- CLUTRR's robustness tasks add noise edges, and
    a walk over a non-path is a different experiment.
    """
    try:
        edges = ast.literal_eval(rec["story_edges"])
        types = ast.literal_eval(rec["edge_types"])
        q0, q1 = ast.literal_eval(rec["query_edge"])
    except (ValueError, SyntaxError, KeyError):
        return None
    names = _names(rec)
    if len(edges) != len(types) or not edges:
        return None
    if max(max(e) for e in edges) >= len(names):
        return None

    # order the edges into a path starting at q0
    adj: dict[int, list[tuple[int, str]]] = {}
    for (i, j), r in zip(edges, types):
        adj.setdefault(i, []).append((j, r))
    out: list[tuple[str, str, str]] = []
    cur, seen = q0, {q0}
    while len(out) < len(edges):
        nxt = [(j, r) for j, r in adj.get(cur, []) if j not in seen]
        if len(nxt) != 1:
            return None                       # a fork or a dead end: not a path
        j, r = nxt[0]
        out.append((names[cur], r, names[j]))
        seen.add(j); cur = j
    if cur != q1:
        return None                           # the path does not end at the query's target
    return out


def fact(a: str, rel: str, b: str) -> str:
    """The edge `(a, rel, b)` as the project's fact template: 'b is the rel of a'."""
    return f"{b} is the {rel} of {a}"


def question(ch: list[tuple[str, str, str]]) -> str:
    """The nested chain question the VM's grammar parses.

    Hops apply outermost-last: `Who is the {r_k} of the {r_k-1} of ... {seed}?`
    """
    body = ch[0][0]
    for _, rel, _ in ch:
        body = f"the {rel} of {body}"
    return f"Who is {body}?"


def gold(ch: list[tuple[str, str, str]]) -> str:
    return ch[-1][2]


def verify_convention(recs: list[dict], n: int = 400,
                      min_agree: float = 0.98) -> dict:
    """Re-derive the `(a, r, b)` == *"b is the r of a"* reading from the data.

    Three checks. Each pins a different thing, and the third is the one that
    actually settles the direction:

      1. **The name map.** Every triple `story_edges` + `edge_types` + `genders`
         build must appear somewhere in `proof_state`. That is what licenses
         reading a node index as a name, and it also pins the ORDER inside the
         tuple -- read the indices the other way round and none of the built
         triples is in the proof.
      2. **The query.** `query_edge`'s two node indices name, under that same
         map and in that same order, the two entities in `query`.
      3. **The direction, from gender.** For an edge `(a, r, b)` where `r` names
         a gendered role, this loader's reading requires b's sex to match r's;
         the inverse reading requires a's. The forward reading must hold on
         essentially EVERY gendered edge -- `min_agree`, not a margin over the
         inverse. The inverse is reported alongside but cannot be the test: it
         sits at chance (~50%) rather than at zero, because an edge between two
         people of the same sex satisfies both readings, and half of CLUTRR's
         pairs are. Measured on the first cut: forward 1486/1486, inverse
         732 -- a 2.03x margin that a 3x rule would have rejected, on data that
         agrees with the loader perfectly. The ratio test was the broken
         instrument, not the reading.

         This is the check that does not depend on anyone reading a story
         correctly -- including the person who wrote this module.

    Raises ConventionError on the first failure of (1) or (2), naming the
    record, and on (3) if the data does not back the reading this loader uses.
    A silent inversion turns every answer in the set into a WRONG that belongs
    to the loader, so this runs on every load and is not optional.
    """
    checked = 0
    fwd = inv = gendered = 0            # gender votes for each reading
    for rec in recs[:n]:
        proof = _proof_triples(rec)
        if not proof:
            continue
        try:
            edges = ast.literal_eval(rec["story_edges"])
            types = ast.literal_eval(rec["edge_types"])
            q = ast.literal_eval(rec["query"])
            qe = ast.literal_eval(rec["query_edge"])
        except (ValueError, SyntaxError, KeyError):
            continue
        names = _names(rec)
        if not edges or max(max(e) for e in edges) >= len(names):
            continue
        built = [(names[i], r, names[j]) for (i, j), r in zip(edges, types)]
        checked += 1

        missing = [t for t in built if t not in proof]
        if missing:
            raise ConventionError(
                f"record {rec.get('id')}: story edge(s) {missing} do not appear "
                f"in proof_state -- the node-index map or the tuple order is "
                f"wrong; DO NOT score against this loader")

        if max(qe) >= len(names) or (names[qe[0]], names[qe[1]]) != (q[0], q[1]):
            raise ConventionError(
                f"record {rec.get('id')}: query {q} does not match query_edge "
                f"{qe} under the node-order name map {names}")

        sex = _gender_map(rec)
        for a, r, b in built:
            want = _REL_GENDER.get(r.lower())
            if want is None or a not in sex or b not in sex:
                continue
            gendered += 1
            fwd += sex[b] == want              # "b is the r of a"
            inv += sex[a] == want              # "a is the r of b"

    if checked == 0:
        raise ConventionError("no record carried a usable proof_state to check against")
    if gendered == 0:
        raise ConventionError("no gendered edge to settle the direction on")
    rate = fwd / gendered
    if rate < min_agree or fwd <= inv:
        raise ConventionError(
            f"the gender check does not back this loader's reading: "
            f"'b is the r of a' holds on {fwd}/{gendered} gendered edges "
            f"({rate:.3f}); the inverse holds on {inv}. Reading the edges "
            f"backwards inverts every answer in the set -- fix `fact()` before "
            f"scoring anything")
    return {"checked": checked, "gendered_edges": gendered,
            "forward": fwd, "forward_rate": round(rate, 4), "inverted": inv}


def items(split: str = "test", task: str = DEFAULT_TASK, root: str | None = None,
          max_hops: int | None = None) -> list[dict]:
    """Loaded, convention-checked, path-only records as chain questions.

    Each item: {id, hops, chain, facts, question, gold, clutrr_target}.
    `clutrr_target` is CLUTRR's own answer (the composite relation name) and is
    carried through unused -- it is the query type the engine registry cannot
    serve yet (WO-2.6), and keeping it in the record means the day it can, the
    harness does not need rebuilding.
    """
    recs = load_raw(split, task, root)
    verify_convention(recs)
    out = []
    for rec in recs:
        ch = chain(rec)
        if ch is None:
            continue
        if max_hops is not None and len(ch) > max_hops:
            continue
        out.append({
            "id": rec.get("id"),
            "hops": len(ch),
            "chain": ch,
            "facts": [fact(*e) for e in ch],
            "question": question(ch),
            "gold": gold(ch),
            "clutrr_target": rec.get("target_text"),
        })
    return out
