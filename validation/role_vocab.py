"""WO-0.3 -- role-vocabulary growth as a build metric.

One integer per SFT build: the count of DISTINCT role identifiers in the
corpus. A generalizing interface has a **flat** count across rounds; a
per-relation interface grows one identifier per relation and the count tracks
the ontology instead of the task.

The measurement that motivated this: **146 distinct roles in v12e -> 407 in
v13f, 402 of them per-relation.** That is the generalization-failure signature,
and it sat in the data unnoticed for several rounds because nobody counted it.
The whole point of this file is that counting is trivial and free, so there is
no excuse for it to go unwatched again.

Usage
-----
Report on one or more corpora::

    python validation/role_vocab.py standin/data/out/emitter_sft_v13f.jsonl

Compare against the previous round and fail on growth (what the build calls)::

    python validation/role_vocab.py NEW.jsonl --baseline OLD.jsonl --max-growth 0.10

Exit code is 1 when the gate fails, so it drops into a build pipeline as-is.

What counts as a role
---------------------
A role identifier is the ROLE argument of the VSA binding surface -- the
mechanism the emitter actually depends on (cubelang docs/DRIFT.md A8/B2)::

    bind frame, ROLE, "filler"       # statement form
    recover(frame, ROLE)             # read back
    BIND_ROLE frame, ROLE, filler    # asm form
    UNBIND frame, ROLE

Roles are matched positionally, not by a spelling convention: an emitter that
started writing lowercase or dotted role names would still be counted, which is
the point -- the metric must not be evadable by a rename.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import re
import sys

# `bind <reg> , <ROLE> ,` and the asm spelling. The role is the SECOND operand.
_BIND = re.compile(
    r"\b(?:bind|BIND_ROLE|bind_role)\s+[A-Za-z_][\w.]*\s*,\s*([A-Za-z_][\w.]*)\s*,",
)
# `recover ( <reg> , <ROLE> )` and `UNBIND <reg>, <ROLE>` -- role is second, last.
_RECOVER = re.compile(
    r"\brecover\s*\(\s*[A-Za-z_][\w.]*\s*,\s*([A-Za-z_][\w.]*)\s*\)",
)
_UNBIND = re.compile(
    r"\b(?:UNBIND|unbind)\s+[A-Za-z_][\w.]*\s*,\s*([A-Za-z_][\w.]*)",
)

_PATTERNS = (_BIND, _RECOVER, _UNBIND)

# The per-relation signature. Measured on v13f: the 261 roles added since v12e
# are almost entirely `H<hop>_<RELATION>` -- H1_DATE_OF_BIRTH, H1_PLACE_OF_BIRTH,
# H1_LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY, one identifier minted per
# Wikidata property. That is what makes the growth a GENERALIZATION failure
# rather than a vocabulary that merely got bigger: the interface is tracking the
# ontology instead of the task, so a relation never seen in training has no role
# to bind to. Reported separately because "the vocabulary grew" and "the
# vocabulary grew one-per-relation" call for completely different responses.
_HOP_ROLE = re.compile(r"^H(\d+)_([A-Z0-9_]+)$")


def _ontology_relations() -> set[str]:
    """Every Wikidata property wording the repo's local table knows, normalized
    to the `UPPER_SNAKE` shape role names use.

    This is the discriminator, and it took two wrong tries to get here, both
    worth recording so nobody repeats them:

      * Matching every `H<n>_<SUFFIX>` counts `H1_STEP` and `H3_NAME` as
        per-relation. They are positional slots -- a small fixed set indexed by
        hop, which is the GENERALIZING shape, the exact opposite of what this
        metric looks for.
      * "A suffix reused across several hop numbers is a slot" looks principled
        and is wrong here: in a two-hop chain the SAME relation can sit at hop
        1 or hop 2, so `DATE_OF_BIRTH`, `SPOUSE` and a hundred others appear
        under both prefixes. That version classified 101 of 403 as
        per-relation and called the rest slots.

    What actually separates them is whether the suffix NAMES A RELATION IN THE
    ONTOLOGY. `DATE_OF_BIRTH` is a Wikidata property label; `STEP` and `NAME`
    are not. The table is already in the repo (`standin/data/out/
    wikidata_properties_en_fr.json`, read by `sources.property_aliases`), so
    this costs nothing.

    Returns an empty set if the table cannot be read, in which case
    `per_relation_share` falls back to the shape test and says so.
    """
    try:
        # `standin/sources.py` imports `cubbyllm.reasoning.planner`, so the
        # repo root has to be importable too -- not just `standin/`.
        root = pathlib.Path(__file__).resolve().parents[1]
        for p in (root, root / "standin"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from sources import property_aliases  # type: ignore
        pa = property_aliases()
    except Exception:  # noqa: BLE001 - the table is optional tooling, not a dependency
        return set()

    out: set[str] = set()
    for texts in getattr(pa, "_texts_of", {}).values():
        for t in texts:
            out.add(re.sub(r"[^A-Z0-9]+", "_", t.upper()).strip("_"))
    return out


_ONTOLOGY: set[str] | None = None


def _ontology() -> set[str]:
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = _ontology_relations()
    return _ONTOLOGY


def is_per_relation(name: str, ontology: set[str] | None = None) -> bool:
    """Does this role name mint an identifier for a specific RELATION?"""
    m = _HOP_ROLE.match(name)
    if not m:
        return False
    onto = _ontology() if ontology is None else ontology
    if not onto:
        return True          # no table: fall back to the shape test
    return m.group(2) in onto


def per_relation_share(counts: "collections.Counter") -> tuple[int, float]:
    """(how many roles name a specific relation, what fraction of the vocabulary)."""
    if not counts:
        return 0, 0.0
    onto = _ontology()
    n = sum(1 for name in counts if is_per_relation(name, onto))
    return n, n / len(counts)


def roles_in_program(source: str) -> list[str]:
    """Every role identifier mentioned in one CubeLang program, with repeats."""
    found: list[str] = []
    for pat in _PATTERNS:
        found.extend(pat.findall(source))
    return found


def role_counts(path: str | pathlib.Path) -> collections.Counter:
    """Role identifier -> number of mentions across every record in the file."""
    counts: collections.Counter = collections.Counter()
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            program = rec.get("program")
            if isinstance(program, str):
                counts.update(roles_in_program(program))
    return counts


def _report(path: str, counts: collections.Counter, top: int) -> None:
    total_mentions = sum(counts.values())
    n_rel, frac_rel = per_relation_share(counts)
    print(f"{path}")
    print(f"  distinct roles : {len(counts)}")
    print(f"  total mentions : {total_mentions:,}")
    print(f"  per-relation   : {n_rel} ({frac_rel:.0%} of the vocabulary)")
    if counts and top:
        print(f"  most common    :")
        for name, n in counts.most_common(top):
            print(f"    {name:<28} {n:>8,}")
        # The tail is the interesting part for this metric: a per-relation
        # interface has a long tail of roles mentioned only a handful of times.
        singles = [name for name, n in counts.items() if n <= 2]
        print(f"  roles mentioned <=2 times: {len(singles)}"
              f" ({len(singles) / len(counts):.0%} of the vocabulary)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+",
                    help="SFT .jsonl file(s) to count. The LAST one is the "
                         "corpus under test when --baseline is given.")
    ap.add_argument("--baseline",
                    help="A previous round's .jsonl to compare against. "
                         "Without it this only reports.")
    ap.add_argument("--max-growth", type=float, default=0.10,
                    help="Fail if the distinct-role count grows by more than "
                         "this fraction over the baseline (default 0.10). "
                         "A generalizing interface should be flat, so any "
                         "real growth is worth a look.")
    ap.add_argument("--top", type=int, default=10,
                    help="How many of the most-mentioned roles to print.")
    ap.add_argument("--json-out",
                    help="Also write the counts for the corpus under test here.")
    a = ap.parse_args(argv)

    counts_by_path = {}
    for path in a.corpora:
        if not pathlib.Path(path).is_file():
            print(f"role_vocab: no such corpus: {path}", file=sys.stderr)
            return 2
        counts_by_path[path] = role_counts(path)
        _report(path, counts_by_path[path], a.top)
        print()

    under_test = a.corpora[-1]
    counts = counts_by_path[under_test]

    if a.json_out:
        payload = {
            "corpus": under_test,
            "distinct_roles": len(counts),
            "total_mentions": sum(counts.values()),
            "counts": dict(counts.most_common()),
        }
        with io.open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        print(f"wrote {a.json_out}")

    if not a.baseline:
        return 0

    if not pathlib.Path(a.baseline).is_file():
        print(f"role_vocab: no such baseline: {a.baseline}", file=sys.stderr)
        return 2

    base = role_counts(a.baseline)
    n_base, n_new = len(base), len(counts)
    added = sorted(set(counts) - set(base))
    dropped = sorted(set(base) - set(counts))

    print(f"baseline {a.baseline}: {n_base} distinct roles")
    print(f"under test {under_test}: {n_new} distinct roles")
    print(f"  added   : {len(added)}")
    print(f"  dropped : {len(dropped)}")
    if added:
        show = added[:20]
        print(f"  new roles (first {len(show)}): {', '.join(show)}")

    if n_base == 0:
        print("baseline has no roles at all -- nothing to gate on.")
        return 0

    growth = (n_new - n_base) / n_base
    print(f"  growth  : {growth:+.1%} (limit {a.max_growth:+.1%})")

    n_added_rel = sum(1 for name in added if is_per_relation(name))
    if added:
        print(f"  of the {len(added)} added roles, {n_added_rel} "
              f"({n_added_rel / len(added):.0%}) are per-relation "
              f"(H<hop>_<RELATION>)")

    if growth > a.max_growth:
        print(
            f"\nFAIL: the role vocabulary grew {growth:.1%}, past the "
            f"{a.max_growth:.1%} limit.\n"
            f"A generalizing interface is FLAT across rounds. Growth this size "
            f"means the corpus is minting one role per relation, which is the "
            f"v12e->v13f failure signature (146 -> 407, 402 per-relation).\n"
            f"Either fix the emitter interface or raise the limit deliberately "
            f"-- do not raise it to make the build pass.",
            file=sys.stderr)
        return 1

    print("\nOK: role vocabulary is within the growth limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
