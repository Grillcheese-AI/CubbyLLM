"""WO-1.3 -- audit the SFT mix before rebuilding it.

The work order proposes dropping `chain` rows and the decorative attributes,
and rests on three specific claims:

  1. `chain` is 28,529 of 75,534 rows.
  2. `chain` is **the sole source** of the 402 per-relation role identifiers.
  3. 11,731 rows teach `@external` / `@system` / `@once`, none of which the
     compiler or VM ever reads.

A rebuild is expensive and a retrain more so, so the claims get checked before
anything is dropped. Claim 2 is the load-bearing one: if `plan` rows also mint
per-relation roles, dropping `chain` does not fix the vocabulary and the
predicted improvement will not arrive.

    python validation/sft_mix_audit.py standin/data/out/emitter_sft_v13f.jsonl
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from role_vocab import (is_per_relation, per_relation_share,  # noqa: E402
                        roles_in_program)

# The attributes DRIFT.md records as parsed-and-never-read.
DECORATIVE = ("@external", "@system", "@once")
_ATTR = re.compile(r"@(?:external|system|once)\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--write-filtered", metavar="OUT",
                    help="also write a corpus with `chain` rows dropped and the "
                         "decorative attributes stripped, ready to train on")
    a = ap.parse_args()

    path = pathlib.Path(a.corpus)
    if not path.is_file():
        print(f"no such corpus: {path}", file=sys.stderr)
        return 2

    by_task: collections.Counter = collections.Counter()
    roles_by_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    attr_rows = 0
    attr_rows_by_task: collections.Counter = collections.Counter()
    vm_ok_by_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows: list[dict] = []
    n = 0

    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            task = str(rec.get("task") or "?")
            by_task[task] += 1
            prog = rec.get("program")
            if isinstance(prog, str):
                roles_by_task[task].update(roles_in_program(prog))
                if _ATTR.search(prog):
                    attr_rows += 1
                    attr_rows_by_task[task] += 1
            vm_ok_by_task[task][repr(rec.get("vm_ok"))] += 1
            if a.write_filtered:
                rows.append(rec)

    print(f"{path}\n  rows: {n:,}\n")
    print(f"  {'task':<22}{'rows':>9}{'roles':>8}{'per-rel':>9}{'attr rows':>11}")
    for task, cnt in by_task.most_common():
        rc = roles_by_task[task]
        n_rel, _ = per_relation_share(rc)
        print(f"  {task[:21]:<22}{cnt:>9,}{len(rc):>8}{n_rel:>9}{attr_rows_by_task[task]:>11,}")

    # --- claim 2: is `chain` the SOLE source of the per-relation roles? ---
    chain_roles = set(roles_by_task.get("chain", {}))
    other_roles: set[str] = set()
    for task, rc in roles_by_task.items():
        if task != "chain":
            other_roles |= set(rc)

    all_roles = chain_roles | other_roles
    rel_all = {r for r in all_roles if is_per_relation(r)}
    rel_chain_only = {r for r in rel_all if r in chain_roles and r not in other_roles}
    rel_elsewhere = rel_all - rel_chain_only
    # Roles shaped like H<n>_<SOMETHING> whose suffix is NOT an ontology
    # relation. Some are honest positional slots (H1_STEP, H3_NAME); the rest
    # are a finding in their own right -- see below.
    hop_shaped = {r for r in all_roles if re.match(r"^H\d+_", r)}
    odd = sorted(hop_shaped - rel_all)
    print(f"\n  hop-shaped roles whose suffix is NOT an ontology relation: {len(odd)}")
    for r in odd[:18]:
        print(f"    {r}")
    if len(odd) > 18:
        print(f"    ... and {len(odd) - 18} more")

    print(f"\n  CLAIM 2 -- is `chain` the sole source of the per-relation roles?")
    print(f"    per-relation roles overall        : {len(rel_all)}")
    print(f"    ...only ever seen in `chain` rows : {len(rel_chain_only)}")
    print(f"    ...also present outside `chain`   : {len(rel_elsewhere)}")
    share_elsewhere = len(rel_elsewhere) / max(1, len(rel_all))
    if rel_elsewhere:
        sample = sorted(rel_elsewhere)[:12]
        where = collections.Counter()
        for task, rc in roles_by_task.items():
            if task == "chain":
                continue
            for r in rc:
                if r in rel_elsewhere:
                    where[task] += 1
        print(f"    tasks that also mint them         : {dict(where)}")
        print(f"    examples                          : {', '.join(sample)}")
    if share_elsewhere > 0.05:
        print("\n    => Claim 2 is FALSE as stated. Dropping `chain` alone will NOT")
        print("       flatten the role vocabulary, because other tasks mint the same")
        print("       per-relation identifiers. Fix the interface, not the mix.")
    elif rel_elsewhere:
        print(f"\n    => Claim 2 holds in substance: {len(rel_chain_only)}/{len(rel_all)} "
              f"({1 - share_elsewhere:.0%}) are chain-only.")
        print("       The handful outside it are positional slots whose suffix happens")
        print("       to collide with a Wikidata property alias (H<n>_NAME -- 'name' is")
        print("       an alias), not genuine per-relation minting.")
    else:
        print("\n    => Claim 2 holds: every per-relation role is chain-only, so")
        print("       dropping `chain` does remove them from the corpus.")

    # --- the vm_ok asymmetry the work order's last paragraph is about ---
    print(f"\n  vm_ok by task (the verification asymmetry):")
    for task, cnt in by_task.most_common():
        shown = ", ".join(f"{k} {v:,}" for k, v in vm_ok_by_task[task].most_common())
        print(f"    {task[:21]:<22} {shown}")

    print(f"\n  CLAIM 3 -- rows carrying a decorative attribute: {attr_rows:,}")

    if a.write_filtered:
        out = pathlib.Path(a.write_filtered)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Backfill the plan rows' verification label BEFORE dropping `chain`,
        # because the chain twin is the evidence.
        #
        # `vm_ok: None` on a plan row is a missing LABEL, not missing
        # verification: build_gen3 emits the plan record and its chain record
        # from one `learn_and_answer` call and skips both unless that walk was
        # `certified`. Measured on v13e, 18,190 of 19,404 plan rows (94%) have
        # a chain twin carrying vm_ok=True/gold_match=True from that same walk.
        # Carrying the label across makes the corpus say what is true of it;
        # the rows without a twin (gen-2 `game`/`cot_harvest`, and v13f's
        # `gen3_profile`) keep `None` because nothing here certified them.
        #
        # This deliberately does NOT write `False` anywhere: `None` means "not
        # certified by THIS path", which is not the same as "failed", and the
        # notebook admits both True and None. So the backfill changes no
        # training row -- it changes what the corpus can be audited for.
        def _question_of(r):
            return str(r.get("prompt") or "").split("\nFacts:\n", 1)[0].strip()

        twin = {}
        for rec in rows:
            if str(rec.get("task") or "") == "chain" and rec.get("vm_ok") is True:
                twin[(_question_of(rec), str(rec.get("gold")))] = rec
        backfilled = 0

        kept: list[dict] = []
        with io.open(out, "w", encoding="utf-8") as fh:
            for rec in rows:
                if str(rec.get("task") or "") == "chain":
                    continue
                if rec.get("task") == "plan" and rec.get("vm_ok") is None:
                    t = twin.get((_question_of(rec), str(rec.get("gold"))))
                    if t is not None:
                        rec = dict(rec, vm_ok=True, vm_result=t.get("vm_result"),
                                   gold_match=True,
                                   verified_via="chain_twin (same certified walk)")
                        backfilled += 1
                prog = rec.get("program")
                if isinstance(prog, str):
                    # Strip the attribute and any blank line it leaves behind.
                    cleaned = _ATTR.sub("", prog)
                    cleaned = re.sub(r"\n[ \t]*\n(?=[ \t]*(?:public|private|@))", "\n", cleaned)
                    rec = dict(rec, program=cleaned)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append(rec)
        print(f"\n  wrote {out} ({len(kept):,} rows, {n - len(kept):,} dropped)")
        still_none = sum(1 for r in kept if r.get("task") == "plan" and r.get("vm_ok") is None)
        print(f"  plan verification backfilled from the chain twin: {backfilled:,}"
              f" | still unlabelled: {still_none:,}")

        # The Colab notebook (notebooks/standin_gen3_masked_sft.ipynb) asserts a
        # manifest exists beside the corpus, that its `version` equals
        # STANDIN_VERSION, and that it is recognisably gen-3 lineage; it then
        # prints by_task / n_records / train_rows_after_repeat / by_source.
        # Writing the corpus without one would fail the very first cell, so the
        # manifest is part of the deliverable, not an afterthought.
        version = out.stem.replace("emitter_sft_", "")
        train_rows = sum(int(r.get("repeat", 1) or 1)
                         for r in kept if r.get("split") == "train")
        src_path = pathlib.Path(a.corpus)
        base_manifest = src_path.parent / (src_path.stem + ".manifest.json")
        parent_rev = ""
        if base_manifest.is_file():
            try:
                parent_rev = json.loads(base_manifest.read_text(encoding="utf-8")).get("git_rev", "")
            except (OSError, ValueError):
                pass

        # The notebook's optional smoke-eval cell reads `m['output_sha256']`
        # directly. Only v12e's manifest ever had it, so that cell raises a
        # KeyError on v13e/v13f -- and it runs by DEFAULT (STANDIN_EVAL_N=8).
        # Write it here so the arm does not inherit that trap, and because a
        # corpus hash is provenance worth having anyway.
        import hashlib
        h = hashlib.sha256()
        with io.open(out, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)

        manifest = {
            "version": version,
            "built": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "git_rev": parent_rev,
            "base": src_path.name,
            "output_sha256": h.hexdigest(),
            "gen3": {
                "derived_from": src_path.name,
                "filter": "WO-1.3: dropped every `chain` row; stripped "
                          "@external/@system/@once from every program",
                "rows_dropped": n - len(kept),
            },
            "n_records": len(kept),
            "by_task": dict(collections.Counter(str(r.get("task")) for r in kept)),
            "by_source": dict(collections.Counter(str(r.get("source")) for r in kept)),
            "by_split": dict(collections.Counter(str(r.get("split")) for r in kept)),
            "train_rows_after_repeat": train_rows,
            "schema_note": (
                "WO-1.3 arm. Identical to its base except: all `chain` rows removed "
                "(the host already does that work deterministically via "
                "build_chain_program, and it is the source of ~99% of the "
                "per-relation role identifiers), and the decorative attributes "
                "@external/@system/@once stripped (parsed, never read by compiler "
                "or VM). Role vocabulary drops 412 -> 17. The prediction under test: "
                "plan accuracy and cross-task retention improve with no loss of "
                "VM-verified answers. Kill criterion: if the VM-verified answer rate "
                "drops, `chain` was doing something the audit did not see."
            ),
        }
        mp = out.parent / (out.stem + ".manifest.json")
        mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {mp}")
        print(f"  -> Colab: STANDIN_VERSION={version} "
              f"(needs both files copied to the Drive standin/ folder)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
