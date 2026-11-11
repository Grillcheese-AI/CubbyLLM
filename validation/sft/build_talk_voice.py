"""build_talk_voice -- Cubby's voice for the talk adapter: picked templates around checked facts.

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

Order 0010, Cowork's addendum of 2026-11-10. Nick's rule: a template may be model-written, a fact never is.
- A TEMPLATE is wording with a hole in it: the panel's lines and reply shapes (`validation/panel_talk_lines.py`,
  `validation/panel_talk_shapes.py`), picked by rule (`talk_voice_pick.py`) and then by Nick.
- A FACT fills the hole: a line of the facts block the host handed over, or Cubby's own record of itself
  (`identity/identity_facts.yaml` in the cubby-lm checkout beside this one, approved by Nick).
Nothing here asks a model for anything, and every row says where its wording came from (`wording`).

The families are in `talk_voice_self.py` (about Cubby, and the social turns) and `talk_voice_facts.py` (over a
facts block). The prompt shapes are `cubbyllm/emit/talk.py`'s, so the host needs no new prompt format except one
`Premise:` line, written the way it writes `Explicit: allowed`. A refusal keeps the host's exact sentence ("The
facts don't say."): the gates read it. The picked `unknown` and `which` lines are for the host to say itself, so
they go to `talk_lines/host_lines.jsonl` and are not trained on.

    python validation/sft/build_talk_voice.py [--identity <identity_facts.yaml>] [--base standin/data/out/talk_v2_2.jsonl]
        [--mix-out standin/data/out/talk_v3a.jsonl] [--drop-provider openai,anthropic] [--seed 0]

Writes `standin/data/out/talk_voice.jsonl` (the voice rows), `talk_lines/picked.jsonl` (the templates kept, for
Nick to read) and `validation/logs/talk_voice.manifest.json` (counts, sources, what the guard refused).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from validation.sft import talk_voice_facts as facts  # noqa: E402
from validation.sft import talk_voice_self as about  # noqa: E402
from validation.sft.talk_voice_pick import HOST_SAYS, PLACES, pick  # noqa: E402
from validation.sft.talk_voice_rows import CLOSER_RATE, JOKE_RATE, Voice  # noqa: E402

__wiring__ = "STANDALONE"

OUT = ROOT / "standin" / "data" / "out"
LINES = OUT / "talk_lines"
SOCIAL = ("voice_identity", "voice_hello", "voice_social")


def host_checks():
    """The host's own guard, value check and value reader, and the builders' opener view with its word list."""
    from cubbyllm.reasoning.ask_words import grounded_prose, has_value, value_check
    from validation.sft.build_chat_sft import COMMON, chat_view
    checks = types.SimpleNamespace(grounded_prose=grounded_prose, has_value=has_value, value_check=value_check,
                                   chat_view=chat_view)
    return checks, set(COMMON.read_text(encoding="utf-8").split())


def social_guard(v: Voice) -> tuple[dict, dict]:
    """How the rows without a facts block read to the guard when Cubby's record and the conversation are the
    facts. Reported, not enforced: it tells the host which openers its word list lacks."""
    seen, tokens = collections.Counter(), collections.Counter()
    for r in v.rows:
        if r["family"] not in SOCIAL:
            continue
        said = [ln.split(": ", 1)[1] for ln in r["prompt"].split("\n") if ": " in ln]
        ok, bad = v.checks.grounded_prose(v.checks.chat_view(r["answer"], v.common), v.record_facts + said, v.ident["name"])
        seen[f"{r['family']}:rows"] += 1
        seen[f"{r['family']}:pass with Cubby's record as the facts"] += int(ok)
        for b in bad[:2]:
            tokens[b] += 1
    return dict(sorted(seen.items())), dict(tokens.most_common(25))


def manifest(v: Voice, a, candidates, picked, dropped, providers) -> dict:
    rows = v.rows
    fam = collections.Counter((r["family"], r["lang"], r["split"]) for r in rows)
    by_model = collections.Counter(w.split(":", 1)[1] for r in rows for w in r["wording"] if w.startswith("template:"))
    seen, tokens = social_guard(v)
    return {
        "seed": a.seed, "rows": len(rows), "train": sum(r["split"] == "train" for r in rows),
        "held": sum(r["split"] == "held" for r in rows),
        "by_family": {f"{f}:{lg}:{s}": n for (f, lg, s), n in sorted(fam.items())},
        "templates": {"read": len(candidates), "picked": len(picked), "dropped": dict(dropped.most_common()),
                      "picked_by_kind": dict(sorted(collections.Counter(f"{r['kind']}:{r['lang']}" for r in picked).items()))},
        "rows_using_a_template_by_model": dict(by_model.most_common()),
        "rows_with_a_joke": sum("joke" in r["wording"] for r in rows),
        "dropped_rows": {k[len("dropped:"):]: n for k, n in sorted(v.stats.items()) if k.startswith("dropped:")},
        "guard_on_fact_rows": dict(sorted(v.guard.items())),
        "guard_refused_tokens": dict(v.bad_tokens.most_common(25)),
        "guard_on_social_rows": seen, "social_tokens_outside_the_record": tokens,
        "place_names_in_social_rows": [r["id"] for r in rows if r["family"] in SOCIAL and PLACES.search(r["answer"])][:20],
        "providers_left_out": sorted(providers), "joke_rate": JOKE_RATE, "closer_rate": CLOSER_RATE,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--identity", default=str(ROOT.parent / "cubby-lm" / "identity" / "identity_facts.yaml"),
                    help="Cubby's record of itself (the cubby-lm checkout beside this one)")
    ap.add_argument("--base", default=str(OUT / "talk_v2_2.jsonl"), help="the current mix: where the fact rows come from")
    ap.add_argument("--out", default=str(OUT / "talk_voice.jsonl"))
    ap.add_argument("--mix-out", default="", help="also write the base mix followed by the voice rows")
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "talk_voice.manifest.json"))
    ap.add_argument("--drop-provider", default="", help="comma-separated provider prefixes whose lines are left out")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every family's target")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import yaml
    checks, common = host_checks()
    providers = {p.strip() for p in a.drop_provider.split(",") if p.strip()}

    candidates = [json.loads(ln) for f in ("candidates.jsonl", "shapes.jsonl") for ln in open(LINES / f, encoding="utf-8")]
    picked, dropped = pick(candidates, providers)
    keys = ("lang", "kind", "cat", "model", "text")
    (LINES / "picked.jsonl").write_text("".join(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n"
                                                for r in picked), encoding="utf-8")
    (LINES / "host_lines.jsonl").write_text("".join(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n"
                                                    for r in picked if r["cat"] in HOST_SAYS), encoding="utf-8")

    doc = yaml.safe_load(pathlib.Path(a.identity).read_text(encoding="utf-8"))
    v = Voice(rng, [r for r in picked if r["cat"] not in HOST_SAYS], doc["identity"], checks, common, a.scale)
    about.identity_rows(v, doc["qa_templates"])
    about.record_rows(v)
    about.hello_rows(v)
    framed, listed, one_relation, values = facts.load(a.base, rng)
    facts.fact_rows_en(v, framed)
    facts.fact_rows_fr(v, one_relation)
    facts.list_rows(v, listed)
    facts.premise_rows(v, one_relation + listed, values)
    about.social_rows(v, [x for x in framed if x[0]["split"] == "train"][:4000])

    rng.shuffle(v.rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in v.rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = manifest(v, a, candidates, picked, dropped, providers)
    pathlib.Path(a.manifest).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=1, ensure_ascii=False))
    if a.mix_out:
        n = 0
        with open(a.mix_out, "w", encoding="utf-8") as f:
            for line in open(a.base, encoding="utf-8"):
                f.write(line if line.endswith("\n") else line + "\n")
                n += 1
            for r in v.rows:
                f.write(json.dumps({k: r[k] for k in ("id", "family", "split", "prompt", "answer")}, ensure_ascii=False) + "\n")
        print(f"mix: {n} base rows + {len(v.rows)} voice rows -> {pathlib.Path(a.mix_out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
