"""build_talk_v3b -- the `talk_v3b` mix: `talk_v3a`'s, rebuilt with the changes of order 0010's addendum of 2026-11-20.

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

    python3 validation/sft/build_talk_v3b.py [--data <the git-ignored out folder>] [--seed 0]

- The voice rows are built as `talk_v3a`'s (`build_talk_voice.py`: the same picked templates, base mix and seed),
  with what Cubby can do said as a sentence (`Voice.sentence`), plus the families of `talk_voice_v3b.py`.
- Every row of the base mix (`talk_v2_2.jsonl`) and every voice row then passes the gate (`talk_voice_v3b.Gate`):
  the host's checks g, h and d, the emoji rule, the repeats. A row that fails is dropped and counted.
- Writes, in the out folder: `talk_voice_v3b.jsonl` (the voice rows, with where their wording came from) and
  `talk_v3b.jsonl` (the mix the trainer reads); and `validation/logs/talk_v3b.manifest.json` (counts only). It writes
  nothing `talk_v3a` was built from (`picked.jsonl` and `host_lines.jsonl` are left as they are).
Nothing here asks a model for anything.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from validation.sft import talk_voice_facts as facts  # noqa: E402
from validation.sft import talk_voice_self as about  # noqa: E402
from validation.sft import talk_voice_v3b as v3b  # noqa: E402
from validation.sft.build_talk_voice import host_checks  # noqa: E402
from validation.sft.talk_voice_pick import HOST_SAYS, pick  # noqa: E402
from validation.sft.talk_voice_rows import JOKE_RATE, Voice  # noqa: E402

__wiring__ = "STANDALONE"

KEYS = ("id", "family", "split", "prompt", "answer")


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def openings(rows, n=100) -> float:
    """Distinct first three words per 100 replies (a proxy, reported)."""
    heads = [" ".join(re.findall(r"[\w'’]+", r["answer"].lower())[:3]) for r in rows]
    return round(100 * len(set(heads)) / max(1, len(heads)), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "standin" / "data" / "out"), help="the git-ignored out folder")
    ap.add_argument("--identity", default=str(ROOT.parent / "cubby-lm" / "identity" / "identity_facts.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "talk_v3b.manifest.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    out_dir, lines_dir = pathlib.Path(a.data), pathlib.Path(a.data) / "talk_lines"
    base = out_dir / "talk_v2_2.jsonl"
    rng = random.Random(a.seed)
    import yaml
    from cubbyllm.speech import cubby_record, identity

    from cubbyllm.reasoning.opener_view import common_words
    checks, _ = host_checks()
    common = set(common_words())          # the host's opener words, interjections included, as at serving
    candidates = [json.loads(ln) for f in ("candidates.jsonl", "shapes.jsonl") for ln in open(lines_dir / f, encoding="utf-8")]
    picked, _ = pick(candidates, set())
    doc = yaml.safe_load(pathlib.Path(a.identity).read_text(encoding="utf-8"))
    v = Voice(rng, [r for r in picked if r["cat"] not in HOST_SAYS], doc["identity"], checks, common, a.scale)
    v.sentence = True
    rec = cubby_record.load()
    v.record_facts = v.record_facts + cubby_record.lines()

    # talk_v3a's families, what Cubby can do said as a sentence
    about.identity_rows(v, doc["qa_templates"])
    about.record_rows(v)
    about.hello_rows(v)
    framed, listed, one_relation, values = facts.load(str(base), rng)
    facts.fact_rows_en(v, framed)
    facts.fact_rows_fr(v, one_relation)
    facts.list_rows(v, listed)
    facts.premise_rows(v, one_relation + listed, values)
    about.social_rows(v, [x for x in framed if x[0]["split"] == "train"][:4000])

    # talk_v3b's families
    users = [json.loads(ln) for ln in open(lines_dir / "user_lines_topical_chat.jsonl", encoding="utf-8")]
    is_about = lambda t: cubby_record.is_about_cubby(t) or identity.is_identity_question(t)  # noqa: E731
    v3b.record_chat_rows(v, rec, cubby_record.block)
    v3b.tc_social_rows(v, users)
    v3b.personal_rows(v, users, is_about)
    v3b.emoji_rows(v, users)

    gate = v3b.Gate()
    voice = gate.run(v.rows, "voice")
    rng.shuffle(voice)
    base_rows = [json.loads(ln) for ln in open(base, encoding="utf-8")]
    kept_base = gate.run(base_rows, "base")

    with open(out_dir / "talk_voice_v3b.jsonl", "w", encoding="utf-8") as f:
        for r in voice:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mix = out_dir / "talk_v3b.jsonl"
    with open(mix, "w", encoding="utf-8") as f:
        for r in kept_base:
            f.write(json.dumps({k: r[k] for k in KEYS + (("explicit",) if "explicit" in r else ())}, ensure_ascii=False) + "\n")
        for r in voice:
            f.write(json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False) + "\n")

    fam = collections.Counter((r["family"], r["lang"], r["split"]) for r in voice)
    drops = collections.Counter()
    for k, n in gate.stats.items():
        if k.startswith(("voice:", "base:")) and not k.endswith(":ok"):
            drops[k] = n
    social = [r for r in voice if r["family"] in v3b.CHAT_SHAPE]
    m = {
        "what": "order 0010 talk_v3b: talk_v2_2 and the voice rows through the host's checks (Cowork's addendum, 2026-11-20)",
        "seed": a.seed, "base": base.name, "base_sha256": sha(base), "base_rows": len(base_rows), "base_kept": len(kept_base),
        "voice_built": len(v.rows), "voice_kept": len(voice),
        "mix_rows": len(kept_base) + len(voice),
        "mix_train": sum(r["split"] == "train" for r in kept_base) + sum(r["split"] == "train" for r in voice),
        "mix_sha256": sha(mix),
        "voice_by_family": {f"{f}:{lg}:{s}": n for (f, lg, s), n in sorted(fam.items())},
        "dropped": dict(sorted(drops.items())),
        "emoji_removed": {k.split(":", 1)[1]: n for k, n in sorted(gate.stats.items()) if k.startswith("emoji removed:")},
        "builder_dropped": {k[len("dropped:"):]: n for k, n in sorted(v.stats.items()) if k.startswith("dropped:")},
        "rows_with_a_joke": sum("joke" in r["wording"] for r in voice), "joke_rate": JOKE_RATE,
        "chat_shape_replies": len(social), "chat_shape_openings_per_100": openings(social),
        "chat_shape_emoji_per_reply": round(sum(len(v3b.EMOJI.findall(r["answer"])) for r in social) / max(1, len(social)), 4),
        "chat_shape_mean_words": round(sum(len(r["answer"].split()) for r in social) / max(1, len(social)), 1),
        "lists_in_chat_shape": sum(bool(re.search(r"^\s*(?:\d+\.|-)\s", r["answer"], re.M)) for r in social),
    }
    pathlib.Path(a.manifest).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
