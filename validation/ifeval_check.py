"""Order 0067, line 3: IFEval's own checker, run over (rule, reply) pairs, in the checker's own environment.

IFEval's checker (google-research ``instruction_following_eval``, Apache-2.0) needs absl, langdetect, immutabledict and
nltk's punkt; it is kept off-repo with its own interpreter. Its folder is ``$CUBBY_IFEVAL_CHECKER`` (it holds the
package ``instruction_following_eval/``), nltk's data ``$CUBBY_IFEVAL_NLTK``.

  <checker's python> validation/ifeval_check.py <pairs.jsonl> <verdicts.jsonl>
      pairs:    {"id", "instruction_id", "kwargs", "reply"} per line
      verdicts: {"id", "followed": bool} per line
  <checker's python> validation/ifeval_check.py --describe <specs.jsonl> <out.jsonl>
      specs:    {"id", "instruction_id", "kwargs"} -> {"id", "instruction"}: the checker's own wording of the rule;
                {"id", "instruction_id", "seed"} (no kwargs) -> {"id", "instruction", "kwargs"}: the checker draws its
                own arguments (Python's random, seeded) and says which

langdetect is seeded (``DetectorFactory.seed = 0``) so a verdict repeats.
"""
from __future__ import annotations

import json
import os
import sys

__wiring__ = "STANDALONE"


def _load():
    sys.path.insert(0, os.environ["CUBBY_IFEVAL_CHECKER"])
    import langdetect
    import nltk
    langdetect.DetectorFactory.seed = 0
    if os.environ.get("CUBBY_IFEVAL_NLTK"):
        nltk.data.path.insert(0, os.environ["CUBBY_IFEVAL_NLTK"])
    from instruction_following_eval import instructions_registry
    return instructions_registry.INSTRUCTION_DICT


def _kwargs(kw: dict) -> dict:
    return {k: v for k, v in (kw or {}).items() if v is not None}


def main() -> None:
    reg = _load()
    describe = sys.argv[1] == "--describe"
    src, dst = sys.argv[2:4] if describe else sys.argv[1:3]
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            p = json.loads(line)
            inst = reg[p["instruction_id"]](p["instruction_id"])
            if describe and "kwargs" not in p:           # the checker draws its own arguments, seeded
                import random
                random.seed(p["seed"])
                text = inst.build_description()
                out.write(json.dumps({"id": p["id"], "instruction": text, "kwargs": inst.get_instruction_args()},
                                     ensure_ascii=False) + "\n")
                continue
            text = inst.build_description(**_kwargs(p["kwargs"]))
            if describe:
                out.write(json.dumps({"id": p["id"], "instruction": text}, ensure_ascii=False) + "\n")
                continue
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=p.get("prompt", ""))
            followed = bool(p["reply"].strip()) and bool(inst.check_following(p["reply"]))
            out.write(json.dumps({"id": p["id"], "followed": followed}) + "\n")


if __name__ == "__main__":
    main()
