"""Order 0067, the reader's fifth round: IFBench's own checker over (prompt, reply) pairs, in the checker's environment.

IFBench's checker (AllenAI, Apache-2.0; https://github.com/allenai/IFBench) is kept off-repo: ``$CUBBY_IFBENCH_CHECKER``
is the folder holding its package ``ifbench/``. It needs nltk, emoji, syllapy, langdetect, immutabledict (installed
into IFEval's checker interpreter, ``$CUBBY_IFEVAL_PYTHON``, which runs this file). On import it fetches nltk's small
data files it lacks (punkt, stopwords, a tagger). Nick allowed running this downloaded code (2026-11-13).

  <checker's python> validation/ifbench_check.py <pairs.jsonl> <verdicts.jsonl>
      pairs:    {"id", "prompt", "instruction_ids": [...], "kwargs": [{...}, ...], "reply"} per line
      verdicts: {"id", "followed": bool (every instruction of the prompt followed), "each": [bool, ...]} per line

langdetect is seeded (``DetectorFactory.seed = 0``) so a verdict repeats.
"""
from __future__ import annotations

import json
import os
import sys

__wiring__ = "STANDALONE"


def _load():
    sys.path.insert(0, os.environ["CUBBY_IFBENCH_CHECKER"])
    import langdetect
    langdetect.DetectorFactory.seed = 0
    from ifbench import instructions_registry
    return instructions_registry.INSTRUCTION_DICT


def main() -> None:
    reg = _load()
    src, dst = sys.argv[1:3]
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            p = json.loads(line)
            each = []
            for iid, kw in zip(p["instruction_ids"], p["kwargs"]):
                inst = reg[iid](iid)
                inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
                args = inst.get_instruction_args()
                if args and "prompt" in args:
                    inst.build_description(prompt=p["prompt"])
                try:
                    each.append(bool(p["reply"].strip()) and bool(inst.check_following(p["reply"])))
                except Exception:                        # a checker that cannot read the reply: not followed
                    each.append(False)
            out.write(json.dumps({"id": p["id"], "followed": all(each), "each": each}) + "\n")


if __name__ == "__main__":
    main()
