"""chat_batches_0079 -- the final run's plans, mended before batch b004 (Cowork's note of 2026-11-28):
- **`reseed`: fact plans the loop answers.** b000-b003's 78 "asks about a fact" plans asked "tell me about / do you
  know X", and 0 rows carried facts. Each not-yet-drafted fact plan gets one question on the wiki world, a name and
  one relation ("When was X born?", "Who founded X?"), kept only when the drafting server's ask loop verifies it
  (`POST /ask`); the user asks it in their own words. No name of the gen-3 held split (a test) is drawn.
- **`rebatch`: smaller sessions, outside the repo.** The plans not yet started go into batches of `--size` (5); the
  old batches holding only unstarted plans move to `batches/_old/<time>/`. The sessions' folder (`--sessions`, outside the
  repo, so its CLAUDE.md files are not read at a session's start) gets `step.sh` and `launch_sessions.ps1`.

Wired: STANDALONE (`chat_step_0079.py reseed | rebatch`).
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import shutil
import sys
import time

from validation.sft import chat_plans_0079 as P
from validation.sft.chat_rows_0079 import STEP, Stop, _read, _write, brief

__wiring__ = "STANDALONE"

# the wiki world's relation -> the question a user would ask (subject first)
TEMPLATES = {"BORN_ON": "When was {s} born?", "BORN_IN": "Where was {s} born?", "DIED_ON": "When did {s} die?",
             "DIED_IN": "Where did {s} die?", "OCCUPATION": "What was {s}'s occupation?", "AUTHORED_BY": "Who wrote {s}?",
             "CREATED_BY": "Who created {s}?", "FOUNDED_BY": "Who founded {s}?", "SPOUSE_OF": "Who was {s} married to?",
             "EDUCATED_AT": "Where did {s} study?", "WORKS_FOR": "Who did {s} work for?",
             "FIELD_OF_WORK": "What field did {s} work in?", "PERFORMED_BY": "Who performed {s}?", "LED_BY": "Who led {s}?"}
_SEED = re.compile(r'bind frame, SEED, "([^"]+)"')


def name(camel: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel).replace("_", " ").strip()


def held_seeds(gen3: str) -> set[str]:
    """The gen-3 held split's entities (a test set): never a drafted user's question."""
    out = set()
    for line in open(gen3, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "held":
            out |= {s.lower() for s in _SEED.findall(r.get("program") or "")}
    return out


def candidates(triples, held: set[str], rng: random.Random):
    """(name, relation, question) for subjects with a templated relation, in a seeded order, held names left out."""
    pool = [(name(s), r) for s, r, _ in triples if r in TEMPLATES and len(name(s)) >= 4]
    rng.shuffle(pool)
    seen = set()
    for n, r in pool:
        if n.lower() in held or n.lower() in seen or not re.match(r"^[A-Z]", n):
            continue
        seen.add(n.lower())
        yield n, r, TEMPLATES[r].format(s=n)


def ask(host: str, question: str) -> dict:
    import urllib.request
    from cubbyllm.speech.audience import configured_token
    tok = configured_token()
    req = urllib.request.Request(host.rstrip("/") + "/ask", data=json.dumps({"text": question}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **({"X-Audit-Token": tok} if tok else {})})
    return json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8"))


def cmd_reseed(work: pathlib.Path, host: str | None, max_asks: int, seed: int, gen3: str) -> str:
    if not host:
        raise Stop("reseed asks the drafting server: give --host")
    from cubbyllm.worlds import wikikg as wk
    plans = _read(work / "plans.json")
    todo = [p for p in plans if p["source"] == "asks about a fact" and not p.get("ask_question")
            and not (work / "convs" / (p["id"].replace(":", "_") + ".json")).exists()]
    wk.ensure_data(("triplets",))
    pool = candidates(wk.load_triples(), held_seeds(gen3), random.Random(seed))
    asked = verified = 0
    t0 = time.time()
    for p in todo:
        while asked < max_asks:
            n, r, q = next(pool)
            asked += 1
            rec = ask(host, q)
            if rec.get("verified") and rec.get("trace") is not None and not rec.get("clarify"):
                p.update(ask=n, ask_relation=r, ask_question=q)
                verified += 1
                break
        else:
            break
    shutil.copy(work / "plans.json", work / f"plans_before_reseed_{int(t0)}.json")
    _write(work / "plans.json", plans)
    return json.dumps({"fact_plans_to_reseed": len(todo), "reseeded": verified, "asks": asked,
                       "minutes": round((time.time() - t0) / 60, 1), "left_as_they_were": len(todo) - verified})


STEP_SH = '#!/usr/bin/env bash\n# Order 0079: the step tool for a drafting session; its only allowed command.\nexec "{py}" "{step}" --work "{work}"{host} "$@"\n'


def cmd_rebatch(work: pathlib.Path, size: int, sessions: str | None, host: str | None) -> str:
    if not sessions:
        raise Stop("rebatch writes the sessions' folder: give --sessions (outside the repo)")
    root = pathlib.Path(__file__).resolve().parents[2]
    sdir = pathlib.Path(sessions).resolve()
    if root in sdir.parents or sdir == root:
        raise Stop(f"{sdir} is inside the repo: its CLAUDE.md files would be read at every session's start")
    plans = _read(work / "plans.json")
    started = {f.stem.replace("_", ":") for f in (work / "convs").glob("*.json")}
    old = sorted((work / "batches").glob("b*.json"))
    keep, last = [], -1
    aside = work / "batches" / "_old" / time.strftime("%Y%m%d_%H%M%S")   # each rebatch its own folder: nothing overwritten
    aside.mkdir(parents=True, exist_ok=True)
    for f in old:
        b = _read(f)
        if set(b["ids"]) & started or b.get("model"):
            keep.append(b)
            last = max(last, int(b["batch"][1:]))
        else:
            for ext in (".json", ".md"):
                g = f.with_suffix(ext)
                if g.exists():
                    shutil.move(str(g), str(aside / g.name))
    kept_ids = {i for b in keep for i in b["ids"]}
    todo = [p for p in plans if p["id"] not in kept_ids and p["id"] not in started]
    new = P.batches(todo, size)
    for k, b in enumerate(new):
        b["batch"] = f"b{last + 1 + k:03d}"
        b["writer"] = ("claude-haiku", "claude-sonnet")[(last + 1 + k) % 2]
        _write(work / "batches" / f"{b['batch']}.json", b)
        (work / "batches" / f"{b['batch']}.md").write_text(brief(work, b["batch"], host), encoding="utf-8")
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "step.sh").write_text(STEP_SH.format(py=pathlib.Path(sys.executable).as_posix(), step=STEP.as_posix(),
                                                 work=work.resolve().as_posix(), host=f' --host "{host}"' if host else ""),
                                  encoding="utf-8", newline="\n")
    launcher = work / "launch_sessions.ps1"
    if launcher.exists():                                # Cowork's launcher, run from the sessions' folder instead
        text = launcher.read_text(encoding="utf-8")
        text = text.replace("-WorkingDirectory $work", "-WorkingDirectory $sessions").replace(
            '$prompt = "$work\\prompt_$b.txt"', '$prompt = "$sessions\\prompt_$b.txt"')
        text = text.replace("$ErrorActionPreference", f'$sessions = "{sdir}"\n$ErrorActionPreference', 1)
        (sdir / "launch_sessions.ps1").write_text(text, encoding="utf-8")
    return json.dumps({"kept_batches": [b["batch"] for b in keep], "moved_aside": len(old) - len(keep),
                       "new_batches": len(new), "first": new[0]["batch"] if new else None,
                       "last": new[-1]["batch"] if new else None, "plans_in_them": len(todo), "sessions": str(sdir)})
