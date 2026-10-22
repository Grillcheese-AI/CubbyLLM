"""plugins_talk_e2e -- order 0028: the `talk` plugin on the real 450M (grilly2, the local card).

Three checks, greedy decoding on fixed prompts:
1. **plugin = flag**: the talk adapter loaded as a plugin role (``Loader.load("talk")`` -> ``Cubby450mTrunk.add_role``)
   says exactly what the same adapter says when the trunk is built with it (the server's ``--talk-adapter`` path);
2. **the other role does not move**: the trunk's first role answers the same before the plugin loads, while it is
   loaded, and after it unloads;
3. **another base is refused**: the plugin pins the base it was trained on (``requires.base``, the export's SHA-256);
   a trunk on another export refuses it with the reason, before any weight moves.

    python validation/plugins_talk_e2e.py [--base DIR] [--other-base DIR] [--seat DIR] [--talk DIR] [--tokens 40]
-> validation/logs/plugins_talk_e2e_<date>.json (the outputs' hashes and equalities, the prompts; no local paths)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "standin")]
MODELS = os.environ.get("CUBBY_MODELS", "")
PROMPTS = ["hi Cubby, how are you?", "I had a rough day at work", "what do you like to do?"]


def say(trunk, role: str, n: int) -> list[str]:
    from emitter import Cubby450mTalk
    t = Cubby450mTalk(trunk, role=role)
    return [t.emit(p, max_new_tokens=n) for p in PROMPTS]


def digest(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=os.path.join(MODELS, "base450m", "grilly"))
    ap.add_argument("--other-base", default=os.path.join(MODELS, "base450m_cont", "grilly"))
    ap.add_argument("--seat", default=os.path.join(MODELS, "base450m", "talk_v2_1"), help="the trunk's first role")
    ap.add_argument("--talk", default=os.environ.get("CUBBY_TALK_ADAPTER") or os.path.join(MODELS, "base450m", "talk_v2_2"))
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    from emitter import Cubby450mTrunk
    import plugins_guard as guard
    from plugins_index import discover
    from plugins_load import Loader
    tok = os.path.join(a.base, "tokenizer.json")
    t0 = time.time()
    rep = {"prompts": PROMPTS, "tokens": a.tokens, "seat": os.path.basename(os.path.normpath(a.seat)),
           "talk": os.path.basename(os.path.normpath(a.talk))}

    # 1. the flag's path: one trunk built with both adapters
    ref = Cubby450mTrunk(a.base, tok, {"seat": a.seat, "talk": a.talk})
    base_hash = ref.base_hash()
    flag_talk, flag_seat = say(ref, "talk", a.tokens), say(ref, "seat", a.tokens)
    del ref

    # 2. the plugin's path: a trunk with the seat alone, the talk plugin pinned to this base
    tmp = tempfile.mkdtemp(prefix="talk_e2e_")
    os.environ["CUBBY_PLUGINS_LOCK"] = os.path.join(tmp, "lock.json")
    os.environ["CUBBY_TALK_ADAPTER"] = a.talk
    d = os.path.join(tmp, "plugins", "talk")
    shutil.copytree(os.path.join(os.path.dirname(HERE), "plugins", "talk"), d)
    md = open(os.path.join(d, "PLUGIN.md"), encoding="utf-8").read()
    if "base: null" in md:
        md = md.replace("base: null", f"base: {base_hash}")
    open(os.path.join(d, "PLUGIN.md"), "w", encoding="utf-8", newline="\n").write(md)
    idx = discover([os.path.dirname(d)])
    guard.approve(idx["talk"])
    trunk = Cubby450mTrunk(a.base, tok, {"seat": a.seat})
    seat_before = say(trunk, "seat", a.tokens)
    loader = Loader(idx, data_root=os.path.join(tmp, "data"), trunk=trunk, log=lambda *_: None)
    loaded = loader.load("talk")
    plug_talk = say(trunk, "talk", a.tokens) if loaded else []
    seat_during = say(trunk, "seat", a.tokens)
    loader.unload("talk") if hasattr(loader, "unload") else trunk.remove_role("talk")
    seat_after = say(trunk, "seat", a.tokens)
    del trunk

    # 3. another export refuses the pinned plugin before its weights move
    other = Cubby450mTrunk(a.other_base, os.path.join(a.other_base, "tokenizer.json"), {"seat": a.seat})
    loader2 = Loader(discover([os.path.dirname(d)]), data_root=os.path.join(tmp, "data2"), trunk=other, log=lambda *_: None)
    refused_ok = not loader2.load("talk")
    rep.update({
        "base_sha256": base_hash,
        "plugin_loaded": loaded, "plugin_status": loader.status("talk"),
        "plugin_equals_flag": plug_talk == flag_talk,
        "seat_unchanged_before_during_after": seat_before == seat_during == seat_after,
        "seat_equals_flag_trunk": seat_before == flag_seat,
        "other_base_refused": refused_ok, "other_base_reason": loader2.refused.get("talk", loader2.status("talk")),
        "digests": {"flag_talk": digest(flag_talk), "plugin_talk": digest(plug_talk), "seat": digest(seat_before)},
        "talk_said": plug_talk, "wall_s": round(time.time() - t0, 1)})
    shutil.rmtree(tmp, ignore_errors=True)
    out = os.path.join(HERE, "logs", f"plugins_talk_e2e_{a.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "talk_said"}, indent=1, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
