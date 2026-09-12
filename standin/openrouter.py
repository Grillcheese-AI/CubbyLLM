"""openrouter — a frontier model as a PROPOSER for probes and data building, never for serving.

Wired: STANDALONE (validation and data-building only). The final model never calls this: it is
the ceiling probe (exp_r11 --proposer) and the gen-3 question writer (build_gen3_partition), and
every output it produces goes through the same disposer, walk and VM as the stand-in's. It never
judges anything.

`OpenRouterProposer(model).emit(question)` returns a CotPlan program (the stand-in emitter's
output shape: SEED + HOP1..HOPk), built from the JSON the model is asked for, so exp_r11 reads
it with the same `emitted_plan`. Responses are cached by (model, prompt) under
data/out/openrouter_cache/ -- a rerun is offline and byte-identical -- and usage is tallied.
The key is read from $OPENROUTER_API_KEY or from an off-repo file named by $OPENROUTER_KEY_FILE.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request

__wiring__ = "STANDALONE"

API = "https://openrouter.ai/api/v1/chat/completions"
CACHE = pathlib.Path(__file__).resolve().parent / "data" / "out" / "openrouter_cache"

PLAN_SYSTEM = """You turn a factual question into a relation-chain plan over a knowledge graph.
Answer with ONE JSON object and nothing else: {"seed": <entity named in the question, or null>, "hops": [<relation>, ...]}.
Rules:
- "seed" is the entity the question starts from, exactly as the question names it.
- "hops" are the relations to follow from the seed, in order (hops[0] applies to the seed, hops[1] to its result, ...); the last hop's value is the answer.
- Name each relation the way a knowledge graph labels it (Wikidata property labels): "date of birth", "place of birth", "country of citizenship", "award received", "position held", "spouse", "inception", "located in the administrative territorial entity", "educated at", "employer", "occupation", "author", "publisher", "publication date", "director", "cast member", "genre", "capital", "head of government", "population".
- Never put a qualifier (a year, "as of", "first", "how many") into a relation name. If the question needs a qualifier or a count that a chain of relations cannot express, output {"seed": null, "hops": []}.
- Never answer the question. Only the plan."""


KEY_FILES = (pathlib.Path(__file__).resolve().parents[1] / "validation" / ".env",)   # gitignored (.gitignore: .env)


def _key() -> str:
    """$OPENROUTER_API_KEY, else a key file ($OPENROUTER_KEY_FILE or validation/.env): either the
    bare key or `OPENROUTER_API_KEY=<key>` lines. The key never enters a log or a commit."""
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        for path in ([os.environ["OPENROUTER_KEY_FILE"]] if os.environ.get("OPENROUTER_KEY_FILE") else []) + list(KEY_FILES):
            p = pathlib.Path(path)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    name, val = line.split("=", 1)
                    if name.strip() == "OPENROUTER_API_KEY":
                        k = val.strip().strip('"').strip("'"); break
                else:
                    k = line; break
            if k:
                break
    if not k:
        raise RuntimeError("no OpenRouter key: set OPENROUTER_API_KEY, OPENROUTER_KEY_FILE, or validation/.env (gitignored)")
    return k


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def cot_plan(seed: str, rels: list[str]) -> str:
    binds = "\n".join([f'        bind frame, SEED, "{_esc(seed)}";'] +
                      [f'        bind frame, HOP{i + 1}, "{_esc(r)}";' for i, r in enumerate(rels)])
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n"
            "    public function solve(mention: str): str {\n"
            "        create frame: number;\n" + binds + "\n"
            "        return recover(frame, SEED);\n    }\n}\n")


class OpenRouterProposer:
    """`emit(question) -> CotPlan program text` ('' when the model declines a plan)."""

    def __init__(self, model: str, cache_dir: pathlib.Path | str | None = CACHE, system: str = PLAN_SYSTEM,
                 temperature: float = 0.0, offline: bool = False, sleep_s: float = 0.0) -> None:
        self.model, self.system, self.temperature, self.offline, self.sleep_s = model, system, temperature, offline, sleep_s
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        self.last: dict = {}

    def chat(self, user: str, system: str | None = None, max_tokens: int = 300) -> str:
        system = self.system if system is None else system
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        elif self.offline:
            return ""
        else:
            req = urllib.request.Request(API, data=json.dumps(body).encode("utf-8"), headers={
                "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
                "HTTP-Referer": "https://grillcheese.ai", "X-Title": "cubbyllm-standin probe"})
            self.calls += 1
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            if path:
                path.write_text(json.dumps(data), encoding="utf-8")
            if self.sleep_s:
                time.sleep(self.sleep_s)
        u = data.get("usage") or {}
        self.usage["prompt_tokens"] += int(u.get("prompt_tokens", 0)); self.usage["completion_tokens"] += int(u.get("completion_tokens", 0))
        self.usage["cost"] += float(u.get("cost", 0.0) or 0.0)
        self.last = data
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def parse_plan(text: str) -> tuple[str | None, list[str]]:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t[t.find("{"):] if "{" in t else t
        try:
            obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
        except (ValueError, TypeError):
            return None, []
        seed = obj.get("seed"); hops = obj.get("hops") or []
        if not isinstance(seed, str) or not isinstance(hops, list) or not all(isinstance(h, str) and h.strip() for h in hops):
            return None, []
        return seed.strip(), [h.strip() for h in hops]

    def emit(self, prompt: str, max_new_tokens: int = 300, **_ignored) -> str:
        seed, hops = self.parse_plan(self.chat(prompt, max_tokens=max_new_tokens))
        if not seed or not hops:
            return ""
        return cot_plan(seed, hops)


def list_models(limit: int = 400) -> list[dict]:
    """The live catalog (id, name, pricing) -- to pick a proposer by today's ids."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for m in (data.get("data") or [])[:limit]:
        pr = m.get("pricing") or {}
        out.append({"id": m.get("id"), "name": m.get("name"), "prompt": pr.get("prompt"), "completion": pr.get("completion")})
    return out
