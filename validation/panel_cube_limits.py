"""panel_cube_limits â€” put the two mapping limitations to a panel of frontier models.

Wired: STANDALONE (validation only; the final model never calls OpenRouter).

Asks several models the SAME well-specified question independently, caches every
answer, and writes them side by side so the agreements and the disagreements are
both visible. A panel is only worth the money if the question is precise enough
that a vague answer is obviously vague, so the prompt below carries the actual
constraints â€” including the ones that rule out the easy answers.

    python validation/panel_cube_limits.py            # ask, cache, write the report
    python validation/panel_cube_limits.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from openrouter import API, _key  # noqa: E402

CACHE = ROOT / "standin" / "data" / "out" / "panel_cache"
OUT = ROOT / "docs" / "panel_cube_limits.md"

# diverse vendors, strong tier, ids read from the live model list rather than
# remembered â€” a guessed model id fails loudly, which is the good case, but it
# also wastes a round trip and there is no reason to guess
PANEL = [
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "moonshotai/kimi-k3",
    "x-ai/grok-4.6",
    "deepseek/deepseek-v3.2",
]

SYSTEM = (
    "You are advising on the design of an affect model for an autonomous agent. "
    "You are one of several independent reviewers; be specific and be willing to say "
    "that a premise is wrong. Prefer a concrete mechanism with a stated failure mode "
    "over a survey of options. If you propose something, say how it would be VALIDATED "
    "â€” what measurement would show it is working, and what would show it is not."
)

QUESTION = r"""
# The system

An autonomous agent (a solitary explorer in a 3-D maze â€” no other agents, no social
partner) has a 5-hormone neurochemical ODE: dopamine, serotonin, noradrenaline,
oxytocin, cortisol, with receptor couplings and per-hormone decay rates. It runs one
update per perception frame.

Emotion is read out in three layers, deliberately separated so that only one classifier
ever decides anything:

  LAYER 1  SOMATIC. Pain and craving are homeostatic drives with their own circuits.
           They (a) skew the monoamine drives inside the ODE, so the skew integrates and
           decays like everything else, and (b) above a threshold, intercept the naming
           entirely. They are handled outside the cube because the cube has no axis for
           tissue damage or appetitive deficit.

  LAYER 2  GEOMETRY. A LÃ¶vheim cube over (5-HT, DA, NE), each axis scaled so that the
           ODE's measured QUIESCENT point is the CENTRE of the cube (0.5, 0.5, 0.5) and
           each axis reaches its own biological clamp at 0 and 1. Nearest-corner by
           Euclidean distance â€” Voronoi cells meeting at the centre. Returns: the corner,
           an INTENSITY (distance from centre, 0 at rest, 1 at the vertex), the runner-up
           corner, and the margin between them.

  LAYER 3  NAMING. The corner picks a Plutchik petal. The intensity picks one of that
           petal's three tiers (e.g. annoyance / anger / rage). A small margin to the
           runner-up yields a Plutchik primary DYAD instead. The name is then offered to
           a language model as 2-3 candidate readings of the same state â€” never asserted â€”
           and the model chooses one and speaks from it.

This layering exists because two classifiers over one space (nearest-corner for the
compass, thresholds for the naming) disagreed in the band between them: the compass said
"annoyance" while the names said "elation" on the same step. There is now exactly one
classifier, and Layer 3 only decides what the place Layer 2 picked can be called.

GoEmotions (27 labels + neutral) is mapped into this cube for corpus work. The mapping is
DERIVED, not asserted: each label's coordinate is its Plutchik petal's corner, reached
from the centre by the label's intensity tier. It is validated by running every derived
coordinate back through the agent's own nearest-corner function and requiring it to land
on the corner its petal implies. That check passes.

# The two limitations we want solved

## LIMITATION 1 â€” Plutchik's eight petals and LÃ¶vheim's eight corners are not the same eight.

Our eight corners map to: joy, trust, fear, surprise, sadness, disgust, anger, and SHAME.
Plutchik's eight petals are: joy, trust, fear, surprise, sadness, disgust, anger, and
ANTICIPATION.

So the cube has a corner we cannot use and lacks one we need:

  * SHAME is social. It requires another agent to be ashamed before. Our agent is alone in
    a maze; the corner is structurally unreachable in any meaningful sense, and we already
    exclude it (along with contempt) from naming for exactly this reason.
  * ANTICIPATION has no corner at all. It is currently handled as a special-case override
    in the classifier ("high novelty and high noradrenaline -> curious"), which is a second
    classifier bolted onto the first â€” precisely the thing this architecture was rebuilt to
    eliminate. And three GoEmotions labels (curiosity, desire, optimism) have nowhere to go
    and are left UNPLACED rather than rounded into a neighbouring corner.

Anticipation matters for this agent more than most emotions: it is an explorer whose whole
loop is forming a hypothesis, testing it, and waiting for the world to answer.

Is the right move to (a) substitute anticipation for shame at that vertex and justify it
neurochemically, (b) treat anticipation as a dyad region rather than a vertex, (c) add a
fourth axis, (d) accept that the cube is the wrong geometry for this and propose what is
right, or (e) something else? Give the neurochemical justification, not just the
engineering convenience â€” the coordinates must be defensible as a claim about monoamines,
because the ODE is what produces them.

## LIMITATION 2 â€” the naming has 24 distinguishable points for 28 labels.

A label's position is (petal x tier) = 8 x 3 = 24, so it cannot separate 28 labels by
construction. Observed collisions: {disappointment, embarrassment, remorse, sadness} share
one point; {excitement, love, pride} share another; {disapproval, disgust} share another.

For the current use â€” bucketing corpus rows by CORNER â€” collisions are harmless. For a
lookup that should tell remorse from sadness, they are not.

Several collisions are genuinely Plutchik dyads (love = joy+trust, remorse = sadness+disgust),
so pulling them toward the second petal's corner would separate them. But that introduces a
second positioning rule alongside petal-x-tier, and we are wary of any second rule over the
same space after what happened last time.

Is the dyad pull the right separation, and how would it be kept from becoming a second
classifier? Or is the collision telling us the labels differ along a dimension the monoamines
genuinely do not encode â€” in which case what is that dimension, and should these labels simply
be treated as the same affective state with different cognitive appraisals attached?

# Constraints that rule out the easy answers

1. NO RETRAINING. The agent's language model is fixed at runtime. Anything proposed must work
   as architecture, data, or computation â€” not as "fine-tune it to handle this".
2. ONE CLASSIFIER PER SPACE. Any proposal that adds a second opinion about where the agent is
   will be rejected. A refinement WITHIN a decision the single classifier already made is fine.
3. DERIVABLE AND CHECKABLE. Every coordinate must be computable from something already
   justified, and must survive being run back through the agent's own classifier. A
   hand-written coordinate table is what we are replacing; it had a duplicate dict key that
   made ANGER unreachable, and nothing detected it.
4. SOLITARY AGENT. Social emotions have no referent. Do not solve limitation 1 by appealing to
   social context.
5. The ODE is the source of truth for the hormone values. A proposal that requires hormone
   levels the ODE cannot produce is not implementable.

# What we want back

For each limitation: your recommended solution, the neurochemical or theoretical
justification for it, the specific failure mode it introduces, and the measurement that
would tell us it is working or not. Disagree with the framing if the framing is wrong.
"""


def ask(model: str, offline: bool, max_tokens: int, effort: str = "low") -> dict:
    """One model, one question.

    REASONING TOKENS COUNT AGAINST `max_tokens`, which is the whole reason this
    takes an `effort` and a large budget. The first run of this panel asked for
    4000 tokens with no reasoning control and three of six models — gpt-5.5,
    kimi-k3, gemini-3.1-pro — returned `finish=length` with ZERO words of
    answer: the entire budget went to thinking and nothing was left to say it
    with. `openrouter.py`'s own docstring warns about this from a previous run;
    the warning was in the tree and I did not read it.

    The settings are part of the cache key, and an empty answer is NEVER
    cached. Both matter: without the key an experiment re-run with a bigger
    budget silently re-serves the old failure, and caching an empty answer
    makes a transient truncation permanent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f"{model}|{SYSTEM}|{QUESTION}|{max_tokens}|{effort}".encode()).hexdigest()[:20]
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        # Rendering, not asking: take any complete answer this model has given,
        # whatever budget it was asked under. The settings belong in the key so
        # a fresh RUN re-asks rather than re-serving a truncation — but they
        # must not make a report cost money to regenerate.
        best = None
        for f in CACHE.glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("model") == model and (r.get("text") or "").strip():
                if best is None or len(r["text"]) > len(best["text"]):
                    best = r
        return best or {"model": model, "text": "", "error": "not cached and --offline"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.3,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": QUESTION}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Grillcheese-AI/CubbyLLM", "X-Title": "CubbyLLM panel"})
    t0 = time.time()
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except Exception as e:                               # a model that fails is a row, not a crash
        return {"model": model, "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    ch = (raw.get("choices") or [{}])[0]
    rec = {"model": model, "text": (ch.get("message") or {}).get("content") or "",
           "finish": ch.get("finish_reason"), "usage": raw.get("usage") or {},
           "wall_s": round(time.time() - t0, 1), "effort": effort, "budget": max_tokens,
           "error": None}
    if rec["text"].strip():                              # never cache a truncation
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        rec["error"] = (f"no answer: finish={rec['finish']}, "
                        f"{(rec['usage'] or {}).get('completion_tokens', '?')} tokens all spent "
                        f"on reasoning - raise --max-tokens or lower --effort")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-render from cache, spend nothing")
    # reasoning tokens are drawn from this budget, so it has to cover thinking
    # AND answering. 4000 left three of six models with nothing to say.
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--effort", default="low", help="OpenRouter reasoning effort; empty to omit")
    ap.add_argument("--models", nargs="*", default=PANEL)
    a = ap.parse_args()

    answers = []
    for m in a.models:
        print(f"  asking {m} ...", flush=True)
        rec = ask(m, a.offline, a.max_tokens, a.effort)
        answers.append(rec)
        u = rec.get("usage") or {}
        print(f"    {'ERROR: ' + rec['error'] if rec.get('error') else ''}"
              f"{len(rec['text'].split()):>6} words  {rec.get('wall_s', 0)}s  "
              f"finish={rec.get('finish')}  out_tok={u.get('completion_tokens', '?')}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Panel: the two limitations of the cube mapping\n\n")
        f.write("Asked independently, same prompt, temperature 0.3. Cached under "
                "`standin/data/out/panel_cache/`; `--offline` re-renders without spending.\n\n")
        f.write("**These are model opinions, not findings.** Nothing here is true because a "
                "panel said it. Each proposal still has to survive the validation it names â€” "
                "and the point of asking several independently is that where they DISAGREE is "
                "where our framing is probably underspecified.\n\n")
        f.write("## The question\n\n<details><summary>full prompt</summary>\n\n```\n")
        f.write(QUESTION.strip() + "\n```\n\n</details>\n\n")
        ok = [r for r in answers if r["text"]]
        f.write(f"## Answers ({len(ok)} of {len(answers)} returned)\n\n")
        for r in answers:
            f.write(f"### {r['model']}\n\n")
            if r.get("error"):
                f.write(f"*failed: {r['error']}*\n\n")
                continue
            if r.get("finish") == "length":
                f.write("*(truncated at the token budget)*\n\n")
            f.write(r["text"].strip() + "\n\n---\n\n")
    spent = sum((r.get("usage") or {}).get("total_tokens", 0) for r in answers)
    print(f"\nwrote {OUT}  ({len(ok)}/{len(answers)} answered, {spent} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
