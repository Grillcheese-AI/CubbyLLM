"""The small history record files -> plain text the history-graph reader takes (build_history_graph.py --src),
with every hypothesis left out: only what a record states as having happened goes in.

    python standin/data/build_history_records.py --src <temporal_timelines_events folder>
    python standin/data/build_history_graph.py --src standin/data/out/history_records --tag _records

Sources (listed for the owner's disclosure), in the local datasets drive's `domains/temporal_timelines_events`:
  historical_events_1800-1850.jsonl, historical_events_1800-1900.jsonl   model-written event records: title, dates,
      participants and a summary are kept; `causal_link`, `precursor_events`, `similar_events`,
      `downstream_influence` and `butterfly_effect_analysis` are the writer's own analysis and counterfactuals
      ("If ... had failed, ... might have ...") -- hypotheses, left out
  arkona_intermediate_3000_detailed.json   student/expert dialogues about people in an encyclopedia: the expert's
      turns only, per person
  historical_high_confidence_831_samples.json, historical_modern_history_771_samples.json,
  historical_wars_conflicts_648_samples.json   question/answer summaries of book passages: the answers, once each;
      their `year` field is not used (it does not match the text: "What happened in 1354?" over the French Revolution)
  inventions.json   dated descriptions of inventions, events and achievements
From every text kept, a sentence that speculates is dropped whole ("might have", "would have", "could have been",
"perhaps", "possibly", "likely", "probably", "it is believed", "legend has it", "if ... had"): the graph holds what
happened, and a branch is where a hypothesis belongs.
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

SPECULATES = re.compile(r"\b(might|would|could|may) (?:have|not have)\b|\bcould have been\b|\bperhaps\b|\bpossibly\b|"
                        r"\blikely\b|\bprobably\b|\bpresumably\b|\bit is (?:believed|thought|said)\b|\blegend(?:s)? (?:has|have|says)\b|"
                        r"\bhypothe|\bcounterfactual\b|\bspeculat|\bwhat if\b|\bif [^.]{0,80}\bhad\b|\bsuggests? that\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def facts_only(text: str, dropped: collections.Counter) -> str:
    keep = []
    for s in _SENT.split(" ".join(str(text or "").split())):
        if SPECULATES.search(s):
            dropped["speculative_sentence"] += 1
        elif s:
            keep.append(s)
    return " ".join(keep)


def load(path: str):
    if path.endswith(".jsonl"):
        return [json.loads(l, strict=False) for l in open(path, encoding="utf-8") if l.strip()]
    d = json.load(open(path, encoding="utf-8"), strict=False)
    return d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])


def records(src: str, dropped: collections.Counter) -> dict[str, list[str]]:
    books = collections.defaultdict(list)
    for name in ("historical_events_1800-1850.jsonl", "historical_events_1800-1900.jsonl"):
        for r in load(os.path.join(src, name)):
            dropped["analysis_fields_left_out"] += sum(1 for k in ("causal_link", "precursor_events", "similar_events",
                                                                    "downstream_influence", "butterfly_effect_analysis") if k in r)
            dates = ", ".join(r.get("cleaned_dates_extended") or r.get("cleaned_dates") or [])
            who = ", ".join(p.get("name", "") for p in r.get("participants") or [] if isinstance(p, dict))
            body = facts_only(r.get("source_text") or r.get("summary"), dropped)
            if body:
                books["model_written_events_1800_1900"].append(
                    f"{r.get('title', '')}{f' ({dates})' if dates else ''}. {body}{f' Participants: {who}.' if who else ''}")
    arkona = load(os.path.join(src, "arkona_intermediate_3000_detailed.json"))
    by_topic = collections.defaultdict(list)
    for r in arkona:
        if r.get("speaker") == "expert":
            body = facts_only(r.get("message"), dropped)
            if body:
                by_topic[r.get("topic", "")].append(body)
    for topic, parts in by_topic.items():
        books["encyclopedia_dialogues_people"].append(f"{topic}. " + " ".join(parts))
    seen = set()
    for name in ("historical_high_confidence_831_samples.json", "historical_modern_history_771_samples.json",
                 "historical_wars_conflicts_648_samples.json"):
        for r in load(os.path.join(src, name)):
            a = " ".join(str(r.get("answer") or "").split())
            if not a or a in seen:
                continue
            seen.add(a)
            body = facts_only(a, dropped)
            if body:
                books["book_passage_summaries"].append(body)
    for r in load(os.path.join(src, "inventions.json")):
        body = facts_only(r.get("description"), dropped)
        if body:
            books["inventions_and_achievements"].append(body)
    return books


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "history_records"))
    args = ap.parse_args()
    dropped = collections.Counter()
    books = records(args.src, dropped)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, paras in books.items():
        (out / f"{name}.txt").write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    stats = {"books": {k: {"records": len(v), "words": sum(len(p.split()) for p in v)} for k, v in books.items()},
             "left_out": dict(dropped)}
    (ROOT / "validation" / "logs" / "history_records.manifest.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
