"""The questions people actually ask, as benchmarks (Nick, 2026-09-13: "more common questions related to usa /
canada ... and europe ... that people will actually ask"; "natural questions are a must"):

  * WebQuestions (Stanford): 5,810 questions from Google Suggest (2013) with Freebase answers;
  * NQ-Open (Google, Natural Questions): 91,535 real Google search queries with short answers from Wikipedia.

Filtered to the USA, Canada and Europe: a row is kept when its question or one of its answers names a
country, state, province or major city of those regions; the region tag is the first match (`--all` keeps
everything, tagging the rest 'world'). Writes a SimpleQA-shaped csv (problem, answer, answers, region, url)
that `exp_r11_search_learn.py --questions` takes; every listed answer counts.

  python standin/data/build_webq_bench.py --dataset webq     # -> E:/datasets/webquestions/webq_na_eu.csv
  python standin/data/build_webq_bench.py --dataset nq       # -> E:/datasets/webquestions/nq_na_eu.csv
"""
from __future__ import annotations

import argparse, collections, csv, pathlib, re

USA = ("united states|usa|u\\.s\\.|america|american|alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
       "florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|"
       "minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|"
       "north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|"
       "virginia|washington|west virginia|wisconsin|wyoming|los angeles|chicago|houston|philadelphia|phoenix|san antonio|"
       "san diego|dallas|san francisco|seattle|boston|denver|miami|atlanta|detroit|las vegas|nashville|new orleans|"
       "washington dc|white house|nfl|nba|mlb|nhl|congress|senate")
CANADA = ("canada|canadian|quebec|québec|ontario|british columbia|alberta|manitoba|saskatchewan|nova scotia|new brunswick|"
          "newfoundland|prince edward island|yukon|nunavut|northwest territories|toronto|montreal|montréal|vancouver|ottawa|"
          "calgary|edmonton|winnipeg|halifax|lévis|levis")
EUROPE = ("europe|european|england|britain|british|united kingdom|uk|scotland|wales|ireland|irish|france|french|paris|germany|"
          "german|berlin|italy|italian|rome|spain|spanish|madrid|portugal|netherlands|holland|amsterdam|belgium|brussels|"
          "switzerland|swiss|austria|vienna|sweden|swedish|norway|denmark|finland|iceland|poland|polish|czech|prague|hungary|"
          "budapest|greece|greek|athens|turkey|russia|russian|moscow|ukraine|romania|bulgaria|serbia|croatia|slovakia|slovenia|"
          "london|edinburgh|dublin|lisbon|barcelona|milan|munich|stockholm|oslo|copenhagen|helsinki|warsaw|vatican|monaco|"
          "luxembourg|malta|cyprus|estonia|latvia|lithuania|belarus|scandinavia|alps|rhine|danube|thames|seine")
REGIONS = [("usa", re.compile(rf"\b({USA})\b", re.I)), ("canada", re.compile(rf"\b({CANADA})\b", re.I)),
           ("europe", re.compile(rf"\b({EUROPE})\b", re.I))]


def region_of(question: str, answers: list[str]) -> str | None:
    for text in [question] + list(answers):
        for name, rx in REGIONS:
            if rx.search(text):
                return name
    return None


NQ_URL = "https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/refs%2Fconvert%2Fparquet/nq_open/{split}/0000.parquet"
WEBQ_URL = "https://huggingface.co/datasets/stanfordnlp/web_questions/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"


def fetch(src: pathlib.Path, dataset: str) -> list[dict]:
    """The parquet files, downloaded once into `src`; rows as {question, answers, url}."""
    import urllib.request
    import pyarrow.parquet as pq
    src.mkdir(parents=True, exist_ok=True)
    splits = ("train", "test") if dataset == "webq" else ("train", "validation")
    url = WEBQ_URL if dataset == "webq" else NQ_URL
    rows = []
    for split in splits:
        path = src / f"{dataset}_{split}.parquet"
        if not path.is_file():
            legacy = src / f"{split}.parquet"                       # the first WebQuestions pull, 2026-09-13
            if dataset == "webq" and legacy.is_file():
                path = legacy
            else:
                urllib.request.urlretrieve(url.format(split=split), str(path))
        for r in pq.read_table(str(path)).to_pylist():
            rows.append({"question": r["question"], "answers": list(r.get("answers") or r.get("answer") or []),
                         "url": r.get("url") or f"nq_open/{split}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="E:/datasets/webquestions")
    ap.add_argument("--dataset", choices=("webq", "nq"), default="webq",
                    help="webq: WebQuestions (5,810, Google Suggest 2013); nq: NQ-Open (91,535 real Google queries, Natural Questions)")
    ap.add_argument("--out", default=None, help="default: <src>/<dataset>_na_eu.csv")
    ap.add_argument("--all", action="store_true", help="keep every question (no region filter); region = 'world' when none matches")
    a = ap.parse_args()
    out = a.out or str(pathlib.Path(a.src) / f"{a.dataset}_na_eu.csv")
    rows = fetch(pathlib.Path(a.src), a.dataset)
    kept = []; c = collections.Counter()
    for r in rows:
        answers = [x for x in r["answers"] if x and not str(x).startswith("http")]
        if not answers:
            c["no_answer"] += 1; continue
        reg = region_of(r["question"], answers)
        if reg is None:
            c["other"] += 1
            if not a.all:
                continue
            reg = "world"
        c[reg] += 1
        kept.append({"problem": r["question"].strip().rstrip("?") + "?", "answer": answers[0], "answers": " | ".join(answers),
                     "region": reg, "url": r["url"]})
    a.out = out
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "answer", "answers", "region", "url"]); w.writeheader(); w.writerows(kept)
    print(f"{len(rows):,} questions -> {len(kept):,} kept ({', '.join(f'{k} {v}' for k, v in c.most_common())}) -> {a.out}")
    for r in kept[:10]:
        print("  ", r["region"], "|", r["problem"], "->", r["answers"][:60])


if __name__ == "__main__":
    main()
