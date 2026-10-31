import io
import sys

path = sys.argv[1]
text = io.open(path, encoding="utf-8").read()

S12_END = "the Unity placement path (gate 3) is the next piece.**"
S12_ADD = (S12_END + " **First read at the 450M (2026-10-20, `notebooks/spatial_facts_450m.ipynb` run 1: `base450m_cont_stable_end` "
  "continued 2,808 steps = 0.37B tokens an arm on an RTX PRO 6000, the two hypertoken shards at ~4.6% each; `link` = "
  "`CB_SPATIAL=1 CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1`, the link added to the trained trunk with `o = 0`; `text` = the same "
  "villages with cells as words, at the link arm's step count; `validation/exp_hypertoken_recall.py`, every held-out question "
  "teacher-forced):** villages -- `link` **0.793** exact with the address on the question, **0.700** with the questions' places "
  "stripped (the trunk's own query, taught by the route loss, finds the address), against `text` **0.995**: **gate 1 fails** -- "
  "in-window, cells written as numbers are answered almost perfectly by attention, and the link is worse there at this budget "
  "(it is not killed by the letter: 0.793 is above 0.75; it is beaten). **Gate 2 passes** on the plain-text sources: held-out "
  "loss 2.2318 (`link`) against 2.2286 (`text`), +0.003, every source within 0.017 (`_mix`, which weights each arm's own "
  "hypertoken shards in, reads 2.1662 / 2.1442); `instruct` and `program` read 0.10-0.13 lower on `link` because that arm still "
  "drew Drive's duplicate copies of them, held-out tails included (fixed since: the Corpus skips such copies). What it says: the "
  "places channel costs the trunk nothing on text and its route is learnable at scale, but where attention can see the words, "
  "the words win; the case for places is what attention cannot see -- beyond the window, or more things than a window holds -- "
  "which run 2 tests (H-K10). Throughput: 54.6k tok/s, 29.5% MFU (`link`); the text arm's profile (`validation/profile_base.py`): "
  "matmuls 44% of device time, the MinGRU scan and its casts ~20%, attention 8.5% (dense masked: no flex on this card), the "
  "128k-vocabulary loss **2.8%** -- not the bottleneck it was assumed to be.")

K10_END = ("then spans (gate 3).**")
K10_ADD = (K10_END + " **Run 1 at the 450M (2026-10-20, the same runs as H-S12's first read):** facts at addresses in-window "
  "(`facts`, 512-token documents) -- `link` **0.016** exact (first token 0.27), `text` **0.987**: the read did not learn, and the "
  "comparison was the wrong one. Causes, in the shard and the scene: every placed token wrote into ONE scene per 1,024-token "
  "window -- two documents, every name piece, and the questions' own `here` and answer pieces at the very addresses they asked "
  "about -- ~150-250 items in d = 1024, past the capacity law's ~55 for 90% (H-K8); and the facts sat inside the attention "
  "window, so `text` copied them. **Built for run 2 (`cad8da2`):** a thing WRITES into the scene and a question only ASKS there "
  "(`.role.u8`; `Tokens.flags`), the scene restarts at every document (`SpatialLink.scene(starts=)`), and the facts move beyond "
  "the window -- `facts_far1k` (6-10 facts, ~800 tokens of real text, then the questions; ~24 written items, the first question "
  "809 tokens past the last fact) against `facts_far1k_text`. Run 1's link final read with the roles: villages 0.836 (12 held-out "
  "documents), `facts_far1k` 0.000 (never trained on it). Status: **OPEN -- run 2 under way.**")

for old, new in ((S12_END, S12_ADD), (K10_END, K10_ADD)):
    assert text.count(old) == 1, old
    text = text.replace(old, new)
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("recorded")
