"""Run 1 of the spatial notebook: the held-out loss on the plain-text sources only, per arm."""
import json
import sys

r = sys.argv[1]
sizes = json.load(open(sys.argv[2]))
val = {}
for a in ("link", "text"):
    rows = [json.loads(l) for l in open(f"{r}/spatial_{a}/spatial_{a}_metrics.jsonl") if '"val"' in l]
    val[a] = rows[-1]["val"]
common = [k for k in val["link"] if not k.startswith(("_", "spatial", "facts")) and k in val["text"]
          and " (" not in k and k in sizes]
W = sum(sizes[k] for k in common)
for a in ("link", "text"):
    print(a, "text-only mix %.4f" % (sum(val[a][k] * sizes[k] for k in common) / W),
          " unweighted %.4f" % (sum(val[a][k] for k in common) / len(common)), " _mix %.4f" % val[a]["_mix"])
print("per source, link - text:", {k: round(val["link"][k] - val["text"][k], 4) for k in common})
