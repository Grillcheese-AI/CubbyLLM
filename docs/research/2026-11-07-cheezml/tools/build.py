"""Build CheezML example records: inline the context, then derive negative records by one mutation each."""
import copy
import json
import os
import sys

SYNTH = sys.argv[1]
SRC = sys.argv[2]
EX = os.path.join(SYNTH, "examples")
NEG = os.path.join(EX, "negative")
os.makedirs(NEG, exist_ok=True)

ctx = json.load(open(os.path.join(SYNTH, "context.jsonld")))["@context"]

POS = {
    "01-certified-answer": "01-certified-answer.json",
    "02-refusal": "02-refusal.json",
    "03-routing-spawn": "03-routing-spawn.json",
    "04-world-split": "04-world-split.json",
    "05-helper-rejected": "05-helper-rejected.json",
    "06-invoice-review": "06-invoice-review.json",
}

U = "urn:uuid:00000000-0000-0000-0000-000000000"


def load(name):
    d = json.load(open(os.path.join(SRC, POS[name])))
    assert d["@context"] == "@@CONTEXT@@"
    d["@context"] = copy.deepcopy(ctx)
    return d


def node(d, suffix):
    for n in d["@graph"]:
        if n["@id"] == U + suffix:
            return n
    raise KeyError(suffix)


def write(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")


positives = {}
for name in POS:
    d = load(name)
    positives[name] = d
    write(os.path.join(EX, name + ".jsonld"), d)

# ---- negatives: each is one positive with one deliberate fault ----
negs = {}

# 1. certificate produced by agreement between two models instead of a VM run
d = copy.deepcopy(positives["01-certified-answer"])
node(d, "104")["prov:wasGeneratedBy"] = U + "199"
d["@graph"] += [
    {"@id": U + "199", "@type": "prov:Activity",
     "prov:wasAssociatedWith": [U + "197", U + "198"],
     "prov:endedAtTime": "2026-11-07T14:00:02Z"},
    {"@id": U + "197", "@type": "prov:SoftwareAgent"},
    {"@id": U + "198", "@type": "prov:SoftwareAgent"},
]
negs["n01-certificate-by-agreement"] = d

# 1b. certificate carrying a truth score
d = copy.deepcopy(positives["01-certified-answer"])
node(d, "104")["cheez:truthScore"] = "0.97"
negs["n01b-certificate-truth-score"] = d

# 2. refusal without a reason
d = copy.deepcopy(positives["02-refusal"])
del node(d, "203")["cubby:refusalReason"]
negs["n02-refusal-without-reason"] = d

# 2b. refusal for a conflict that does not name the conflict kind
d = copy.deepcopy(positives["02-refusal"])
node(d, "203")["cubby:refusalReason"] = "cubby:Conflict"
negs["n02b-conflict-without-kind"] = d

# 3. routing decision whose margin threshold lacks its calibration
d = copy.deepcopy(positives["03-routing-spawn"])
del node(d, "321")["prov:wasGeneratedBy"]
negs["n03-threshold-without-calibration"] = d

# 3b. routing decision that routes although the margin did not clear tau_margin
d = copy.deepcopy(positives["03-routing-spawn"])
r = node(d, "301")
r["cheez:outcome"] = "mowm:Route"
r["mowm:routedTo"] = U + "310"
d["@graph"] = [n for n in d["@graph"] if n["@id"] not in (U + "340", U + "341")]
negs["n03b-route-against-margin"] = d

# 4. split beyond z_max: the parent already sits at z_max
d = copy.deepcopy(positives["04-world-split"])
node(d, "402")["mowm:zDepth"] = 3
node(d, "403")["mowm:zDepth"] = 4
negs["n04-split-beyond-zmax"] = d

# 5. rejected helper without the case that broke it
d = copy.deepcopy(positives["05-helper-rejected"])
del node(d, "501")["cubby:breakingCase"]
negs["n05-rejection-without-breaking-case"] = d

# 6. committed field with no source
d = copy.deepcopy(positives["06-invoice-review"])
del node(d, "620")["prov:hadPrimarySource"]
negs["n06-commit-without-source"] = d

# 6b. committed field whose number does not appear in the region's text
d = copy.deepcopy(positives["06-invoice-review"])
node(d, "604")["prov:value"] = "1350.00"
node(d, "620")["prov:value"] = "1350.00"
negs["n06b-number-not-in-region"] = d

for name, d in negs.items():
    write(os.path.join(NEG, name + ".jsonld"), d)

print("positives:", len(positives), "negatives:", len(negs))
