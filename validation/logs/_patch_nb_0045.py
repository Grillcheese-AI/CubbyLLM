"""Scratch (untracked): the notebook's readout cell calls grow_mem_readout (order 0045)."""
p = "notebooks/_build_grow_mem.py"
lines = open(p, encoding="utf-8").read().split("\n")
assert lines[346].startswith("checks = {}"), lines[346]
assert lines[361].startswith("print('") and "check 2 (the memory)" in lines[361], lines[361]
new = ["# the gates, from validation/grow_mem_readout.py (order 0045): one place for this cell and the pod runner",
       "import sys; sys.path.insert(0, f'{REPO}/validation')",
       "import grow_mem_readout as gr",
       "R = gr.gates(rows_of(mpath('grown')), rows_of(mpath('control')), cont_rows, SEQ)",
       "print()",
       "for k, ok in R['checks'].items():",
       "    print(f\"  {'PASS' if ok else 'fail' if ok is False else 'not yet'}  {k}\")",
       "for k, ok in R['kills'].items():",
       "    print(f\"  {'KILLED' if ok else 'not killed'}  {k}\")"]
lines[346:361] = new
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("patched")
