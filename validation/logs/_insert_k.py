import re, sys, io
path, section_path = sys.argv[1], sys.argv[2]
text = io.open(path, encoding="utf-8").read()
section = io.open(section_path, encoding="utf-8").read().rstrip("\n") + "\n\n"
assert "## 9d. Group K" not in text, "already inserted"
i = text.index("\n## 10. Prioritized validation order")
text = text[:i + 1] + section + text[i + 1:]
# validation order: a new numbered item before "## 11."
j = text.index("\n## 11. What this document does not claim")
sec10 = text[text.index("\n## 10. Prioritized validation order"):j]
nums = [int(m) for m in re.findall(r"^(\d+)\. ", sec10, flags=re.M)]
n = max(nums) + 1
item = (f"{n}. ⏳ **Group K (added 2026-10-20) — the research pause.** H-K8's law and chunk store measured the same day "
        f"(`validation/exp_k8_holo.py`: gate 1 failed as written — the law marks the 0.68 point, 90% holds to half of it; "
        f"gate 2 passed at the line, 0.90); H-K1 running (`validation/exp_k1_map.py`, map vs TransE on the wiki world). "
        f"Next, cheapest first: H-K7 gate 1 (the exposure curve on our trunk, a CPU afternoon), H-K4's host side "
        f"(`Scene` with dims = 1), H-K8 gate 3 (a `CubbyMemory` sibling on a synthetic recall task), then H-K2 and H-K4's "
        f"trunk sides on the grown model's second round, H-K9 on BlockWorld traces, H-K5's rule generator.\n")
text = text[:j].rstrip("\n") + "\n" + item + text[j:]
# appendix row
row = ("| K — Knowledge, the map, the state (research pause) | H-K1–H-K9 | Nick's request and questions 2026-10-20; H-S12's algebra; "
       "inspiration only: Titans (arXiv:2501.00663), Nested Learning / HOPE (research.google blog), Gated DeltaNet-2 (NVIDIA Research), "
       "Kimi Linear (arXiv:2510.26692), Attention as Binding (arXiv:2512.14709), Haga (bioRxiv 2023.05.11.540307), grid-cell path "
       "integration (`ruiqigao/grid-cell-path`, NeurIPS 2021), Language Models Need Sleep (arXiv:2605.26099), HeLa-Mem (ACL 2026), "
       "Hrrformer (ICML 2023, arXiv:2305.19534), Capacity Analysis of VSA (arXiv:2301.10352), Physics of LMs 3.3 (arXiv:2404.05405), "
       "Memory Layers at Scale (arXiv:2412.09764), ARC-AGI-3 leaderboard (arcprize.org), Raschka \"Beyond Standard LLMs\", "
       "`google-research/swirl-dynamics` (DySLIM, flow-map matching / MeanFlow, Evolve Smoothly, debiasing, WENO-NN) |\n")
k = text.index("| P — Prospection")
end = text.index("\n", k) + 1
text = text[:end] + row + text[end:]
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("inserted; lines:", text.count("\n"))
