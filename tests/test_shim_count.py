"""Order 0033 (Cowork's answer 7 in 0031's review): the shims order 0031 left in ``standin/`` go when no in-repo import
names them and the Colab notebooks are updated. ``tools/shim_count.py`` counts; this reports the count and fails
only if it grows (``CEILING``: lower it as the count falls; at 0 the shims can go, with 0032's report).
Run: python -m pytest tests/test_shim_count.py -s       (-s prints the count by name)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("shim_count", ROOT / "tools" / "shim_count.py")
shim_count = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim_count)

CEILING = {"code": 338, "notebooks": 1}                  # 2026-10-22, order 0033: the count the day it was first taken


def test_in_repo_imports_of_shimmed_names_only_go_down():
    c = shim_count.count(ROOT)
    print(f"\n{c['total']} in-repo imports of shimmed names in code, {c['notebooks']} in notebooks ({c['shims']} shims); "
          "by name: " + ", ".join(f"{k} {n}" for k, n in c["by_name"].items()))
    assert c["shims"] > 0 or c["total"] == 0
    assert c["total"] <= CEILING["code"], f"new imports of shimmed names: import the new homes ({c['by_file']})"
    assert c["notebooks"] <= CEILING["notebooks"]


def test_the_counter_finds_a_planted_import(tmp_path):
    (tmp_path / "standin").mkdir()
    (tmp_path / "standin" / "old.py").write_text('"""standin/old.py moved to cubbyllm/x/old.py (order 0031)."""\n'
                                                '_NEW = "cubbyllm.x.old"\n', encoding="utf-8")
    (tmp_path / "use.py").write_text("import json\nfrom old import f\n\ndef g():\n    import standin.old\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("from cubbyllm.x.old import f\nfrom .old import f\n", encoding="utf-8")
    c = shim_count.count(tmp_path)
    assert c["total"] == 2 and {h["name"] for h in c["hits"]} == {"old", "standin.old"} and c["by_file"] == {"use.py": 2}
