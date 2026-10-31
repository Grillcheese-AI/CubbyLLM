"""Cowork's review probes for 0027 P1 and 0028 (scratch, not committed)."""
import sys, tempfile, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standin"))
import plugins_guard as g

# 1. the code review: bypasses a reviewer would expect the scan to catch
cases = {
    "eval": "eval(\"__import__('socket')\")\n",
    "exec": "exec('import socket')\n",
    "getattr_os": "import os\ngetattr(os, 'sys' + 'tem')('echo hi')\n",
    "builtins_import": "import builtins\nbuiltins.__import__('socket')\n",
    "posix_spawn": "import os\nos.posix_spawn('/bin/sh', ['sh'], {})\n",
    "spawnlp": "import os\nos.spawnlp(os.P_WAIT, 'sh', 'sh')\n",
    "execlpe": "import os\nos.execlpe('sh', 'sh', {})\n",
    "sys_modules": "import sys\nsys.modules['socket']\n",
    "write_outside": "open('C:/Windows/x.txt', 'w').write('x')\n",
    "shutil_rmtree": "import shutil\nshutil.rmtree('C:/tmp/x')\n",
    "from_os_system": "from os import system\nsystem('x')\n",
    "importlib_mod": "import importlib\nimportlib.import_module('socket')\n",
}
class P:  # the smallest plugin check() reads
    def __init__(self, folder): self.folder = folder
out = {}
for k, src in cases.items():
    d = pathlib.Path(tempfile.mkdtemp()); (d / "m.py").write_text(src, encoding="utf-8")
    out[k] = "caught" if g.check(P(d)) else "MISSED"
print(json.dumps(out, indent=1))

# 2. the public scrub: ordinary words that are also world names
import board_api as ba
names = {"facts", "wiki", "self", "plugin:game"} | {n for n in __import__("audience").SOURCE_NAMES}
samples = ["3.4M facts held", "Cubby talks about itself", "refused when the fact is not held",
           "how the 450M was trained on wikis", "the self world", "Weather warnings in effect"]
print({s: ba.scrub(s, {n.lower() for n in names if len(n) > 3}) for s in samples})
