#!/usr/bin/env bash
# setup.sh -- a fresh pod to a running H-E20 (order 0045). Checks the card and the stack, installs what the image
# lacks, clones the repo at the manifest's commit, starts run_pod.py in tmux. Needs: HF_TOKEN, CB_POD_REPO,
# HOURS_PER_ARM (7 on demand, 12-14 spot), CB_REPO_URL (the repo's git URL); optional CB_POD_STOP (the provider's stop command), POD_WORK (default
# /workspace/pod). Usage: bash setup.sh
set -euo pipefail
WORK="${POD_WORK:-/workspace/pod}"
REPO_URL="${CB_REPO_URL:?set CB_REPO_URL (the git URL of this repo; no account name lives in a committed file)}"
: "${HF_TOKEN:?set HF_TOKEN (the token of the private repo; it goes nowhere else)}"
: "${CB_POD_REPO:?set CB_POD_REPO (the private dataset repo holding the inputs)}"
: "${HOURS_PER_ARM:?set HOURS_PER_ARM (7 on demand, 12-14 spot)}"
mkdir -p "$WORK"

echo "== card"
if ! command -v nvidia-smi >/dev/null; then echo "no nvidia-smi: not a GPU pod"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
case "$CARD" in *H100*|*H200*|*B200*) ;; *) echo "not an H100 / H200 / B200: $CARD"; exit 1;; esac
if [ "$MEM" -lt 80000 ]; then echo "less than 80 GB on the card ($MEM MiB)"; exit 1; fi

echo "== stack"
python - <<'EOF'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
name = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
maj, mi = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
cu = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
if "B200" in name and ((maj, mi) < (2, 7) or cu < (12, 8)):
    print(f"MISSING: a B200 needs torch >= 2.7 with CUDA >= 12.8 (this image: torch {torch.__version__}, CUDA {torch.version.cuda}).")
    print("Install the cu128 build yourself -- pip install --index-url https://download.pytorch.org/whl/cu128 torch -- then rerun.")
    sys.exit(2)
EOF
for pkg in tokenizers numpy huggingface_hub pytest; do
  python -c "import $pkg" 2>/dev/null || pip install -q "$pkg"
done
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)

echo "== code"
cd "$WORK"
if [ ! -d CubbyLLM/.git ]; then git clone -q "$REPO_URL" CubbyLLM; fi
cd CubbyLLM
COMMIT=$(python - <<'EOF'
import json, os
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["CB_POD_REPO"], "inputs/pod_inputs.json", repo_type="dataset", local_dir="/tmp/pod_manifest")
print(json.load(open(p))["commit"])
EOF
)
git fetch -q origin && git checkout -q "$COMMIT"
echo "repo at $COMMIT"
pip install -q -e . 2>/dev/null || true

echo "== run"
# a dedicated tmux server (-L pod), started fresh, inherits this shell's exported variables: the token reaches
# run_pod.py through the environment, never through a command line (ps shows arguments, not the environment)
export HF_TOKEN CB_POD_REPO HOURS_PER_ARM CB_POD_STOP="${CB_POD_STOP:-}"
mkdir -p "$WORK/logs"
tmux -L pod kill-server 2>/dev/null || true
tmux -L pod new-session -d -s pod "cd $WORK/CubbyLLM && python -u validation/pod/run_pod.py --work $WORK 2>&1 | tee -a $WORK/logs/tmux.log"
echo "started. Attach: tmux -L pod attach -t pod   (detach: Ctrl-b d)"
echo "watch:   tail -f $WORK/logs/run_pod.log   and the arm's logs/train_<arm>.log; metrics in $WORK/runs/<arm>/grow_mem_<arm>_metrics.jsonl"
echo "remote:  https://huggingface.co/datasets/$CB_POD_REPO  (pod_state.json, runs/, logs/)"
