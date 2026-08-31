# probe_v3_gpu.ps1 — run the v3-or-not probe on the GPU, in YOUR terminal.
#
#   cd C:\Users\grill\Documents\GitHub\CubbyLLM
#   .\standin\scripts\probe_v3_gpu.ps1              # probe with the installed llama-cpp-python
#   .\standin\scripts\probe_v3_gpu.ps1 -Upgrade     # upgrade llama-cpp-python first (Vulkan wheel), then probe
#   .\standin\scripts\probe_v3_gpu.ps1 -Cpu         # same probe, CPU-only (safe fallback)
#
# The probe loads the Q4 GGUF (copied from Drive if missing) and runs two short
# generations; see standin/scripts/probe_v3.py for what they discriminate.
param([switch]$Upgrade, [switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\grill\Documents\GitHub\CubbyLLM"

$dst = "standin\models\emitter_maybe_v3.Q4_K_M.gguf"
$src = "D:\My Drive\cubbyllm\standin\emitter_lfm25_2p6b\gguf_gguf\LFM2.5-2.6B.Q4_K_M.gguf"
if (-not (Test-Path $dst)) {
    Write-Host "copying Q4 GGUF from Drive (1.7 GB) ..."
    Copy-Item $src $dst
}

if ($Upgrade) {
    Write-Host "upgrading llama-cpp-python (Vulkan wheel index first, plain PyPI as fallback) ..."
    python -m pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Vulkan wheel index failed; trying plain PyPI (may build from source or lose the Vulkan backend)"
        python -m pip install -U llama-cpp-python
    }
}

Write-Host "--- backend check:"
python -c "import llama_cpp, os, glob; d=os.path.join(os.path.dirname(llama_cpp.__file__),'lib'); print('llama-cpp-python', llama_cpp.__version__); print('vulkan dll:', bool(glob.glob(os.path.join(d,'*vulkan*'))))"

$ngl = if ($Cpu) { 0 } else { -1 }
Write-Host "--- probe (n_gpu_layers=$ngl). This window owns the process; Ctrl+C kills it."
python standin\scripts\probe_v3.py --gguf $dst --n-gpu-layers $ngl
