cd /c/Users/grill/Documents/GitHub/CubbyLLM
while pgrep -f exp_refuse_450m >/dev/null 2>&1 || tasklist //FI "IMAGENAME eq python.exe" //V 2>/dev/null | grep -q exp_refuse; do sleep 30; done
until grep -q '"wall_s"' validation/logs/_out_refuse_450m.txt 2>/dev/null || grep -q Traceback validation/logs/_out_refuse_450m.txt 2>/dev/null; do sleep 30; done
for s in 0 1 2; do OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 .venv-dml/Scripts/python.exe validation/k10_refuse.py --resume validation/logs/_k10_ckpt/P12_s$s.pt --threads 2 --tag _t12 > validation/logs/_out_k10_refuse_P12_s$s.txt 2>&1 & done; wait
echo done
