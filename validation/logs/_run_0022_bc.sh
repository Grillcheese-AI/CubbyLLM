cd /c/Users/grill/Documents/GitHub/CubbyLLM
until [ -f validation/logs/_k10_ckpt/P12_s0.pt ] && [ -f validation/logs/_k10_ckpt/P12_s1.pt ] && [ -f validation/logs/_k10_ckpt/P12_s2.pt ]; do sleep 30; done
sleep 20
for s in 0 1 2; do OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 .venv-dml/Scripts/python.exe validation/k10_refuse.py --resume validation/logs/_k10_ckpt/P12_s$s.pt --threads 2 --tag _t12 > validation/logs/_out_k10_refuse_P12_s$s.txt 2>&1 & done; wait
OMP_NUM_THREADS=6 CUBBY_MODELS='C:\CUBBY-TRAINED-MODELS' PYTHONIOENCODING=utf-8 .venv-dml/Scripts/python.exe validation/exp_refuse_450m.py --ckpt "I:/My Drive/cubbyllm/runs/spatial_link_5/spatial_link_5_final.pt" --probe D:/cubby_worlds/_probe_0022 --docs 400 --batch 4 --threads 6 > validation/logs/_out_refuse_450m.txt 2>&1
echo done
