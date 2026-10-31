@echo off
rem %1 = data jsonl, %2 = split, %3 = n, %4 = out npz, %5 = tag
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
%PY% validation\exp_he19_pointer.py extract --data %1 --split %2 --n %3 --out %4 --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter I:\CUBBY-TRAINED-MODELS\emitter\cubby450m_v12e_w_step_cont --tokenizer "%TOK%" > validation\logs\exp_he19_pointer_extract_%5.out 2>&1
echo done > validation\logs\exp_he19_pointer_extract_%5.done
