@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
%PY% validation\exp_he19_rank.py --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter I:\CUBBY-TRAINED-MODELS\emitter\cubby450m_v12e_w_step_cont --tokenizer "%TOK%" --n %1 --tag %2 > validation\logs\exp_he19_rank_%2.log 2>&1
echo done > validation\logs\exp_he19_rank_%2.done
