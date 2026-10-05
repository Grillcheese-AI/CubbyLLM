@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set AD=H:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_step_rank_lr5e5
%PY% validation\exp_he19_step_loop.py --export C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "%AD%" --tokenizer "%TOK%" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag rankft5e5_held > validation\logs\exp_he19_step_loop_rankft5e5_held.log 2>&1
%PY% validation\exp_he19_rank.py --export C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "%AD%" --tokenizer "%TOK%" --n 400 --tag rankft5e5_held400 > validation\logs\exp_he19_rank_rankft5e5_held400.log 2>&1
echo done > validation\logs\exp_he19_rankft5e5_read.done
