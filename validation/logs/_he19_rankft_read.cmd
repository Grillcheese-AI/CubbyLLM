@echo off
cd /d %CUBBY_REPO%
set PY=%CUBBY_GITHUB%\grilly2\.venv\Scripts\python.exe
set TOK=%CUBBY_DRIVE%\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set AD=%CUBBY_DRIVE%\cubbyllm\emitter\cubby450m_v12e_w_step_rank
%PY% validation\exp_he19_step_loop.py --export %CUBBY_MODELS%\base450m_cont\grilly --adapter "%AD%" --tokenizer "%TOK%" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag rankft_held > validation\logs\exp_he19_step_loop_rankft_held.log 2>&1
%PY% validation\exp_he19_rank.py --export %CUBBY_MODELS%\base450m_cont\grilly --adapter "%AD%" --tokenizer "%TOK%" --n 400 --tag rankft_held400 > validation\logs\exp_he19_rank_rankft_held400.log 2>&1
echo done > validation\logs\exp_he19_rankft_read.done
