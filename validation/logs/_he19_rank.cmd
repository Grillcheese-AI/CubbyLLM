@echo off
cd /d %CUBBY_REPO%
set PY=%CUBBY_GITHUB%\grilly2\.venv\Scripts\python.exe
set TOK=%CUBBY_DRIVE%\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
%PY% validation\exp_he19_rank.py --export %CUBBY_MODELS%\base450m_cont\grilly --adapter %CUBBY_MODELS%\emitter\cubby450m_v12e_w_step_cont --tokenizer "%TOK%" --n %1 --tag %2 > validation\logs\exp_he19_rank_%2.log 2>&1
echo done > validation\logs\exp_he19_rank_%2.done
