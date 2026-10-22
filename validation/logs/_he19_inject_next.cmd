@echo off
cd /d %CUBBY_REPO%
:wait
if not exist validation\logs\exp_he19_next_step_standin.done (timeout /t 60 /nobreak >nul & goto wait)
set PY=%CUBBY_GITHUB%\grilly2\.venv\Scripts\python.exe
%PY% validation\exp_he19_next_step.py --export %CUBBY_MODELS%\base450m_cont\grilly --adapter "%CUBBY_DRIVE%\cubbyllm\emitter\cubby450m_v12e_w_tg_cont" --tokenizer "%CUBBY_DRIVE%\cubbyllm\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag tg_held_inject --inject-state > validation\logs\exp_he19_next_step_tg_held_inject.log 2>&1
%PY% validation\exp_he19_next_step.py --export %CUBBY_MODELS%\base450m_cont\grilly --adapter "%CUBBY_DRIVE%\cubbyllm\emitter\cubby450m_v12e_w_tg_cont" --tokenizer "%CUBBY_DRIVE%\cubbyllm\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag tg_held_fixed > validation\logs\exp_he19_next_step_tg_held_fixed.log 2>&1
echo done > validation\logs\exp_he19_next_step_inject.done
