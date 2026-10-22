@echo off
cd /d %CUBBY_REPO%
:wait
if not exist validation\logs\exp_he19_next_step.done (timeout /t 60 /nobreak >nul & goto wait)
set PY=%CUBBY_REPO%\<venv>\Scripts\python.exe
%PY% validation\exp_he19_next_step.py --gguf standin\models\emitter_v12e.Q4_K_M.gguf --data standin\data\out\pf_heldout_eval.jsonl --tag standin_held > validation\logs\exp_he19_next_step_standin_held.log 2>&1
echo done > validation\logs\exp_he19_next_step_standin.done
