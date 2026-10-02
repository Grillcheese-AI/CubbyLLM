@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set AD=I:\CUBBY-TRAINED-MODELS\emitter\cubby450m_v12e_w_step_cont
:wait
if not exist "%AD%\emitter_lora.safetensors" (timeout /t 60 /nobreak >nul & goto wait)
if not exist "%AD%\emitter_lora.json" (timeout /t 60 /nobreak >nul & goto wait)

set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set EXP=I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly
set TK=D:\My Drive\cubbyllm\grillcheese_bbpe128k.json
%PY% validation\exp_he19_step_loop.py --export %EXP% --adapter "%AD%" --tokenizer "%TK%" --teacher-forced standin\data\out\pf_heldout_eval_w_step.jsonl --tag step_tf_held > validation\logs\exp_he19_step_loop_step_tf_held.log 2>&1
%PY% validation\exp_he19_step_loop.py --export %EXP% --adapter "%AD%" --tokenizer "%TK%" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag step_held > validation\logs\exp_he19_step_loop_step_held.log 2>&1
echo done > validation\logs\exp_he19_step_read.done
