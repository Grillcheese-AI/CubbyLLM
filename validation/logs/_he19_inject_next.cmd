@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
:wait
if not exist validation\logs\exp_he19_next_step_standin.done (timeout /t 60 /nobreak >nul & goto wait)
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
%PY% validation\exp_he19_next_step.py --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "D:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_tg_cont" --tokenizer "D:\My Drive\cubbyllm\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag tg_held_inject --inject-state > validation\logs\exp_he19_next_step_tg_held_inject.log 2>&1
%PY% validation\exp_he19_next_step.py --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "D:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_tg_cont" --tokenizer "D:\My Drive\cubbyllm\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag tg_held_fixed > validation\logs\exp_he19_next_step_tg_held_fixed.log 2>&1
echo done > validation\logs\exp_he19_next_step_inject.done
