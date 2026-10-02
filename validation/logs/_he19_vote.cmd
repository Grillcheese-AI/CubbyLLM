@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set AD=I:\CUBBY-TRAINED-MODELS\emitter\cubby450m_v12e_w_step_cont
%PY% validation\exp_he19_step_loop.py --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "%AD%" --tokenizer "D:\My Drive\cubbyllm\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag vote5_held --vote 5 --temperature 0.7 > validation\logs\exp_he19_step_loop_vote5_held.log 2>&1
echo done > validation\logs\exp_he19_vote.done
