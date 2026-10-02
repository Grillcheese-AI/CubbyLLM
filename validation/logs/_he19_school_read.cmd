@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set AD=H:\My Drive\cubbyllm\emitter\school_step
%PY% validation\exp_he19_step_loop.py --export I:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "%AD%" --tokenizer "H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json" --data standin\data\out\pf_heldout_eval_w_slots.jsonl --tag school_step_held > validation\logs\exp_he19_step_loop_school_step_held.log 2>&1
echo done > validation\logs\exp_he19_school_read.done
