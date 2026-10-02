@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set O=standin\data\out
move /y %O%\emitter_sft_v12e_w_slots.jsonl %O%\emitter_sft_v12e_w_slots.regbug.jsonl >nul
move /y %O%\pf_heldout_eval_w_slots.jsonl %O%\pf_heldout_eval_w_slots.regbug.jsonl >nul
python validation\emitter_data.py %O%\emitter_sft_v12e.jsonl %O%\emitter_sft_v12e_w_slots.jsonl --drop-step-values --world > validation\logs\he19_reslot.log 2>&1
python validation\emitter_data.py %O%\pf_heldout_eval.jsonl %O%\pf_heldout_eval_w_slots.jsonl --drop-step-values --world >> validation\logs\he19_reslot.log 2>&1
python validation\emitter_data.py %O%\tinygsm\tinygsm_s0_head30k.jsonl %O%\tinygsm\tinygsm_s0_head30k_slots.jsonl --drop-step-values --world >> validation\logs\he19_reslot.log 2>&1
python -m cubbyllm.reasoning.step_loop %O%\emitter_sft_v12e_w_slots.jsonl %O%\emitter_sft_v12e_w_step.jsonl >> validation\logs\he19_reslot.log 2>&1
python -m cubbyllm.reasoning.step_loop %O%\pf_heldout_eval_w_slots.jsonl %O%\pf_heldout_eval_w_step.jsonl >> validation\logs\he19_reslot.log 2>&1
python -m cubbyllm.reasoning.step_loop %O%\tinygsm\tinygsm_s0_head30k_slots.jsonl %O%\tinygsm\tinygsm_s0_head30k_step.jsonl --only-arithmetic >> validation\logs\he19_reslot.log 2>&1
copy /b %O%\emitter_sft_v12e_w_step.jsonl + %O%\tinygsm\tinygsm_s0_head30k_step.jsonl %O%\emitter_sft_v12e_w_tg30_step.jsonl >nul
echo done >> validation\logs\he19_reslot.log
