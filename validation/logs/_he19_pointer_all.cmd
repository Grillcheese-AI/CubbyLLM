@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
call validation\logs\_he19_pointer_extract.cmd standin\data\out\pf_heldout_eval_w_step.jsonl val 0 standin\data\out\he19_ptr_held.npz held
call validation\logs\_he19_pointer_extract.cmd standin\data\out\emitter_sft_v12e_w_tg30_step.jsonl train 24000 standin\data\out\he19_ptr_train.npz train
C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe validation\exp_he19_pointer.py train --train standin\data\out\he19_ptr_train.npz --held standin\data\out\he19_ptr_held.npz --tag v1 > validation\logs\exp_he19_pointer_train_v1.out 2>&1
echo done > validation\logs\exp_he19_pointer_all.done
