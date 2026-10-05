@echo off
rem H-M1 Attract-Repel: specialised teachers -> tables (same vocabulary and projection) -> the same screens
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=
set SH=G:\My Drive\cubbyllm\token_cache_base\wiki_full.u32
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set D=C:\CUBBY-TRAINED-MODELS
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
%PY% validation\hm1_attract_repel.py --teacher %D%\teacher_paraminilm.npz --out %D%\teacher_paraminilm_ar4.npz --epochs 2 --lr 1e-2 --reg 0.001 --keep 2 --rep-w 4 --tag paraminilm_ar4 > validation\logs\hm1_attract_repel_paraminilm_ar4.out 2>&1
for %%T in (potion_ar paraminilm_ar paraminilm_ar4) do (
  cd /d C:\Users\grill\Documents\GitHub\mowm
  %PY% scripts\build_fastword_table.py --shard "%SH%" --teacher npz:%D%\teacher_%%T.npz --top-words 60000 --out %D%\fastword_table_%%T.npz > %D%\fastword_table_%%T.build.out 2>&1
  cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
  %PY% validation\exp_hm1_ngram_hv.py --table %D%\fastword_table_%%T.npz --shard "%SH%" --tokenizer "%TOK%" --tag %%T > validation\logs\exp_hm1_ngram_hv_%%T.out 2>&1
  %PY% validation\exp_hm1b_soft_address.py --table %D%\fastword_table_%%T.npz --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tag %%T_t3m8 > validation\logs\exp_hm1b_soft_address_%%T_t3m8.out 2>&1
  %PY% validation\exp_hm1b_soft_address.py --table %D%\fastword_table_%%T.npz --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tables 26 --tag %%T_t3m8_tab26 > validation\logs\exp_hm1b_soft_address_%%T_t3m8_tab26.out 2>&1
)
echo done > validation\logs\exp_hm1_ar_tables.done
