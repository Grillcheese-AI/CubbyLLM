@echo off
rem H-M1 teachers, round 2: dump the potion and paraphrase-MiniLM teacher vectors (for Attract-Repel), then the
rem Qwen3-Embedding-0.6B table and its screens. CPU only.
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=
set SH=G:\My Drive\cubbyllm\token_cache_base\wiki_full.u32
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set D=C:\CUBBY-TRAINED-MODELS
cd /d C:\Users\grill\Documents\GitHub\mowm
%PY% scripts\build_fastword_table.py --shard "%SH%" --teacher m2v:minishlab/potion-retrieval-32M --top-words 60000 --dump-teacher %D%\teacher_potion.npz --out %D%\scratch_table.npz > %D%\teacher_potion.out 2>&1
%PY% scripts\build_fastword_table.py --shard "%SH%" --teacher st:sentence-transformers/paraphrase-MiniLM-L6-v2 --top-words 60000 --dump-teacher %D%\teacher_paraminilm.npz --out %D%\scratch_table.npz > %D%\teacher_paraminilm.out 2>&1
echo dumps > C:\Users\grill\Documents\GitHub\CubbyLLM\validation\logs\exp_hm1_dumps.done
%PY% scripts\build_fastword_table.py --shard "%SH%" --teacher st:Qwen/Qwen3-Embedding-0.6B --batch 128 --top-words 60000 --dump-teacher %D%\teacher_qwen3e.npz --out %D%\fastword_table_v6q.npz > %D%\fastword_table_v6q.build.out 2>&1
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
%PY% validation\exp_hm1_ngram_hv.py --table %D%\fastword_table_v6q.npz --shard "%SH%" --tokenizer "%TOK%" --tag v6q > validation\logs\exp_hm1_ngram_hv_v6q.out 2>&1
%PY% validation\exp_hm1b_soft_address.py --table %D%\fastword_table_v6q.npz --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tag v6q_t3m8 > validation\logs\exp_hm1b_soft_address_v6q_t3m8.out 2>&1
%PY% validation\exp_hm1b_soft_address.py --table %D%\fastword_table_v6q.npz --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tables 26 --tag v6q_t3m8_tab26 > validation\logs\exp_hm1b_soft_address_v6q_t3m8_tab26.out 2>&1
echo done > validation\logs\exp_hm1_teachers2.done
