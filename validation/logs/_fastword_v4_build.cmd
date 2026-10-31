@echo off
cd /d C:\Users\grill\Documents\GitHub\mowm
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
%PY% scripts\build_fastword_table.py --shard "G:\My Drive\cubbyllm\token_cache_base\wiki_full.u32" --teacher m2v:minishlab/potion-retrieval-32M --top-words 60000 --out C:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz > C:\CUBBY-TRAINED-MODELS\fastword_table_v4.build.out 2>&1
echo done > C:\CUBBY-TRAINED-MODELS\fastword_table_v4.done
