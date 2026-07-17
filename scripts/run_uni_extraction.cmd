@echo off
setlocal
cd /d "%~dp0.."
venv\Scripts\python.exe -u scripts\extract_uni_embeddings.py --threads 6 --batch-size 4 1>data\mhist\uni_extraction.log 2>data\mhist\uni_extraction_error.log
