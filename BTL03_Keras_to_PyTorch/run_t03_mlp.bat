@echo off
cd /d "%~dp0"
python -m unittest discover -s tests -v
if errorlevel 1 goto failed
python -m src.train_mlp --task all
if errorlevel 1 goto failed
python scripts/verify_mlp.py
if errorlevel 1 goto failed
echo T03 experiments and checkpoint verification passed.
goto end
:failed
echo T03 failed. Read the traceback above.
:end
pause
