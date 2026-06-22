@echo off
set PYTHONIOENCODING=utf-8
set "DIR=%~dp0"
python "%DIR%main.py"
start explorer "%DIR%results"
pause
