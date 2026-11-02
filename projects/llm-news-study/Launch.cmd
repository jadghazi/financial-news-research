@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup.cmd first to install the collector in this project.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m newscollector dashboard
if errorlevel 1 pause
