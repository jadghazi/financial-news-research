@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.11 or later from python.org, then run Setup.cmd again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m newscollector doctor
if errorlevel 1 goto failed
echo.
echo Setup complete. Open Launch.cmd to start the dashboard.
pause
exit /b 0
:failed
echo Setup failed. Keep this output for troubleshooting; see README.md.
pause
exit /b 1
