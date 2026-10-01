@echo off
setlocal
set "PILOT_BASE=%~dp0"
for %%F in (requirements.txt app.py train_experiment.py experiment\scores.csv data\selection.json experiment\models\Cam0_autoencoder.npz experiment\models\Cam1_autoencoder.npz experiment\models\Cam2_autoencoder.npz) do (
    if not exist "%PILOT_BASE%%%F" goto incomplete
)
pushd "%PILOT_BASE%"
if errorlevel 1 goto failed
set "PYTHONUTF8=1"
echo Pretrained exploratory autoencoder and Streamlit app. No original Image ZIP needed.
echo Preparing Python environment. First run requires internet.
if exist ".venv\Scripts\python.exe" goto install
py -3 -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>nul
if not errorlevel 1 goto usepy
python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>nul
if not errorlevel 1 goto usepython
echo Python 3.11-3.14 is required. Python 3.12 or 3.13 is recommended.
echo Install Python with Tcl/Tk and the Python launcher, then run this file again.
echo https://www.python.org/downloads/windows/
goto failed
:usepy
py -3 -m venv ".venv"
if errorlevel 1 goto failed
goto install
:usepython
python -m venv ".venv"
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r "%PILOT_BASE%requirements.txt"
if errorlevel 1 goto failed
echo Open http://localhost:8501 if a browser does not open.
echo Keep this window open while using the app. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m streamlit run "%PILOT_BASE%app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
set "RUNRESULT=%ERRORLEVEL%"
echo.
if "%RUNRESULT%"=="0" (echo App stopped.) else (echo App stopped with an error. Capture this window and send the error text.)
pause
popd
exit /b %RUNRESULT%
:incomplete
echo Required package files are missing.
echo Do not run this file from inside a ZIP preview.
echo Right-click the downloaded ZIP and choose Extract All.
echo Open the extracted folder, then run START_STREAMLIT_WINDOWS.cmd.
echo Expected folder: "%PILOT_BASE%"
pause
exit /b 1
:failed
echo Setup failed. Copy or capture this window's error text and send it to the conversation.
pause
exit /b 1
