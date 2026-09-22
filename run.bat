@echo off
cd /d "%~dp0"
title ML CI Pipeline
set HF_HUB_DISABLE_PROGRESS_BARS=1
set TRANSFORMERS_VERBOSITY=error
set PY=.venv\Scripts\python.exe

if exist "%PY%" goto menu
echo Virtual environment not found - creating one and installing dependencies...
echo (first run only - downloads PyTorch, takes a few minutes)
python -m venv .venv
"%PY%" -m pip install -q -r requirements.txt -r requirements-dev.txt
if errorlevel 1 goto install_failed

:menu
rem "run.bat 2" runs one option and exits instead of showing the menu.
if not "%~1"=="" (
    set choice=%~1
    set ONESHOT=1
    goto dispatch
)
cls
echo ============================================
echo   ML CI Pipeline - local test runner
echo ============================================
echo.
echo   1. Run the full pipeline (what CI does on a PR)
echo        tests -^> train -^> evaluate -^> gate    ~5 min
echo   2. Simulate a regression (re-ranking off)
echo        the gate should FAIL                   ~2 min
echo   3. Run unit tests only                       ~30 sec
echo   4. Show the current baseline
echo   Q. Quit
echo.
set choice=
set /p choice=Choose an option: 
:dispatch
if /i "%choice%"=="1" goto full
if /i "%choice%"=="2" goto regression
if /i "%choice%"=="3" goto tests
if /i "%choice%"=="4" goto baseline
if /i "%choice%"=="q" exit /b 0
if defined ONESHOT echo Unknown option "%choice%" - use 1, 2, 3 or 4.& exit /b 1
goto menu

:full
echo.
echo === [1/4] Unit tests ===
"%PY%" -m pytest -q
if errorlevel 1 goto step_failed
echo.
echo === [2/4] Train (build index artifact) ===
"%PY%" -m model.train
if errorlevel 1 goto step_failed
echo.
echo === [3/4] Evaluate on 500 held-out queries ===
"%PY%" -m model.evaluate --out metrics\latest.json
if errorlevel 1 goto step_failed
echo.
echo === [4/4] Regression gate ===
"%PY%" -m model.gate check --latest metrics\latest.json
if errorlevel 1 goto gate_failed
echo RESULT: PASSED - in CI this model would be allowed to deploy.
goto done

:regression
if exist "artifacts\manifest.json" goto regression_eval
echo.
echo === Building the index artifact first (not built yet) ===
"%PY%" -m model.train
if errorlevel 1 goto step_failed
:regression_eval
echo.
echo === Evaluating a deliberately worse model (no cross-encoder re-ranking) ===
"%PY%" -m model.evaluate --no-rerank --out metrics\demo_regression.json
if errorlevel 1 goto step_failed
echo.
echo === Regression gate ===
"%PY%" -m model.gate check --latest metrics\demo_regression.json
if errorlevel 1 goto gate_failed
echo RESULT: PASSED - unexpected for this demo, the gate should have caught it.
goto done

:tests
echo.
"%PY%" -m pytest -v
if errorlevel 1 goto step_failed
goto done

:baseline
echo.
type metrics\baseline.json
goto done

:gate_failed
echo RESULT: BLOCKED - a gated metric had a significant drop of more than 0.01.
echo In CI this PR would get a red check and could not deploy.
goto done

:step_failed
echo.
echo A step failed - see the error above.
goto done

:install_failed
echo.
echo Installing dependencies failed - see the error above.
echo Delete the .venv folder and run this again to retry.
pause
exit /b 1

:done
echo.
if defined ONESHOT exit /b 0
pause
goto menu
