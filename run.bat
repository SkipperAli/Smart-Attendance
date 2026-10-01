@echo off
setlocal
cd /d "%~dp0"
title Smart Attendance

rem ---------------------------------------------------------------
rem  Smart Attendance launcher (Windows)
rem  First run: creates .venv and installs dependencies.
rem  Usage:  run.bat                 -> menu
rem          run.bat run --manual    -> passes arguments straight through
rem ---------------------------------------------------------------

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 - 3.12 from https://python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo [WARNING] Python 3.10 - 3.12 is recommended. TensorFlow may fail to install on other versions.
    echo.
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :venv_failed
)
set "PY=.venv\Scripts\python.exe"

if not exist ".venv\installed.flag" (
    echo Installing dependencies - first run only, this can take a few minutes...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :install_failed
    echo ok> ".venv\installed.flag"
)

rem Health check: OpenCV 5 (or a broken install) has no CascadeClassifier,
rem which DeepFace needs. Reinstall OpenCV 4.x if so.
"%PY%" -c "import cv2; cv2.CascadeClassifier; cv2.data.haarcascades" >nul 2>&1
if errorlevel 1 (
    echo Installing a compatible OpenCV version - 4.x...
    "%PY%" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >nul 2>&1
    "%PY%" -m pip install --force-reinstall --no-deps --quiet "opencv-python>=4.8,<5"
    "%PY%" -c "import cv2; cv2.CascadeClassifier; cv2.data.haarcascades" >nul 2>&1
    if errorlevel 1 goto :opencv_failed
    echo OpenCV fixed.
)

rem Arguments given? Run that command directly instead of showing the menu.
if not "%~1"=="" (
    "%PY%" attendance.py %*
    exit /b
)

:menu
cls
echo.
echo   ==========================================
echo              SMART ATTENDANCE
echo   ==========================================
echo.
echo     1  Start live attendance
echo     2  Start in manual mode  (SPACE to scan)
echo     3  Enroll a student
echo     4  Today's report
echo     5  List students
echo     6  Quit
echo.
set "choice="
set /p "choice=  Choose 1-6: "

if "%choice%"=="1" goto :run
if "%choice%"=="2" goto :run_manual
if "%choice%"=="3" goto :enroll
if "%choice%"=="4" goto :report
if "%choice%"=="5" goto :students
if "%choice%"=="6" exit /b 0
goto :menu

:run
"%PY%" attendance.py run
pause
goto :menu

:run_manual
"%PY%" attendance.py run --manual
pause
goto :menu

:enroll
echo.
set "name="
set /p "name=  Student name: "
if "%name%"=="" goto :menu
"%PY%" attendance.py enroll "%name%"
pause
goto :menu

:report
"%PY%" attendance.py report
pause
goto :menu

:students
echo.
"%PY%" attendance.py students
pause
goto :menu

:venv_failed
echo [ERROR] Could not create the virtual environment.
pause
exit /b 1

:opencv_failed
echo [ERROR] OpenCV is still broken. Delete the .venv folder and run this file again.
pause
exit /b 1

:install_failed
echo [ERROR] Installing dependencies failed. Check the messages above.
echo Fix the problem, then run this file again.
pause
exit /b 1
