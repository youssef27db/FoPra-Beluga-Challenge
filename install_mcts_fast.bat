@echo off
REM Install script for mcts_fast on Windows

echo Installing mcts_fast module...

REM Check Python version
python --version
echo.

REM Check for Visual Studio C++ Build Tools
echo Checking for Visual Studio C++ Build Tools...

where cl >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Visual Studio C++ Build Tools not found in PATH.
    echo Please make sure you run this from a "Developer Command Prompt for VS"
    echo or install Visual Studio with C++ Build Tools from:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo Do you want to continue anyway? (Y/N)
    set /p ANSWER=
    if /i "%ANSWER%" NEQ "Y" goto :END
)

REM Install dependencies first
echo Installing dependencies...
pip install --upgrade pip setuptools wheel numpy pybind11

REM Install the module
echo Building and installing mcts_fast...
pip install --no-build-isolation .\rl\mcts\mcts_fast

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation successful!
    echo You can now import mcts_fast in your Python code.
) ELSE (
    echo.
    echo Installation failed. Please check the error messages above.
    exit /b 1
)

:END
