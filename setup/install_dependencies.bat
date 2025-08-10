@echo off
echo Installing Beluga Challenge dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install core dependencies
echo Installing PyTorch...
python -m pip install torch>=2.0.0

echo Installing NumPy and Matplotlib...
python -m pip install numpy>=1.21.0 matplotlib>=3.5.0

echo Installing RL libraries...
python -m pip install gymnasium>=0.29.0

REM Install additional utilities
echo Installing additional utilities...
python -m pip install typing-extensions>=4.0.0 pytest>=7.0.0 jupyter>=1.0.0

REM Optional dependencies
echo Installing optional dependencies...
python -m pip install pandas>=1.5.0 seaborn>=0.11.0

echo.
echo Installation completed!
echo.
echo You can now run:
echo   python -m rl.main --help
echo.
pause
