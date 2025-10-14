@echo off
REM Quick Start Script for IR Spectroscopy Streamlit Dashboard (Windows)

echo ==============================================
echo IR Spectroscopy ML Project - Dashboard Launcher
echo ==============================================
echo.

REM Check if Dataset folder exists
if not exist "Dataset\" (
    echo ERROR: Dataset folder not found!
    echo Please ensure the Dataset folder with required files exists.
    pause
    exit /b 1
)

REM Check for required data files
if not exist "Dataset\qm9s_irdata.csv" (
    echo ERROR: Dataset\qm9s_irdata.csv not found!
    pause
    exit /b 1
)

if not exist "Dataset\smiles.txt" (
    echo ERROR: Dataset\smiles.txt not found!
    pause
    exit /b 1
)

echo All data files found!
echo.
echo Starting Streamlit Dashboard...
echo.
echo The dashboard will open in your browser at:
echo    http://localhost:8501
echo.
echo Tips for presentation:
echo    - Start with 500-1000 samples for quick demo
echo    - Use sidebar to navigate between sections
echo    - Try the 'Your Own SMILES' page for interactive demo
echo.
echo Press Ctrl+C to stop the dashboard
echo ==============================================
echo.

REM Install requirements if needed
pip install -r requirements.txt

REM Run streamlit
streamlit run streamlit_app.py

pause
