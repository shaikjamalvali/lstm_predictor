@echo off
echo ============================================
echo LSTM Glucose Prediction System
echo ============================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo ============================================
echo Step 1: Installing Requirements
echo ============================================
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo.

echo ============================================
echo Step 2: Training LSTM Model
echo ============================================
echo This will take several minutes...
echo.
python lstm_glucose_prediction.py
if errorlevel 1 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo.

echo ============================================
echo Step 3: Running Predictions
echo ============================================
echo.
python predict_realtime.py
if errorlevel 1 (
    echo ERROR: Prediction failed
    pause
    exit /b 1
)
echo.

echo ============================================
echo COMPLETE!
echo ============================================
echo Check the generated files:
echo - lstm_glucose_model.keras
echo - scaler_X.pkl, scaler_y.pkl
echo - training_history.png
echo - predictions_comparison.png
echo - error_analysis.png
echo ============================================
pause
