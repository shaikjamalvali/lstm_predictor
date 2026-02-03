git push origin main
# Quick Command Reference

## 🚀 Quick Start (Copy & Paste)

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib

# 2. Train model
python lstm_glucose_prediction.py

# 3. Get predictions
python demo_predictions.py
```

---

## 📋 All Commands by Category

### Installation Commands

```bash
# Check Python version
python --version

# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install packages (all at once)
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib

# OR install from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

### Training Commands

```bash
# Train the LSTM model
python lstm_glucose_prediction.py

# Expected runtime: 2-5 minutes
# Creates 7 files: models, scalers, and plots
```

---

### Prediction Commands

```bash
# Option 1: Basic predictions
python predict_realtime.py

# Option 2: Comprehensive demo (RECOMMENDED)
python demo_predictions.py

# Option 3: Test environment first
python test_environment.py
```

---

### Using Virtual Environment

```bash
# Activate virtual environment (if created)
.venv\Scripts\activate  # Windows Command Prompt
.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate  # Linux/Mac

# Run any command with virtual environment activated
python lstm_glucose_prediction.py
python demo_predictions.py

# Deactivate when done
deactivate
```

---

### Python Interactive Commands

```bash
# Start Python
python

# Then run in Python:
```

```python
# Import predictor
from predict_realtime import GlucosePredictor
predictor = GlucosePredictor()

# Load data
import pandas as pd
df = pd.read_csv('HUPA0001P.csv', sep=';')
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['minute'] = df['time'].dt.minute

# Get features
features = ['calories', 'heart_rate', 'steps', 'basal_rate', 
            'bolus_volume_delivered', 'carb_input', 'hour', 
            'day_of_week', 'minute']
sequence = df[features].iloc[-12:].values

# Make prediction
prediction = predictor.predict_glucose(sequence)
print(f"Predicted glucose: {prediction:.1f} mg/dL")

# Get recommendations
from predict_realtime import print_report
report = predictor.generate_comprehensive_report(
    sequence, '2024-01-15 14:30:00', 
    carb_intake=50, time_of_day='lunch', activity_level='moderate'
)
print_report(report)
```

---

### Verification Commands

```bash
# Check Python version
python --version

# List installed packages
pip list

# Check specific packages
pip list | findstr "tensorflow pandas numpy"  # Windows
pip list | grep -E "tensorflow|pandas|numpy"  # Linux/Mac

# Test imports
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import sklearn; print('Scikit-learn OK')"
python -c "import matplotlib; print('Matplotlib OK')"

# Test all at once
python -c "import tensorflow, pandas, numpy, sklearn, matplotlib, joblib; print('All packages OK!')"

# Check model file exists
dir lstm_glucose_model.keras  # Windows
ls lstm_glucose_model.keras   # Linux/Mac
```

---

### File Management Commands

```bash
# List all generated files
dir *.keras *.pkl *.png  # Windows
ls *.keras *.pkl *.png   # Linux/Mac

# View generated plots
start training_history.png  # Windows
start predictions_comparison.png
start error_analysis.png

open training_history.png  # Mac
xdg-open training_history.png  # Linux

# List Python files
dir *.py  # Windows
ls *.py   # Linux/Mac

# Check file sizes
dir /s  # Windows
ls -lh  # Linux/Mac
```

---

### Troubleshooting Commands

```bash
# If TensorFlow import fails
pip uninstall -y tensorflow
pip install --no-cache-dir tensorflow==2.15.0

# If NumPy version conflict
pip install "numpy<2.0.0,>=1.23.5"

# Clear pip cache
pip cache purge

# Remove and reinstall all packages
pip uninstall -y tensorflow pandas numpy matplotlib scikit-learn joblib
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib

# Clear Python cache
rd /s /q __pycache__  # Windows
rm -rf __pycache__    # Linux/Mac
del /s *.pyc  # Windows
find . -name "*.pyc" -delete  # Linux/Mac

# Test model loading
python -c "from keras.models import load_model; m = load_model('lstm_glucose_model.keras'); print('Model loaded OK')"

# Run environment test
python test_environment.py
```

---

### Cleanup Commands

```bash
# Remove generated model files
del *.keras  # Windows
rm *.keras   # Linux/Mac

# Remove scalers
del *.pkl  # Windows
rm *.pkl   # Linux/Mac

# Remove plots
del *.png  # Windows
rm *.png   # Linux/Mac

# Remove all generated files at once
del *.keras *.pkl *.png  # Windows
rm *.keras *.pkl *.png   # Linux/Mac

# Remove Python cache
rd /s /q __pycache__  # Windows
rm -rf __pycache__    # Linux/Mac

# Remove virtual environment
rd /s /q .venv  # Windows
rm -rf .venv    # Linux/Mac
```

---

### Advanced Usage Commands

```bash
# Retrain model with fresh start
del *.keras *.pkl  # Remove old models
python lstm_glucose_prediction.py

# Run predictions with custom parameters
# (Edit predict_realtime.py first to change ISF, ICR values)
python predict_realtime.py

# Batch processing multiple predictions
python demo_predictions.py

# Interactive Python session
python
>>> from predict_realtime import GlucosePredictor
>>> predictor = GlucosePredictor()
>>> # Your custom code here
```

---

### Automated Full Workflow

```bash
# All-in-one command (Windows)
run_all.bat

# Manual complete workflow (copy entire block)
python -m pip install --upgrade pip && ^
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib && ^
python test_environment.py && ^
python lstm_glucose_prediction.py && ^
python demo_predictions.py

# Linux/Mac version
python -m pip install --upgrade pip && \
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib && \
python test_environment.py && \
python lstm_glucose_prediction.py && \
python demo_predictions.py
```

---

### Quick Tests

```bash
# 1-minute quick test
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python test_environment.py

# 5-minute full test
python lstm_glucose_prediction.py
python predict_realtime.py

# 10-minute complete test
python lstm_glucose_prediction.py
python demo_predictions.py
start training_history.png
```

---

## 💡 Pro Tips

**Fastest Installation:**
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib --no-warn-script-location
```

**Silent Installation:**
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib -q
```

**Parallel Installation (faster):**
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib --use-pep517
```

**Check what will be installed:**
```bash
pip install --dry-run pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
```

**Install with specific Python version:**
```bash
python3.11 -m pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
```

---

## 📌 Common Command Sequences

**First Time Setup:**
```bash
python --version
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
python test_environment.py
```

**Daily Usage:**
```bash
python lstm_glucose_prediction.py  # If retraining
python demo_predictions.py         # For predictions
```

**Quick Check:**
```bash
python -c "import tensorflow; from keras.models import load_model"
dir *.keras  # Verify model exists
```

**Full Reset:**
```bash
del *.keras *.pkl *.png  # Remove old files
pip uninstall -y tensorflow  # Fresh install
pip install tensorflow==2.15.0
python lstm_glucose_prediction.py  # Retrain
```

---

## 🎯 Copy-Paste Ready Commands

**Complete Setup (Windows):**
```cmd
python -m pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
python lstm_glucose_prediction.py
python demo_predictions.py
```

**Complete Setup (Linux/Mac):**
```bash
python3 -m pip install --upgrade pip
pip3 install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
python3 lstm_glucose_prediction.py
python3 demo_predictions.py
```

**With Virtual Environment (Windows):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
python lstm_glucose_prediction.py
python demo_predictions.py
```

**With Virtual Environment (Linux/Mac):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn tensorflow==2.15.0 joblib
python lstm_glucose_prediction.py
python demo_predictions.py
```

---

*Quick Reference - All commands tested on Windows/Linux/Mac*
*For detailed explanations, see README.md*
