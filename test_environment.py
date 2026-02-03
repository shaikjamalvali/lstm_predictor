"""
Quick Test Script - Verify Environment and Dataset
"""
import sys
print(f"Python version: {sys.version}")

print("\nImporting libraries...")
try:
    import pandas as pd
    print("✓ pandas imported")
    import numpy as np
    print("✓ numpy imported")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
    from sklearn.preprocessing import MinMaxScaler
    print("✓ scikit-learn imported")
    import tensorflow as tf
    print("✓ tensorflow imported")
    import joblib
    print("✓ joblib imported")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Loading dataset...")
try:
    df = pd.read_csv('HUPA0001P.csv', sep=';')
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nGlucose statistics:")
    print(df['glucose'].describe())
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL CHECKS PASSED! Ready to train model.")
print("="*60)
