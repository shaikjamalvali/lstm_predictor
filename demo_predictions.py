"""
Demo Script: Glucose Prediction with Comprehensive Recommendations
Shows various glucose scenarios with insulin, meal, and exercise recommendations
"""

import numpy as np
import pandas as pd
from predict_realtime import GlucosePredictor, print_report
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GLUCOSE PREDICTION & RECOMMENDATION SYSTEM - DEMO")
print("="*80)
print("\nInitializing system...")

# Initialize predictor
predictor = GlucosePredictor()

# Load actual data
df = pd.read_csv('HUPA0001P.csv', sep=';')
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['minute'] = df['time'].dt.minute

features = ['calories', 'heart_rate', 'steps', 'basal_rate', 
            'bolus_volume_delivered', 'carb_input', 'hour', 'day_of_week', 'minute']

print("System ready!\n")

# ============================================================================
# SCENARIO 1: Normal glucose - planning lunch
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 1: Normal Glucose Level - Planning Lunch")
print("="*80)

# Get a sequence with relatively normal glucose
idx = 1500
sequence = df[features].iloc[idx:idx+12].values
timestamp = df['time'].iloc[idx+11].strftime('%Y-%m-%d %H:%M:%S')

report = predictor.generate_comprehensive_report(
    input_sequence=sequence,
    current_time=timestamp,
    carb_intake=50,  # Planning 50g carbs for lunch
    time_of_day='lunch',
    activity_level='moderate'
)

print_report(report)

# ============================================================================
# SCENARIO 2: High glucose - post-meal
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 2: High Glucose Level - Post-Meal Management")
print("="*80)

# Find a high glucose reading
high_glucose_idx = df[df['glucose'] > 200].index[5] if len(df[df['glucose'] > 200]) > 5 else 2000
idx = max(12, high_glucose_idx - 6)
sequence = df[features].iloc[idx:idx+12].values
timestamp = df['time'].iloc[idx+11].strftime('%Y-%m-%d %H:%M:%S')

report = predictor.generate_comprehensive_report(
    input_sequence=sequence,
    current_time=timestamp,
    carb_intake=0,  # No immediate carb intake
    time_of_day='snack',
    activity_level='low'
)

print_report(report)

# ============================================================================
# SCENARIO 3: Low glucose - need intervention
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 3: Low Glucose Level - Requires Immediate Attention")
print("="*80)

# Find a low glucose reading
low_glucose_idx = df[df['glucose'] < 100].index[10] if len(df[df['glucose'] < 100]) > 10 else 1000
idx = max(12, low_glucose_idx - 6)
sequence = df[features].iloc[idx:idx+12].values
timestamp = df['time'].iloc[idx+11].strftime('%Y-%m-%d %H:%M:%S')

report = predictor.generate_comprehensive_report(
    input_sequence=sequence,
    current_time=timestamp,
    carb_intake=0,
    time_of_day='snack',
    activity_level='low'
)

print_report(report)

# ============================================================================
# SCENARIO 4: Good glucose - planning exercise
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 4: Good Glucose Level - Planning Exercise")
print("="*80)

# Find a moderate glucose reading (120-150)
moderate_idx = df[(df['glucose'] >= 120) & (df['glucose'] <= 150)].index
if len(moderate_idx) > 15:
    idx = moderate_idx[15] - 6
else:
    idx = 800
    
idx = max(12, idx)
sequence = df[features].iloc[idx:idx+12].values
timestamp = df['time'].iloc[idx+11].strftime('%Y-%m-%d %H:%M:%S')

report = predictor.generate_comprehensive_report(
    input_sequence=sequence,
    current_time=timestamp,
    carb_intake=0,
    time_of_day='snack',
    activity_level='high'
)

print_report(report)

# ============================================================================
# SUMMARY: Show glucose predictions over time
# ============================================================================
print("\n" + "="*80)
print("GLUCOSE PREDICTION SUMMARY - Multiple Time Points")
print("="*80)

print("\nPredicting glucose levels for next 30 minutes at different starting points:\n")

test_indices = [500, 1000, 1500, 2000, 2500]
for i, idx in enumerate(test_indices, 1):
    if idx + 12 >= len(df):
        continue
    
    sequence = df[features].iloc[idx:idx+12].values
    current_glucose = df['glucose'].iloc[idx+11]
    timestamp = df['time'].iloc[idx+11].strftime('%H:%M')
    
    # Predict next 30 minutes
    predicted_next = predictor.predict_glucose(sequence)
    future_predictions = predictor.predict_future(sequence, steps_ahead=6)
    
    print(f"{i}. Time: {timestamp}")
    print(f"   Current Glucose: {current_glucose:.1f} mg/dL")
    print(f"   Predicted (5 min): {predicted_next:.1f} mg/dL")
    print(f"   Predicted (30 min): {', '.join([f'{p:.1f}' for p in future_predictions])} mg/dL")
    
    # Trend analysis
    if predicted_next > current_glucose + 10:
        trend = "↑ Rising"
    elif predicted_next < current_glucose - 10:
        trend = "↓ Falling"
    else:
        trend = "→ Stable"
    print(f"   Trend: {trend}")
    print()

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS FROM THE LSTM MODEL")
print("="*80)

print("""
✓ GLUCOSE PREDICTIONS:
  - Model predicts glucose levels 5-30 minutes ahead
  - Uses 1 hour of historical data (12 time steps)
  - Considers: calories, heart rate, steps, insulin, carbs

✓ INSULIN RECOMMENDATIONS:
  - Correction dose: Adjusts high glucose to target
  - Carb coverage: Covers carbohydrate intake
  - Safety checks: Prevents dosing during hypoglycemia

✓ MEAL RECOMMENDATIONS:
  - Carb ranges based on glucose status
  - Specific food suggestions for each scenario
  - Adjusts for time of day and activity level

✓ EXERCISE RECOMMENDATIONS:
  - Safety assessment before exercise
  - Exercise type, duration, and intensity
  - Pre-exercise carb requirements
  - Specific activity suggestions

⚠️  IMPORTANT REMINDERS:
  - This system is for educational purposes only
  - Always consult healthcare professionals
  - Individual insulin sensitivity varies
  - Monitor glucose according to your care plan
""")

print("="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nAll model files saved:")
print("  • lstm_glucose_model.keras - Trained model")
print("  • scaler_X.pkl & scaler_y.pkl - Data scalers")
print("  • training_history.png - Training performance")
print("  • predictions_comparison.png - Predictions vs actual")
print("  • error_analysis.png - Error distribution")
print("\nYou can now use predict_realtime.py for custom predictions!")
print("="*80)
