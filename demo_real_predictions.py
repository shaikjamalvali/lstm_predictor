"""
Demonstration script showing real LSTM-based glucose predictions
with insulin, meal, and exercise recommendations
"""

import numpy as np
import pandas as pd
from predict_realtime import GlucosePredictor
from datetime import datetime, timedelta

def main():
    print("=" * 80)
    print("LSTM Glucose Prediction System - Real Predictions Demo")
    print("=" * 80)
    
    # Initialize predictor
    print("\nInitializing LSTM predictor...")
    predictor = GlucosePredictor()
    
    # Load actual data
    print("Loading real patient data...")
    df = pd.read_csv('HUPA0001P.csv', sep=';')
    df['time'] = pd.to_datetime(df['time'])
    
    # Create time-based features (same as training)
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['minute'] = df['time'].dt.minute
    
    # Forward fill missing values
    df = df.ffill().bfill()
    
    # Feature columns (same order as training)
    feature_cols = ['calories', 'heart_rate', 'steps', 'basal_rate', 
                   'bolus_volume_delivered', 'carb_input', 
                   'hour', 'day_of_week', 'minute']
    
    print(f"Total records available: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Run predictions on different real scenarios from the dataset
    scenarios = [
        {
            'name': 'Normal Morning Scenario',
            'start_idx': 500,  # Random starting point
            'carb_intake': 45,
            'time_of_day': 'breakfast',
            'activity_level': 'light'
        },
        {
            'name': 'Post-Lunch Scenario',
            'start_idx': 1500,
            'carb_intake': 60,
            'time_of_day': 'lunch',
            'activity_level': 'moderate'
        },
        {
            'name': 'Evening Exercise Scenario',
            'start_idx': 2500,
            'carb_intake': 30,
            'time_of_day': 'dinner',
            'activity_level': 'vigorous'
        },
        {
            'name': 'Late Night Scenario',
            'start_idx': 3500,
            'carb_intake': 15,
            'time_of_day': 'snack',
            'activity_level': 'sedentary'
        }
    ]
    
    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 80)
        
        idx = scenario['start_idx']
        
        # Get 12 timesteps of data (1 hour history at 5-minute intervals)
        if idx + 12 >= len(df):
            print(f"Skipping - not enough data from index {idx}")
            continue
            
        # Extract the sequence
        sequence_data = df.iloc[idx:idx+12]
        current_data = df.iloc[idx+11]  # The last (current) timestep
        
        # Get current glucose value
        current_glucose = current_data['glucose']
        current_time = current_data['time']
        
        # Prepare input sequence (normalize like in training)
        input_sequence = sequence_data[feature_cols].values
        
        # Normalize the sequence using the loaded scalers
        input_sequence_scaled = predictor.scaler_X.transform(input_sequence)
        input_sequence_scaled = input_sequence_scaled.reshape(1, 12, 9)
        
        print(f"\nCurrent Time: {current_time}")
        print(f"Current Glucose: {current_glucose:.1f} mg/dL")
        print(f"Heart Rate: {current_data['heart_rate']:.0f} bpm")
        print(f"Steps (last 5 min): {current_data['steps']:.0f}")
        print(f"Recent Carbs: {current_data['carb_input']:.1f} g")
        
        # Generate comprehensive report with REAL data
        report = predictor.generate_comprehensive_report(
            input_sequence=input_sequence_scaled[0],  # Remove batch dimension
            current_glucose=current_glucose,
            current_time=current_time.strftime('%Y-%m-%d %H:%M:%S'),
            carb_intake=scenario['carb_intake'],
            time_of_day=scenario['time_of_day'],
            activity_level=scenario['activity_level']
        )
        
        # Print predictions
        print("\n--- Glucose Predictions (LSTM) ---")
        print(f"  Next 5 minutes: {report['predictions']['next_5_min']} mg/dL")
        print("  Next 30 minutes forecast:")
        for i, glucose_pred in enumerate(report['predictions']['next_30_min']):
            time_ahead = 5 * (i + 1)
            print(f"    +{time_ahead} min: {glucose_pred} mg/dL")
        
        # Print insulin recommendation
        print("\n--- Insulin Recommendation ---")
        insulin = report['insulin_recommendation']
        print(f"  Type: {insulin['recommendation_type']}")
        print(f"  Urgency: {insulin['urgency']}")
        print(f"  Recommended Dose: {insulin['dosage']} units")
        print(f"  Advice: {insulin['advice']}")
        print(f"  Breakdown:")
        print(f"    - Correction: {insulin['breakdown']['correction_dose']} units")
        print(f"    - Carb Coverage: {insulin['breakdown']['carb_dose']} units")
        
        # Print meal recommendation
        print("\n--- Meal Recommendation ---")
        meal = report['meal_recommendation']
        print(f"  Meal Type: {meal['meal_type']}")
        print(f"  Recommended Carbs: {meal['carb_range']}")
        print(f"  Timing: {meal['timing']}")
        print(f"  Suggestions:")
        for suggestion in meal['suggestions'][:3]:
            print(f"    - {suggestion}")
        if meal['foods_to_avoid']:
            print(f"  Foods to Avoid: {', '.join(meal['foods_to_avoid'])}")
        
        # Print exercise recommendation
        print("\n--- Exercise Recommendation ---")
        exercise = report['exercise_recommendation']
        print(f"  Can Exercise: {exercise['can_exercise']}")
        if exercise['can_exercise']:
            print(f"  Exercise Type: {exercise['exercise_type']}")
            print(f"  Duration: {exercise['duration']}")
            print(f"  Intensity: {exercise['intensity']}")
            if 'suggested_activities' in exercise and exercise['suggested_activities']:
                print(f"  Suggested Activities:")
                for activity in exercise['suggested_activities'][:3]:
                    print(f"    - {activity}")
        if exercise['carb_needed'] > 0:
            print(f"  Pre-Exercise Carbs Needed: {exercise['carb_needed']}g")
        if exercise['precautions']:
            print(f"  Precautions:")
            for precaution in exercise['precautions'][:3]:
                print(f"    - {precaution}")
        
        print("\n" + "-" * 80)
    
    # Show prediction accuracy by comparing with actual future values
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY VERIFICATION")
    print("=" * 80)
    
    test_idx = 1000
    if test_idx + 24 < len(df):  # Need 12 history + 12 future
        print(f"\nTesting predictions from index {test_idx}...")
        
        # Get input sequence
        sequence_data = df.iloc[test_idx:test_idx+12]
        current_glucose = df.iloc[test_idx+11]['glucose']
        current_time = df.iloc[test_idx+11]['time']
        
        input_sequence = sequence_data[feature_cols].values
        input_sequence_scaled = predictor.scaler_X.transform(input_sequence)
        
        # Make predictions (pass 2D array for predict_future)
        predictions = predictor.predict_future(input_sequence_scaled, steps_ahead=12)
        
        # Get actual future values
        actual_future = df.iloc[test_idx+12:test_idx+24]['glucose'].values
        
        print(f"\nCurrent Glucose: {current_glucose:.1f} mg/dL")
        print(f"Current Time: {current_time}")
        print("\nPredicted vs Actual (next 1 hour):")
        print(f"{'Time':>6} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10}")
        print("-" * 50)
        
        errors = []
        for i, (pred, actual) in enumerate(zip(predictions, actual_future)):
            future_time = current_time + timedelta(minutes=5*(i+1))
            error = abs(pred - actual)
            errors.append(error)
            print(f"{5*(i+1):>4}min | {pred:>8.1f} | {actual:>8.1f} | {error:>8.1f}")
        
        print("\n" + "-" * 50)
        print(f"Mean Absolute Error: {np.mean(errors):.2f} mg/dL")
        print(f"Root Mean Square Error: {np.sqrt(np.mean(np.array(errors)**2)):.2f} mg/dL")
        print(f"Max Error: {np.max(errors):.2f} mg/dL")
    
    print("\n" + "=" * 80)
    print("Demo complete! All predictions are from the trained LSTM model.")
    print("=" * 80)

if __name__ == "__main__":
    main()
