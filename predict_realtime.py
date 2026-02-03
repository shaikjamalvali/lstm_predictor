import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

class GlucosePredictor:
    """
    Real-time glucose prediction and recommendation system
    """
    
    def __init__(self, model_path='lstm_glucose_model.keras', 
                 scaler_x_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
        """Initialize the predictor with trained model and scalers"""
        print("Loading model and scalers...")
        self.model = load_model(model_path)
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.time_steps = 12  # 1 hour of history
        self.feature_names = ['calories', 'heart_rate', 'steps', 'basal_rate', 
                             'bolus_volume_delivered', 'carb_input', 'hour', 
                             'day_of_week', 'minute']
        print("Model loaded successfully!")
        
    def predict_glucose(self, input_sequence):
        """
        Predict glucose level from input sequence
        
        Parameters:
        -----------
        input_sequence: array-like, shape (time_steps, n_features)
            Historical data for the last time_steps (12 by default)
            
        Returns:
        --------
        predicted_glucose: float
            Predicted glucose level in mg/dL
        """
        # Ensure correct shape
        if len(input_sequence) != self.time_steps:
            raise ValueError(f"Input sequence must have {self.time_steps} time steps")
        
        # Scale input
        input_scaled = self.scaler_X.transform(input_sequence)
        input_scaled = input_scaled.reshape(1, self.time_steps, -1)
        
        # Predict
        prediction_scaled = self.model.predict(input_scaled, verbose=0)
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction[0][0]
    
    def predict_future(self, input_sequence, steps_ahead=6):
        """
        Predict glucose levels for multiple steps ahead
        
        Parameters:
        -----------
        input_sequence: array-like, shape (time_steps, n_features)
            Historical data for prediction
        steps_ahead: int
            Number of future steps to predict (default 6 = 30 minutes)
            
        Returns:
        --------
        predictions: list
            List of predicted glucose values
        """
        predictions = []
        current_sequence = input_sequence.copy()
        
        for _ in range(steps_ahead):
            # Predict next value
            pred = self.predict_glucose(current_sequence)
            predictions.append(pred)
            
            # Update sequence for next prediction
            # Keep the last features constant (simplified approach)
            next_step = current_sequence[-1].copy()
            current_sequence = np.vstack([current_sequence[1:], next_step])
        
        return predictions
    
    def get_insulin_recommendation(self, current_glucose, predicted_glucose, 
                                   carb_intake=0, target_glucose=120):
        """
        Calculate insulin recommendation based on glucose levels and carb intake
        
        Parameters:
        -----------
        current_glucose: float
            Current glucose level (mg/dL)
        predicted_glucose: float
            Predicted glucose level (mg/dL)
        carb_intake: float
            Expected carbohydrate intake (grams)
        target_glucose: float
            Target glucose level (mg/dL), default 120
            
        Returns:
        --------
        recommendation: dict
            Insulin dosage recommendation and explanation
        """
        # Insulin sensitivity factor (ISF): how much 1 unit of insulin lowers glucose
        ISF = 50  # mg/dL per unit (typical value, should be personalized)
        
        # Insulin-to-carb ratio (ICR): how many grams of carbs 1 unit covers
        ICR = 10  # grams per unit (typical value, should be personalized)
        
        # Calculate correction dose
        glucose_diff = max(predicted_glucose - target_glucose, 0)
        correction_dose = glucose_diff / ISF
        
        # Calculate carb coverage dose
        carb_dose = carb_intake / ICR
        
        # Total recommended dose
        total_dose = correction_dose + carb_dose
        
        # Safety checks
        if current_glucose < 70:
            recommendation_type = "URGENT: Hypoglycemia Risk"
            advice = "DO NOT TAKE INSULIN. Consume 15-20g of fast-acting carbs immediately."
            dosage = 0
            urgency = "CRITICAL"
        elif predicted_glucose < 80:
            recommendation_type = "WARNING: Predicted Low Glucose"
            advice = "Reduce insulin dose or consume carbs to prevent hypoglycemia."
            dosage = max(0, total_dose * 0.5)
            urgency = "HIGH"
        elif predicted_glucose > 180:
            recommendation_type = "ALERT: Predicted High Glucose"
            advice = f"Consider {total_dose:.2f} units of rapid-acting insulin."
            dosage = total_dose
            urgency = "MEDIUM"
        elif predicted_glucose > 250:
            recommendation_type = "URGENT: Severe Hyperglycemia Risk"
            advice = f"Take {total_dose:.2f} units and monitor closely. Check for ketones."
            dosage = total_dose
            urgency = "HIGH"
        else:
            recommendation_type = "NORMAL: Glucose in Target Range"
            if carb_intake > 0:
                advice = f"Consider {carb_dose:.2f} units to cover {carb_intake}g carbs."
                dosage = carb_dose
            else:
                advice = "No insulin needed. Continue monitoring."
                dosage = 0
            urgency = "LOW"
        
        return {
            'recommendation_type': recommendation_type,
            'dosage': round(dosage, 2),
            'advice': advice,
            'urgency': urgency,
            'breakdown': {
                'correction_dose': round(correction_dose, 2),
                'carb_dose': round(carb_dose, 2),
                'total_dose': round(total_dose, 2)
            }
        }
    
    def get_meal_recommendation(self, current_glucose, predicted_glucose, 
                                time_of_day, activity_level='moderate'):
        """
        Provide meal recommendations based on glucose levels
        
        Parameters:
        -----------
        current_glucose: float
            Current glucose level (mg/dL)
        predicted_glucose: float
            Predicted glucose level (mg/dL)
        time_of_day: str
            'breakfast', 'lunch', 'dinner', or 'snack'
        activity_level: str
            'low', 'moderate', or 'high'
            
        Returns:
        --------
        recommendation: dict
            Meal recommendations
        """
        recommendations = {
            'meal_type': time_of_day,
            'carb_range': None,
            'suggestions': [],
            'foods_to_avoid': [],
            'timing': None,
            'notes': []
        }
        
        # Adjust recommendations based on glucose levels
        if current_glucose < 70:
            recommendations['carb_range'] = (15, 20)
            recommendations['suggestions'] = [
                "4 glucose tablets or 4 oz fruit juice",
                "1 tablespoon honey or sugar",
                "15-20g of fast-acting carbs",
                "Wait 15 min and recheck glucose"
            ]
            recommendations['foods_to_avoid'] = ["High-fat foods (slow absorption)"]
            recommendations['timing'] = "Immediately"
            recommendations['notes'] = ["URGENT: Treat hypoglycemia first"]
            
        elif current_glucose < 100:
            recommendations['carb_range'] = (30, 45)
            recommendations['suggestions'] = [
                "Whole grain toast with peanut butter",
                "Oatmeal with nuts and berries",
                "Greek yogurt with fruit and granola",
                "Include protein to stabilize glucose"
            ]
            recommendations['foods_to_avoid'] = ["High-sugar cereals", "White bread alone"]
            recommendations['timing'] = "Within 30 minutes"
            recommendations['notes'] = ["Include protein and healthy fats"]
            
        elif predicted_glucose > 180:
            recommendations['carb_range'] = (15, 30)
            recommendations['suggestions'] = [
                "Large salad with grilled chicken",
                "Steamed vegetables with lean protein",
                "Cauliflower rice with fish",
                "Avoid high-carb foods temporarily"
            ]
            recommendations['foods_to_avoid'] = [
                "Bread, pasta, rice", "Sugary drinks", 
                "Desserts", "Fruit juices"
            ]
            recommendations['timing'] = "Delay meal by 30-60 min if possible"
            recommendations['notes'] = ["Focus on low-carb, high-fiber options"]
            
        else:
            # Normal glucose range
            if time_of_day == 'breakfast':
                recommendations['carb_range'] = (45, 60)
                recommendations['suggestions'] = [
                    "Whole grain toast with eggs and avocado",
                    "Steel-cut oats with nuts and berries",
                    "Greek yogurt parfait with granola",
                    "Vegetable omelet with whole wheat toast"
                ]
            elif time_of_day == 'lunch':
                recommendations['carb_range'] = (45, 60)
                recommendations['suggestions'] = [
                    "Quinoa bowl with vegetables and protein",
                    "Whole grain sandwich with lean meat",
                    "Lentil soup with whole grain bread",
                    "Mixed salad with chickpeas and whole grain"
                ]
            elif time_of_day == 'dinner':
                recommendations['carb_range'] = (45, 60)
                recommendations['suggestions'] = [
                    "Grilled salmon with sweet potato and vegetables",
                    "Chicken breast with brown rice and broccoli",
                    "Turkey meatballs with whole wheat pasta",
                    "Stir-fry with tofu and vegetables over quinoa"
                ]
            else:  # snack
                recommendations['carb_range'] = (15, 20)
                recommendations['suggestions'] = [
                    "Apple with almond butter",
                    "Carrots and hummus",
                    "Mixed nuts (1/4 cup)",
                    "String cheese with whole grain crackers"
                ]
            
            recommendations['foods_to_avoid'] = [
                "Highly processed foods", "Sugary beverages",
                "White bread and pasta", "Fried foods"
            ]
            recommendations['timing'] = "Regular meal schedule"
            recommendations['notes'] = ["Balance carbs with protein and healthy fats"]
        
        # Adjust for activity level
        if activity_level == 'high':
            low, high = recommendations['carb_range']
            recommendations['carb_range'] = (low + 15, high + 15)
            recommendations['notes'].append("Increased carbs for high activity")
        elif activity_level == 'low':
            low, high = recommendations['carb_range']
            recommendations['carb_range'] = (max(15, low - 10), high - 10)
        
        return recommendations
    
    def get_exercise_recommendation(self, current_glucose, predicted_glucose, 
                                    recent_steps, heart_rate):
        """
        Provide exercise recommendations based on glucose levels
        
        Parameters:
        -----------
        current_glucose: float
            Current glucose level (mg/dL)
        predicted_glucose: float
            Predicted glucose level (mg/dL)
        recent_steps: float
            Steps in last hour
        heart_rate: float
            Current heart rate
            
        Returns:
        --------
        recommendation: dict
            Exercise recommendations
        """
        recommendation = {
            'can_exercise': False,
            'exercise_type': None,
            'duration': None,
            'intensity': None,
            'precautions': [],
            'carb_needed': 0
        }
        
        # Check if safe to exercise
        if current_glucose < 100:
            recommendation['can_exercise'] = False
            recommendation['precautions'] = [
                "WAIT: Glucose too low for exercise",
                "Consume 15-20g carbs and wait 15 minutes",
                "Recheck glucose before starting exercise",
                "Target glucose > 100 mg/dL before exercise"
            ]
            recommendation['carb_needed'] = 15
            
        elif current_glucose > 250:
            recommendation['can_exercise'] = False
            recommendation['precautions'] = [
                "CAUTION: Glucose too high for exercise",
                "Check for ketones",
                "Take correction insulin if needed",
                "Wait until glucose < 250 mg/dL",
                "Stay hydrated"
            ]
            
        elif predicted_glucose < 80:
            recommendation['can_exercise'] = True
            recommendation['exercise_type'] = "Light activity only"
            recommendation['duration'] = "10-15 minutes"
            recommendation['intensity'] = "Low"
            recommendation['precautions'] = [
                "Risk of hypoglycemia during exercise",
                "Consume 15-20g carbs before starting",
                "Monitor glucose every 15 minutes",
                "Keep fast-acting carbs nearby",
                "Stop if you feel symptoms of low glucose"
            ]
            recommendation['carb_needed'] = 20
            
        else:
            # Safe to exercise
            recommendation['can_exercise'] = True
            
            # Determine intensity based on recent activity
            if recent_steps > 500:  # Already active
                recommendation['exercise_type'] = "Continue current activity or light exercise"
                recommendation['duration'] = "15-20 minutes"
                recommendation['intensity'] = "Low to Moderate"
            else:  # Not very active recently
                if current_glucose > 180:
                    recommendation['exercise_type'] = "Moderate to vigorous exercise"
                    recommendation['duration'] = "30-45 minutes"
                    recommendation['intensity'] = "Moderate to High"
                    recommendation['precautions'] = [
                        "Exercise can help lower glucose",
                        "Good time for cardio or strength training"
                    ]
                else:
                    recommendation['exercise_type'] = "Moderate exercise"
                    recommendation['duration'] = "20-30 minutes"
                    recommendation['intensity'] = "Moderate"
            
            # General precautions
            recommendation['precautions'].extend([
                "Monitor glucose before, during, and after exercise",
                "Stay hydrated",
                "Have fast-acting carbs available",
                "Wear medical ID"
            ])
            
            # Suggested activities
            if recommendation['intensity'] == "Low":
                recommendation['suggested_activities'] = [
                    "Walking", "Gentle yoga", "Light stretching",
                    "Easy cycling", "Tai chi"
                ]
            elif recommendation['intensity'] in ["Low to Moderate", "Moderate"]:
                recommendation['suggested_activities'] = [
                    "Brisk walking", "Swimming", "Cycling",
                    "Dancing", "Moderate hiking", "Yoga"
                ]
            else:  # High intensity
                recommendation['suggested_activities'] = [
                    "Running", "HIIT training", "Vigorous cycling",
                    "Sports (basketball, soccer)", "Intense aerobics"
                ]
        
        return recommendation
    
    def generate_comprehensive_report(self, input_sequence, current_glucose, current_time, 
                                     carb_intake=0, time_of_day='breakfast',
                                     activity_level='moderate'):
        """
        Generate a comprehensive report with predictions and all recommendations
        
        Parameters:
        -----------
        input_sequence: array-like
            Historical data for prediction (12 timesteps, 9 features)
        current_glucose: float
            Current glucose level in mg/dL
        current_time: str
            Current time for context
        carb_intake: float
            Expected carb intake
        time_of_day: str
            Meal time context
        activity_level: str
            Activity level
            
        Returns:
        --------
        report: dict
            Comprehensive report with all recommendations
        """
        # Get current values from last row (unscale the features)
        last_values = input_sequence[-1]
        
        # Unscale heart rate and steps from the input features
        # Features order: calories, heart_rate, steps, basal_rate, bolus, carb_input, hour, day_of_week, minute
        heart_rate = last_values[1] * (self.scaler_X.data_max_[1] - self.scaler_X.data_min_[1]) + self.scaler_X.data_min_[1]
        steps = last_values[2] * (self.scaler_X.data_max_[2] - self.scaler_X.data_min_[2]) + self.scaler_X.data_min_[2]
        
        # Make predictions
        predicted_glucose = self.predict_glucose(input_sequence)
        future_predictions = self.predict_future(input_sequence, steps_ahead=6)
        
        # Get recommendations
        insulin_rec = self.get_insulin_recommendation(
            current_glucose, predicted_glucose, carb_intake
        )
        meal_rec = self.get_meal_recommendation(
            current_glucose, predicted_glucose, time_of_day, activity_level
        )
        exercise_rec = self.get_exercise_recommendation(
            current_glucose, predicted_glucose, steps, heart_rate
        )
        
        report = {
            'timestamp': current_time,
            'current_status': {
                'glucose': round(current_glucose, 1),
                'heart_rate': round(heart_rate, 1),
                'recent_steps': round(steps, 0)
            },
            'predictions': {
                'next_5_min': round(predicted_glucose, 1),
                'next_30_min': [round(p, 1) for p in future_predictions]
            },
            'insulin_recommendation': insulin_rec,
            'meal_recommendation': meal_rec,
            'exercise_recommendation': exercise_rec
        }
        
        return report


def print_report(report):
    """Pretty print the comprehensive report"""
    print("\n" + "="*80)
    print("GLUCOSE MANAGEMENT REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print()
    
    print("CURRENT STATUS:")
    print(f"  Glucose Level: {report['current_status']['glucose']:.1f} mg/dL")
    print(f"  Heart Rate: {report['current_status']['heart_rate']:.0f} bpm")
    print(f"  Recent Steps: {report['current_status']['recent_steps']:.0f}")
    print()
    
    print("PREDICTIONS:")
    print(f"  Next 5 minutes: {report['predictions']['next_5_min']:.1f} mg/dL")
    print(f"  Next 30 minutes: {', '.join([f'{p:.1f}' for p in report['predictions']['next_30_min']])} mg/dL")
    print()
    
    print("-"*80)
    print("INSULIN RECOMMENDATION:")
    insulin = report['insulin_recommendation']
    print(f"  Status: {insulin['recommendation_type']}")
    print(f"  Urgency: {insulin['urgency']}")
    print(f"  Recommended Dosage: {insulin['dosage']} units")
    print(f"  Advice: {insulin['advice']}")
    print(f"  Breakdown:")
    print(f"    - Correction: {insulin['breakdown']['correction_dose']} units")
    print(f"    - Carb coverage: {insulin['breakdown']['carb_dose']} units")
    print()
    
    print("-"*80)
    print("MEAL RECOMMENDATION:")
    meal = report['meal_recommendation']
    print(f"  Meal Type: {meal['meal_type'].title()}")
    print(f"  Carb Range: {meal['carb_range'][0]}-{meal['carb_range'][1]}g")
    print(f"  Timing: {meal['timing']}")
    print(f"  Suggestions:")
    for suggestion in meal['suggestions']:
        print(f"    - {suggestion}")
    print(f"  Foods to Avoid:")
    for food in meal['foods_to_avoid']:
        print(f"    - {food}")
    print(f"  Notes:")
    for note in meal['notes']:
        print(f"    - {note}")
    print()
    
    print("-"*80)
    print("EXERCISE RECOMMENDATION:")
    exercise = report['exercise_recommendation']
    print(f"  Can Exercise: {'YES' if exercise['can_exercise'] else 'NO'}")
    if exercise['can_exercise']:
        print(f"  Type: {exercise['exercise_type']}")
        print(f"  Duration: {exercise['duration']}")
        print(f"  Intensity: {exercise['intensity']}")
        if 'suggested_activities' in exercise:
            print(f"  Suggested Activities:")
            for activity in exercise['suggested_activities']:
                print(f"    - {activity}")
    if exercise['carb_needed'] > 0:
        print(f"  Carbs Needed Before Exercise: {exercise['carb_needed']}g")
    print(f"  Precautions:")
    for precaution in exercise['precautions']:
        print(f"    - {precaution}")
    print("="*80)


# Example usage
if __name__ == "__main__":
    print("Glucose Prediction and Recommendation System")
    print("=" * 80)
    
    # Initialize predictor
    predictor = GlucosePredictor()
    
    # Load sample data for demonstration
    df = pd.read_csv('HUPA0001P.csv', sep=';')
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['minute'] = df['time'].dt.minute
    
    # Get a sample sequence (last 12 time steps from middle of dataset)
    features = ['calories', 'heart_rate', 'steps', 'basal_rate', 
                'bolus_volume_delivered', 'carb_input', 'hour', 'day_of_week', 'minute']
    
    # Use data from middle of dataset
    start_idx = len(df) // 2
    sample_sequence = df[features].iloc[start_idx:start_idx+12].values
    current_time = df['time'].iloc[start_idx+11].strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = predictor.generate_comprehensive_report(
        input_sequence=sample_sequence,
        current_time=current_time,
        carb_intake=45,  # Planning to eat 45g carbs
        time_of_day='lunch',
        activity_level='moderate'
    )
    
    # Print report
    print_report(report)
    
    print("\n" + "="*80)
    print("EXAMPLE SCENARIOS")
    print("="*80)
    
    # Scenario 1: Low glucose
    print("\n1. LOW GLUCOSE SCENARIO:")
    low_glucose_seq = sample_sequence.copy()
    low_glucose_seq[-1, 0] = 0.1  # Low glucose value (scaled)
    report1 = predictor.generate_comprehensive_report(
        low_glucose_seq, current_time, 0, 'snack', 'low'
    )
    print(f"Current Glucose: {report1['current_status']['glucose']:.1f} mg/dL")
    print(f"Insulin: {report1['insulin_recommendation']['advice']}")
    print(f"Meal: {', '.join(report1['meal_recommendation']['suggestions'][:2])}")
    print(f"Exercise: {', '.join(report1['exercise_recommendation']['precautions'][:2])}")
    
    # Scenario 2: High glucose
    print("\n2. HIGH GLUCOSE SCENARIO:")
    high_glucose_seq = sample_sequence.copy()
    high_glucose_seq[-1, 0] = 0.9  # High glucose value (scaled)
    report2 = predictor.generate_comprehensive_report(
        high_glucose_seq, current_time, 30, 'dinner', 'low'
    )
    print(f"Current Glucose: {report2['current_status']['glucose']:.1f} mg/dL")
    print(f"Insulin: {report2['insulin_recommendation']['advice']}")
    print(f"Meal: Carbs {report2['meal_recommendation']['carb_range']}")
    print(f"Exercise: {report2['exercise_recommendation']['exercise_type']}")
    
    print("\n" + "="*80)
    print("System ready for real-time predictions!")
    print("="*80)
