import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('HUPA0001P.csv', sep=';')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Handle missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
df = df.ffill().bfill()

# Feature engineering - extract time features
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['minute'] = df['time'].dt.minute

# Select features for modeling
features = ['calories', 'heart_rate', 'steps', 'basal_rate', 
            'bolus_volume_delivered', 'carb_input', 'hour', 'day_of_week', 'minute']
target = 'glucose'

print(f"\nFeatures used: {features}")
print(f"Target: {target}")

# Prepare data
X = df[features].values
y = df[target].values

# Normalize the features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Save scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("\nScalers saved!")

# Create sequences for LSTM
def create_sequences(X, y, time_steps=12):
    """
    Create sequences for LSTM input
    time_steps: number of previous time steps to use (12 = 1 hour with 5-min intervals)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences (using 12 time steps = 1 hour of history)
TIME_STEPS = 12
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

print(f"\nSequence shape: X={X_seq.shape}, y={y_seq.shape}")

# Split data into train and test sets (80-20 split)
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"\nTrain set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# Build LSTM model
print("\nBuilding LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', 
                            save_best_only=True, mode='min', verbose=1)

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Save final model
model.save('lstm_glucose_model.keras')
print("\nModel saved as 'lstm_glucose_model.keras'")

# Make predictions on test set
print("\nMaking predictions on test set...")
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE ON TEST SET")
print("="*50)
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} mg/dL")
print(f"MAE:  {mae:.2f} mg/dL")
print(f"R² Score: {r2:.4f}")
print("="*50)

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved as 'training_history.png'")
plt.close()

# Plot predictions vs actual
plt.figure(figsize=(15, 6))
plot_points = min(500, len(y_test_actual))

plt.subplot(1, 2, 1)
plt.plot(y_test_actual[:plot_points], label='Actual', alpha=0.7, linewidth=2)
plt.plot(y_pred[:plot_points], label='Predicted', alpha=0.7, linewidth=2)
plt.title('Glucose Level: Actual vs Predicted (First 500 points)')
plt.xlabel('Time Steps')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_pred, alpha=0.3)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.title('Prediction Scatter Plot')
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Predicted Glucose (mg/dL)')
plt.grid(True)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
print("Predictions comparison saved as 'predictions_comparison.png'")
plt.close()

# Error analysis
errors = y_test_actual - y_pred
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(errors, bins=50, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (mg/dL)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(errors[:plot_points])
plt.title('Prediction Errors Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Error (mg/dL)')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_test_actual, errors, alpha=0.3)
plt.title('Residual Plot')
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Error (mg/dL)')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
print("Error analysis saved as 'error_analysis.png'")
plt.close()

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nFiles generated:")
print("1. lstm_glucose_model.keras - Final trained model")
print("2. best_lstm_model.keras - Best model checkpoint")
print("3. scaler_X.pkl - Feature scaler")
print("4. scaler_y.pkl - Target scaler")
print("5. training_history.png - Training metrics")
print("6. predictions_comparison.png - Predictions vs actual")
print("7. error_analysis.png - Error analysis plots")
print("="*50)
