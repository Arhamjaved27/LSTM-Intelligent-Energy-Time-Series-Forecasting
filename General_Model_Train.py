
import pandas as pd
import numpy as np
import holidays
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import math

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_features(df):
    """    
    Features added:
    - Hour, DayOfWeek, DayOfYear
    - Cyclical transformations (Sin/Cos) for Hour and DayOfYear
    - IsHoliday (Spain)
    - IsWeekend
    """
    df = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    print("Feature Engineering: Adding Time & Holiday features...")
    
    # 1. Time Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    
    # 2. Cyclical Features (Critical for Time Series)
    # 24 hours in a day
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 365 (or 366) days in a year
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # 3. Holidays (Spain)
    # "Energy consumption is less in Spain holidays"
    es_holidays = holidays.Spain()
    df['is_holiday'] = df.index.map(lambda x: x in es_holidays).astype(int)
    
    # 4. Weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    
    # Let's drop the raw integers to reduce noise, keeping only encoded versions.
    df.drop(['hour', 'dayofweek', 'dayofyear'], axis=1, inplace=True)
    
    return df

def create_sequences(data, input_steps, output_steps):
    """
    Creates Input (X) and Output (y) sequences for the LSTM.
    
    X: [Batch, input_steps, n_features] -> History (e.g., 90 days)
    y: [Batch, output_steps, 2] -> Future Targets (e.g., 30 days of Power & Consumption)
    
    Note: We only predict the 2 columns (Produccion, Consum), not the features.
    """
    X, y = [], []
    
    # We need to predict 'output_steps' into the future.
    # We use 'input_steps' from the past.
    # data is expected to be a numpy array where the first 2 columns are the targets.
    
    full_len = len(data)
    
    # We iterate such that we have enough history and enough future
    for i in range(full_len - input_steps - output_steps + 1):
        # Input: i to i + input_steps
        X.append(data[i : i + input_steps])
        
        # We only take the first 2 columns (Produccion, Consum) as targets
        y.append(data[i + input_steps : i + input_steps + output_steps, 0:2])
        
    return np.array(X), np.array(y)

def train_model():
    input_file = 'School_Data_Cleaned.csv'
    if not os.path.exists(input_file):
        print(f"File {input_file} not found!")
        return

    # Load Data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    # Feature Engineering
    df_enhanced = create_features(df)
    print("Data Columns after enhancement:", df_enhanced.columns.tolist())
    
    # Scaling
    # We scale EVERYTHING to 0-1 for LSTM stability.
    # However, we must be careful: we want to verify plots later in Real Units.
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = feature_scaler.fit_transform(df_enhanced)
    
    # Save scaler for future inference
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    
    # We also need a scaler just for the target variables to invert predictions easily
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[['Produccion_kWh', 'Consum_kWh']]) # Fit only on original targets
    joblib.dump(target_scaler, 'target_scaler_train.pkl')
    
    # Sequence Generation
    input_months = 3
    output_months = 1
    hours_per_month = 30 * 24 # 720 hours
    
    n_input = input_months * hours_per_month  # 2160 hours
    n_output = output_months * hours_per_month # 720 hours
    
    print(f"Creating sequences... Input: {n_input} hrs, Output: {n_output} hrs")
    # This might take a moment and consume RAM
    X, y = create_sequences(data_scaled, n_input, n_output)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    if len(X) == 0:
        print("Error: Not enough data to create sequences! Dataset is too short for 3 months history + 1 month future.")
        return

    # Train/Test Split
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build LSTM Model
    n_features = X.shape[2]
    
    model = Sequential([
        # Encoder
        Input(shape=(n_input, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2), # Prevent overfitting
        LSTM(32, return_sequences=False), # Compress context
        Dropout(0.2),
        
        # Decoder / Output Adapter
        # The output is (720, 2). 
        # A simple Dense layer outputs flat vector. We verify shape.
        Dense(n_output * 2), # 720 * 2 = 1440 units
        
        # Reshape to (720, 2)
        tf.keras.layers.Reshape((n_output, 2))
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    
    # Training
    print("Starting Training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Using a smaller batch size to update weights more often, or larger for speed. 
    # 32 is standard.
    history = model.fit(
        X_train, y_train,
        epochs=50, # Set high, early stopping will catch it
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('best_lstm_model.keras')
    print("Model saved to 'best_lstm_model.keras'")
    
    # Evaluation & Plotting
    print("Evaluating on Test Set...")
    
    # Predict on one sample from test set for visualization
    X_sample = X_test[-1:] 
    
    y_pred_scaled = model.predict(X_sample) # Shape (1, 720, 2)
    y_pred_scaled = y_pred_scaled[0]        # Shape (720, 2)
    
    # Inverse Transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot Production
    plt.subplot(2, 1, 1)
    plt.plot(y_pred[:, 0], label='Predicted Production', color='red', linestyle='-')
    plt.title('Predicted Production (Next 30 Days)')
    plt.legend()
    
    # Plot Consumption
    plt.subplot(2, 1, 2)
    plt.plot(y_pred[:, 1], label='Predicted Consumption', color='cyan', linestyle='-')
    plt.title('Predicted Consumption (Next 30 Days)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction.png')
    print("Plot saved to 'prediction.png'")
    
    # Accuracy Metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {score[0]}")
    print(f"Test MAE: {score[1]}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
