
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import holidays
import os
import matplotlib.pyplot as plt

def create_features(df):
    # Adds cyclical time features and holiday flags for single-column prediction.

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Time Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear

    # Cyclical Features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Holidays (Spain)
    es_holidays = holidays.Spain()
    df['is_holiday'] = df.index.map(lambda x: x in es_holidays).astype(int)

    # Weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Drop raw ordinal time columns
    df.drop(['hour', 'dayofweek', 'dayofyear'], axis=1, inplace=True)

    return df

def predict_next_month(data_path, model_dir):
    # Extract name after / and before _
    # e.g., "Models/SportsCenter_model" -> "SportsCenter"
    basename = os.path.basename(model_dir.rstrip('/\\'))
    extracted_name = basename.split('_')[0]
    
    # Paths
    model_path = os.path.join(model_dir, f"{extracted_name}_lstm.keras")
    feature_scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    
    # Check existence
    if not os.path.exists(model_path):
        # Fallback if the naming convention in training was different
        model_path = os.path.join(model_dir, os.listdir(model_dir)[0]) if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0 else model_path
        if not os.path.exists(model_path):
            print(f"Error: Model not found in {model_dir}")
            return

    print(f"Loading model and scalers for {extracted_name}...")
    model = tf.keras.models.load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Sequence length from training (3 months)
    n_input = 2160 
    if len(df) < n_input:
        print(f"Error: Not enough data. Need {n_input} rows.")
        return
        
    # Get last N hours
    last_data = df.tail(n_input).copy()
    
    # Feature Engineering
    last_data_enhanced = create_features(last_data)
    
    # Scaling
    input_scaled = feature_scaler.transform(last_data_enhanced)
    
    # Reshape for LSTM: (1, 2160, n_features)
    X_input = input_scaled.reshape(1, n_input, input_scaled.shape[1])
    
    print(f"Predicting next 30 days for {extracted_name}...")
    # Prediction
    y_pred_scaled = model.predict(X_input)[0] # Shape (720, 1)
    
    # Inverse Scaling
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    # Create Future Timeline
    last_timestamp = df.index[-1]
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=720, freq='h')
    
    # Create DataFrame
    pred_df = pd.DataFrame(y_pred, columns=['Consum_kWh'], index=future_dates)
    pred_df[pred_df < 0] = 0 # Safety rule
    pred_df = pred_df.round(2)

    # Save outputs
    os.makedirs('Model_output', exist_ok=True)
    
    csv_path = f'Model_output/{extracted_name}_SingleCol_Prediction.csv'
    pred_df.to_csv(csv_path)
    print(f"Predictions saved to {csv_path}")
    
    plot_path = f'Model_output/{extracted_name}_SingleCol_Plot.png'
    plt.figure(figsize=(15, 6))
    plt.plot(pred_df.index, pred_df['Consum_kWh'], color='cyan', label='Predicted Consumption')
    plt.title(f'Forecast: {extracted_name} Energy Consumption (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('kWh')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    print(f"\nFinal Total Consumption prediction: {pred_df['Consum_kWh'].sum():.2f} kWh")

if __name__ == "__main__":
    
    # predict next month for Civil Center:
    # data_path = 'Data_Cleaning/CivilCenter_Data_Cleaned.csv'
    # model_path = "Models/CivilCenter_model"
    
    # predict next month for Sports Center:
    # data_path = 'Data_Cleaning/SportsCenter_Data_Cleaned.csv'
    # model_path = "Models/SportsCenter_model"


     # predict next month for Sports Area:
    data_path = 'Data_Cleaning/SportsArea_Data_Cleaned.csv'
    model_path = "Models/SportsArea_model"

    if os.path.exists(data_path):
        predict_next_month(data_path, model_path)
    else:
        print(f"Wait: {data_path} not found. Please check paths in __main__.")
