
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import holidays
import os
import matplotlib.pyplot as plt

def create_features(df):
    """
    Same feature engineering as training.
    Must be identical to what the model was trained on.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # 1. Time Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    
    # 2. Cyclical Features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # 3. Holidays
    es_holidays = holidays.Spain()
    df['is_holiday'] = df.index.map(lambda x: x in es_holidays).astype(int)
    
    # 4. Weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    df.drop(['hour', 'dayofweek', 'dayofyear'], axis=1, inplace=True)
    return df

def predict_next_month(data_path, model_dir):
    # Extract model name (after / and before _)
    # e.g., "Models/SchoolE_model" -> "SchoolE"
    basename = os.path.basename(model_dir.rstrip('/\\'))
    extracted_name = basename.split('_')[0]
    
    # Paths
    model_path = os.path.join(model_dir, "best_lstm_model.keras")
    feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler_train.pkl")
    
    # Check existence
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please wait for training to finish.")
        return
        
    print(f"Loading model and scalers for {extracted_name}...")
    model = tf.keras.models.load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    print("Loading recent data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # We need the last 3 months (2160 hours) to predict the next 1 month
    input_hours = 2160
    if len(df) < input_hours:
        print(f"Error: Not enough data. Need {input_hours} hours. Found {len(df)}.")
        return
        
    # Get last N hours
    last_data = df.tail(input_hours).copy()
    
    # Feature Engineering (Must match training)
    last_data_enhanced = create_features(last_data)
    
    # Scaling
    # Note: feature_scaler expects all columns (targets + features)
    input_scaled = feature_scaler.transform(last_data_enhanced)
    
    # Reshape for LSTM: (1, 2160, n_features)
    X_input = input_scaled.reshape(1, input_hours, input_scaled.shape[1])
    
    print(f"Predicting next 30 days for {extracted_name}...")
    # Prediction -> Shape (1, 720, 2)
    y_pred_scaled = model.predict(X_input)
    y_pred_scaled = y_pred_scaled[0] # (720, 2)
    
    # Inverse Transform to get real kWh values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    # Create Future Timeline
    last_timestamp = df.index[-1]
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=720, freq='H')
    
    # Creates DataFrame
    pred_df = pd.DataFrame(y_pred, columns=['Produccion_kWh', 'Consum_kWh'], index=future_dates)
    
    # POST-PROCESSING (Safety Rules)
    print("Applying Post-Processing Rules...")
    
    # Rule 1: Night Solar = 0
    # "Production = 0 if Hour > 20 or Hour < 6" (Strict 0 for 21-23 and 0-5)
    # *User correction: "at 6 it 0"* -> So 0-6 inclusive are 0.
    night_mask = (pred_df.index.hour > 20) | (pred_df.index.hour <= 6)
    pred_df.loc[night_mask, 'Produccion_kWh'] = 0
    
    # Rule 2: Cannot be negative
    pred_df[pred_df < 0] = 0
    
    # Rule 3: Rounding
    pred_df = pred_df.round(2)

    # Save
    os.makedirs('Model_output', exist_ok=True)
    output_csv = f'Model_output/{extracted_name}_Prediction.csv'
    pred_df.to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(pred_df.index, pred_df['Produccion_kWh'], color='orange', label='Predicted Solar Production')
    plt.title(f'Forecast: {extracted_name} Solar Production (Next 30 Days)')
    plt.ylabel('kWh')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(pred_df.index, pred_df['Consum_kWh'], color='cyan', label='Predicted Consumption')
    plt.title(f'Forecast: {extracted_name} Energy Consumption (Next 30 Days)')
    plt.ylabel('kWh')
    plt.legend()
    
    plt.tight_layout()
    plot_name = f'Model_output/{extracted_name}_Forecast_Plot.png'
    plt.savefig(plot_name)
    print(f"Forecast plot saved to '{plot_name}'")
    
    print("\nSample Forecast (First 24 Hours):")
    print(pred_df.head(24))

    print(f"Total Consumption ({extracted_name}):", pred_df["Consum_kWh"].sum())
    print(f"Total Production ({extracted_name}):", pred_df["Produccion_kWh"].sum())

if __name__ == "__main__":
    

    # Uncomment this if you want to predict school E
    # data_path = 'Data_Cleaning/SchoolE_Data_Cleaned.csv'
    # model_path = "Models/SchoolE_model"
    # predict_next_month(data_path, model_path)


    # Uncomment this if you want to predict school P
    data_path = 'Data_Cleaning/SchoolP_Data_Cleaned.csv'
    model_path = "Models/SchoolP_model"
    predict_next_month(data_path, model_path)