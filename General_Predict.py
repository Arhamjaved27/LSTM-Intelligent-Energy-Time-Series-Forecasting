
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import holidays
import os
import matplotlib.pyplot as plt
import requests


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

def send_data(RMAIN_IP, PORT, SITE_NAME, predicted):
    # Construct URL
    URL = f"http://{RMAIN_IP}:{PORT}/upload_json"

    # Reset index to convert 'Hora' from index to column
    predicted = predicted.reset_index()

    # Ensure timestamp is datetime
    predicted["Hora"] = pd.to_datetime(predicted["Hora"])

    # Extract month automatically (YYYY-MM)
    MONTH = predicted["Hora"].iloc[0].strftime("%Y-%m")

    # Convert Timestamp to string for JSON serialization
    predicted["Hora"] = predicted["Hora"].astype(str)

    # SEND DATAFRAME AS JSON
    payload = {
        "site": SITE_NAME,
        "month": MONTH,
        "data": predicted.to_dict(orient="records")
    }

    response = requests.post(URL, json=payload, timeout=15)

    if response.status_code == 200:
        print("✅ DataFrame sent successfully")
        print(response.json())
    else:
        print("❌ Error sending data:", response.text)


def predict_next_month(model_name, data_path, model_dir):
       
    # Paths
    model_path = os.path.join(model_dir, f"{model_name}_lstm.keras")
    feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    
    # Check existence
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please wait for training to finish.")
        return
        
    print(f"Loading model and scalers for {model_name}...")
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
    
    print(f"Predicting next 30 days for {model_name}...")
    # Prediction -> Shape (1, 720, 2)
    y_pred_scaled = model.predict(X_input)
    y_pred_scaled = y_pred_scaled[0] # (720, 2)
    
    # Inverse Transform to get real kWh values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    # Create Future Timeline
    last_timestamp = df.index[-1]
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=720, freq='h')
    
    # Creates DataFrame
    pred_df = pd.DataFrame(y_pred, columns=['Produccion_kWh', 'Consum_kWh'], index=future_dates)
    
    # POST-PROCESSING (Safety Rules)
    print("Applying Post-Processing Rules...")
    
    # Rule 1: Night Solar = 0
    # "Production = 0 if Hour > 20 or Hour < 6" (Strict 0 for 21-23 and 0-5)
    # *User correction: "at 6 it 0"* -> So 0-6 inclusive are 0.
    night_mask = (pred_df.index.hour > 20) | (pred_df.index.hour <= 6)
    pred_df.loc[night_mask, 'Produccion_kWh'] = 0
    
    pred_df.index.name = 'Hora'
    # Rule 2: Cannot be negative
    pred_df[pred_df < 0] = 0
    
    # Rule 3: Rounding
    pred_df = pred_df.round(2)

    # Save
    os.makedirs('Model_output', exist_ok=True)
    output_csv = f'Model_output/{model_name}_Prediction.csv'
    pred_df.to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(pred_df.index, pred_df['Produccion_kWh'], color='red', label='Predicted Solar Production')
    plt.title(f'Forecast: {model_name} Solar Production (Next 30 Days)')
    plt.ylabel('kWh')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(pred_df.index, pred_df['Consum_kWh'], color='cyan', label='Predicted Consumption')
    plt.title(f'Forecast: {model_name} Energy Consumption (Next 30 Days)')
    plt.ylabel('kWh')
    plt.legend()
    
    plt.tight_layout()
    # plot_name = f'Model_output/{model_name}_Forecast_Plot.png'
    plot_name = f'C:/Users/Arham/Downloads/{model_name}_Forecast_Plot.png'
    plt.savefig(plot_name)
    print(f"Forecast plot saved to '{plot_name}'")
    
    # print("\nSample Forecast (First 24 Hours):")
    # print(pred_df.head(24))

    print(f"Total Consumption ({model_name}):", pred_df["Consum_kWh"].sum())
    print(f"Total Production ({model_name}):", pred_df["Produccion_kWh"].sum())

    return pred_df

if __name__ == "__main__":
    
    # CONFIGURE HERE
    # model_name = "SchoolP"
    model_name = "SchoolE"
    
    RMAIN_IP = "192.168.1.15"   # CHANGE THIS acording to your R-MAIN IP
    PORT = 5000


    data_path = f"Data_Cleaning/{model_name}_Data_Cleaned.csv"
    model_path = f"Models/{model_name}_model"

    if os.path.exists(data_path):
        result = predict_next_month(model_name, data_path, model_path)
        # send_data(RMAIN_IP, PORT, model_name, result)
    else:
        print(f"Wait: {data_path} not found. Please check paths in __main__.")