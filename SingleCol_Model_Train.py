import pandas as pd
import numpy as np
import holidays
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_features(df):
    """
    Adds cyclical time features and holiday flags for single-column prediction.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Time Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear

    # Cyclical Features (Critical for Time Series)
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

def create_sequences(data, input_steps, output_steps):
    """
    X: [Batch, input_steps, n_features]
    y: [Batch, output_steps, 1] (Target is Consum_kWh Only)
    """
    X, y = [], []
    full_len = len(data)

    for i in range(full_len - input_steps - output_steps + 1):
        X.append(data[i : i + input_steps])
        # Target is the first column (Consum_kWh)
        y.append(data[i + input_steps : i + input_steps + output_steps, 0:1])

    return np.array(X), np.array(y)

def train_model(dataset_name):
    # Configuration
    input_file = f'{dataset_name}_Data_Cleaned.csv'
    output_dir = f'Models/{dataset_name}_model/'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"File {input_file} not found!")
        return

    # 1. Load Data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # 2. Feature Engineering
    df_enhanced = create_features(df)
    print("Data Columns after enhancement:", df_enhanced.columns.tolist())

    # 3. Scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = feature_scaler.fit_transform(df_enhanced)
    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[['Consum_kWh']])
    joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))

    # 4. Sequence Generation
    n_input = 2160  # 3 months
    n_output = 720  # 1 month

    print(f"Creating sequences... Input: {n_input} hrs, Output: {n_output} hrs")
    X, y = create_sequences(data_scaled, n_input, n_output)

    if len(X) == 0:
        print("Error: Not enough data for sequences.")
        return

    # 5. Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 6. Build Model
    n_features = X.shape[2]
    model = Sequential([
        Input(shape=(n_input, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(n_output),
        tf.keras.layers.Reshape((n_output, 1))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 7. Training
    print("Starting Training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    model_path = os.path.join(output_dir, f'{dataset_name}_lstm.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 8. Plot Last Test Sample
    X_sample = X_test[-1:]
    y_pred_scaled = model.predict(X_sample)[0]

    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(15, 5))
    plt.plot(y_pred, label='Predicted Consumption', color='cyan', linestyle='-')
    plt.title(f'{dataset_name} Consumption Prediction (Next 30 Days)')
    plt.xlabel('Hours')
    plt.ylabel('kWh')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prediction_plot.png'))
    print("Plot saved.")



if __name__ == "__main__":

    #this is use to train Sports Area model
    # dataset_name = "SportsArea"
    # train_model(dataset_name) 

    #this is use to train Sports Center model
    # dataset_name = "SportsCenter"
    # train_model(dataset_name) 

    #this is use to train Civil Center model 
    dataset_name = "CivilCenter"
    train_model(dataset_name) 
