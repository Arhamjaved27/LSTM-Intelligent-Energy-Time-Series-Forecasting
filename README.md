# LSTM Energy Prediction System

A powerful Deep Learning system designed to forecast energy consumption and solar production using LSTM (Long Short-Term Memory) networks. This project handles data cleaning, model training with advanced feature engineering, and future forecasting.

## 🚀 Project Features
- **Dual Tracking**: Support for both Solar Production (`Produccion_kWh`) and Energy Consumption (`Consum_kWh`).
- **Flexible Models**: 
  - `General`: For buildings with both solar and consumption.
  - `Single Column`: Specialized for buildings with consumption data only (e.g., Civic Centers, Sports Areas).
- **Advanced Feature Engineering**: Includes Sin/Cos cyclical time encoding, holiday detection (Spain), and weekend analysis.
- **Robust Training**: Utilizes Early Stopping, Huber/MSE loss functions, and dataset normalization.
- **Synthetic Generator**: A realistic data generator for testing scenarios with solar cycles and usage spikes.

---

## 📂 Directory Structure
- `Data_Cleaning/`: Contains cleaned CSV files ready for training.
- `Models/`: Stores trained `.keras` files and `.pkl` scalers for each building.
- `Model_output/`: Stores final prediction CSVs and forecast graphs.
- `SyntheticDataGen.py`: Class to generate 3 months of dummy data for testing.
- `requirements.txt`: List of Python dependencies.

---

## 🛠️ Installation & Setup

1. **Activate Virtual Environment** (Recommended):
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 📖 Execution Guide

### Step 1: Data Preparation
Ensure your raw data is cleaned and placed in the `Data_Cleaning/` folder as a CSV with a datetime index. The format should be:
- General: `Hora, Produccion_kWh, Consum_kWh`
- Single: `Hora, Consum_kWh`

### Step 2: Training the Model
Choose the script based on your dataset type:

- **For General Data (Production + Consumption):**
  Open `General_Model_Train.py` and run it to train on your building data.
  
- **For Single-Column (Consumption Only):**
  Open `SingleCol_Model_Train.py`, set the `dataset_name` at the bottom, and run.
  ```python
  dataset_name = "CivilCenter" # Change this to yours
  ```

### Step 3: Making Predictions
After training, generate the next 30 days of forecast:

- **General Forecast:**
  Run `General_Predict.py`. It will save a CSV and a 30-day plot in `Model_output/`.
  
- **Single-Column Forecast:**
  Run `SingleCol_Predict.py`. It automatically detects the building name from the model path.

### Step 4: Testing with Synthetic Data
If you don't have real data yet, use `SyntheticDataGen.py` to create a realistic 3-month dataset that follows physical solar cycles and sharp energy spikes.

---

## 🧠 Model Specifications
- **Input Window**: 720 hours (30 Days) or 2160 hours (90 Days) depending on script configuration.
- **Output Window**: 720 hours (Next 30 Days).
- **Features**: 
  - `sin_hour`, `cos_hour` (Daily Seasonality)
  - `sin_doy`, `cos_doy` (Yearly Seasonality)
  - `is_holiday`, `is_weekend` (Event Flags)
- **scaling**: MinMaxScaler (0 to 1).

---

## 📊 Outputs
All results are stored in `Model_output/` with naming conventions based on the building name.
- **CSV**: Detailed hour-by-hour forecast for the next month.
- **PNG**: Visual graph of the predicted energy trends.

---
*Created for Advanced Energy Analytics.*
