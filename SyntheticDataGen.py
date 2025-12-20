
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnergyDataGenerator:
    """
    Generates realistic synthetic energy data for model input testing.
    Standard requirement: 3 months (2160 hours) of history.
    """
    
    def __init__(self, start_date=None):
        # Default start date is 3 months ago from today if not provided
        if start_date is None:
            self.end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
            self.start_date = self.end_date - timedelta(hours=2159)
        else:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = self.start_date + timedelta(hours=2159)

    def generate_general_data(self):
        """
        Generates 3 months of data with: Hora, Produccion_kWh, Consum_kWh
        Includes solar cycles (0 production at night) and daily consumption patterns.
        """
        timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
        df = pd.DataFrame({'Hora': timestamps})
        
        # Generate Realistic Production (Solar Cycle)
        # Peak at noon, 0 at night
        hour = df['Hora'].dt.hour
        # Simple seasonal peak (higher in middle of day)
        solar_base = np.sin(np.pi * (hour - 6) / 12) 
        solar_base = np.clip(solar_base, 0, 1) # Force 0 outside 6am-6pm
        
        # Add some random weather noise (0.7 to 1.1 multiplier)
        weather_noise = np.random.uniform(0.7, 1.1, size=len(df))
        df['Produccion_kWh'] = (solar_base * 50 * weather_noise).round(2)
        
        # Generate Realistic Consumption
        # Basic pattern: High in morning and evening, low at night
        consumption_base = 5 + 3 * np.sin(2 * np.pi * (hour - 7) / 24) + 2 * np.sin(4 * np.pi * (hour - 7) / 24)
        # Add weekend effect (slightly lower)
        is_weekend = df['Hora'].dt.dayofweek >= 5
        consumption_base[is_weekend] *= 0.8
        
        # Add random noise
        cons_noise = np.random.normal(1, 0.1, size=len(df))
        df['Consum_kWh'] = (consumption_base * cons_noise).round(2)
        
        return df

    def generate_single_col_data(self):
        """
        Generates 3 months of data with: Hora, Consum_kWh
        Focuses on consumption patterns for single-column models.
        """
        timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
        df = pd.DataFrame({'Hora': timestamps})
        hour = df['Hora'].dt.hour
        
        # Pattern: Peak around business hours (9am-8pm)
        consumption_base = 10 + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        
        # Add some sharp spikes occasionally (Realistic for Civic/Sports centers)
        spikes = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        spike_values = np.random.uniform(5, 15, size=len(df))
        
        # Add random noise
        cons_noise = np.random.normal(1, 0.05, size=len(df))
        df['Consum_kWh'] = ((consumption_base * cons_noise) + (spikes * spike_values)).round(2)
        
        return df

if __name__ == "__main__":
    generator = EnergyDataGenerator()
    
    print("Generating General Dataset (3 months)...")
    gen_df = generator.generate_general_data()
    print(gen_df.head())
    print(f"Shape: {gen_df.shape}\n")
    
    print("Generating Single Column Dataset (3 months)...")
    single_df = generator.generate_single_col_data()
    print(single_df.head())
    print(f"Shape: {single_df.shape}")
