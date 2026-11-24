"""
Synthetic Data Generator for AI WaterWise
Generates realistic hourly water consumption and environmental parameter data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_water_dataset(num_records=5500, output_path="data/water_data.csv", seed=42):
    """
    Generates synthetic hourly water consumption dataset.
    
    Parameters:
    -----------
    num_records : int
        Number of hourly records to generate (>= 5000)
    output_path : str
        Relative filepath to save the output CSV
    seed : int
        Fixed random seed for academic reproducibility
    """
    np.random.seed(seed)
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_records)]
    
    data = []
    
    # Initialize tank level state (percentage 0 - 100)
    current_tank_level = 75.0
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday() # 0 = Monday, 6 = Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Simple holiday indicator (e.g. fixed holiday dates or random ~5% chance)
        is_holiday = 1 if (ts.month == 1 and ts.day in [1, 26]) or (ts.month == 8 and ts.day == 15) or (is_weekend and np.random.rand() < 0.1) else 0
        
        # Temperature: sinusoidal daily cycle with seasonal trend + noise
        base_temp = 25.0 + 5.0 * np.sin(2 * np.pi * ts.timetuple().tm_yday / 365)
        daily_temp_var = 6.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        temp_noise = np.random.normal(0, 1.2)
        temperature = round(float(np.clip(base_temp + daily_temp_var + temp_noise, 12.0, 42.0)), 2)
        
        # Humidity: inversely correlated with temperature
        humidity = round(float(np.clip(80.0 - 1.2 * (temperature - 20.0) + np.random.normal(0, 3.0), 30.0, 95.0)), 2)
        
        # Occupancy (number of people in facility / building, max ~100)
        if 0 <= hour <= 5:
            base_occupancy = 10 if is_weekend else 5
        elif 6 <= hour <= 8:
            base_occupancy = 40 if is_weekend else 75
        elif 9 <= hour <= 17:
            base_occupancy = 60 if is_weekend else 95
        elif 18 <= hour <= 22:
            base_occupancy = 50 if is_weekend else 40
        else:
            base_occupancy = 20 if is_weekend else 10
            
        if is_holiday:
            base_occupancy = int(base_occupancy * 0.4)
            
        occupancy = int(np.clip(base_occupancy + np.random.randint(-5, 6), 0, 100))
        
        # Normal Diurnal Water Consumption (Liters per hour)
        # Base demand per person ~ 8-12 L/hr + HVAC/Irrigation based on temp
        per_person_demand = 8.5
        temp_factor = max(0, (temperature - 25.0) * 12.0)
        
        # Time-of-day demand multiplier
        if 6 <= hour <= 9: # Morning peak
            tod_mult = 1.6
        elif 18 <= hour <= 21: # Evening peak
            tod_mult = 1.4
        elif 1 <= hour <= 4: # Night lull
            tod_mult = 0.2
        else:
            tod_mult = 1.0
            
        normal_consumption = (occupancy * per_person_demand * tod_mult) + temp_factor + np.random.normal(0, 15.0)
        normal_consumption = max(10.0, normal_consumption)
        
        # Inject Abnormalities & Leaks (~4% of the dataset)
        is_leak = False
        is_spike = False
        
        # Periodic leak condition (e.g. hours 1200-1248, 2500-2530, or random midnight leaks)
        if (1200 <= i <= 1248) or (3400 <= i <= 3440) or (hour in [1, 2, 3] and np.random.rand() < 0.04):
            is_leak = True
            # Continuous unexplained flow (e.g., pipe rupture or running valve)
            consumption = normal_consumption + np.random.uniform(250.0, 450.0)
        elif np.random.rand() < 0.02: # Random high demand spike
            is_spike = True
            consumption = normal_consumption + np.random.uniform(150.0, 300.0)
        else:
            consumption = normal_consumption
            
        water_consumption = round(float(consumption), 2)
        
        # Tank level dynamics simulation (0 - 100%)
        # Inflow comes from automated pump, outflow is consumption
        # Average tank capacity ~ 20,000 Liters, 1% tank ~ 200 Liters
        outflow_pct = water_consumption / 200.0
        
        # Pump logic simulation for realistic tank dynamics
        if current_tank_level < 30.0:
            inflow_pct = 15.0 # Pump ON
        elif current_tank_level < 60.0 and (6 <= hour <= 22):
            inflow_pct = 8.0 # Maintenance pumping
        else:
            inflow_pct = 0.0 # Pump OFF
            
        current_tank_level = current_tank_level + inflow_pct - outflow_pct + np.random.normal(0, 0.5)
        current_tank_level = round(float(np.clip(current_tank_level, 5.0, 98.0)), 2)
        
        data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': temperature,
            'humidity': humidity,
            'occupancy': occupancy,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday,
            'tank_level': current_tank_level,
            'water_consumption': water_consumption
        })
        
    df = pd.DataFrame(data)
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset generated successfully with {len(df)} records at '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_water_dataset()
