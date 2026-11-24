"""
Data Preprocessing Service for AI WaterWise
Handles CSV validation, dynamic active dataset loading, cleaning, feature engineering, and sequence formation.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from backend.utils.config import (
    get_active_csv_path, REQUIRED_COLUMNS, DNN_FEATURES, AE_FEATURES
)

class DataPreprocessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None

    @staticmethod
    def validate_dataframe(df):
        """
        Validates whether a pandas DataFrame contains all required columns.
        
        Returns:
        --------
        tuple: (is_valid: bool, missing_columns: list)
        """
        # Normalize column names (strip whitespace and convert to lowercase)
        col_map = {col: str(col).strip().lower() for col in df.columns}
        df.rename(columns=col_map, inplace=True)
        
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, missing
        return True, []

    def load_data(self):
        """Loads and cleans the active uploaded dataset."""
        target_path = self.data_path or get_active_csv_path()
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError("No dataset uploaded. Please upload a CSV file to begin.")
            
        self.df = pd.read_csv(target_path)
        
        # Validate columns
        is_valid, missing = self.validate_dataframe(self.df)
        if not is_valid:
            raise ValueError(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
            
        # Clean & format timestamps with mixed format and dayfirst=True to handle DD-MM-YYYY formats properly
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed', dayfirst=True, errors='coerce')
        self.df.dropna(subset=['timestamp'], inplace=True)
        self.df.sort_values(by='timestamp', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Handle string day_of_week variations if any (e.g., "Monday" / "Mon" -> 0)
        if self.df['day_of_week'].dtype == 'object':
            day_map = {
                'monday': 0, 'mon': 0,
                'tuesday': 1, 'tue': 1,
                'wednesday': 2, 'wed': 2,
                'thursday': 3, 'thu': 3,
                'friday': 4, 'fri': 4,
                'saturday': 5, 'sat': 5,
                'sunday': 6, 'sun': 6
            }
            self.df['day_of_week'] = self.df['day_of_week'].astype(str).str.strip().str.lower().map(day_map).fillna(0).astype(int)
        else:
            self.df['day_of_week'] = pd.to_numeric(self.df['day_of_week'], errors='coerce').fillna(0).astype(int)

        # Convert remaining numerical columns
        num_cols = ['temperature', 'humidity', 'occupancy', 'is_holiday', 'tank_level', 'water_consumption']
        for col in num_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Drop rows with NaN in critical numerical values
        self.df.dropna(subset=num_cols, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        if len(self.df) == 0:
            raise ValueError("Uploaded CSV file contains no valid data rows after cleaning.")
            
        # Feature Engineering: Lag 1 feature (previous hour's water consumption)
        self.df['prev_consumption'] = self.df['water_consumption'].shift(1)
        self.df['prev_consumption'] = self.df['prev_consumption'].fillna(self.df['water_consumption'])
        
        return self.df

    def prepare_dnn_data(self):
        """
        Prepares normalized features and target split for DNN model training using chronological split:
        70% Training, 15% Validation, 15% Testing.
        """
        if self.df is None:
            self.load_data()
            
        X = self.df[DNN_FEATURES].values
        y = self.df['water_consumption'].values
        n = len(X)
        
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        if train_end == 0:
            train_end = max(1, n - 2) if n >= 3 else 1
            val_end = max(train_end, n - 1) if n >= 2 else train_end
            
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Fallbacks if val or test is empty
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train
        if len(X_test) == 0:
            X_test, y_test = X_val, y_val
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        split_info = {
            "train_records": len(X_train),
            "val_records": len(X_val),
            "test_records": len(X_test)
        }
        
        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_y, split_info
        )

    def prepare_lstm_data(self, lookback=24):
        """
        Creates 3D time-series windows (samples, time_steps, 1) for LSTM demand forecasting.
        Requires dataset length to be sufficient for lookback sequence formation.
        Applies chronological split: 70% Train, 15% Validation, 15% Test.
        """
        if self.df is None:
            self.load_data()
            
        n_raw = len(self.df)
        min_required = lookback + 5
        if n_raw < min_required:
            raise ValueError(
                f"Insufficient sequential data for LSTM forecasting. "
                f"Uploaded dataset contains {n_raw} records, but minimum required is {min_required}."
            )
            
        series = self.df[['water_consumption']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler.fit_transform(series)
        
        X_seq, y_seq = [], []
        for i in range(len(scaled_series) - lookback):
            X_seq.append(scaled_series[i : i + lookback])
            y_seq.append(scaled_series[i + lookback])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        n_seq = len(X_seq)
        train_end = int(n_seq * 0.70)
        val_end = int(n_seq * 0.85)
        
        if train_end == 0:
            train_end = max(1, n_seq - 2) if n_seq >= 3 else 1
            val_end = max(train_end, n_seq - 1) if n_seq >= 2 else train_end
            
        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train
        if len(X_test) == 0:
            X_test, y_test = X_val, y_val
            
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    def prepare_autoencoder_data(self):
        """
        Prepares normalized feature array for Autoencoder training using chronological split:
        70% Train, 15% Validation, 15% Test.
        """
        if self.df is None:
            self.load_data()
            
        X = self.df[AE_FEATURES].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        n = len(X_scaled)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        if train_end == 0:
            train_end = max(1, n - 2) if n >= 3 else 1
            val_end = max(train_end, n - 1) if n >= 2 else train_end
            
        X_train = X_scaled[:train_end]
        X_val = X_scaled[train_end:val_end]
        X_test = X_scaled[val_end:]
        
        if len(X_val) == 0:
            X_val = X_train
        if len(X_test) == 0:
            X_test = X_val
            
        return X_train, X_val, X_test, scaler

