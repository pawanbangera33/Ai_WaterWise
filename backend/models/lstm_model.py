"""
LSTM Time-Series Model for Future Water Demand Forecasting
Built using TensorFlow / Keras.
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.utils.config import (
    LSTM_MODEL_PATH, LSTM_SCALER_PATH, LSTM_METRICS_PATH
)

class LSTMForecastingModel:
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.metrics = {}

    def build_model(self, lookback):
        """Constructs an LSTM sequential architecture for time series demand forecasting."""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        self.model = model
        return model

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test, scaler, epochs=25, batch_size=32):
        """Trains LSTM on train split, validates on val split, and evaluates performance metrics strictly on test set."""
        self.scaler = scaler
        self.build_model(self.lookback)
        
        val_data = (X_val, y_val) if len(X_val) > 0 else None
        self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Evaluate performance metrics strictly on TEST DATA ONLY
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler.inverse_transform(y_test).flatten()
        
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        self.metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        }
        
        self.save()
        return self.metrics


    def forecast_next_hours(self, last_sequence, horizon=24):
        """
        Recursively predicts the next N hours of demand given the last sequence.
        last_sequence shape: (lookback,) raw consumption values.
        """
        if self.model is None or self.scaler is None:
            self.load()
            
        scaled_seq = self.scaler.transform(np.array(last_sequence).reshape(-1, 1)).flatten()
        curr_seq = list(scaled_seq[-self.lookback:])
        
        predictions = []
        for _ in range(horizon):
            in_arr = np.array(curr_seq[-self.lookback:]).reshape(1, self.lookback, 1)
            pred_scaled = self.model.predict(in_arr, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            curr_seq.append(pred_scaled)
            
        unscaled_preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return [round(float(val), 2) for val in unscaled_preds]

    def save(self):
        """Saves model weights and evaluation metrics."""
        os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
        self.model.save(LSTM_MODEL_PATH)
        joblib.dump(self.scaler, LSTM_SCALER_PATH)
        with open(LSTM_METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load(self):
        """Loads serialized model and scaler."""
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
            self.model = load_model(LSTM_MODEL_PATH, compile=False)
            self.scaler = joblib.load(LSTM_SCALER_PATH)
            if os.path.exists(LSTM_METRICS_PATH):
                with open(LSTM_METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
            return True
        return False
