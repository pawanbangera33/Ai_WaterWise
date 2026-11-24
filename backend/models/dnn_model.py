"""
DNN / MLP Model for Water Consumption Regression
Built using TensorFlow / Keras.
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.utils.config import (
    DNN_MODEL_PATH, DNN_SCALER_PATH, DNN_METRICS_PATH, DNN_FEATURES
)

class DNNConsumptionModel:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        
    def build_model(self, input_dim):
        """Constructs a feed-forward neural network (MLP)."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        self.model = model
        return model

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, epochs=30, batch_size=32):
        """Trains the DNN model on train split, validates on val split, and evaluates performance metrics strictly on test split."""
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        input_dim = X_train.shape[1]
        self.build_model(input_dim)
        
        val_data = (X_val, y_val) if len(X_val) > 0 else None
        self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Evaluate performance metrics strictly on TEST DATA ONLY
        y_pred_scaled = self.model.predict(X_test, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        
        self.metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }
        
        self.save()
        return self.metrics


    def predict(self, feature_dict):
        """
        Predicts consumption given a dictionary of raw input features.
        Expects keys: ['temperature', 'humidity', 'occupancy', 'day_of_week', 'is_holiday', 'tank_level', 'prev_consumption']
        """
        if self.model is None or self.scaler_X is None:
            self.load()
            
        raw_vals = [feature_dict[feat] for feat in DNN_FEATURES]
        X_in = np.array(raw_vals).reshape(1, -1)
        X_scaled = self.scaler_X.transform(X_in)
        
        pred_scaled = self.model.predict(X_scaled, verbose=0).flatten()
        pred_val = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        return round(float(pred_val), 2)

    def save(self):
        """Saves model weight artifacts and metric benchmarks."""
        os.makedirs(os.path.dirname(DNN_MODEL_PATH), exist_ok=True)
        self.model.save(DNN_MODEL_PATH)
        joblib.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, DNN_SCALER_PATH)
        with open(DNN_METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load(self):
        """Loads serialized model and pre-fitted scalers."""
        if os.path.exists(DNN_MODEL_PATH) and os.path.exists(DNN_SCALER_PATH):
            self.model = load_model(DNN_MODEL_PATH, compile=False)
            scalers = joblib.load(DNN_SCALER_PATH)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
            if os.path.exists(DNN_METRICS_PATH):
                with open(DNN_METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
            return True
        return False
