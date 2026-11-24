"""
Autoencoder Anomaly Detection Model
Built using TensorFlow / Keras.
Calculates dynamic reconstruction error threshold to detect abnormal consumption patterns.
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

from backend.utils.config import (
    AE_MODEL_PATH, AE_SCALER_PATH, AE_THRESHOLD_PATH, AE_FEATURES
)

class AutoencoderAnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = 0.05
        
    def build_model(self, input_dim):
        """Constructs an Encoder-Decoder Autoencoder network."""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(16, activation='relu')(input_layer)
        bottleneck = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(bottleneck)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        
        self.model = model
        return model

    def train(self, X_train, X_val, X_test, scaler, epochs=30, batch_size=32):
        """
        Trains the Autoencoder on baseline data.
        Derives reconstruction threshold dynamically from validation/test loss distribution.
        """
        self.scaler = scaler
        input_dim = X_train.shape[1]
        self.build_model(input_dim)
        
        val_data = (X_val, X_val) if len(X_val) > 0 else None
        self.model.fit(
            X_train, X_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Determine dynamic threshold from test reconstruction error (e.g., 97th percentile)
        eval_set = X_test if len(X_test) > 0 else X_val
        reconstructions = self.model.predict(eval_set, verbose=0)
        test_loss = np.mean(np.square(eval_set - reconstructions), axis=1)
        
        # Dynamic calculation: 97th percentile threshold
        self.threshold = float(np.percentile(test_loss, 97)) if len(test_loss) > 0 else 0.05
        
        self.save()
        return self.threshold


    def detect(self, feature_dict):
        """
        Evaluates a single sample observation for anomaly status.
        Returns dict containing: reconstruction_error, threshold, status ('NORMAL' or 'ANOMALY')
        """
        if self.model is None or self.scaler is None:
            self.load()
            
        raw_vals = [feature_dict[feat] for feat in AE_FEATURES]
        X_in = np.array(raw_vals).reshape(1, -1)
        X_scaled = self.scaler.transform(X_in)
        
        reconstruction = self.model.predict(X_scaled, verbose=0)
        rec_error = float(np.mean(np.square(X_scaled - reconstruction)))
        
        is_anomaly = rec_error > self.threshold
        status = "ANOMALY" if is_anomaly else "NORMAL"
        
        return {
            "reconstruction_error": round(rec_error, 4),
            "threshold": round(self.threshold, 4),
            "status": status,
            "is_anomaly": is_anomaly
        }

    def save(self):
        """Saves Autoencoder model, scaler, and dynamic threshold."""
        os.makedirs(os.path.dirname(AE_MODEL_PATH), exist_ok=True)
        self.model.save(AE_MODEL_PATH)
        joblib.dump(self.scaler, AE_SCALER_PATH)
        with open(AE_THRESHOLD_PATH, 'w') as f:
            json.dump({"threshold": self.threshold}, f, indent=2)

    def load(self):
        """Loads Autoencoder model, scaler, and dynamic threshold."""
        if os.path.exists(AE_MODEL_PATH) and os.path.exists(AE_SCALER_PATH):
            self.model = load_model(AE_MODEL_PATH, compile=False)
            self.scaler = joblib.load(AE_SCALER_PATH)
            if os.path.exists(AE_THRESHOLD_PATH):
                with open(AE_THRESHOLD_PATH, 'r') as f:
                    self.threshold = json.load(f).get("threshold", 0.05)
            return True
        return False
