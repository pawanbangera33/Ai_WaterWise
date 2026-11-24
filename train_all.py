"""
Master Model Training Script for AI WaterWise
Generates data (if missing), trains DNN, LSTM, Autoencoder, and Q-Learning models,
and saves artifacts and evaluation metrics into saved_models/
"""

import os
import sys
import numpy as np

# Ensure parent directory is in Python path for backend imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import hashlib

from data.generate_data import generate_water_dataset
from backend.services.preprocessing import DataPreprocessor
from backend.models.dnn_model import DNNConsumptionModel
from backend.models.lstm_model import LSTMForecastingModel
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.models.q_learning import QLearningPumpAgent
from backend.utils.config import DATA_FILE, set_models_metadata, set_active_dataset_info, get_active_dataset_info

def train_all_models():
    print("==================================================")
    print("      AI WaterWise - Master Model Training        ")
    print("==================================================")
    
    # Step 1: Data Check / Generation
    if not os.path.exists(DATA_FILE):
        print("[1/5] Dataset not found. Generating synthetic water dataset...")
        generate_water_dataset()
    else:
        print("[1/5] Dataset found at 'data/water_data.csv'.")
        
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    metrics_summary = {}
    
    # Step 2: Train DNN Regression Model
    print("\n[2/5] Training DNN (Multi-Layer Perceptron) Model...")
    X_tr_dnn, X_val_dnn, X_te_dnn, y_tr_dnn, y_val_dnn, y_te_dnn, s_X_dnn, s_y_dnn, split_info = preprocessor.prepare_dnn_data()
    dnn_model = DNNConsumptionModel()
    dnn_metrics = dnn_model.train(X_tr_dnn, X_val_dnn, X_te_dnn, y_tr_dnn, y_val_dnn, y_te_dnn, s_X_dnn, s_y_dnn, epochs=30)
    metrics_summary["dnn"] = dnn_metrics
    print(f"    -> DNN Metrics: MAE = {dnn_metrics['MAE']}, RMSE = {dnn_metrics['RMSE']}, R² = {dnn_metrics['R2']}")
    
    # Step 3: Train LSTM Demand Forecasting Model
    print("\n[3/5] Training LSTM Time-Series Forecasting Model...")
    try:
        X_tr_lstm, X_val_lstm, X_te_lstm, y_tr_lstm, y_val_lstm, y_te_lstm, s_lstm = preprocessor.prepare_lstm_data(lookback=24)
        lstm_model = LSTMForecastingModel(lookback=24)
        lstm_metrics = lstm_model.train(X_tr_lstm, X_val_lstm, X_te_lstm, y_tr_lstm, y_val_lstm, y_te_lstm, s_lstm, epochs=25)
        metrics_summary["lstm"] = lstm_metrics
        print(f"    -> LSTM Metrics: MAE = {lstm_metrics['MAE']}, RMSE = {lstm_metrics['RMSE']}")
    except Exception as lstm_err:
        metrics_summary["lstm"] = {"error": str(lstm_err), "status": "Insufficient records"}
        print(f"    -> LSTM Warning: {lstm_err}")
    
    # Step 4: Train Autoencoder Anomaly Detector
    print("\n[4/5] Training Autoencoder Anomaly Detector...")
    X_tr_ae, X_val_ae, X_te_ae, s_ae = preprocessor.prepare_autoencoder_data()
    ae_model = AutoencoderAnomalyDetector()
    threshold = ae_model.train(X_tr_ae, X_val_ae, X_te_ae, s_ae, epochs=30)
    metrics_summary["autoencoder"] = {"threshold": round(threshold, 4)}
    print(f"    -> Dynamic Anomaly Loss Threshold = {round(threshold, 4)}")
    
    # Step 5: Train Q-Learning Intelligent Agent
    print("\n[5/5] Training Tabular Q-Learning Agent...")
    q_agent = QLearningPumpAgent()
    q_agent.train_agent(episodes=3000)
    metrics_summary["q_learning"] = {"episodes": 3000, "states_learned": len(q_agent.q_table)}
    
    # Save active dataset info and models metadata linkage
    with open(DATA_FILE, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    ds_info = get_active_dataset_info() or {}
    dataset_id = ds_info.get("dataset_id", "default_realtime_ds")
    filename = ds_info.get("filename", "water_data.csv (Real-Time Dataset)")
    
    active_metadata = {
        "dataset_id": dataset_id,
        "filename": filename,
        "filepath": DATA_FILE,
        "upload_timestamp": ds_info.get("upload_timestamp", "System Default"),
        "file_hash": file_hash,
        "records": len(df),
        "columns": list(df.columns),
        "upload_status": "success",
        "is_processed": True,
        "is_trained": True,
        "split_info": split_info,
        "start_date": str(df['timestamp'].min()),
        "end_date": str(df['timestamp'].max())
    }
    set_active_dataset_info(active_metadata)
    
    set_models_metadata({
        "dataset_id": dataset_id,
        "dataset_hash": file_hash,
        "trained_at": datetime.now().isoformat(),
        "filename": filename,
        "metrics": metrics_summary
    })
    
    print("\n==================================================")
    print("      ALL MODELS TRAINED & SAVED SUCCESSFULLY     ")
    print("==================================================")
    print(f"Saved location: {os.path.abspath('saved_models/')}\n")

if __name__ == "__main__":
    train_all_models()
