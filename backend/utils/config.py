"""
Configuration and constants for AI WaterWise.
Handles file paths, dataset schema, active dataset tracking, and model parameters.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "water_data.csv")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
ACTIVE_DATASET_JSON = os.path.join(DATA_DIR, "active_dataset.json")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Required CSV Columns for validation
REQUIRED_COLUMNS = [
    'timestamp',
    'temperature',
    'humidity',
    'occupancy',
    'day_of_week',
    'is_holiday',
    'tank_level',
    'water_consumption'
]

# Feature definitions for DNN regression
DNN_FEATURES = [
    'temperature',
    'humidity',
    'occupancy',
    'day_of_week',
    'is_holiday',
    'tank_level',
    'prev_consumption'
]
DNN_TARGET = 'water_consumption'

# Feature definitions for Autoencoder Anomaly Detection
AE_FEATURES = [
    'temperature',
    'humidity',
    'occupancy',
    'tank_level',
    'water_consumption'
]

# Saved artifact paths
DNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "dnn_model.keras")
DNN_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "dnn_scaler.pkl")
DNN_METRICS_PATH = os.path.join(SAVED_MODELS_DIR, "dnn_metrics.json")

LSTM_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_model.keras")
LSTM_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_scaler.pkl")
LSTM_METRICS_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_metrics.json")

AE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "autoencoder_model.keras")
AE_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "ae_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(SAVED_MODELS_DIR, "ae_threshold.json")

Q_TABLE_PATH = os.path.join(SAVED_MODELS_DIR, "q_table.json")
MODELS_METADATA_JSON = os.path.join(SAVED_MODELS_DIR, "models_metadata.json")

# Helper functions for managing active uploaded dataset metadata
def get_default_dataset_info():
    """Generates default tracking metadata for real-time dataset (water_data.csv)."""
    csv_path = os.path.join(DATA_DIR, "water_data.csv")
    if not os.path.exists(csv_path):
        try:
            from data.generate_data import generate_water_dataset
            generate_water_dataset()
        except Exception as e:
            print(f"[!] Default dataset generation warning: {e}")

    file_hash = ""
    records = 0
    start_date = None
    end_date = None

    if os.path.exists(csv_path):
        try:
            import hashlib
            with open(csv_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            import pandas as pd
            df = pd.read_csv(csv_path)
            records = len(df)
            if 'timestamp' in df.columns and len(df) > 0:
                start_date = str(df['timestamp'].min())
                end_date = str(df['timestamp'].max())
        except Exception:
            pass

    model_meta = get_models_metadata()
    is_trained = False
    if model_meta and model_meta.get("dataset_hash") == file_hash and model_meta.get("dataset_id") == "default_realtime_ds":
        is_trained = True

    return {
        "dataset_id": "default_realtime_ds",
        "filename": "water_data.csv (Real-Time Dataset)",
        "filepath": csv_path,
        "upload_timestamp": "System Default",
        "file_hash": file_hash,
        "records": records,
        "upload_status": "success",
        "is_processed": True,
        "is_trained": is_trained,
        "is_default": True,
        "start_date": start_date,
        "end_date": end_date,
        "split_info": {
            "train_records": int(records * 0.70),
            "val_records": int(records * 0.15),
            "test_records": records - int(records * 0.70) - int(records * 0.15)
        } if records > 0 else None
    }

def get_active_dataset_info():
    """Reads tracking metadata from active_dataset.json or returns default real-time dataset info."""
    if os.path.exists(ACTIVE_DATASET_JSON):
        try:
            with open(ACTIVE_DATASET_JSON, 'r') as f:
                info = json.load(f)
                if isinstance(info, dict) and info.get("has_active_dataset", True) and "filename" in info:
                    return info
        except Exception:
            pass
    return get_default_dataset_info()

def set_active_dataset_info(info_dict):
    """Saves metadata tracking dictionary to active_dataset.json."""
    with open(ACTIVE_DATASET_JSON, 'w') as f:
        json.dump(info_dict, f, indent=2)

def clear_active_dataset_info():
    """Clears active dataset tracking."""
    if os.path.exists(ACTIVE_DATASET_JSON):
        try:
            os.remove(ACTIVE_DATASET_JSON)
        except Exception:
            pass

def get_active_csv_path():
    """Returns absolute file path to the currently active CSV dataset."""
    info = get_active_dataset_info()
    if info and "filepath" in info:
        path = info["filepath"]
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        if os.path.exists(path):
            return path
    default_path = os.path.join(DATA_DIR, "water_data.csv")
    if os.path.exists(default_path):
        return default_path
    return None

# Helper functions for model versioning and dataset-model linkage metadata
def get_models_metadata():
    """Reads model metadata linking trained models to dataset_id and file_hash."""
    if os.path.exists(MODELS_METADATA_JSON):
        try:
            with open(MODELS_METADATA_JSON, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def set_models_metadata(meta_dict):
    """Saves model metadata linking trained models to dataset_id and file_hash."""
    with open(MODELS_METADATA_JSON, 'w') as f:
        json.dump(meta_dict, f, indent=2)

def clear_models_metadata():
    """Invalidates all trained model metadata on new dataset upload."""
    if os.path.exists(MODELS_METADATA_JSON):
        try:
            os.remove(MODELS_METADATA_JSON)
        except Exception:
            pass

def verify_models_for_current_dataset():
    """
    Verifies that saved trained models were trained on the CURRENT active dataset hash and ID.
    Returns True if current dataset matches model dataset hash, False otherwise.
    """
    ds_info = get_active_dataset_info()
    if not ds_info or not ds_info.get("is_trained", False):
        return False
    
    current_hash = ds_info.get("file_hash")
    current_id = ds_info.get("dataset_id")
    if not current_hash or not current_id:
        return False
        
    model_meta = get_models_metadata()
    if not model_meta:
        return False
        
    if model_meta.get("dataset_hash") == current_hash and model_meta.get("dataset_id") == current_id:
        return True
        
    return False

