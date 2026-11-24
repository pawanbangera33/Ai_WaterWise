import os
import shutil
import io
import time
import uuid
import hashlib
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from backend.utils.config import (
    UPLOADS_DIR, DATA_DIR, set_active_dataset_info, get_active_dataset_info,
    get_active_csv_path, clear_active_dataset_info, REQUIRED_COLUMNS, clear_models_metadata, set_models_metadata,
    get_models_metadata, verify_models_for_current_dataset
)
from backend.services.preprocessing import DataPreprocessor
from backend.models.dnn_model import DNNConsumptionModel
from backend.models.lstm_model import LSTMForecastingModel
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.models.q_learning import QLearningPumpAgent

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Endpoint for uploading user water-consumption CSV dataset.
    Generates dataset_id, computes file SHA-256 hash, invalidates previous model metadata,
    and sets new dataset as active.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files (.csv) are accepted.")
        
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
        
    # Validate required columns
    is_valid, missing = DataPreprocessor.validate_dataframe(df)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV structure. Missing required columns: {', '.join(missing)}"
        )
        
    # Generate unique dataset_id and SHA-256 file hash
    file_hash = hashlib.sha256(contents).hexdigest()
    dataset_id = f"ds_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    saved_filename = file.filename
    dest_path = os.path.join(UPLOADS_DIR, saved_filename)
    
    with open(dest_path, "wb") as f:
        f.write(contents)
        
    # IMMEDIATELY INVALIDATE PREVIOUS MODEL RESULTS
    clear_models_metadata()
    
    metadata = {
        "dataset_id": dataset_id,
        "filename": saved_filename,
        "filepath": dest_path,
        "upload_timestamp": datetime.now().isoformat(),
        "file_hash": file_hash,
        "records": len(df),
        "columns": list(df.columns),
        "upload_status": "success",
        "is_processed": False,
        "is_trained": False,
        "split_info": None,
        "start_date": None,
        "end_date": None
    }
    set_active_dataset_info(metadata)
    
    return metadata

@router.post("/process")
def process_dataset():
    """
    Cleans, parses datetimes, sorts chronologically, and prepares 70/15/15 chronological data splits.
    """
    active_info = get_active_dataset_info()
    if not active_info or not get_active_csv_path():
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a CSV file to begin.")
        
    preprocessor = DataPreprocessor()
    try:
        df = preprocessor.load_data()
        
        # Chronological splits preparation
        X_tr_dnn, X_val_dnn, X_te_dnn, y_tr_dnn, y_val_dnn, y_te_dnn, s_X_dnn, s_y_dnn, split_info = preprocessor.prepare_dnn_data()
        
        # Prepare Autoencoder split
        X_tr_ae, X_val_ae, X_te_ae, s_ae = preprocessor.prepare_autoencoder_data()
        
        # Prepare LSTM split if record count permits
        lstm_ready = True
        lstm_error_msg = None
        try:
            X_tr_lstm, X_val_lstm, X_te_lstm, y_tr_lstm, y_val_lstm, y_te_lstm, s_lstm = preprocessor.prepare_lstm_data(lookback=24)
            lstm_seq_shape = list(X_tr_lstm.shape)
        except Exception as lstm_err:
            lstm_ready = False
            lstm_error_msg = str(lstm_err)
            lstm_seq_shape = []
            
        latest_row = df.iloc[-1].to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")
        
    # Update active dataset metadata
    active_info["is_processed"] = True
    active_info["records"] = len(df)
    active_info["start_date"] = str(df['timestamp'].min())
    active_info["end_date"] = str(df['timestamp'].max())
    active_info["split_info"] = split_info
    set_active_dataset_info(active_info)
    
    return {
        "status": "success",
        "message": "Dataset processed successfully with 70/15/15 chronological split.",
        "dataset_id": active_info["dataset_id"],
        "filename": active_info["filename"],
        "records_total": len(df),
        "records_valid": len(df),
        "records_removed": 0,
        "date_range": f"{active_info['start_date']} to {active_info['end_date']}",
        "columns": list(df.columns),
        "split_info": split_info,
        "lstm_ready": lstm_ready,
        "lstm_status_note": lstm_error_msg,
        "dnn_features_count": X_tr_dnn.shape[1],
        "ae_features_count": X_tr_ae.shape[1],
        "processing_status": "PROCESSED"
    }

@router.post("/train")
def train_models():
    """
    Trains DNN, LSTM, Autoencoder, and Q-Learning agent on current active real-time dataset.
    Evaluates metrics strictly on TEST split and saves models with dataset ID and file hash metadata.
    """
    active_info = get_active_dataset_info()
    csv_path = get_active_csv_path()
    if not active_info or not csv_path or not os.path.exists(csv_path):
        from data.generate_data import generate_water_dataset
        generate_water_dataset()
        active_info = get_active_dataset_info()
        csv_path = get_active_csv_path()

    preprocessor = DataPreprocessor()
    try:
        df = preprocessor.load_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data loading error: {str(e)}")
        
    # Auto-process if not marked as processed
    if not active_info.get("is_processed", False):
        try:
            _, _, _, _, _, _, _, _, split_info = preprocessor.prepare_dnn_data()
            active_info["is_processed"] = True
            active_info["records"] = len(df)
            active_info["start_date"] = str(df['timestamp'].min())
            active_info["end_date"] = str(df['timestamp'].max())
            active_info["split_info"] = split_info
            set_active_dataset_info(active_info)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")
        
    metrics_summary = {}
    
    # 1. Train DNN Model (Evaluates strictly on Test set)
    X_tr_dnn, X_val_dnn, X_te_dnn, y_tr_dnn, y_val_dnn, y_te_dnn, s_X_dnn, s_y_dnn, split_info = preprocessor.prepare_dnn_data()
    dnn = DNNConsumptionModel()
    dnn_metrics = dnn.train(X_tr_dnn, X_val_dnn, X_te_dnn, y_tr_dnn, y_val_dnn, y_te_dnn, s_X_dnn, s_y_dnn, epochs=25)
    metrics_summary["dnn"] = dnn_metrics
    
    # 2. Train LSTM Model (If dataset sequence length permits)
    try:
        X_tr_lstm, X_val_lstm, X_te_lstm, y_tr_lstm, y_val_lstm, y_te_lstm, s_lstm = preprocessor.prepare_lstm_data(lookback=24)
        lstm = LSTMForecastingModel(lookback=24)
        lstm_metrics = lstm.train(X_tr_lstm, X_val_lstm, X_te_lstm, y_tr_lstm, y_val_lstm, y_te_lstm, s_lstm, epochs=20)
        metrics_summary["lstm"] = lstm_metrics
    except Exception as lstm_err:
        metrics_summary["lstm"] = {"error": str(lstm_err), "status": "Insufficient records"}
    
    # 3. Train Autoencoder Anomaly Detector
    X_tr_ae, X_val_ae, X_te_ae, s_ae = preprocessor.prepare_autoencoder_data()
    ae = AutoencoderAnomalyDetector()
    threshold = ae.train(X_tr_ae, X_val_ae, X_te_ae, s_ae, epochs=25)
    metrics_summary["autoencoder"] = {"threshold": round(threshold, 4)}
    
    # 4. Train Q-Learning Agent
    q_agent = QLearningPumpAgent()
    q_agent.train_agent(episodes=3000)
    metrics_summary["q_learning"] = {"episodes": 3000, "states_learned": len(q_agent.q_table)}
    
    # Link models metadata to dataset_id and file_hash
    set_models_metadata({
        "dataset_id": active_info["dataset_id"],
        "dataset_hash": active_info["file_hash"],
        "trained_at": datetime.now().isoformat(),
        "filename": active_info["filename"],
        "metrics": metrics_summary
    })
    
    active_info["is_trained"] = True
    set_active_dataset_info(active_info)
    
    return {
        "status": "success",
        "message": "All models trained successfully for current dataset.",
        "dataset_id": active_info["dataset_id"],
        "filename": active_info["filename"],
        "metrics": metrics_summary
    }

@router.get("/dataset-status")
def get_dataset_status():
    """Returns metadata status for the currently active uploaded dataset."""
    info = get_active_dataset_info()
    if not info:
        return {
            "has_active_dataset": False,
            "message": "Please upload a CSV dataset to begin."
        }
    
    models_valid = verify_models_for_current_dataset()
    return {
        "has_active_dataset": True,
        "dataset_info": info,
        "models_valid_for_current": models_valid
    }

@router.get("/dataset")
def get_dataset():
    """Returns active dataset tracking metadata."""
    info = get_active_dataset_info()
    if not info:
        return {"has_active_dataset": False, "message": "No dataset uploaded."}
    return info

@router.get("/model-status")
def get_model_status():
    """Returns whether saved model artifacts belong to current dataset hash."""
    info = get_active_dataset_info()
    models_meta = get_models_metadata()
    is_valid = verify_models_for_current_dataset()
    
    return {
        "is_trained_for_current_dataset": is_valid,
        "active_dataset_id": info.get("dataset_id") if info else None,
        "model_dataset_id": models_meta.get("dataset_id") if models_meta else None,
        "active_dataset_hash": info.get("file_hash") if info else None,
        "model_dataset_hash": models_meta.get("dataset_hash") if models_meta else None
    }

@router.get("/download-sample")
def download_sample_csv():
    """Serves sample water data CSV strictly as a downloadable template file."""
    sample_path = os.path.join(DATA_DIR, "water_data.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample CSV template not found on server.")
    return FileResponse(
        sample_path,
        filename="sample_water_data_format.csv",
        media_type="text/csv"
    )

@router.post("/reset-dataset")
def reset_dataset():
    """
    Resets the active dataset tracking and model metadata,
    allowing the user to start fresh with a clean state.
    """
    clear_active_dataset_info()
    clear_models_metadata()
    return {
        "status": "success",
        "message": "Dataset and model state cleared. You can start fresh now."
    }


