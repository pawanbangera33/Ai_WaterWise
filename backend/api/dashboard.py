"""
FastAPI Dashboard Router
Aggregates real-time view from the currently active uploaded dataset.
"""

from fastapi import APIRouter, HTTPException
from backend.utils.config import (
    get_active_dataset_info, get_active_csv_path, verify_models_for_current_dataset, get_models_metadata
)
from backend.services.preprocessing import DataPreprocessor
from backend.models.dnn_model import DNNConsumptionModel
from backend.models.lstm_model import LSTMForecastingModel
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.services.bayesian import BayesianLeakageEstimator
from backend.services.recommendation import PumpRecommendationService

router = APIRouter()

dnn_model = DNNConsumptionModel()
lstm_model = LSTMForecastingModel()
ae_model = AutoencoderAnomalyDetector()
bayes_estimator = BayesianLeakageEstimator()
rec_service = PumpRecommendationService()
preprocessor = DataPreprocessor()

@router.get("/dashboard")
def get_dashboard_summary():
    """
    Returns real-time system state from the latest valid row of the active uploaded dataset,
    combined with DNN, LSTM, Autoencoder, Bayesian, and Q-Agent outputs.
    Strictly enforces dataset-to-model hash verification.
    """
    dataset_info = get_active_dataset_info()
    csv_path = get_active_csv_path()
    
    if not dataset_info or not csv_path:
        return {
            "has_active_dataset": False,
            "state": "NO_DATASET",
            "message": "Please upload a CSV dataset to begin."
        }

    try:
        df = preprocessor.load_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Latest row in the uploaded dataset represents current reading
    latest_row = df.iloc[-1].to_dict()

    dnn_features = {
        'temperature': float(latest_row['temperature']),
        'humidity': float(latest_row['humidity']),
        'occupancy': int(latest_row['occupancy']),
        'day_of_week': int(latest_row['day_of_week']),
        'is_holiday': int(latest_row['is_holiday']),
        'tank_level': float(latest_row['tank_level']),
        'prev_consumption': float(latest_row['prev_consumption'])
    }

    ae_features = {
        'temperature': float(latest_row['temperature']),
        'humidity': float(latest_row['humidity']),
        'occupancy': int(latest_row['occupancy']),
        'tank_level': float(latest_row['tank_level']),
        'water_consumption': float(latest_row['water_consumption'])
    }

    # Verify model dataset hash matches current active dataset
    models_trained = verify_models_for_current_dataset()

    # 1. DNN Prediction
    if models_trained and dnn_model.load():
        predicted_consumption = dnn_model.predict(dnn_features)
    else:
        predicted_consumption = None

    # 2. LSTM Forecast
    forecast_values = []
    forecast_note = None
    if models_trained and lstm_model.load():
        if len(df['water_consumption']) >= 24:
            last_24 = df['water_consumption'].values[-24:].tolist()
            forecast_values = lstm_model.forecast_next_hours(last_24, horizon=24)
        else:
            forecast_note = f"Insufficient sequential data for LSTM forecasting. Dataset has {len(df)} records, minimum required is 29."

    # 3. Autoencoder Anomaly Detection
    if models_trained and ae_model.load():
        ae_res = ae_model.detect(ae_features)
        try:
            rec_err = float(ae_res['reconstruction_error'])
            thresh = float(ae_res['threshold'])
            ae_loss_pct = round(min(100.0, (rec_err / thresh) * 100.0), 1) if thresh > 0 else 0.0
        except Exception:
            ae_loss_pct = 0.0
            
        if ae_res['status'] == "ANOMALY":
            ae_explanation = "Anomaly Detected: Current water sensor parameters deviate significantly from learned normal operating patterns (Loss exceeds threshold)."
        else:
            ae_explanation = "Normal Pattern: Sensor parameters closely match learned normal baseline water consumption behavior."
    else:
        ae_res = {"reconstruction_error": "--", "threshold": "--", "status": "UNTRAINED", "is_anomaly": False}
        ae_loss_pct = 0.0
        ae_explanation = "Train models to compute Autoencoder pattern reconstruction."

    # 4. Bayesian Leakage Risk Estimation
    bayes_res = bayes_estimator.calculate_leakage_probability(
        is_anomaly=ae_res.get('is_anomaly', False),
        consumption=float(latest_row['water_consumption']),
        occupancy=int(latest_row['occupancy']),
        tank_level=float(latest_row['tank_level']),
        temperature=float(latest_row['temperature'])
    )

    # 5. Q-Agent Pump Recommendation
    if models_trained:
        pred_val = predicted_consumption if predicted_consumption is not None else float(latest_row['water_consumption'])
        rec_res = rec_service.get_pump_recommendation(
            tank_level=float(latest_row['tank_level']),
            predicted_demand=float(pred_val),
            current_consumption=float(latest_row['water_consumption']),
            leakage_probability=bayes_res['probability']
        )
    else:
        rec_res = {
            "recommended_action": "Models Not Trained For Current Dataset",
            "reason": "Models have not been trained for this dataset. Click 'Train Models' to process.",
            "state": None
        }

    # Metrics retrieved from models_metadata.json
    models_meta = get_models_metadata()
    metrics = models_meta.get("metrics") if (models_trained and models_meta) else None

    # System status calculation
    if not models_trained:
        system_status = "UNTRAINED"
    elif ae_res['status'] == "ANOMALY" or bayes_res['probability'] > 0.6:
        system_status = "ANOMALY"
    elif bayes_res['probability'] > 0.35 or float(latest_row['tank_level']) < 20.0:
        system_status = "WARNING"
    else:
        system_status = "NORMAL"

    return {
        "has_active_dataset": True,
        "dataset_name": dataset_info["filename"],
        "dataset_id": dataset_info.get("dataset_id"),
        "records_count": len(df),
        "is_processed": dataset_info.get("is_processed", False),
        "models_trained": models_trained,
        "timestamp": str(latest_row['timestamp']),
        "current_consumption": float(latest_row['water_consumption']),
        "temperature": float(latest_row['temperature']),
        "humidity": float(latest_row['humidity']),
        "occupancy": int(latest_row['occupancy']),
        "tank_level": float(latest_row['tank_level']),
        "predicted_consumption": predicted_consumption,
        "forecast": forecast_values,
        "forecast_note": forecast_note,
        "anomaly_status": ae_res['status'],
        "anomaly_score": ae_res['reconstruction_error'],
        "anomaly_threshold": ae_res['threshold'],
        "ae_loss_pct": ae_loss_pct,
        "ae_explanation": ae_explanation,
        "leakage_probability": bayes_res['probability'],
        "leakage_percentage": bayes_res['percentage'],
        "leakage_explanation": bayes_res['explanation'],
        "system_status": system_status,
        "recommended_action": rec_res['recommended_action'],
        "reason": rec_res['reason'],
        "q_state": rec_res.get('state'),
        "metrics": metrics,
        "split_info": dataset_info.get("split_info")
    }

@router.get("/history")
def get_historical_data(limit: int = 50):
    """Returns historical records from active dataset for dashboard line charts."""
    dataset_info = get_active_dataset_info()
    csv_path = get_active_csv_path()
    if not dataset_info or not csv_path:
        return {"count": 0, "data": []}
    try:
        df = preprocessor.load_data()
        records = df.tail(limit).to_dict(orient='records')
        return {"count": len(records), "data": records}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

