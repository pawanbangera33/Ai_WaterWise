"""
FastAPI Router for Autoencoder Anomaly Detection & Bayesian Risk Estimation
"""

from fastapi import APIRouter, HTTPException
from backend.services.preprocessing import DataPreprocessor
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.services.bayesian import BayesianLeakageEstimator
from backend.utils.config import verify_models_for_current_dataset

router = APIRouter()
ae_model = AutoencoderAnomalyDetector()
bayes_estimator = BayesianLeakageEstimator()
preprocessor = DataPreprocessor()

@router.get("/anomaly")
def get_anomaly_status():
    """Returns anomaly detection scores and Bayesian risk probability for latest sample."""
    if not verify_models_for_current_dataset():
        raise HTTPException(
            status_code=400,
            detail="Models must be retrained for the current dataset."
        )
        
    try:
        df = preprocessor.load_data()
        latest = df.iloc[-1].to_dict()
        
        ae_features = {
            'temperature': float(latest['temperature']),
            'humidity': float(latest['humidity']),
            'occupancy': int(latest['occupancy']),
            'tank_level': float(latest['tank_level']),
            'water_consumption': float(latest['water_consumption'])
        }
        
        ae_result = ae_model.detect(ae_features)
        bayes_result = bayes_estimator.calculate_leakage_probability(
            is_anomaly=ae_result['is_anomaly'],
            consumption=float(latest['water_consumption']),
            occupancy=int(latest['occupancy']),
            tank_level=float(latest['tank_level']),
            temperature=float(latest['temperature'])
        )
        
        return {
            "anomaly_analysis": ae_result,
            "bayesian_leakage_risk": bayes_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

