"""
FastAPI Router for Q-Learning Pump Action Recommendation
"""

from fastapi import APIRouter, HTTPException
from backend.services.preprocessing import DataPreprocessor
from backend.models.dnn_model import DNNConsumptionModel
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.services.bayesian import BayesianLeakageEstimator
from backend.services.recommendation import PumpRecommendationService
from backend.utils.config import verify_models_for_current_dataset

router = APIRouter()
dnn_model = DNNConsumptionModel()
ae_model = AutoencoderAnomalyDetector()
bayes_estimator = BayesianLeakageEstimator()
rec_service = PumpRecommendationService()
preprocessor = DataPreprocessor()

@router.get("/recommendation")
def get_recommendation():
    """Returns Q-agent pump decision and explanation."""
    if not verify_models_for_current_dataset():
        raise HTTPException(
            status_code=400,
            detail="Models must be retrained for the current dataset."
        )
        
    try:
        df = preprocessor.load_data()
        latest = df.iloc[-1].to_dict()
        
        dnn_features = {
            'temperature': float(latest['temperature']),
            'humidity': float(latest['humidity']),
            'occupancy': int(latest['occupancy']),
            'day_of_week': int(latest['day_of_week']),
            'is_holiday': int(latest['is_holiday']),
            'tank_level': float(latest['tank_level']),
            'prev_consumption': float(latest['prev_consumption'])
        }
        
        ae_features = {
            'temperature': float(latest['temperature']),
            'humidity': float(latest['humidity']),
            'occupancy': int(latest['occupancy']),
            'tank_level': float(latest['tank_level']),
            'water_consumption': float(latest['water_consumption'])
        }
        
        try:
            pred_demand = dnn_model.predict(dnn_features)
        except Exception:
            pred_demand = float(latest['water_consumption'])
            
        try:
            ae_res = ae_model.detect(ae_features)
        except Exception:
            ae_res = {"is_anomaly": False}
            
        bayes_res = bayes_estimator.calculate_leakage_probability(
            is_anomaly=ae_res.get('is_anomaly', False),
            consumption=float(latest['water_consumption']),
            occupancy=int(latest['occupancy']),
            tank_level=float(latest['tank_level']),
            temperature=float(latest['temperature'])
        )
        
        rec = rec_service.get_pump_recommendation(
            tank_level=float(latest['tank_level']),
            predicted_demand=float(pred_demand),
            current_consumption=float(latest['water_consumption']),
            leakage_probability=bayes_res['probability']
        )
        return rec
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

