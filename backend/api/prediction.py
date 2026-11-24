"""
FastAPI Router for DNN Water Consumption Prediction
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.models.dnn_model import DNNConsumptionModel
from backend.utils.config import verify_models_for_current_dataset

router = APIRouter()
dnn_model = DNNConsumptionModel()

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    occupancy: int
    day_of_week: int
    is_holiday: int
    tank_level: float
    prev_consumption: float

@router.post("/predict")
def predict_consumption(req: PredictionRequest):
    """Predicts consumption using DNN for user-specified input parameters."""
    if not verify_models_for_current_dataset():
        raise HTTPException(
            status_code=400,
            detail="Models must be retrained for the current dataset."
        )
        
    feat_dict = req.model_dump()
    try:
        prediction = dnn_model.predict(feat_dict)
        return {
            "status": "success",
            "inputs": feat_dict,
            "predicted_consumption": prediction,
            "metrics": dnn_model.metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

