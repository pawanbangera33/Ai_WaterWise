"""
FastAPI Router for LSTM Time-Series Demand Forecasting
"""

from fastapi import APIRouter, HTTPException
from backend.services.preprocessing import DataPreprocessor
from backend.models.lstm_model import LSTMForecastingModel
from backend.utils.config import verify_models_for_current_dataset

router = APIRouter()
lstm_model = LSTMForecastingModel()
preprocessor = DataPreprocessor()

@router.get("/forecast")
def get_demand_forecast(horizon: int = 24):
    """Returns N-hour ahead demand forecast from LSTM model."""
    if not verify_models_for_current_dataset():
        raise HTTPException(
            status_code=400,
            detail="Models must be retrained for the current dataset."
        )
        
    try:
        df = preprocessor.load_data()
        if len(df['water_consumption']) < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient sequential data for LSTM forecasting. Uploaded dataset has {len(df)} records, minimum required is 29."
            )
        recent_seq = df['water_consumption'].values[-24:].tolist()
        forecast = lstm_model.forecast_next_hours(recent_seq, horizon=horizon)
        return {
            "horizon_hours": horizon,
            "forecast": forecast,
            "metrics": lstm_model.metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

