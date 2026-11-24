"""
Main FastAPI Application Entry Point for AI WaterWise
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.dataset_pipeline import router as dataset_pipeline_router
from backend.api.dashboard import router as dashboard_router
from backend.api.prediction import router as prediction_router
from backend.api.forecasting import router as forecasting_router
from backend.api.anomaly import router as anomaly_router
from backend.api.recommendation import router as recommendation_router
from backend.api.simulation import router as simulation_router

app = FastAPI(
    title="AI WaterWise API",
    description="Intelligent Water Demand Forecasting and Resource Management System",
    version="1.2.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(dataset_pipeline_router, prefix="/api", tags=["Dataset Pipeline"])
app.include_router(dashboard_router, prefix="/api", tags=["Dashboard"])
app.include_router(prediction_router, prefix="/api", tags=["Prediction"])
app.include_router(forecasting_router, prefix="/api", tags=["Forecasting"])
app.include_router(anomaly_router, prefix="/api", tags=["Anomaly & Bayes"])
app.include_router(recommendation_router, prefix="/api", tags=["Q-Agent"])
app.include_router(simulation_router, prefix="/api", tags=["Simulation Engine"])

@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify backend service status."""
    return {"status": "healthy", "project": "AI WaterWise", "version": "1.1.0"}

# Mount Frontend static files at root / so index.html, style.css, script.js, chart.min.js load seamlessly
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
