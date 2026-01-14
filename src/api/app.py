from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import sys
import os
import uvicorn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict import MaternalRiskPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Maternal Health Risk Prediction API",
    description="REST API for predicting maternal health risk levels",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded on startup)
predictor = None


# Pydantic models for request/response validation
class PatientData(BaseModel):
    # Patient input data with validation constraints
    Age: float = Field(..., ge=10, le=100, description="Patient age in years")
    SystolicBP: float = Field(..., ge=70, le=200, description="Systolic blood pressure in mmHg")
    DiastolicBP: float = Field(..., ge=40, le=120, description="Diastolic blood pressure in mmHg")
    BS: float = Field(..., ge=3.0, le=25.0, description="Blood sugar level in mmol/L")
    BodyTemp: float = Field(..., ge=95.0, le=105.0, description="Body temperature in Fahrenheit")
    HeartRate: float = Field(..., ge=40, le=150, description="Heart rate in beats per minute")
    
    class Config:
        schema_extra = {
            "example": {
                "Age": 25,
                "SystolicBP": 120,
                "DiastolicBP": 80,
                "BS": 7.5,
                "BodyTemp": 98.6,
                "HeartRate": 72
            }
        }


class PredictionResponse(BaseModel):
    # Prediction result structure
    risk_level: str
    risk_level_numeric: int
    confidence: float
    probabilities: Dict[str, float]
    interpretation: str


class BatchPatientData(BaseModel):
    # Batch prediction input
    patients: List[PatientData]


class HealthResponse(BaseModel):
    # Health check response structure
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    # Load model when API starts
    global predictor
    try:
        predictor = MaternalRiskPredictor(model_type='best')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("⚠ Warning: Model not found. Please train models first.")
        predictor = None


@app.get("/", response_model=Dict[str, str])
async def root():
    # API root endpoint
    return {
        "message": "Maternal Health Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Health check endpoint
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    # Predict risk level for single patient
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure models are trained and available."
        )
    
    try:
        patient_data = patient.dict()
        result = predictor.predict(patient_data)
        interpretation = predictor.get_risk_interpretation(result)
        result['interpretation'] = interpretation
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(data: BatchPatientData):
    # Predict risk levels for multiple patients at once
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure models are trained and available."
        )
    
    try:
        # Convert to DataFrame for batch processing
        patients_data = [patient.dict() for patient in data.patients]
        import pandas as pd
        df = pd.DataFrame(patients_data)
        
        results = predictor.predict_batch(df)
        
        # Add interpretation to each result
        for result in results:
            result['interpretation'] = predictor.get_risk_interpretation(result)
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    # Get information about loaded model
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "best",
        "is_keras": predictor.is_keras,
        "features": predictor.feature_names,
        "risk_levels": list(predictor.risk_labels.values())
    }


@app.get("/features")
async def get_features():
    # Get list of required input features
    return {
        "features": [
            {
                "name": "Age",
                "description": "Patient age in years",
                "min": 10,
                "max": 100,
                "unit": "years"
            },
            {
                "name": "SystolicBP",
                "description": "Systolic blood pressure",
                "min": 70,
                "max": 200,
                "unit": "mmHg"
            },
            {
                "name": "DiastolicBP",
                "description": "Diastolic blood pressure",
                "min": 40,
                "max": 120,
                "unit": "mmHg"
            },
            {
                "name": "BS",
                "description": "Blood sugar level",
                "min": 3.0,
                "max": 25.0,
                "unit": "mmol/L"
            },
            {
                "name": "BodyTemp",
                "description": "Body temperature",
                "min": 95.0,
                "max": 105.0,
                "unit": "°F"
            },
            {
                "name": "HeartRate",
                "description": "Heart rate",
                "min": 40,
                "max": 150,
                "unit": "bpm"
            }
        ]
    }


if __name__ == "__main__":
    # Start API server with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

