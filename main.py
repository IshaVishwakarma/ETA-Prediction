from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Taxi ETA Prediction API",
    description="Predicts taxi trip duration in seconds using an XGBoost model trained on NYC Yellow Taxi 2019 data.",
    version="1.0.0",
)

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "eta_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run the training notebook first.")

# ── Schemas ───────────────────────────────────────────────────────────────────

class TripInput(BaseModel):
    trip_distance: float = Field(..., ge=0, description="Trip distance in miles")
    hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Saturday or Sunday, else 0")
    is_rush_hour: int = Field(..., ge=0, le=1, description="1 if 8–9 AM or 5–7 PM, else 0")
    pickup_hour_traffic: float = Field(..., ge=0, description="Number of trips from this zone at this hour (busyness proxy)")

    @validator("is_weekend", always=True)
    def validate_weekend(cls, v, values):
        dow = values.get("day_of_week")
        if dow is not None:
            expected = 1 if dow in (5, 6) else 0
            if v != expected:
                raise ValueError(
                    f"is_weekend={v} is inconsistent with day_of_week={dow}. "
                    f"Expected is_weekend={expected}."
                )
        return v

    class Config:
        schema_extra = {
            "example": {
                "trip_distance": 2.5,
                "hour": 17,
                "day_of_week": 1,
                "is_weekend": 0,
                "is_rush_hour": 1,
                "pickup_hour_traffic": 45000,
            }
        }


class PredictionResponse(BaseModel):
    predicted_duration_seconds: float = Field(..., description="Predicted trip duration in seconds")
    predicted_duration_minutes: float = Field(..., description="Predicted trip duration in minutes (rounded to 1 decimal)")
    input_summary: dict = Field(..., description="Echo of the features used for prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """API root — points to docs."""
    return {
        "message": "NYC Taxi ETA Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Health check — confirms the model is loaded and ready."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(trip: TripInput):
    """
    Predict the duration of a NYC taxi trip.

    Pass trip features and receive a predicted duration in seconds and minutes.
    """
    features = np.array([[
        trip.trip_distance,
        trip.hour,
        trip.day_of_week,
        trip.is_weekend,
        trip.is_rush_hour,
        trip.pickup_hour_traffic,
    ]])

    try:
        prediction = float(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Clamp to reasonable range (60s–7200s) — same bounds used during training
    prediction = max(60.0, min(prediction, 7200.0))

    return {
        "predicted_duration_seconds": round(prediction, 1),
        "predicted_duration_minutes": round(prediction / 60, 1),
        "input_summary": trip.dict(),
    }


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
def predict_batch(trips: list[TripInput]):
    """
    Predict trip durations for multiple trips in one request (max 100).
    """
    if len(trips) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 trips.")

    features = np.array([[
        t.trip_distance, t.hour, t.day_of_week,
        t.is_weekend, t.is_rush_hour, t.pickup_hour_traffic
    ] for t in trips])

    try:
        predictions = model.predict(features).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    results = []
    for trip, pred in zip(trips, predictions):
        pred = max(60.0, min(float(pred), 7200.0))
        results.append({
            "predicted_duration_seconds": round(pred, 1),
            "predicted_duration_minutes": round(pred / 60, 1),
            "input_summary": trip.dict(),
        })

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
