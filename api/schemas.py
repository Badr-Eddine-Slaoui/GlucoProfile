from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20, description="Nombre de grossesses")
    Glucose: float = Field(..., gt=0, le=600, description="Glycémie (mg/dL)")
    BloodPressure: float = Field(..., gt=0, le=250, description="Pression artérielle diastolique (mm Hg)")
    SkinThickness: float = Field(..., gt=0, le=120, description="Épaisseur du pli cutané (mm)")
    Insulin: float = Field(..., gt=0, le=1500, description="Insuline sérique à 2 h (µU/mL)")
    BMI: float = Field(..., gt=0, le=80, description="Indice de masse corporelle (kg/m²)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=5, description="Fonction de pedigree du diabète")
    Age: float = Field(..., gt=0, le=120, description="Âge (années)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 219,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50,
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class ThresholdAlert(BaseModel):
    feature: str
    value: float
    threshold: float
    message: str


class PredictionResponse(BaseModel):
    cluster: int = Field(..., description="Cluster prédit par le classifieur")
    risk_category: str = Field(..., description="Libellé : 'Risque Élevé' ou 'Risque Faible'")
    risk_level: Literal["high", "low"]
    badge_color: Literal["red", "green"]
    risk_probability: float = Field(..., ge=0, le=1, description="Probabilité d'appartenir au groupe à risque élevé")
    confidence: float = Field(..., ge=0, le=1, description="Confiance du modèle dans la classe prédite")
    alerts: list[ThresholdAlert] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    model_name: str
    model_version: str
    predicted_at: datetime


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    stage: str
    model_uri: str
    run_id: str | None = None
    features: list[str]
    cluster_mapping: dict[str, str]
    loaded_at: datetime
    tracking_uri: str


class ErrorResponse(BaseModel):
    detail: str